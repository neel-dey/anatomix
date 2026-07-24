"""Volume/label warping and transform export for the pipeline.

All warping uses the canonical cumulative sampling grid produced by
:mod:`~anatomix.registration.registration_infrastructure.register` (normalized
``[-1, 1]`` coordinates mapping fixed voxels to the original moving image), so
the original moving image and label are each resampled exactly once and written
on the fixed image's geometry. Pure filesystem helpers (CSV, naming) live in
:mod:`~anatomix.registration.registration_infrastructure.io_utils`.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from ._fireants import (
    BatchedImages,
    DeformableMixin,
    FakeBatchedImages,
    Image,
    torch_grid_sampler_3d,
)


def load_image(path, device, is_segmentation=False):
    """Load a NIfTI volume as a FireANTs :class:`Image` on ``device``."""
    return Image.load_file(path, is_segmentation=is_segmentation).to(device)


def as_batch(image):
    """Wrap a single :class:`Image` in a :class:`BatchedImages`."""
    return BatchedImages([image])


def array_spacing(image):
    """Voxel spacing of ``image`` in array-axis order ``(H, W, D)``.

    SimpleITK reports spacing in ``(x, y, z)`` while the array axes are
    ``(z, y, x)``, so the ITK spacing is reversed.
    """
    return tuple(float(s) for s in reversed(image.itk_image.GetSpacing()))


def warp_volume(moving_tensor, grid, mode):
    """Resample ``moving_tensor`` by a cumulative sampling ``grid``.

    Parameters
    ----------
    moving_tensor : torch.Tensor
        Original moving volume ``(1, 1, H, W, D)`` (float).
    grid : torch.Tensor
        Cumulative sampling grid ``(1, H, W, D, 3)`` (absolute normalized
        coordinates, not a displacement).
    mode : {'bilinear', 'nearest'}
        ``'bilinear'`` (trilinear in 3D) for intensities, ``'nearest'`` for
        labels.

    Returns
    -------
    torch.Tensor
        The moved volume on the fixed grid ``(1, 1, H, W, D)``.
    """
    return torch_grid_sampler_3d(
        moving_tensor, grid=grid, mode=mode,
        padding_mode="zeros", align_corners=True, is_displacement=False,
    )


def write_on_geometry(tensor, reference_batch, out_path):
    """Write ``tensor`` to ``out_path`` on ``reference_batch``'s geometry."""
    FakeBatchedImages(tensor, reference_batch).write_image(out_path)


class _CumulativeWarp(DeformableMixin):
    """Present the canonical cumulative grid as a FireANTs deformable transform.

    FireANTs' ANTs/SciPy writers serialize a single stage's own ``{affine,
    grid}``, where ``grid`` is a *displacement* and the full moving coordinates
    are ``affine_grid(affine) + grid`` (see ``torch_affine_warp_3d``). A
    *composed* multi-stage transform is not held by any single FireANTs object,
    so it is presented here as an identity affine plus the displacement ``W -
    identity_grid`` (``W`` = the canonical cumulative sampling grid). Substituting
    this into FireANTs' own conversion yields exactly the physical (ANTs) / voxel
    (SciPy) displacement it emits for a native single-stage warp -- the correct
    cumulative transform, in FireANTs' own conventions -- so composed chains
    export losslessly instead of dropping their linear component.
    """

    def __init__(self, grid, fixed_images, moving_images):
        self._grid = grid
        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.dims = grid.shape[-1]
        self.opt_size = 1
        self.dtype = grid.dtype

    def get_warp_parameters(self, fixed_images, moving_images, shape=None):
        n = self._grid.shape[0]
        eye = torch.zeros(
            n, self.dims, self.dims + 1,
            device=self._grid.device, dtype=self._grid.dtype,
        )
        for axis in range(self.dims):
            eye[:, axis, axis] = 1.0
        identity = F.affine_grid(
            eye, [n, 1] + list(self._grid.shape[1:-1]), align_corners=True,
        )
        # FireANTs' SciPy writer requires a contiguous grid; the cumulative grid
        # may be a non-contiguous view (e.g. from grid composition).
        displacement = (self._grid - identity).contiguous()
        return {"affine": eye, "grid": displacement}

    def get_inverse_warp_parameters(self, *args, **kwargs):
        raise NotImplementedError(
            "Inverse cumulative-transform export is not supported."
        )


def _linear_matrix(stage):
    """Homogeneous physical-space matrix of a linear/init stage."""
    if stage.name == "affine":
        return stage.registration.get_affine_matrix()
    if stage.name == "rigid":
        return stage.registration.get_rigid_matrix()
    return stage.registration.get_affine_init()  # moment initialization


def _save_one(stage, grid, convention, base, fixed_images, moving_images):
    """Save one stage's cumulative transform in the requested convention.

    ``grid`` is the cumulative sampling grid up to (and including) ``stage``. A
    *composed* deformable stage is exported through :class:`_CumulativeWarp` (its
    FireANTs object holds only the residual); every other stage is exported by
    its own FireANTs writer, which already holds the cumulative transform (a
    linear ``.mat``, or a native deformable displacement field).
    """
    if convention == "pytorch":
        torch.save(grid.detach().cpu(), base + ".pt")
        return
    if convention not in ("ants", "scipy"):
        raise ValueError(f"Unknown transform convention {convention!r}.")

    if stage.composed:
        warp = _CumulativeWarp(grid, fixed_images, moving_images)
        if convention == "ants":
            warp.save_as_ants_transforms(base + ".nii.gz")
        else:
            warp.save_as_scipy_transforms(base + ".npz")
        return

    if convention == "ants":
        ext = ".nii.gz" if stage.is_deformable else ".mat"
        stage.registration.save_as_ants_transforms(base + ext)
    elif stage.is_deformable:
        stage.registration.save_as_scipy_transforms(base + ".npz")
    else:
        matrix = _linear_matrix(stage).detach().cpu().numpy()
        np.savez(base + ".npz", affine=matrix)


def save_transforms(
    result, convention, collapse, out_dir, prefix, stem,
):
    """Export the registration transform(s).

    With ``collapse`` truthy a single cumulative transform is written; otherwise
    one cumulative snapshot is written per stage (and after a non-``none``
    initialization).

    All three conventions represent the full cumulative transform for every
    chain, including composed (linear->deformable and repeated-deformable) ones:
    ``pytorch`` saves the canonical sampling grid directly; ``ants``/``scipy``
    use FireANTs' writers, and a composed deformable stage is routed through
    :class:`_CumulativeWarp` so its linear/prior component is not dropped.

    Returns
    -------
    list of str
        The transform file basenames written (without extension).
    """
    fixed_images = result.fixed_images
    moving_images = result.moving_images

    written = []
    if collapse:
        base = os.path.join(out_dir, f"{prefix}warp-{stem}")
        _save_one(
            result.stages[-1], result.warped_coordinates, convention, base,
            fixed_images, moving_images,
        )
        written.append(base)
    else:
        for (label, grid), stage in zip(result.snapshots, result.stages):
            base = os.path.join(out_dir, f"{prefix}warp-{stem}-{label}")
            _save_one(stage, grid, convention, base,
                      fixed_images, moving_images)
            written.append(base)
    return written
