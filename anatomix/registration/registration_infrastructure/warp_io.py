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

from ._fireants import (
    BatchedImages,
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


def _linear_matrix(stage):
    """Homogeneous physical-space matrix of a linear/init stage."""
    if stage.name == "affine":
        return stage.registration.get_affine_matrix()
    if stage.name == "rigid":
        return stage.registration.get_rigid_matrix()
    return stage.registration.get_affine_init()  # moment initialization


def _save_one(stage, grid, convention, base):
    """Save one stage's cumulative transform in the requested convention."""
    if convention == "pytorch":
        torch.save(grid.detach().cpu(), base + ".pt")
    elif convention == "ants":
        if stage.is_deformable:
            stage.registration.save_as_ants_transforms(base + ".nii.gz")
        else:
            stage.registration.save_as_ants_transforms(base + ".mat")
    elif convention == "scipy":
        if stage.is_deformable:
            stage.registration.save_as_scipy_transforms(base + ".npz")
        else:
            matrix = _linear_matrix(stage).detach().cpu().numpy()
            np.savez(base + ".npz", affine=matrix)
    else:
        raise ValueError(f"Unknown transform convention {convention!r}.")


def save_transforms(
    result, convention, collapse, out_dir, prefix, stem, verbose=False,
):
    """Export the registration transform(s).

    With ``collapse`` truthy a single cumulative transform is written; otherwise
    one cumulative snapshot is written per stage (and after a non-``none``
    initialization). The ``pytorch`` convention always saves the exact composed
    sampling grid; ``ants``/``scipy`` use FireANTs' writers, which capture the
    full composed transform for chains with at most one deformable stage.

    Returns
    -------
    list of str
        The transform file basenames written (without extension).
    """
    if result.num_deformable > 1 and convention != "pytorch" and verbose:
        print(
            "  [warp] WARNING: more than one deformable stage; ants/scipy export "
            "captures only the final residual deformation. Use "
            "--output-transformation-convention pytorch for the exact composed "
            "transform.",
            flush=True,
        )

    written = []
    if collapse:
        base = os.path.join(out_dir, f"{prefix}warp-{stem}")
        _save_one(result.stages[-1], result.warped_coordinates, convention, base)
        written.append(base)
    else:
        for (label, grid), stage in zip(result.snapshots, result.stages):
            base = os.path.join(out_dir, f"{prefix}warp-{stem}-{label}")
            _save_one(stage, grid, convention, base)
            written.append(base)
    return written
