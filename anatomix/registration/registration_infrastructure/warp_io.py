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
    AffineRegistration,
    BatchedImages,
    DeformableMixin,
    FakeBatchedImages,
    Image,
    torch_grid_sampler_3d,
)
from .io_utils import KEYPOINT_CONVENTIONS


def load_image(path, device, is_segmentation=False):
    """Load a NIfTI volume as a FireANTs :class:`Image` on ``device``.

    ``device`` goes to the constructor rather than a later ``.to()``, which
    moves only ``.array`` and would leave the coordinate matrices
    (``torch2phy``, ``phy2torch``, ...) on FireANTs' default ``'cuda'``.
    """
    return Image.load_file(path, device=device, is_segmentation=is_segmentation)


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


# Keypoints. FireANTs' Image supplies the coordinate matrices: phy2torch maps an
# ITK/LPS point to the normalized [-1, 1] coordinates that index the sampling
# grid, torch2phy inverts it, and both use ITK (i, j, k) order.
# nibabel/NIfTI world coordinates are RAS; ITK/SimpleITK physical space is LPS.
_RAS_TO_LPS = (-1.0, -1.0, 1.0)


def _apply_affine(matrix, points):
    """Apply a ``(1, 4, 4)`` homogeneous matrix to ``(K, 3)`` points."""
    matrix = matrix[0].to(points.dtype)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def keypoints_to_physical(points, convention, image):
    """Convert keypoints from ``convention`` to ITK/LPS physical coordinates.

    Parameters
    ----------
    points : torch.Tensor
        Keypoints ``(K, 3)`` in ``convention``, on ``image``'s device.
    convention : {'lps', 'ras', 'voxel'}
        World coordinates in mm, or the ITK continuous index ``(i, j, k)`` for
        ``'voxel'`` (the reverse of the NumPy array axes).
    image : fireants.io.Image
        The image the keypoints belong to.

    Returns
    -------
    torch.Tensor
        Keypoints ``(K, 3)`` in ITK/LPS physical coordinates.
    """
    if convention == "lps":
        return points
    if convention == "ras":
        return points * torch.tensor(
            _RAS_TO_LPS, device=points.device, dtype=points.dtype,
        )
    if convention == "voxel":
        return _apply_affine(image.px2phy, points)
    raise ValueError(
        f"keypoint convention must be one of {KEYPOINT_CONVENTIONS}, "
        f"got {convention!r}."
    )


def keypoints_from_physical(points, convention, image):
    """Inverse of :func:`keypoints_to_physical`."""
    if convention == "lps":
        return points
    if convention == "ras":
        return points * torch.tensor(
            _RAS_TO_LPS, device=points.device, dtype=points.dtype,
        )
    if convention == "voxel":
        return _apply_affine(image.phy2px, points)
    raise ValueError(
        f"keypoint convention must be one of {KEYPOINT_CONVENTIONS}, "
        f"got {convention!r}."
    )


def warp_keypoints(points_phys, grid, fixed_image, moving_image):
    """Map fixed-image keypoints through a cumulative grid into moving space.

    Parameters
    ----------
    points_phys : torch.Tensor
        Fixed-image keypoints ``(K, 3)`` in ITK/LPS physical coordinates.
    grid : torch.Tensor
        Cumulative sampling grid ``(1, H, W, D, 3)``: absolute normalized
        moving-image coordinates indexed on the fixed grid, so it transports
        *fixed*-image points the same way the moving image is resampled onto
        the fixed grid.
    fixed_image, moving_image : fireants.io.Image
        The pair being registered, for their coordinate matrices.

    Returns
    -------
    torch.Tensor
        The warped keypoints ``(K, 3)`` in the moving image's ITK/LPS physical
        coordinates. Points outside the fixed field of view are clamped to the
        grid border.
    """
    normalized = _apply_affine(
        fixed_image.phy2torch, points_phys.to(grid.dtype))
    # grid_sample wants the coordinate field channels-first and the (continuous)
    # keypoint queries as a (1, 1, 1, K, 3) grid.
    field = grid.permute(0, 4, 1, 2, 3)
    query = normalized.reshape(1, 1, 1, -1, 3)
    sampled = F.grid_sample(
        field, query, mode="bilinear", padding_mode="border",
        align_corners=True,
    )
    moved = sampled.reshape(3, -1).T
    return _apply_affine(moving_image.torch2phy, moved)


class _CumulativeWarp(DeformableMixin):
    """Present the canonical cumulative grid as a FireANTs deformable transform.

    FireANTs' ANTs/SciPy writers serialize a single stage's own ``{affine,
    grid}``, where ``grid`` is a *displacement* and the moving coordinates are
    ``affine_grid(affine) + grid``. No single FireANTs object holds a composed
    multi-stage transform, so it is presented here as an identity affine plus
    the displacement ``W - identity_grid`` (``W`` = the cumulative sampling
    grid). Fed through FireANTs' own conversion that yields exactly the
    displacement it writes for a native single-stage warp, so composed chains
    export without dropping their linear component.
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


class _CumulativeLinear:
    """Present a composed linear matrix to FireANTs' ANTs ``.mat`` writer.

    A warm-started rigid/affine stage optimizes a *residual* against
    re-extracted features, so its FireANTs object no longer holds the
    cumulative matrix. FireANTs' rigid, affine and moments ANTs writers are the
    same routine and read only ``get_affine_matrix`` and ``dims``, so supplying
    those two reuses it verbatim rather than restating the ``.mat`` layout.
    """

    save_as_ants_transforms = AffineRegistration.save_as_ants_transforms

    def __init__(self, matrix):
        self.matrix = matrix
        self.dims = matrix.shape[-1] - 1

    def get_affine_matrix(self, homogenous=True):
        return self.matrix if homogenous else self.matrix[:, :self.dims]


def _save_one(stage, grid, convention, base, fixed_images, moving_images):
    """Save one stage's cumulative transform in the requested convention.

    ``grid`` is the cumulative sampling grid up to (and including) ``stage``.
    Every deformable stage is exported through :class:`_CumulativeWarp`, which
    carries the cumulative transform and the pair's real :class:`BatchedImages`;
    a stage's own FireANTs object holds only a residual, and in a partly masked
    chain its images are a ``FakeBatchedImages`` that FireANTs' ANTs writer
    cannot read geometry from. Linear stages export their cumulative matrix
    (``stage.linear_matrix``) instead, which needs no image geometry; the moment
    initialization has no residual to compose and keeps its native writer.
    """
    if convention == "pytorch":
        torch.save(grid.detach().cpu(), base + ".pt")
        return
    if convention not in ("ants", "scipy"):
        raise ValueError(f"Unknown transform convention {convention!r}.")

    if stage.is_deformable:
        warp = _CumulativeWarp(grid, fixed_images, moving_images)
        if convention == "ants":
            warp.save_as_ants_transforms(base + ".nii.gz")
        else:
            warp.save_as_scipy_transforms(base + ".npz")
        return

    if stage.linear_matrix is None:  # moment initialization
        if convention == "ants":
            stage.registration.save_as_ants_transforms(base + ".mat")
        else:
            matrix = stage.registration.get_affine_init()
            np.savez(base + ".npz", affine=matrix.detach().cpu().numpy())
        return

    if convention == "ants":
        _CumulativeLinear(stage.linear_matrix).save_as_ants_transforms(
            base + ".mat")
    else:
        np.savez(
            base + ".npz", affine=stage.linear_matrix.detach().cpu().numpy())


def save_transforms(
    result, convention, collapse, out_dir, prefix, stem,
):
    """Export the registration transform(s).

    With ``collapse`` truthy a single cumulative transform is written; otherwise
    one cumulative snapshot is written per stage (and after a non-``none``
    initialization).

    All three conventions carry the full cumulative transform for every chain,
    composed ones included: ``pytorch`` saves the sampling grid directly, while
    ``ants``/``scipy`` use FireANTs' writers with composed deformable stages
    routed through :class:`_CumulativeWarp`.

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
