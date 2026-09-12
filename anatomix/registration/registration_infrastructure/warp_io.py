"""Warp moving volumes onto the fixed grid and export fixed-to-moving transforms."""
import os

import numpy as np
import SimpleITK as sitk
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
from .ome_zarr import is_ome_zarr, open_ome_zarr


def load_image(path, device, is_segmentation=False):
    """Load an image on device, including its physical-coordinate matrices."""
    if is_ome_zarr(path):
        return Image(open_ome_zarr(path).read_image(), device=device,
                     is_segmentation=is_segmentation)
    return Image.load_file(path, device=device, is_segmentation=is_segmentation)


def as_batch(image):
    """Wrap a single :class:`Image` in a :class:`BatchedImages`."""
    return BatchedImages([image])


def array_spacing(image):
    """Voxel spacing in tensor array order (z,y,x), reversed from ITK (x,y,z)."""
    return tuple(float(s) for s in reversed(image.itk_image.GetSpacing()))


def warp_volume(moving_tensor, grid, mode):
    """Sample a (1,C,Z,Y,X) tensor with a normalized (1,Z,Y,X,3) grid.

    Use bilinear for intensities, nearest for labels. Out-of-FOV samples use zero
    padding; coordinates follow align_corners=True."""
    return torch_grid_sampler_3d(
        moving_tensor, grid=grid, mode=mode,
        padding_mode="zeros", align_corners=True, is_displacement=False,
    )


def write_on_geometry(tensor, reference_batch, out_path):
    """Write ``tensor`` to ``out_path`` on ``reference_batch``'s geometry."""
    FakeBatchedImages(tensor, reference_batch).write_image(out_path)


_RAS_LPS_4x4 = np.diag([-1.0, -1.0, 1.0, 1.0])


def _lta_vox2ras(geometry):
    """Scanner vox2ras of an LTA volume-info block (FreeSurfer's MRIxfmCRS2XYZ)."""
    axes = np.stack([geometry["xras"], geometry["yras"], geometry["zras"]], axis=1)
    matrix = np.eye(4)
    matrix[:3, :3] = axes * geometry["voxelsize"][None, :]
    matrix[:3, 3] = geometry["cras"] - matrix[:3, :3] @ (geometry["volume"] / 2.0)
    return matrix


def _read_lta(path):
    """Return (src-to-dst RAS matrix, src geometry, dst geometry) of a FreeSurfer LTA."""
    lines = [line.split("#")[0].strip() for line in open(path)]
    lines = [line for line in lines if line]
    values = {}
    geometry = {"src": {}, "dst": {}}
    block = None
    matrix_rows = []
    for index, line in enumerate(lines):
        if line in ("src volume info", "dst volume info"):
            block = line[:3]
            continue
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            target = geometry[block] if block else values
            target[key] = value
        elif line == "1 4 4":
            matrix_rows = [np.array(row.split(), dtype=float) for row in lines[index + 1:index + 5]]
    if len(matrix_rows) != 4:
        raise ValueError(f"{path}: no 4x4 matrix found.")
    matrix = np.stack(matrix_rows)
    kind = int(values.get("type", "-1"))
    for name in ("src", "dst"):
        block = geometry[name]
        block["valid"] = block.get("valid", "0").startswith("1")
        for key in ("volume", "voxelsize", "xras", "yras", "zras", "cras"):
            if key in block:
                block[key] = np.array(block[key].split(), dtype=float)
    if kind == 0:  # LINEAR_VOX_TO_VOX
        if not (geometry["src"]["valid"] and geometry["dst"]["valid"]):
            raise ValueError(f"{path}: vox-to-vox LTA without valid volume geometry.")
        matrix = _lta_vox2ras(geometry["dst"]) @ matrix @ np.linalg.inv(_lta_vox2ras(geometry["src"]))
    elif kind != 1:  # LINEAR_RAS_TO_RAS
        raise ValueError(f"{path}: unsupported LTA type {kind}; use RAS-to-RAS or vox-to-vox.")
    return matrix, geometry["src"], geometry["dst"]


def _lta_geometry_matches(block, image):
    """Does an LTA volume-info block describe ``image`` = (shape_xyz, RAS affine)?"""
    if image is None or not block.get("valid"):
        return False
    shape, affine = image
    if not np.array_equal(block["volume"], np.array(shape, dtype=float)):
        return False
    center = affine[:3, :3] @ (np.array(shape, dtype=float) / 2.0) + affine[:3, 3]
    return bool(np.allclose(block["cras"], center, atol=0.5))


def read_linear_transform(path, fixed=None, moving=None):
    """Read a linear transform file as a (1, 4, 4) fixed-to-moving LPS-mm matrix.

    ITK/ANTs files (.mat/.txt/.tfm/.h5, e.g. ANTs' ``0GenericAffine.mat``)
    already map fixed to moving. FreeSurfer ``.lta`` files map their ``src``
    volume to their ``dst`` volume in RAS; ``fixed``/``moving`` are
    ``(shape_xyz, RAS affine)`` pairs used to tell which of them is ``src``.
    Without a match the file is taken as ``mri_coreg --mov moving --ref fixed``
    writes it (``src`` = moving) and inverted."""
    if path.endswith(".lta"):
        ras, src, dst = _read_lta(path)
        lps = _RAS_LPS_4x4 @ ras @ _RAS_LPS_4x4
        src_is_fixed = (
            _lta_geometry_matches(src, fixed) and _lta_geometry_matches(dst, moving)
            and not _lta_geometry_matches(src, moving)
        )
        matrix = lps if src_is_fixed else np.linalg.inv(lps)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-8:
            raise ValueError(f"{path}: transform is singular or non-finite.")
        return torch.tensor(matrix, dtype=torch.float32)[None]
    transform = sitk.ReadTransform(path)
    origin = np.array(transform.TransformPoint((0.0, 0.0, 0.0)))
    columns = [
        np.array(transform.TransformPoint(tuple(axis))) - origin
        for axis in np.eye(3)
    ]
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack(columns, axis=1)
    matrix[:3, 3] = origin
    probe = np.array([3.0, -7.0, 11.0])
    predicted = matrix[:3, :3] @ probe + origin
    if not np.allclose(predicted, transform.TransformPoint(tuple(probe)), atol=1e-3):
        raise ValueError(f"{path}: not a linear transform.")
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-8:
        raise ValueError(f"{path}: transform is singular or non-finite.")
    return torch.tensor(matrix, dtype=torch.float32)[None]


# ITK/SimpleITK physical space is LPS; nibabel/NIfTI world coordinates are RAS.
_RAS_TO_LPS = (-1.0, -1.0, 1.0)


def _apply_affine(matrix, points):
    """Apply a ``(1, 4, 4)`` homogeneous matrix to ``(K, 3)`` points."""
    matrix = matrix[0].to(points.dtype)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def keypoints_to_physical(points, convention, image):
    """Convert (K,3) LPS/RAS mm or voxel (i,j,k) points to ITK/LPS mm."""
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
    """Map fixed LPS-mm points through the grid to moving LPS-mm points.

    Points outside the fixed FOV clamp to the grid border."""
    normalized = _apply_affine(
        fixed_image.phy2torch, points_phys.to(grid.dtype))
    field = grid.permute(0, 4, 1, 2, 3)
    query = normalized.reshape(1, 1, 1, -1, 3)
    sampled = F.grid_sample(
        field, query, mode="bilinear", padding_mode="border",
        align_corners=True,
    )
    moved = sampled.reshape(3, -1).T
    return _apply_affine(moving_image.torch2phy, moved)


def _physical_points(matrix, grid):
    """Apply a (1,4,4) matrix to a (1,Z,Y,X,3) grid of points."""
    linear = matrix[0, :3, :3].to(grid.dtype)
    return grid @ linear.T + matrix[0, :3, 3].to(grid.dtype)


def _identity_grid(shape, device, dtype):
    eye = torch.eye(3, 4, device=device, dtype=dtype)[None]
    return F.affine_grid(eye, [1, 1] + list(shape), align_corners=True)


def invert_grid(grid, fixed_images, moving_images, iterations=500, damping=0.25, tol_mm=2e-4):
    """Invert a fixed-to-moving sampling grid by damped fixed-point iteration in physical space.

    Returns a moving-to-fixed grid (1,Zm,Ym,Xm,3) on the moving image grid,
    holding normalized fixed coordinates, and the per-voxel inverse-consistency
    residual |T(T^-1(y)) - y| in mm (non-finite where the preimage leaves the
    fixed FOV). Converges for fold-free fields; folds leave a large residual."""
    fixed_t2p = fixed_images.get_torch2phy()
    fixed_p2t = fixed_images.get_phy2torch()
    moving_t2p = moving_images.get_torch2phy()
    # Forward displacement in mm on the fixed grid.
    fixed_phys = _physical_points(fixed_t2p, _identity_grid(grid.shape[1:-1], grid.device, grid.dtype))
    forward = _physical_points(moving_t2p, grid) - fixed_phys
    forward = forward.permute(0, 4, 1, 2, 3).contiguous()
    # Solve v(y) = -u(y + v(y)) on the moving grid; damping keeps large
    # deformations from oscillating.
    target = _physical_points(moving_t2p, _identity_grid(moving_images.shape[2:], grid.device, grid.dtype))
    inverse = torch.zeros_like(target)
    for _ in range(iterations):
        query = _physical_points(fixed_p2t, target + inverse)
        sampled = F.grid_sample(
            forward, query, mode="bilinear", padding_mode="border", align_corners=True,
        ).permute(0, 2, 3, 4, 1)
        update = damping * (inverse + sampled)
        inverse = inverse - update
        if update.norm(dim=-1).max() < tol_mm:
            break
    grid_inv = _physical_points(fixed_p2t, target + inverse)
    # Residual: push the inverse points back through the forward grid.
    roundtrip = F.grid_sample(
        grid.permute(0, 4, 1, 2, 3), grid_inv, mode="bilinear",
        padding_mode="border", align_corners=True,
    ).permute(0, 2, 3, 4, 1)
    residual = (_physical_points(moving_t2p, roundtrip) - target).norm(dim=-1)
    inside = (grid_inv.abs() <= 1).all(dim=-1)
    residual = torch.where(inside, residual, torch.full_like(residual, float("nan")))
    return grid_inv, residual


class _CumulativeWarp(DeformableMixin):
    """Adapt a cumulative sampling grid to FireANTs displacement-field export."""

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
        displacement = (self._grid - identity).contiguous()
        return {"affine": eye, "grid": displacement}

    def get_inverse_warp_parameters(self, *args, **kwargs):
        raise NotImplementedError(
            "Inverse cumulative-transform export is not supported."
        )

    @torch.no_grad()
    def save_as_scipy_transforms(self, filename):
        """Save voxel offsets: moving_index = fixed_index + arr_0[x,y,z]."""
        grid = self._grid[0].cpu().numpy().transpose(2, 1, 0, 3)
        moving_size = np.asarray(self.moving_images.shape[2:][::-1])
        moving_index = (grid + 1) * (moving_size - 1) / 2
        fixed_index = np.moveaxis(np.indices(grid.shape[:3]), 0, -1)
        np.savez(filename, (moving_index - fixed_index).astype(np.float32))


class _CumulativeLinear:
    """Adapt a cumulative physical matrix to FireANTs' ANTs .mat writer."""

    save_as_ants_transforms = AffineRegistration.save_as_ants_transforms

    def __init__(self, matrix):
        self.matrix = matrix
        self.dims = matrix.shape[-1] - 1

    def get_affine_matrix(self, homogenous=True):
        return self.matrix if homogenous else self.matrix[:, :self.dims]


def _save_one(stage, grid, convention, base, fixed_images, moving_images):
    """Export one cumulative snapshot; linear stages use their composed physical matrix."""
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

    if convention == "ants":
        _CumulativeLinear(stage.linear_matrix).save_as_ants_transforms(
            base + ".mat")
    else:
        np.savez(
            base + ".npz", affine=stage.linear_matrix.detach().cpu().numpy())


def save_inverse_transform(result, grid_inv, convention, out_dir, prefix, stem):
    """Write the moving-to-fixed transform on the moving grid; returns the basename."""
    base = os.path.join(out_dir, f"{prefix}inverse-warp-{stem}")
    last = result.stages[-1]
    if result.num_deformable == 0:
        matrix = torch.linalg.inv(last.linear_matrix.double()).to(last.linear_matrix.dtype)
        stage = last._replace(linear_matrix=matrix)
    else:
        stage = last
    # Roles are swapped: the inverse lives on the moving grid and maps into the fixed image.
    _save_one(stage, grid_inv, convention, base, result.moving_images, result.fixed_images)
    return base


def save_transforms(
    result, convention, collapse, out_dir, prefix, stem,
):
    """Write one cumulative transform, or one cumulative snapshot per stage.

    Returns basenames without extensions. Snapshots include the initialization
    when requested; they must not be composed again."""
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
