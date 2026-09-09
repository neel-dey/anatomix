"""Registration quality metrics: label Dice, keypoint TRE, and fold count."""
import numpy as np
import torch
from sklearn.metrics import f1_score

from ._fireants import jacobian


def dice_score(fixed_seg, moved_seg):
    """Mean Dice over nonzero labels present in fixed_seg; NaN if none exist."""
    gt = np.asarray(fixed_seg).astype(np.int64).flatten()
    pred = np.asarray(moved_seg).astype(np.int64).flatten()
    labels = [int(v) for v in np.unique(gt) if v != 0]
    if not labels:
        return float("nan")
    return float(
        f1_score(gt, pred, labels=labels, average="macro", zero_division=0)
    )


def keypoint_metrics(warped, target, source):
    """Compute TRE in mm and fraction improved for corresponding (K,3) point arrays.

    Source and target must share a physical frame; correspondence is by row."""
    warped = np.asarray(warped, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    final = np.linalg.norm(warped - target, axis=1)
    initial = np.linalg.norm(source - target, axis=1)
    return {
        "tre_median": float(np.median(final)),
        "tre_mean": float(final.mean()),
        "tre_initial_median": float(np.median(initial)),
        "robustness": float((final < initial).mean()),
    }


def count_folds(warped_coordinates, fixed_images=None, moving_images=None):
    """Count interior voxels with non-positive physical Jacobian determinant.

    Pass both image geometries when their orientations may differ. Without them,
    count in normalized coordinates (assumes matching handedness). Excludes the
    one-voxel border and rejects non-finite determinants."""
    jac = jacobian(warped_coordinates).permute(0, 2, 3, 4, 1, 5)
    jac = jac[:, 1:-1, 1:-1, 1:-1, :]
    det = torch.linalg.det(jac)
    if (fixed_images is None) != (moving_images is None):
        raise ValueError("Provide both fixed and moving geometries, or neither.")
    if fixed_images is not None:
        # Header handedness can reverse normalized axes without a physical fold.
        fixed_det = torch.linalg.det(fixed_images.get_torch2phy()[:, :3, :3])
        moving_det = torch.linalg.det(moving_images.get_torch2phy()[:, :3, :3])
        det = det * (moving_det / fixed_det)[:, None, None, None]
    if not torch.isfinite(det).all():
        raise ValueError("Transform has non-finite Jacobian determinants.")
    return int((det <= 0).sum().item())
