"""Registration quality metrics: label Dice, keypoint TRE, and fold count."""
import numpy as np
import torch
from sklearn.metrics import f1_score

from ._fireants import jacobian


def dice_score(fixed_seg, moved_seg):
    """Macro-averaged Dice between a fixed and a warped moving segmentation.

    Parameters
    ----------
    fixed_seg, moved_seg : array_like
        Integer label volumes of identical shape (on the fixed grid). Dice is
        computed over the nonzero labels present in ``fixed_seg`` (background
        label 0 is excluded).

    Returns
    -------
    float
        Macro-averaged Dice (``sklearn.metrics.f1_score`` with
        ``average='macro'``), or ``nan`` if ``fixed_seg`` has no foreground
        label.
    """
    gt = np.asarray(fixed_seg).astype(np.int64).flatten()
    pred = np.asarray(moved_seg).astype(np.int64).flatten()
    labels = [int(v) for v in np.unique(gt) if v != 0]
    if not labels:
        return float("nan")
    return float(
        f1_score(gt, pred, labels=labels, average="macro", zero_division=0)
    )


def keypoint_metrics(warped, target, source):
    """Target-registration-error statistics for one keypoint set.

    Parameters
    ----------
    warped, target, source : array_like
        Corresponding physical (mm) coordinates ``(K, 3)``: the fixed keypoints
        mapped through the transform, the ground-truth moving keypoints, and
        the fixed keypoints before warping. The initial error compares the last
        two directly, so it assumes both images share a world frame.

    Returns
    -------
    dict
        ``tre_median``, ``tre_mean``, ``tre_initial_median`` (mm) and
        ``robustness``, the fraction of keypoints whose error decreased.
    """
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


def count_folds(warped_coordinates):
    """Count folded voxels (non-positive Jacobian determinant) in a warp.

    Parameters
    ----------
    warped_coordinates : torch.Tensor
        Cumulative sampling grid ``(1, H, W, D, 3)`` (normalized coordinates).

    Returns
    -------
    int
        Number of interior voxels whose Jacobian determinant is ``<= 0``; the
        one-voxel border is excluded before the determinant.
    """
    jac = jacobian(warped_coordinates).permute(0, 2, 3, 4, 1, 5)
    jac = jac[:, 1:-1, 1:-1, 1:-1, :]
    det = torch.linalg.det(jac)
    return int((det <= 0).sum().item())
