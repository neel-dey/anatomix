"""Registration quality metrics: label Dice and deformation fold count."""
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


def count_folds(warped_coordinates):
    """Count folded voxels (non-positive Jacobian determinant) in a warp.

    Parameters
    ----------
    warped_coordinates : torch.Tensor
        Cumulative sampling grid ``(1, H, W, D, 3)`` (normalized coordinates).

    Returns
    -------
    int
        Number of interior voxels whose Jacobian determinant is ``<= 0``. The
        one-voxel border is excluded before the determinant, following the
        reference implementation.
    """
    jac = jacobian(warped_coordinates).permute(0, 2, 3, 4, 1, 5)
    jac = jac[:, 1:-1, 1:-1, 1:-1, :]
    det = torch.linalg.det(jac)
    return int((det <= 0).sum().item())
