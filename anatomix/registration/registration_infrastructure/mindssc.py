"""Device-safe MIND-SSC descriptor for the FireANTs registration backend.

MIND-SSC (Modality-Independent Neighbourhood Descriptor -- Self-Similarity
Context) is a hand-crafted 12-channel local descriptor that is robust to
contrast differences and therefore well suited to multi-modal registration
(e.g. MR-to-CT). See Heinrich et al., MICCAI 2013
(http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf).

This is a self-contained copy of the descriptor used by the ConvexAdam backend
(``registration_backend/convexadam/convex_adam_utils.py``), with the only change
being that every tensor is allocated on the *input's* device instead of a
hard-coded ``.cuda()``. The numerical result is identical to the original.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist_squared(x):
    """Pairwise squared Euclidean distances between a set of points.

    Parameters
    ----------
    x : torch.Tensor
        Point coordinates of shape ``(1, dims, n_points)``.

    Returns
    -------
    torch.Tensor
        Matrix of pairwise squared distances of shape ``(1, n_points,
        n_points)``.
    """
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


def MINDSSC(img, radius=1, dilation=2):
    """Compute the 12-channel MIND-SSC descriptor of a 3D volume.

    Parameters
    ----------
    img : torch.Tensor
        Input volume of shape ``(1, 1, H, W, D)``. The descriptor is allocated
        on ``img.device`` and returned in ``img.dtype``.
    radius : int, optional
        Radius of the self-similarity patch (patch side is ``2 * radius + 1``).
        Default 1, matching the AbdomenMRCT reference pipeline.
    dilation : int, optional
        Dilation of the six-neighbourhood sampling pattern. Default 2.

    Returns
    -------
    torch.Tensor
        MIND-SSC descriptor of shape ``(1, 12, H, W, D)`` on ``img.device``.

    Notes
    -----
    The channel ordering matches the reference C++ implementation via the final
    permutation, so descriptors are interchangeable with the ConvexAdam backend
    and the published AbdomenMRCT results.
    """
    device = img.device
    dtype = img.dtype

    # Kernel size of the self-similarity patch.
    kernel_size = radius * 2 + 1

    # Six-neighbourhood used to define the self-similarity pattern.
    six_neighbourhood = torch.tensor([
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 2],
        [2, 1, 1],
        [1, 2, 1],
    ]).long()

    # Squared distances between neighbourhood offsets.
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # Comparison mask: keep the ordered pairs at squared distance 2.
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # Build the two shifted convolution kernels selecting the compared voxels.
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(
        1, 6, 1,
    ).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(
        6, 1, 1,
    ).view(-1, 3)[mask, :]

    # Kernels are scattered on CPU (with CPU index tensors) and then moved to the
    # input's device, keeping the computation device-agnostic.
    mshift1 = torch.zeros(12, 1, 3, 3, 3)
    mshift1.view(-1)[
        torch.arange(12) * 27
        + idx_shift1[:, 0] * 9
        + idx_shift1[:, 1] * 3
        + idx_shift1[:, 2]
    ] = 1
    mshift1 = mshift1.to(device=device, dtype=dtype)

    mshift2 = torch.zeros(12, 1, 3, 3, 3)
    mshift2.view(-1)[
        torch.arange(12) * 27
        + idx_shift2[:, 0] * 9
        + idx_shift2[:, 1] * 3
        + idx_shift2[:, 2]
    ] = 1
    mshift2 = mshift2.to(device=device, dtype=dtype)

    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # Patch sum-of-squared-differences between the shifted samples.
    img_padded = rpad1(img)
    conv1 = F.conv3d(img_padded, mshift1, dilation=dilation)
    conv2 = F.conv3d(img_padded, mshift2, dilation=dilation)
    ssd = F.avg_pool3d(
        rpad2((conv1 - conv2) ** 2),
        kernel_size,
        stride=1,
    )

    # MIND equation with per-voxel variance normalization.
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(
        mind_var,
        mind_var.mean().item() * 0.001,
        mind_var.mean().item() * 1000,
    )
    mind /= mind_var
    mind = torch.exp(-mind)

    # Permute channels to match the reference C++ descriptor ordering.
    mind = mind[
        :,
        torch.tensor(
            [6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], device=device,
        ).long(),
        :, :, :,
    ]

    return mind
