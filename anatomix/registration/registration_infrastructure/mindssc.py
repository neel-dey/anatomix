"""MIND-SSC descriptors (Heinrich et al., MICCAI 2013), adapted from ConvexAdam.

Tensors follow the input device; channel order matches the reference implementation."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist_squared(x):
    """Pairwise squared Euclidean distances between a set of points."""
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


def MINDSSC(img, radius=1, dilation=2):
    """Return 12-channel MIND-SSC on the input tensor's device and dtype.

    Input shape is (1,1,Z,Y,X); radius controls patch size and dilation the
    neighbourhood offsets."""
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

    # Scattered on CPU (with CPU index tensors), then moved to the input's
    # device, so the descriptor works on CPU and GPU alike.
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
