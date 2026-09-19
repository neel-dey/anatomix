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


_CHANNEL_ORDER = [6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]  # the reference C++ descriptor ordering


def MINDSSC(img, radius=1, dilation=2):
    """Return 12-channel MIND-SSC on the input tensor's device and dtype.

    Input shape is (1,1,Z,Y,X); radius controls patch size and dilation the
    neighbourhood offsets."""
    mind = _patch_distances(img, radius, dilation)
    return _normalize_distances(mind, torch.mean(mind, 1, keepdim=True).mean().item())


def MINDSSC_tiled(img, radius=1, dilation=2, tile=64, out_device="cpu"):
    """MIND-SSC computed in slabs of ``tile`` slices along z, collected on ``out_device``.

    Slabs overlap by the descriptor's reach, so the patch distances equal those
    of :func:`MINDSSC`; the volume-wide mean variance is gathered in a first pass."""
    reach = radius + dilation
    depth = img.shape[2]
    out = torch.empty((1, 12) + tuple(img.shape[2:]), dtype=img.dtype, device=out_device)
    variance_sum, slabs = 0.0, []
    for z0 in range(0, depth, tile):
        z1 = min(z0 + tile, depth)
        lo, hi = max(z0 - reach, 0), min(z1 + reach, depth)
        mind = _patch_distances(img[:, :, lo:hi], radius, dilation)[:, :, z0 - lo:z0 - lo + (z1 - z0)]
        variance_sum += torch.mean(mind, 1, keepdim=True).sum(dtype=torch.float64).item()
        out[:, :, z0:z1] = mind.to(out_device)
        slabs.append((z0, z1))
    variance_mean = variance_sum / (depth * img.shape[3] * img.shape[4])
    for z0, z1 in slabs:
        out[:, :, z0:z1] = _normalize_distances(
            out[:, :, z0:z1].to(img.device), variance_mean).to(out_device)
    return out


def _patch_distances(img, radius, dilation):
    """Patch distances to the twelve neighbour pairs, minus their per-voxel minimum."""
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

    return ssd - torch.min(ssd, 1, keepdim=True)[0]


def _normalize_distances(mind, variance_mean):
    """The MIND equation: per-voxel variance normalization, clamped around the volume mean."""
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, variance_mean * 0.001, variance_mean * 1000)
    mind /= mind_var
    mind = torch.exp(-mind)
    return mind[:, torch.tensor(_CHANNEL_ORDER, device=mind.device).long(), :, :, :]
