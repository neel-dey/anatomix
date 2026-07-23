"""Feature preparation for the FireANTs registration backend.

This module turns a pair of 3D volumes into the multi-channel feature tensors
that FireANTs registers. The steps mirror the AbdomenMRCT reference pipeline:

1. per-image intensity clip + min-max normalization to ``[0, 1]``;
2. (optionally) resample to an isotropic grid at the finest voxel spacing before
   feature extraction, then resample the features back to the original grid;
3. dense anatomix network features via MONAI sliding-window inference;
4. per-voxel feature normalization (L2 / standardized / none) of the *network*
   features only;
5. a hand-crafted MIND-SSC descriptor (unnormalized);
6. optional masking of the network features and concatenation of the selected
   feature families.

Everything here operates on plain ``torch`` tensors and the anatomix models; no
FireANTs import is required.
"""
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from anatomix.model.load_from_hf import (
    ANATOMIX_VARIANTS,
    load_from_hf,
    _load_handling_compile,
)
from anatomix.model.network import Unet

from .mindssc import MINDSSC


def load_backbone(
    backbone,
    device,
    *,
    custom_arch=None,
    custom_weights=None,
    unet_kwargs=None,
    vit_kwargs=None,
):
    """Load a pretrained anatomix feature extractor in frozen eval mode.

    Parameters
    ----------
    backbone : {'anatomix', 'anatomix-dev', 'anatomix-dev-vit', 'custom'}
        Which feature extractor to load. The first three are downloaded from the
        HuggingFace Hub via :func:`anatomix.model.load_from_hf.load_from_hf`.
    device : torch.device
        Device to move the model onto.
    custom_arch : {'unet', 'vit'}, optional
        Architecture to build when ``backbone == 'custom'``.
    custom_weights : str, optional
        Path to the ``.pth`` checkpoint for a custom backbone.
    unet_kwargs, vit_kwargs : dict, optional
        Constructor keyword arguments for the custom UNet / ViT (``dimension``
        and ``input_channels`` are fixed at 3 and 1 respectively).

    Returns
    -------
    model : torch.nn.Module
        The loaded model on ``device``, in ``eval`` mode with gradients
        disabled.
    """
    if backbone == "custom":
        if custom_arch == "unet":
            model = Unet(3, 1, **(unet_kwargs or {}))
        elif custom_arch == "vit":
            from anatomix.model.vit3d import PrimusV2

            model = PrimusV2(input_channels=1, **(vit_kwargs or {}))
        else:
            raise ValueError(
                f"custom_arch must be 'unet' or 'vit', got {custom_arch!r}."
            )
        state_dict = torch.load(custom_weights, map_location="cpu")
        model = _load_handling_compile(model, state_dict)
    elif backbone in ANATOMIX_VARIANTS:
        model = load_from_hf(backbone)
    else:
        raise ValueError(
            f"Unknown backbone {backbone!r}. "
            f"Known: {sorted(ANATOMIX_VARIANTS) + ['custom']}."
        )

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def minmax_normalize(arr, minclip=None, maxclip=None):
    """Clip and min-max normalize a volume to ``[0, 1]``.

    Parameters
    ----------
    arr : torch.Tensor
        Input volume.
    minclip, maxclip : float, optional
        Lower / upper intensity clip bounds applied before normalization. If
        both are given, ``minclip`` must be strictly below ``maxclip``.

    Returns
    -------
    torch.Tensor
        The clipped, min-max normalized volume (same shape / device / dtype).

    Raises
    ------
    ValueError
        If the volume is constant after clipping (min == max).
    """
    if minclip is not None and maxclip is not None and not minclip < maxclip:
        raise ValueError(
            f"minclip ({minclip}) must be strictly below maxclip ({maxclip})."
        )
    if minclip is not None or maxclip is not None:
        arr = torch.clamp(arr, min=minclip, max=maxclip)
    lo = arr.min()
    hi = arr.max()
    if not (hi > lo):
        raise ValueError(
            "Image is constant after clipping; cannot min-max normalize. "
            "Check the clip bounds."
        )
    return (arr - lo) / (hi - lo)


def _isotropic_shape(shape, spacing):
    """Endpoint-preserving isotropic grid shape at the finest voxel spacing.

    Parameters
    ----------
    shape : sequence of int
        Spatial size ``(H, W, D)`` of the volume.
    spacing : sequence of float
        Voxel spacing ``(sH, sW, sD)`` in array-axis order.

    Returns
    -------
    tuple of int
        The isotropic spatial shape, using ``round((n - 1) * s / t) + 1`` per
        axis where ``t = min(spacing)``.
    """
    target = float(min(spacing))
    return tuple(
        int(round((n - 1) * float(s) / target)) + 1
        for n, s in zip(shape, spacing)
    )


def _sliding_window_features(
    volume, model, window, sw_batch, overlap, mode, sigma, verbose=False,
):
    """Dense network features via MONAI sliding-window inference (OOM-safe).

    On a CUDA out-of-memory error the sliding-window batch size is halved and the
    inference retried, down to a batch size of 1.
    """
    batch = int(sw_batch)
    while True:
        try:
            with torch.no_grad():
                return sliding_window_inference(
                    volume,
                    (window, window, window),
                    batch,
                    model,
                    overlap=overlap,
                    mode=mode,
                    sigma_scale=sigma,
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() or batch <= 1:
                raise
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            if verbose:
                print(
                    f"    [features] CUDA OOM; retrying sliding window with "
                    f"sw_batch={batch}",
                    flush=True,
                )


def normalize_features(feats, method):
    """Per-voxel normalization of network features across channels.

    Parameters
    ----------
    feats : torch.Tensor
        Network features of shape ``(1, C, H, W, D)``.
    method : {'l2', 'standardized', 'none'}
        ``'l2'`` scales each voxel's channel vector to unit L2 norm;
        ``'standardized'`` gives it zero mean and unit standard deviation
        (biased variance); ``'none'`` returns the features unchanged.

    Returns
    -------
    torch.Tensor
        The normalized features.
    """
    if method == "l2":
        return F.normalize(feats, p=2, dim=1, eps=1e-12)
    if method == "standardized":
        mean = feats.mean(dim=1, keepdim=True)
        var = feats.var(dim=1, keepdim=True, unbiased=False)
        return (feats - mean) / torch.sqrt(var + 1e-5)
    if method == "none":
        return feats
    raise ValueError(
        f"feature_normalization must be 'l2', 'standardized' or 'none', "
        f"got {method!r}."
    )


def prepare_feature_channels(
    image_norm,
    spacing,
    model,
    *,
    use_mindssc,
    isotropic,
    window,
    sw_batch,
    overlap,
    sw_mode,
    sigma,
    feature_normalization,
    mindssc_radius,
    mindssc_dilation,
    verbose=False,
):
    """Extract network and/or MIND-SSC feature channels on the original grid.

    Parameters
    ----------
    image_norm : torch.Tensor
        Min-max normalized intensity volume of shape ``(1, 1, H, W, D)``.
    spacing : sequence of float
        Voxel spacing in array-axis order (used only when ``isotropic``).
    model : torch.nn.Module or None
        Feature extractor. ``None`` is allowed only for
        ``use_mindssc == 'mindssc-only'``.
    use_mindssc : {'both', 'feats-only', 'mindssc-only'}
        Which feature families to compute.
    isotropic : bool
        If True, features are extracted on an isotropic grid (finest spacing)
        and resampled back to the original grid.
    window, sw_batch, overlap, sw_mode, sigma
        Sliding-window inference parameters.
    feature_normalization : {'l2', 'standardized', 'none'}
        Network-feature normalization (does not affect MIND-SSC).
    mindssc_radius, mindssc_dilation : int
        MIND-SSC hyperparameters.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    feats : torch.Tensor or None
        Normalized network features ``(1, Cf, H, W, D)`` on the original grid,
        or ``None`` for ``use_mindssc == 'mindssc-only'``.
    mind : torch.Tensor or None
        MIND-SSC descriptor ``(1, 12, H, W, D)`` on the original grid, or
        ``None`` for ``use_mindssc == 'feats-only'``.
    """
    orig_shape = tuple(image_norm.shape[-3:])
    if isotropic:
        iso_shape = _isotropic_shape(orig_shape, spacing)
    else:
        iso_shape = orig_shape
    resample = iso_shape != orig_shape

    volume = image_norm
    if resample:
        volume = F.interpolate(
            image_norm, size=iso_shape, mode="trilinear", align_corners=True,
        )

    feats = None
    if use_mindssc in ("both", "feats-only"):
        if model is None:
            raise ValueError(
                "A backbone model is required unless use_mindssc == "
                "'mindssc-only'."
            )
        feats = _sliding_window_features(
            volume, model, window, sw_batch, overlap, sw_mode, sigma, verbose,
        )
        feats = normalize_features(feats, feature_normalization)
        if resample:
            feats = F.interpolate(
                feats, size=orig_shape, mode="trilinear", align_corners=True,
            )

    mind = None
    if use_mindssc in ("both", "mindssc-only"):
        mind = MINDSSC(volume, mindssc_radius, mindssc_dilation)
        if resample:
            mind = F.interpolate(
                mind, size=orig_shape, mode="trilinear", align_corners=True,
            )

    return feats, mind


def combine_feature_channels(feats, mind, mask, use_mindssc):
    """Mask and concatenate the selected feature families.

    When a mask is supplied it multiplies the *network* features only (MIND-SSC
    is left unmasked), matching the reference pipeline. The channel order for
    ``'both'`` is network features followed by MIND-SSC.

    Parameters
    ----------
    feats : torch.Tensor or None
        Network features ``(1, Cf, H, W, D)``.
    mind : torch.Tensor or None
        MIND-SSC descriptor ``(1, 12, H, W, D)``.
    mask : torch.Tensor or None
        Binary mask ``(1, 1, H, W, D)`` broadcast across the network channels.
    use_mindssc : {'both', 'feats-only', 'mindssc-only'}
        Which feature families to keep.

    Returns
    -------
    torch.Tensor
        The combined feature tensor ``(1, C, H, W, D)``.
    """
    if mask is not None and feats is not None:
        feats = feats * mask
    if use_mindssc == "mindssc-only":
        return mind
    if use_mindssc == "feats-only":
        return feats
    return torch.cat([feats, mind], dim=1)
