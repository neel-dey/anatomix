"""Normalize intensities and extract network/MIND-SSC features on the image grid."""
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
    """Load a frozen pretrained backbone, or custom UNet/ViT weights.

    Custom architecture kwargs must match the checkpoint. Returns an eval-mode
    model on the requested device; built-in variants load from Hugging Face."""
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


def minmax_normalize(arr, minclip=None, maxclip=None, name="Image"):
    """Clip and normalize to [0,1]; reject non-finite or constant volumes."""
    if minclip is not None and maxclip is not None and not minclip < maxclip:
        raise ValueError(
            f"minclip ({minclip}) must be strictly below maxclip ({maxclip})."
        )
    nonfinite = int((~torch.isfinite(arr)).sum())
    if nonfinite:
        raise ValueError(
            f"{name} contains {nonfinite} non-finite voxel(s); repair the "
            "volume before registering it."
        )
    if minclip is not None or maxclip is not None:
        arr = torch.clamp(arr, min=minclip, max=maxclip)
    lo = arr.min()
    hi = arr.max()
    if not (hi > lo):
        raise ValueError(
            f"{name} is constant after clipping; cannot min-max normalize. "
            "Check the clip bounds."
        )
    return (arr - lo) / (hi - lo)


def _isotropic_shape(shape, spacing):
    """Preserve voxel-centre endpoints at the finest spacing: round((n-1)*s/min(s))+1."""
    target = float(min(spacing))
    return tuple(
        int(round((n - 1) * float(s) / target)) + 1
        for n, s in zip(shape, spacing)
    )


def _sliding_window_features(
    volume, model, window, sw_batch, overlap, mode, sigma, verbose=False,
):
    """Dense network features via MONAI sliding windows; halves the batch on CUDA OOM."""
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
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or any(
                token in str(exc).lower()
                for token in ("out of memory", "alloc", "cudnn_status_alloc")
            )
            if not is_oom or batch <= 1:
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
    """Normalize each voxel across channels: L2, zero mean/unit variance, or none."""
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
    features,
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
    """Return (primary, mind) tensors on the original (1,C,Z,Y,X) grid.

    Primary is network features or normalized intensity; MIND-SSC has 12 channels.
    Unused families return None. Spacing is in array order (z,y,x). Isotropic
    extraction uses the finest spacing, then resamples features back. Intensity
    mode uses the original grid directly."""
    if features == "intensity":
        return image_norm, None

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

    primary = None
    if features in ("anatomix+mindssc", "anatomix"):
        if model is None:
            raise ValueError(
                "A backbone model is required unless --features is 'mindssc' "
                "or 'intensity'."
            )
        primary = _sliding_window_features(
            volume, model, window, sw_batch, overlap, sw_mode, sigma, verbose,
        )
        if resample:
            primary = F.interpolate(
                primary, size=orig_shape, mode="trilinear", align_corners=True,
            )
        primary = normalize_features(primary, feature_normalization)

    mind = None
    if features in ("anatomix+mindssc", "mindssc"):
        mind = MINDSSC(volume, mindssc_radius, mindssc_dilation)
        if resample:
            mind = F.interpolate(
                mind, size=orig_shape, mode="trilinear", align_corners=True,
            )

    return primary, mind


def combine_feature_channels(primary, mind, mask, features, append_mask=False):
    """Build the registered image: gated primary channels, then MIND-SSC, then the mask.

    The mask gates the primary channels only. With ``append_mask`` the binary
    mask becomes the last channel, which FireANTs' masked losses split off.
    One concatenation keeps the peak memory at a single copy of the volume."""
    parts = []
    if primary is not None and features != "mindssc":
        if mask is not None:
            primary = primary * (mask > 0).to(primary.dtype)
        parts.append(primary)
    if mind is not None and features in ("anatomix+mindssc", "mindssc"):
        parts.append(mind)
    if append_mask:
        parts.append((mask > 0).to(parts[0].dtype))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
