"""Feature preparation for the FireANTs registration backend.

This module turns a 3D volume into the multi-channel feature tensor that
FireANTs registers:

1. per-image intensity clip + min-max normalization to ``[0, 1]``;
2. (optionally) resample to an isotropic grid at the finest voxel spacing before
   feature extraction, then resample the features back to the original grid;
3. dense anatomix network features via MONAI sliding-window inference;
4. per-voxel feature normalization (L2 / standardized / none) of the *network*
   features only;
5. a hand-crafted MIND-SSC descriptor (unnormalized);
6. optional masking of the primary family and concatenation of the selected
   channel families.

``--features intensity`` skips steps 2-5 and registers the normalized intensity
directly, the classic single-channel MI/CC baseline.

Everything here operates on plain ``torch`` tensors and the anatomix models; no
FireANTs import is required.
"""
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


def minmax_normalize(arr, minclip=None, maxclip=None, name="Image"):
    """Clip and min-max normalize a volume to ``[0, 1]``.

    Parameters
    ----------
    arr : torch.Tensor
        Input volume.
    minclip, maxclip : float, optional
        Lower / upper intensity clip bounds applied before normalization. If
        both are given, ``minclip`` must be strictly below ``maxclip``.
    name : str, optional
        How to refer to this volume in error messages.

    Returns
    -------
    torch.Tensor
        The clipped, min-max normalized volume (same shape / device / dtype).

    Raises
    ------
    ValueError
        If the volume holds a NaN or infinity, or is constant after clipping.
    """
    if minclip is not None and maxclip is not None and not minclip < maxclip:
        raise ValueError(
            f"minclip ({minclip}) must be strictly below maxclip ({maxclip})."
        )
    # Checked before clipping: clamping an infinity to a bound would hide it,
    # and a NaN survives clamping to poison min/max and every voxel downstream.
    nonfinite = int((~torch.isfinite(arr)).sum())
    if nonfinite:
        raise ValueError(
            f"{name} contains {nonfinite} non-finite voxel(s) (NaN or "
            "infinity); registration would silently produce a meaningless "
            "result. Repair the volume before registering it."
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
    """Extract the requested channel families on the original grid.

    Parameters
    ----------
    image_norm : torch.Tensor
        Min-max normalized intensity volume of shape ``(1, 1, H, W, D)``.
    spacing : sequence of float
        Voxel spacing in array-axis order (used only when ``isotropic``).
    model : torch.nn.Module or None
        Feature extractor. ``None`` is allowed only for the model-free families
        (``'mindssc'`` and ``'intensity'``).
    features : {'anatomix+mindssc', 'anatomix', 'mindssc', 'intensity'}
        Which channel families to compute.
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
    primary : torch.Tensor or None
        The maskable channel family on the original grid: normalized network
        features ``(1, Cf, H, W, D)`` when ``features`` includes ``anatomix``,
        the min-max normalized intensity ``(1, 1, H, W, D)`` for
        ``'intensity'``, and ``None`` for ``'mindssc'``.
    mind : torch.Tensor or None
        MIND-SSC descriptor ``(1, 12, H, W, D)`` on the original grid, or
        ``None`` when ``features`` does not include MIND-SSC.
    """
    if features == "intensity":
        # The isotropic detour would only cost a round-trip interpolation of
        # the very channel being registered.
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
        # Normalize on the (original) registration grid so the L2/standardized
        # invariant holds exactly there, not on the isotropic extraction grid.
        primary = normalize_features(primary, feature_normalization)

    mind = None
    if features in ("anatomix+mindssc", "mindssc"):
        mind = MINDSSC(volume, mindssc_radius, mindssc_dilation)
        if resample:
            mind = F.interpolate(
                mind, size=orig_shape, mode="trilinear", align_corners=True,
            )

    return primary, mind


def combine_feature_channels(primary, mind, mask, features):
    """Mask and concatenate the selected channel families.

    A supplied mask multiplies the *primary* family (network features, or the
    raw intensity for ``'intensity'``); MIND-SSC is left unmasked. The channel
    order for ``'anatomix+mindssc'`` is network features followed by MIND-SSC.

    Parameters
    ----------
    primary : torch.Tensor or None
        Network features ``(1, Cf, H, W, D)`` or the normalized intensity
        ``(1, 1, H, W, D)``.
    mind : torch.Tensor or None
        MIND-SSC descriptor ``(1, 12, H, W, D)``.
    mask : torch.Tensor or None
        Binary mask ``(1, 1, H, W, D)`` broadcast across the primary channels.
    features : {'anatomix+mindssc', 'anatomix', 'mindssc', 'intensity'}
        Which channel families to keep.

    Returns
    -------
    torch.Tensor
        The combined feature tensor ``(1, C, H, W, D)``.
    """
    if mask is not None and primary is not None:
        primary = primary * mask
    if features == "mindssc":
        return mind
    if features == "anatomix+mindssc":
        return torch.cat([primary, mind], dim=1)
    return primary  # 'anatomix' or 'intensity'
