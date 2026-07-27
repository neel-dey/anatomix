"""End-to-end orchestration for the FireANTs registration CLI.

Loads the backbone once (when ``--features`` needs one), then processes each
fixed/moving pair sequentially: preprocess -> feature extraction -> feature
image assembly (with optional mask channel) -> multi-stage FireANTs
registration -> warp the original moving image and label onto the fixed
geometry and the fixed keypoints into moving space -> Dice, TRE and fold
metrics -> a per-pair row in the metrics CSV. Batch pairs are processed one at
a time so arbitrary shapes are supported and GPU memory stays bounded.
"""
import csv
import math
import os
import random
from collections import Counter

import numpy as np
import torch

from ._fireants import FFO_AVAILABLE, FakeBatchedImages, apply_mask_to_image
from .features import (
    combine_feature_channels,
    load_backbone,
    minmax_normalize,
    prepare_feature_channels,
)
from .io_utils import (
    METRIC_COLUMNS,
    read_keypoints,
    strip_nifti_ext,
    write_keypoints,
)
from .metrics import count_folds, dice_score, keypoint_metrics
from .register import run_registration
from .warp_io import (
    array_spacing,
    as_batch,
    keypoints_from_physical,
    keypoints_to_physical,
    load_image,
    save_transforms,
    warp_keypoints,
    warp_volume,
    write_on_geometry,
)


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs (deterministic cuDNN)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(spec):
    """Resolve a ``--device`` spec to a :class:`torch.device`.

    ``'auto'`` picks the visible CUDA device with the most free memory, so a
    shared multi-GPU box does not default onto a busy GPU; ``'cpu'``, ``'cuda'``
    and ``'cuda:N'`` are honored verbatim. ``CUDA_VISIBLE_DEVICES`` restricts the
    candidates considered by ``'auto'`` (and the meaning of any index).
    """
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        best, best_free = 0, -1
        for index in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(index)
            if free > best_free:
                best_free, best = free, index
        return torch.device(f"cuda:{best}")
    return torch.device(spec)


def resolve_stage_losses(stages, has_masks):
    """Fill in per-stage default losses for one pair.

    An omitted loss (``None``) becomes ``masked_cc`` when the pair has both masks
    and ``cc`` otherwise; an explicit loss is left unchanged.
    """
    resolved = []
    for stage in stages:
        spec = dict(stage)
        if spec["loss"] is None:
            spec["loss"] = "masked_cc" if has_masks else "cc"
        resolved.append(spec)
    return resolved


def _mass_image(normalized, mask):
    """Non-negative, intensity-like volume for the moment initialization.

    ``normalized`` is already min-max normalized to ``[0, 1]``. When a mask is
    available the mass is restricted to it, so the center of mass follows the
    anatomy of interest rather than everything inside the field of view.
    """
    if mask is None:
        return normalized
    return normalized * (mask > 0).to(normalized.dtype)


def _validate_pair(pair, label):
    """Validate mask/segmentation/keypoint availability for one pair."""
    has_fixed_mask = pair.get("fixed_mask") is not None
    has_moving_mask = pair.get("moving_mask") is not None
    if has_fixed_mask != has_moving_mask:
        raise ValueError(
            f"{label}: provide both fixed and moving masks, or neither."
        )
    if pair.get("fixed_seg") is not None and pair.get("moving_seg") is None:
        raise ValueError(
            f"{label}: a fixed segmentation requires a moving segmentation."
        )
    if (pair.get("moving_keypoints") is not None
            and pair.get("fixed_keypoints") is None):
        raise ValueError(
            f"{label}: moving keypoints require fixed keypoints."
        )
    return has_fixed_mask and has_moving_mask


def process_pair(pair, args, stages, feat_cfg, model, device, prefix, stem,
                 label):
    """Register one fixed/moving pair and write its outputs.

    ``stem`` is the (batch-disambiguated) output filename stem for this pair.

    Returns
    -------
    dict
        The :data:`METRIC_COLUMNS` for this pair; the Dice and TRE entries are
        ``""`` when the pair lacks the segmentations or keypoints they need.
    """
    has_masks = _validate_pair(pair, label)
    stage_specs = resolve_stage_losses(stages, has_masks)
    any_masked = any(s["loss"].startswith("masked_") for s in stage_specs)
    if any_masked and not has_masks:
        raise ValueError(
            f"{label}: a masked loss was requested but the pair has no masks."
        )

    if args.verbose:
        print(f"[pair] {label}", flush=True)
        print(f"  fixed  = {pair['fixed']}", flush=True)
        print(f"  moving = {pair['moving']}", flush=True)
        print(
            f"  masks={has_masks} losses={[s['loss'] for s in stage_specs]}",
            flush=True,
        )

    fixed_img = load_image(pair["fixed"], device)
    moving_img = load_image(pair["moving"], device)
    fixed_spacing = array_spacing(fixed_img)
    moving_spacing = array_spacing(moving_img)
    # Coordinate matrices for the keypoints; the images themselves are rebound
    # to feature tensors below.
    fixed_geom, moving_geom = fixed_img, moving_img

    # Keep the original moving intensity for the final (single) warp.
    moving_raw = moving_img.array.clone()

    fixed_norm = minmax_normalize(
        fixed_img.array, args.fixed_minclip, args.fixed_maxclip,
        name=f"{label} [fixed]",
    )
    moving_norm = minmax_normalize(
        moving_img.array, args.moving_minclip, args.moving_maxclip,
        name=f"{label} [moving]",
    )

    fixed_mask_img = moving_mask_img = None
    fixed_mask_t = moving_mask_t = None
    if has_masks:
        fixed_mask_img = load_image(pair["fixed_mask"], device)
        moving_mask_img = load_image(pair["moving_mask"], device)
        fixed_mask_t = fixed_mask_img.array
        moving_mask_t = moving_mask_img.array

    fixed_primary, fixed_mind = prepare_feature_channels(
        fixed_norm, fixed_spacing, model, **feat_cfg,
    )
    moving_primary, moving_mind = prepare_feature_channels(
        moving_norm, moving_spacing, model, **feat_cfg,
    )
    fixed_comb = combine_feature_channels(
        fixed_primary, fixed_mind, fixed_mask_t, args.features,
    )
    moving_comb = combine_feature_channels(
        moving_primary, moving_mind, moving_mask_t, args.features,
    )
    if args.verbose:
        print(
            f"  feature channels: fixed={fixed_comb.shape[1]} "
            f"moving={moving_comb.shape[1]}",
            flush=True,
        )

    # Reuse the loaded images as the feature images (geometry is preserved).
    fixed_img.array = fixed_comb
    moving_img.array = moving_comb
    if any_masked:
        fixed_img = apply_mask_to_image(fixed_img, fixed_mask_img)
        moving_img = apply_mask_to_image(moving_img, moving_mask_img)

    fixed_batch = as_batch(fixed_img)
    moving_batch = as_batch(moving_img)

    # The moment initialization needs a non-negative mass field, so it runs on
    # the intensities rather than the features (see ``register``). Same
    # geometry, so its physical-space transform initializes the stages directly.
    init_batches = None
    if args.initialization != "none":
        init_batches = (
            FakeBatchedImages(_mass_image(fixed_norm, fixed_mask_t),
                              fixed_batch),
            FakeBatchedImages(_mass_image(moving_norm, moving_mask_t),
                              moving_batch),
        )

    def reextract_moving(grid):
        """Warp the moving *image* by ``grid`` and re-extract features.

        A warm-started deformable stage needs the moving features in the
        already-transformed frame. The anatomix extractor is not
        warp-equivariant, so they are recomputed from the warped moving image
        (and mask) rather than resampled from the moving feature maps. The
        result lives on the fixed grid and carries the fixed geometry.
        """
        with torch.no_grad():
            warped_norm = warp_volume(moving_norm, grid, "bilinear")
            warped_mask_t = (
                warp_volume(moving_mask_t, grid, "nearest")
                if moving_mask_t is not None else None
            )
            primary, mind = prepare_feature_channels(
                warped_norm, fixed_spacing, model, **feat_cfg,
            )
            comb = combine_feature_channels(
                primary, mind, warped_mask_t, args.features,
            )
        # fixed geometry template
        warped_img = load_image(pair["fixed"], device)
        warped_img.array = comb
        if any_masked:
            warped_mask_img = load_image(pair["fixed_mask"], device)
            warped_mask_img.array = warped_mask_t
            warped_img = apply_mask_to_image(warped_img, warped_mask_img)
        return as_batch(warped_img)

    result = run_registration(
        fixed_batch, moving_batch, stage_specs,
        initialization=args.initialization, verbose=args.verbose,
        reextract_moving=reextract_moving, has_mask_channel=any_masked,
        init_images=init_batches,
    )
    grid = result.warped_coordinates

    moved = warp_volume(moving_raw, grid, "bilinear")
    moved_path = os.path.join(args.output_dir, f"{prefix}moved-{stem}.nii.gz")
    write_on_geometry(moved, fixed_batch, moved_path)

    save_transforms(
        result, args.output_transformation_convention, args.collapse,
        args.output_dir, prefix, stem,
    )

    num_folds = count_folds(grid)
    dice = ""
    if pair.get("moving_seg") is not None:
        moving_seg = load_image(
            pair["moving_seg"], device, is_segmentation=False,
        ).array.float()
        # Nearest warp on float32 preserves integer labels exactly; int32 holds
        # any realistic label id (int16 would wrap ids >= 32768).
        moved_seg = warp_volume(moving_seg, grid, "nearest").round().to(
            torch.int32
        )
        moved_seg_path = os.path.join(
            args.output_dir, f"{prefix}moved-seg-{stem}.nii.gz",
        )
        write_on_geometry(moved_seg, fixed_batch, moved_seg_path)
        if pair.get("fixed_seg") is not None:
            fixed_seg = load_image(pair["fixed_seg"], device).array
            dice = dice_score(fixed_seg.cpu().numpy(), moved_seg.cpu().numpy())
            if isinstance(dice, float) and math.isnan(dice):
                dice = ""  # no foreground in the fixed seg -> blank, not "nan"

    metrics = {col: "" for col in METRIC_COLUMNS}
    metrics["dice"] = dice
    metrics["num_folds"] = num_folds

    moved_kp_path = None
    if pair.get("fixed_keypoints") is not None:
        convention = args.keypoint_convention
        kp_columns, kp_rows, kp_coords = read_keypoints(
            pair["fixed_keypoints"])
        source = keypoints_to_physical(
            torch.tensor(kp_coords, device=device, dtype=torch.float32),
            convention, fixed_geom,
        )
        warped = warp_keypoints(source, grid, fixed_geom, moving_geom)
        moved_kp_path = os.path.join(
            args.output_dir, f"{prefix}moved-keypoints-{stem}.csv",
        )
        write_keypoints(
            moved_kp_path, kp_columns, kp_rows,
            keypoints_from_physical(warped, convention, moving_geom)
            .cpu().numpy(),
        )
        if pair.get("moving_keypoints") is not None:
            _, _, target_coords = read_keypoints(pair["moving_keypoints"])
            target = keypoints_to_physical(
                torch.tensor(
                    target_coords, device=device, dtype=torch.float32,
                ),
                convention, moving_geom,
            )
            metrics.update(keypoint_metrics(
                warped.cpu().numpy(), target.cpu().numpy(),
                source.cpu().numpy(),
            ))

    if args.verbose:
        print(f"  -> moved:  {moved_path}", flush=True)
        if moved_kp_path is not None:
            print(f"  -> keypoints: {moved_kp_path}", flush=True)
        print(
            f"  -> dice={metrics['dice']} folds={num_folds} "
            f"tre={metrics['tre_median']} (initial "
            f"{metrics['tre_initial_median']}) "
            f"robustness={metrics['robustness']}",
            flush=True,
        )

    del (
        fixed_img, moving_img, fixed_geom, moving_geom, fixed_batch,
        moving_batch, result, grid, moved, fixed_comb, moving_comb,
        fixed_primary, moving_primary, fixed_mind, moving_mind, moving_raw,
    )
    torch.cuda.empty_cache()
    return metrics


def run(args, pairs, input_columns, stages):
    """Run the pipeline over all pairs and write the metrics CSV.

    Parameters
    ----------
    args : argparse.Namespace
        Fully-resolved CLI arguments.
    pairs : list of dict
        Per-pair path dicts (values are absolute paths or ``None``), keyed by
        ``input_columns``.
    input_columns : list of str
        Metrics-CSV input columns, in order.
    stages : list of dict
        Per-stage registration specs (a stage's ``loss`` may be ``None`` to
        request the per-pair default).

    Returns
    -------
    list of dict
        The metrics rows that were written.
    """
    seed_everything(args.seed)
    if not FFO_AVAILABLE:
        print(
            "[note] fireants_fused_ops is not available; FireANTs is using its "
            "pure-PyTorch fallback, which is numerically equivalent but slower. "
            "Build the kernels with registration_backend/install_fireants.sh.",
            flush=True,
        )
    device = select_device(args.device)
    if device.type == "cuda":
        # FireANTs allocates some internal tensors on the *default* CUDA device,
        # so make that the selected one rather than an unrelated GPU.
        torch.cuda.set_device(device)
    if args.verbose:
        name = (
            f" ({torch.cuda.get_device_name(device)})"
            if device.type == "cuda" else ""
        )
        print(f"[device] {device}{name}", flush=True)

    model = None
    if args.features in ("anatomix+mindssc", "anatomix"):
        model = load_backbone(
            args.backbone, device,
            custom_arch=getattr(args, "custom_arch", None),
            custom_weights=getattr(args, "custom_weights", None),
            unet_kwargs=getattr(args, "unet_kwargs", None),
            vit_kwargs=getattr(args, "vit_kwargs", None),
        )

    feat_cfg = dict(
        features=args.features,
        isotropic=bool(args.isotropic_features),
        window=args.sw_window,
        sw_batch=args.sw_batch,
        overlap=args.sw_overlap,
        sw_mode=args.sw_mode,
        sigma=args.sw_sigma,
        feature_normalization=args.feature_normalization,
        mindssc_radius=args.mindssc_radius,
        mindssc_dilation=args.mindssc_dilation,
        verbose=args.verbose,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"{args.exp_name}-" if args.exp_name else ""

    # Output stems come from the moving basename, with a zero-padded pair index
    # added when the same basename appears in more than one directory.
    raw_stems = [strip_nifti_ext(pair["moving"]) for pair in pairs]
    counts = Counter(raw_stems)
    width = max(1, len(str(len(pairs) - 1)))
    stems = [
        f"{index:0{width}d}-{stem}" if counts[stem] > 1 else stem
        for index, stem in enumerate(raw_stems)
    ]

    # Write the metrics CSV incrementally (one flushed row per completed pair) so
    # a mid-batch failure preserves the results computed so far.
    fieldnames = list(input_columns) + list(METRIC_COLUMNS)
    csv_path = os.path.join(args.output_dir, f"{prefix}metrics.csv")
    rows = []
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        for index, pair in enumerate(pairs):
            label = f"pair {index}"
            metrics = process_pair(
                pair, args, stages, feat_cfg, model, device, prefix,
                stems[index], label,
            )
            row = {col: (pair.get(col) or "") for col in input_columns}
            row.update(metrics)
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            handle.flush()
            rows.append(row)

    if args.verbose:
        print(f"[done] wrote metrics: {csv_path}", flush=True)
    return rows
