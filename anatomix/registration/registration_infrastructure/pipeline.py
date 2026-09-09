"""Register pairs sequentially, then write images, transforms, labels and metrics."""
import csv
import math
import os
import random
from collections import Counter

import nibabel as nib
import numpy as np
import torch

from ._fireants import FFO_AVAILABLE, FakeBatchedImages, generate_image_mask_allones
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
    invert_grid,
    keypoints_from_physical,
    keypoints_to_physical,
    load_image,
    read_linear_transform,
    save_inverse_transform,
    save_transforms,
    warp_keypoints,
    warp_volume,
    write_on_geometry,
)


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch, with deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(spec):
    """Resolve cpu/cuda/cuda:N; auto chooses the visible GPU with most free memory."""
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
    """Default to masked_cc when any mask exists, otherwise cc."""
    resolved = []
    for stage in stages:
        spec = dict(stage)
        if spec["loss"] is None:
            spec["loss"] = "masked_cc" if has_masks else "cc"
        resolved.append(spec)
    return resolved


def _nifti_geometry(path):
    """(shape_xyz, RAS affine) from a NIfTI header."""
    image = nib.load(path)
    return tuple(int(n) for n in image.shape[:3]), image.affine


def _mass_image(normalized, mask):
    """Non-negative intensity mass for moments, restricted to mask foreground if supplied."""
    if mask is None:
        return normalized
    return normalized * (mask > 0).to(normalized.dtype)


def _validate_pair(pair, label):
    """Validate mask/segmentation/keypoint availability for one pair."""
    has_fixed_mask = pair.get("fixed_mask") is not None
    has_moving_mask = pair.get("moving_mask") is not None
    if pair.get("fixed_seg") is not None and pair.get("moving_seg") is None:
        raise ValueError(
            f"{label}: a fixed segmentation requires a moving segmentation."
        )
    if (pair.get("moving_keypoints") is not None
            and pair.get("fixed_keypoints") is None):
        raise ValueError(
            f"{label}: moving keypoints require fixed keypoints."
        )
    return has_fixed_mask or has_moving_mask


def process_pair(pair, args, stages, feat_cfg, model, device, prefix, stem,
                 label):
    """Register one pair, write outputs using its unique stem, and return metrics."""
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fixed_img = load_image(pair["fixed"], device)
    moving_img = load_image(pair["moving"], device)
    fixed_spacing = array_spacing(fixed_img)
    fixed_geom, moving_geom = fixed_img, moving_img  # keypoints need the matrices only

    # Keep the original intensities for the final (single) warps.
    moving_raw = moving_img.array.clone()
    fixed_raw = fixed_img.array.clone() if args.save_inverse else None

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
        fixed_mask_img = (
            load_image(pair["fixed_mask"], device) if pair.get("fixed_mask")
            else generate_image_mask_allones(fixed_img)
        )
        moving_mask_img = (
            load_image(pair["moving_mask"], device) if pair.get("moving_mask")
            else generate_image_mask_allones(moving_img)
        )
        for mask_img, role in ((fixed_mask_img, "fixed"), (moving_mask_img, "moving")):
            if not torch.isfinite(mask_img.array).all():
                raise ValueError(f"{label} [{role} mask]: values must be finite.")
            mask_img.array = (mask_img.array > 0).to(mask_img.array.dtype)
            if not mask_img.array.any():
                raise ValueError(f"{label} [{role} mask]: mask has no foreground.")
        fixed_mask_t = fixed_mask_img.array
        moving_mask_t = moving_mask_img.array

    def feature_image(image, normalized, spacing, mask_t):
        """Rebind a loaded image to its feature channels (geometry is preserved)."""
        primary, mind = prepare_feature_channels(
            normalized, spacing, model, **feat_cfg,
        )
        image.array = combine_feature_channels(
            primary, mind, mask_t, args.features, append_mask=any_masked,
        )
        image.channels = image.array.shape[1]
        return image

    fixed_img = feature_image(fixed_img, fixed_norm, fixed_spacing, fixed_mask_t)
    if args.verbose:
        print(f"  feature channels: {fixed_img.channels}", flush=True)

    fixed_batch = as_batch(fixed_img)
    # The moving image only supplies geometry here; its features are always
    # extracted after resampling onto the fixed grid (see register.py).
    moving_batch = as_batch(moving_img)

    # Moments use non-negative intensities, not network features.
    init_batches = None
    if args.initialization in ("center-of-mass", "moments"):
        init_batches = (
            FakeBatchedImages(_mass_image(fixed_norm, fixed_mask_t),
                              fixed_batch),
            FakeBatchedImages(_mass_image(moving_norm, moving_mask_t),
                              moving_batch),
        )

    def reextract_moving(grid):
        """Warp original moving intensities/mask, then extract features on the fixed grid."""
        with torch.no_grad():
            warped_norm = warp_volume(moving_norm, grid, "bilinear")
            warped_mask_t = (
                warp_volume(moving_mask_t, grid, "nearest")
                if moving_mask_t is not None else None
            )
            # Reloaded only for its geometry; the array is replaced.
            warped_img = load_image(pair["fixed"], device)
            return as_batch(feature_image(
                warped_img, warped_norm, fixed_spacing, warped_mask_t))

    initial_transform = None
    if pair.get("initial_transform") is not None:
        initial_transform = read_linear_transform(
            pair["initial_transform"],
            fixed=_nifti_geometry(pair["fixed"]),
            moving=_nifti_geometry(pair["moving"]),
        )

    result = run_registration(
        fixed_batch, moving_batch, stage_specs,
        initialization=args.initialization, verbose=args.verbose,
        reextract_moving=reextract_moving, has_mask_channel=any_masked,
        init_images=init_batches, initial_transform=initial_transform,
    )
    grid = result.warped_coordinates

    moved = warp_volume(moving_raw, grid, "bilinear")
    moved_path = os.path.join(args.output_dir, f"{prefix}moved-{stem}.nii.gz")
    write_on_geometry(moved, fixed_batch, moved_path)

    save_transforms(
        result, args.output_transformation_convention, args.collapse,
        args.output_dir, prefix, stem,
    )

    num_folds = count_folds(grid, fixed_batch, moving_batch)
    dice = ""
    if pair.get("moving_seg") is not None:
        moving_seg = load_image(
            pair["moving_seg"], device, is_segmentation=False,
        ).array.float()
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

    if args.save_inverse:
        grid_inv, residual = invert_grid(grid, fixed_batch, moving_batch)
        save_inverse_transform(
            result, grid_inv, args.output_transformation_convention,
            args.output_dir, prefix, stem,
        )
        write_on_geometry(
            warp_volume(fixed_raw, grid_inv, "bilinear"), moving_batch,
            os.path.join(args.output_dir, f"{prefix}inverse-moved-{stem}.nii.gz"),
        )
        if pair.get("fixed_seg") is not None:
            fixed_seg_raw = load_image(pair["fixed_seg"], device).array.float()
            write_on_geometry(
                warp_volume(fixed_seg_raw, grid_inv, "nearest").round().to(torch.int32),
                moving_batch,
                os.path.join(args.output_dir, f"{prefix}inverse-moved-seg-{stem}.nii.gz"),
            )
        finite = residual[torch.isfinite(residual)]
        metrics["inverse_residual_mm"] = (
            float(finite.max()) if finite.numel() else float("nan"))
        if args.verbose and finite.numel():
            print(
                f"  -> inverse: residual median={float(finite.median()):.4f} "
                f"max={metrics['inverse_residual_mm']:.4f} mm; "
                f"{float((finite > 1).float().mean()):.2%} of {finite.numel()} "
                "moving voxels inside the fixed FOV exceed 1 mm",
                flush=True,
            )

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
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
            print(f"  -> peak GPU memory: {peak:.1f} GiB", flush=True)

    del (
        fixed_img, moving_img, fixed_geom, moving_geom, fixed_batch,
        moving_batch, result, grid, moved, moving_raw, fixed_raw,
    )
    torch.cuda.empty_cache()
    return metrics


def run(args, pairs, input_columns, stages):
    """Load the backbone once and process pairs sequentially, flushing each metrics row."""
    seed_everything(args.seed)
    if not FFO_AVAILABLE:
        print(
            "[note] fireants_fused_ops is not available; FireANTs is using its "
            "pure-PyTorch fallback. "
            "Build the kernels with registration_backend/install_fireants.sh.",
            flush=True,
        )
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)  # FireANTs allocates on the default device
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

    # Output stems: the moving basename, indexed when it repeats within the batch.
    raw_stems = [strip_nifti_ext(pair["moving"]) for pair in pairs]
    counts = Counter(raw_stems)
    width = max(1, len(str(len(pairs) - 1)))
    stems = [
        f"{index:0{width}d}-{stem}" if counts[stem] > 1 else stem
        for index, stem in enumerate(raw_stems)
    ]

    # One flushed row per pair, so a mid-batch failure keeps earlier results.
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
