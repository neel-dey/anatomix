#!/usr/bin/env python
"""Register 3D volume pairs with FireANTs on anatomix features.

    bash registration_backend/install_fireants.sh      # once
    python anatomix-register.py --fixed F.nii.gz --moving M.nii.gz --output-dir out
    python anatomix-register.py --help

``main`` parses and validates the options, loads the feature network once,
and calls ``register_pair`` for every pair. ``register_pair`` is the whole
per-pair pipeline: load, extract features, run the transform stages, write
the outputs. The modules it uses live in ``registration_infrastructure/``:

    cli.py        options, per-stage schedules, input validation
    features.py   intensity normalization, anatomix and MIND-SSC features
    register.py   initialization and the rigid/affine/deformable stages
    warp_io.py    resampling, keypoint mapping, transform export and inversion
    metrics.py    Dice, landmark error, fold count
    pipeline.py   the helpers called below
"""
import torch

from anatomix.registration.registration_infrastructure import cli, pipeline
from anatomix.registration.registration_infrastructure.features import load_backbone
from anatomix.registration.registration_infrastructure.metrics import count_folds
from anatomix.registration.registration_infrastructure.register import run_registration
from anatomix.registration.registration_infrastructure.warp_io import (
    as_batch,
    load_image,
    save_transforms,
    warp_volume,
    write_on_geometry,
)


def register_pair(pair, index, args, stages, extract_features, device, paths):
    """Register one pair, write its outputs, and return its metrics row."""
    label = f"pair {index}"
    if args.verbose:
        print(f"[{label}] fixed={pair['fixed']} moving={pair['moving']}", flush=True)

    # 1. Load the images, normalize intensities, binarize masks.
    inputs = pipeline.load_pair(pair, args, device, label)
    stage_specs = pipeline.resolve_stage_losses(stages, inputs.has_masks)
    masked = any(spec["loss"].startswith("masked_") for spec in stage_specs)

    # 2. Features of the fixed image, on its own grid. The moving features are
    #    extracted by ``moving_features``: before every stage, the moving image
    #    is resampled onto the fixed grid with the transform so far and its
    #    features are computed there, so the two images may have different
    #    grids, spacings and orientations.
    fixed_batch = extract_features(
        inputs.fixed, inputs.fixed_norm, inputs.fixed_spacing, inputs.fixed_mask, masked)
    moving_batch = as_batch(inputs.moving)  # geometry only; see moving_features

    def moving_features(grid):
        with torch.no_grad():
            warped = warp_volume(inputs.moving_norm, grid, "bilinear")
            warped_mask = (
                warp_volume(inputs.moving_mask, grid, "nearest")
                if inputs.moving_mask is not None else None
            )
            on_fixed_grid = load_image(pair["fixed"], device)  # reloaded for its geometry
            return extract_features(
                on_fixed_grid, warped, inputs.fixed_spacing, warped_mask, masked)

    if args.verbose:
        print(f"  feature channels: {fixed_batch().shape[1]}", flush=True)

    # 3. The transform: a closed-form or file initialization, then one FireANTs
    #    stage per --transform entry. The result holds the cumulative
    #    fixed-to-moving sampling grid and every stage's transform.
    result = run_registration(
        fixed_batch, moving_batch, stage_specs,
        initialization=args.initialization,
        init_images=(
            pipeline.moment_images(inputs, fixed_batch, moving_batch)
            if args.initialization in ("center-of-mass", "moments") else None
        ),
        initial_transform=pipeline.load_initial_transform(pair),
        reextract_moving=moving_features,
        has_mask_channel=masked,
        verbose=args.verbose,
    )
    grid = result.warped_coordinates

    # 4. Outputs: the moving image on the fixed grid, the transform, labels,
    #    the inverse, keypoints, and the metrics row.
    write_on_geometry(
        warp_volume(inputs.moving_raw, grid, "bilinear"), fixed_batch,
        paths.path(index, "moved", ".nii.gz"),
    )
    save_transforms(
        result, args.output_transformation_convention, args.collapse,
        paths.output_dir, paths.prefix, paths.stems[index],
    )
    metrics = pipeline.empty_metrics()
    metrics["num_folds"] = count_folds(grid, fixed_batch, moving_batch)
    if pair.get("moving_seg") is not None:
        metrics["dice"] = pipeline.propagate_labels(
            pair, grid, fixed_batch, device, paths.path(index, "moved-seg", ".nii.gz"))
    if args.save_inverse:
        metrics["inverse_residual_mm"] = pipeline.write_inverse(
            pair, result, inputs, fixed_batch, moving_batch, args, paths, index)
    if pair.get("fixed_keypoints") is not None:
        metrics.update(pipeline.map_keypoints(
            pair, grid, inputs, args.keypoint_convention,
            paths.path(index, "moved-keypoints", ".csv")))

    if args.verbose:
        print(
            f"  -> moved: {paths.path(index, 'moved', '.nii.gz')}\n"
            f"  -> dice={metrics['dice']} folds={metrics['num_folds']} "
            f"tre={metrics['tre_median']} (initial {metrics['tre_initial_median']}) "
            f"robustness={metrics['robustness']}",
            flush=True,
        )
    return metrics


def main(argv=None):
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    try:
        pairs, input_columns, stages = cli.prepare(args)  # headers only; no GPU yet
    except ValueError as error:
        parser.error(str(error))

    pipeline.seed_everything(args.seed)
    pipeline.warn_if_no_fused_ops()
    device = pipeline.select_device(args.device)
    if args.verbose:
        name = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
        print(f"[device] {device}{name}", flush=True)

    # The feature network is loaded once for the whole batch. 'mindssc' and
    # 'intensity' features need no network.
    model = None
    if args.features in cli.MODEL_FEATURES:
        model = load_backbone(
            args.backbone, device,
            custom_arch=args.custom_arch, custom_weights=args.custom_weights,
            unet_kwargs=args.unet_kwargs, vit_kwargs=args.vit_kwargs,
        )
    extract_features = pipeline.FeatureExtractor(args, model)

    paths = pipeline.OutputPaths(args.output_dir, args.exp_name, pairs)
    with pipeline.MetricsWriter(paths.metrics_csv, input_columns) as writer:
        for index, pair in enumerate(pairs):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            metrics = register_pair(
                pair, index, args, stages, extract_features, device, paths)
            writer.write(pair, metrics)
            if args.verbose and device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
                print(f"  -> peak GPU memory: {peak:.1f} GiB", flush=True)
            torch.cuda.empty_cache()
    if args.verbose:
        print(f"[done] wrote metrics: {paths.metrics_csv}", flush=True)
    return writer.rows


if __name__ == "__main__":
    main()
