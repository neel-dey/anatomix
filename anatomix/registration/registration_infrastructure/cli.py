"""Argument parsing and validation for ``anatomix-register.py``.

This module defines the full command-line interface and resolves it into the
inputs the pipeline consumes: a list of fixed/moving (and optional mask/seg)
pairs, the per-stage registration specs, and the feature/output settings.
FireANTs is imported only lazily (inside :func:`main`, after all validation),
so ``--help`` and every argument/validation error work without the backend
installed.
"""
import argparse
import os

import nibabel as nib

from .io_utils import NIFTI_EXTS, read_pairs_csv

TRANSFORM_RANK = {"rigid": 0, "affine": 1, "deformable": 2}
VALID_LOSSES = {"cc", "mi", "mse", "masked_cc", "masked_mi", "masked_mse"}
# Losses that consume a CC kernel schedule (auto default resolves to a CC loss).
CC_LOSSES = {None, "cc", "masked_cc"}


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser():
    """Construct the ``anatomix-register.py`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="anatomix-register.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Register 3D volume pairs with FireANTs on anatomix network "
            "features (and/or MIND-SSC descriptors). Supports rigid/affine/"
            "deformable stages, masked and unmasked losses, label warping, "
            "transform export, Dice, and fold counting, in single-pair and "
            "batch modes."
        ),
    )

    mode = parser.add_argument_group("input mode (choose exactly one)")
    mode.add_argument("--fixed", help="Single-pair fixed image (.nii/.nii.gz).")
    mode.add_argument("--moving", help="Single-pair moving image (.nii/.nii.gz).")
    mode.add_argument("--fixed-dir", help="Batch: directory of fixed images.")
    mode.add_argument("--moving-dir", help="Batch: directory of moving images.")
    mode.add_argument(
        "--registration-pairs-csv",
        help="Batch: CSV with a header and columns fixed,moving"
        "[,fixed_mask,moving_mask,fixed_seg,moving_seg].",
    )

    aux = parser.add_argument_group("masks and segmentations")
    aux.add_argument("--fixed-mask", help="Fixed registration mask.")
    aux.add_argument("--moving-mask", help="Moving registration mask.")
    aux.add_argument("--fixed-seg", help="Fixed segmentation (enables Dice).")
    aux.add_argument("--moving-seg", help="Moving segmentation (warped to fixed).")
    aux.add_argument("--fixed-mask-dir", help="Batch directory of fixed masks.")
    aux.add_argument("--moving-mask-dir", help="Batch directory of moving masks.")
    aux.add_argument("--fixed-seg-dir", help="Batch directory of fixed segs.")
    aux.add_argument("--moving-seg-dir", help="Batch directory of moving segs.")

    clip = parser.add_argument_group("intensity clipping (reused across a batch)")
    clip.add_argument("--fixed-minclip", type=float, help="Fixed lower clip.")
    clip.add_argument("--fixed-maxclip", type=float, help="Fixed upper clip.")
    clip.add_argument("--moving-minclip", type=float, help="Moving lower clip.")
    clip.add_argument("--moving-maxclip", type=float, help="Moving upper clip.")

    tf = parser.add_argument_group(
        "transform chain (per-stage lists are comma-separated, one entry per "
        "--transform stage)"
    )
    tf.add_argument(
        "--initialization", choices=["none", "center-of-mass", "moments"],
        default="none",
        help="Closed-form moment initialization run before the stage chain.",
    )
    tf.add_argument(
        "--transform", default="deformable",
        help="Comma-separated stages from {rigid,affine,deformable}, ordered "
        "rigid<=affine<=deformable (e.g. 'affine,deformable').",
    )
    tf.add_argument(
        "--loss", default=None,
        help="Per-stage loss from {cc,mi,mse,masked_cc,masked_mi,masked_mse}. "
        "Default: masked_cc if the pair has masks, else cc.",
    )
    tf.add_argument(
        "--step-size", default=None,
        help="Per-stage Adam learning rate. Default 1.0 for deformable stages, "
        "0.1 for rigid/affine stages.",
    )
    tf.add_argument(
        "--shrink-factors", default=None,
        help="Per-stage 'AxBx...' resolution schedule. Default 8x4x2x1.",
    )
    tf.add_argument(
        "--iterations", default=None,
        help="Per-stage 'AxBx...' iteration schedule (matches shrink lengths). "
        "Default 100 per scale.",
    )
    tf.add_argument(
        "--cc-kernel-widths", default=None,
        help="Per-stage 'AxBx...' CC kernel widths (odd, one per pyramid level), "
        "'na' for non-CC stages; each stage's widths must match its "
        "--shrink-factors level count. Omitted: FireANTs' own default kernel "
        "size per stage.",
    )
    tf.add_argument(
        "--smooth-grad-sigma", default=None,
        help="Per-stage gradient-smoothing sigma for deformable stages, 'na' "
        "for linear stages. Default 1.0.",
    )
    tf.add_argument(
        "--smooth-warp-sigma", default=None,
        help="Per-stage warp-smoothing sigma for deformable stages, 'na' for "
        "linear stages. Default 0.5.",
    )

    feat = parser.add_argument_group("features")
    feat.add_argument(
        "--backbone",
        choices=["anatomix", "anatomix-dev", "anatomix-dev-vit", "custom"],
        default="anatomix-dev-vit",
        help="anatomix feature extractor (custom: see custom-backbone flags).",
    )
    feat.add_argument(
        "--isotropic-features", type=int, choices=[0, 1], default=1,
        help="Extract features on an isotropic (finest-spacing) grid.",
    )
    feat.add_argument(
        "--sliding-window-params", default="128,4,0.8,gaussian,0.25",
        help="window,sw_batch,overlap,mode,sigma for MONAI sliding-window "
        "inference (mode in {constant,gaussian}). anatomix-dev-vit needs "
        "window=128.",
    )
    feat.add_argument(
        "--feature-normalization",
        choices=["l2", "standardized", "none"], default="l2",
        help="Per-voxel network-feature normalization (not applied to MIND).",
    )
    feat.add_argument(
        "--use-mindssc", choices=["both", "feats-only", "mindssc-only"],
        default="both",
        help="Which feature families to register (mindssc-only loads no model).",
    )
    feat.add_argument(
        "--mindssc-params", default="1,2",
        help="MIND-SSC radius,dilation.",
    )

    _add_custom_backbone_args(parser)

    out = parser.add_argument_group("outputs")
    out.add_argument("--output-dir", default="./", help="Output directory.")
    out.add_argument(
        "--exp-name", default=None,
        help="Optional prefix prepended to every output and the metrics CSV.",
    )
    out.add_argument(
        "--output-transformation-convention",
        choices=["ants", "scipy", "pytorch"], default="ants",
        help="Transform export format.",
    )
    out.add_argument(
        "--collapse-output-transforms", dest="collapse", type=int,
        choices=[0, 1], default=1,
        help="1: one composed transform; 0: one cumulative snapshot per stage.",
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument("--seed", type=int, default=12345, help="Random seed.")
    misc.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=True,
        help="Print resolved inputs, stage progress, outputs, and metrics.",
    )
    return parser


def _add_custom_backbone_args(parser):
    """Add the ``--backbone custom`` sub-CLI (consumed only for custom)."""
    grp = parser.add_argument_group(
        "custom backbone (only used with --backbone custom)"
    )
    grp.add_argument("--custom-arch", choices=["unet", "vit"], default=None)
    grp.add_argument("--custom-weights", default=None, help="Path to .pth.")
    # UNet architecture (dimension=3, input_nc=1 are fixed); defaults mirror the
    # built-in 'anatomix' UNet.
    grp.add_argument("--unet-output-nc", type=int, default=16)
    grp.add_argument("--unet-num-downs", type=int, default=4)
    grp.add_argument("--unet-ngf", type=int, default=16)
    grp.add_argument("--unet-norm", default="batch")
    grp.add_argument("--unet-final-act", default="none")
    grp.add_argument("--unet-activation", default="relu")
    grp.add_argument("--unet-pad-type", default="reflect")
    grp.add_argument(
        "--unet-doubleconv", action=argparse.BooleanOptionalAction, default=True
    )
    grp.add_argument(
        "--unet-residual-connection",
        action=argparse.BooleanOptionalAction, default=False,
    )
    grp.add_argument("--unet-pooling", default="Max")
    grp.add_argument("--unet-interp", default="nearest")
    grp.add_argument(
        "--unet-use-skip-connection",
        action=argparse.BooleanOptionalAction, default=True,
    )
    grp.add_argument("--unet-norm-eps", type=float, default=1e-5)
    # ViT architecture (input_channels=1 fixed); defaults mirror
    # 'anatomix-dev-vit'.
    grp.add_argument("--vit-num-classes", type=int, default=32)
    grp.add_argument("--vit-embed-dim", type=int, default=396)
    grp.add_argument("--vit-eva-depth", type=int, default=12)
    grp.add_argument("--vit-eva-numheads", type=int, default=6)
    grp.add_argument("--vit-patch-embed-size", default="8x8x8")
    grp.add_argument("--vit-input-shape", default="128x128x128")
    grp.add_argument("--vit-num-register-tokens", type=int, default=8)
    grp.add_argument("--vit-init-values", type=float, default=0.1)
    grp.add_argument(
        "--vit-scale-attn-inner",
        action=argparse.BooleanOptionalAction, default=True,
    )
    grp.add_argument(
        "--vit-qk-norm", action=argparse.BooleanOptionalAction, default=True,
    )
    grp.add_argument("--vit-out-norm", default="demean")
    grp.add_argument("--vit-out-norm-eps", type=float, default=1e-2)
    grp.add_argument("--vit-register-init-std", type=float, default=0.02)
    grp.add_argument("--vit-in-eps", type=float, default=1e-2)


# --------------------------------------------------------------------------- #
# Per-stage schedule parsing
# --------------------------------------------------------------------------- #
def _axtuple(value):
    return tuple(int(x) for x in value.split("x"))


def _int_schedule(token, name):
    try:
        return [int(x) for x in token.split("x")]
    except ValueError:
        raise ValueError(f"{name}: expected 'AxBx...' integers, got {token!r}.")


def _split_stages(value, n, name):
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != n:
        raise ValueError(
            f"{name}: expected {n} comma-separated entries (one per --transform "
            f"stage), got {len(parts)}."
        )
    return parts


def _resolve_step_sizes(value, kinds, n):
    """Per-stage Adam learning rate.

    When omitted, the default is stage-aware: 1.0 for deformable stages (the
    SOTA setting) and 0.1 for the far more sensitive linear (rigid/affine)
    stages, which diverge at a deformable-scale learning rate.
    """
    if value is None:
        return [1.0 if kinds[i] == "deformable" else 0.1 for i in range(n)]
    values = [float(p) for p in _split_stages(value, n, "--step-size")]
    if any(v <= 0 for v in values):
        raise ValueError("--step-size: values must be positive.")
    return values


def _resolve_shrink(value, n):
    tokens = [ "8x4x2x1" ] * n if value is None else _split_stages(
        value, n, "--shrink-factors"
    )
    schedules = [_int_schedule(t, "--shrink-factors") for t in tokens]
    for sched in schedules:
        if any(s <= 0 for s in sched):
            raise ValueError("--shrink-factors: values must be positive.")
        if any(b > a for a, b in zip(sched, sched[1:])):
            raise ValueError(
                "--shrink-factors: each schedule must be monotonically "
                f"non-increasing, got {sched}."
            )
    return schedules


def _resolve_iterations(value, n, shrinks):
    if value is None:
        schedules = [[100] * len(s) for s in shrinks]
    else:
        tokens = _split_stages(value, n, "--iterations")
        schedules = [_int_schedule(t, "--iterations") for t in tokens]
    for i, sched in enumerate(schedules):
        if len(sched) != len(shrinks[i]):
            raise ValueError(
                f"--iterations: stage {i} has {len(sched)} levels but "
                f"--shrink-factors has {len(shrinks[i])}."
            )
        if any(v < 0 for v in sched):
            raise ValueError("--iterations: values must be non-negative.")
    return schedules


def _resolve_cc_kernels(value, losses, shrinks, n):
    is_cc = [losses[i] in CC_LOSSES for i in range(n)]
    if value is None:
        # No schedule requested: leave each stage to FireANTs' own default CC
        # kernel size. Task-appropriate schedules (which vary by dataset and
        # pyramid) are passed explicitly via --cc-kernel-widths.
        return [None] * n
    tokens = _split_stages(value, n, "--cc-kernel-widths")
    kernels = []
    for i in range(n):
        if not is_cc[i]:
            kernels.append(None)
            continue
        if tokens[i] == "na":
            raise ValueError(
                f"--cc-kernel-widths: stage {i} uses a CC loss; provide widths, "
                "not 'na'."
            )
        widths = _int_schedule(tokens[i], "--cc-kernel-widths")
        if len(widths) != len(shrinks[i]):
            raise ValueError(
                f"--cc-kernel-widths: stage {i} has {len(widths)} widths but "
                f"--shrink-factors has {len(shrinks[i])} levels."
            )
        if any(w <= 0 or w % 2 == 0 for w in widths):
            raise ValueError(
                "--cc-kernel-widths: widths must be positive and odd."
            )
        kernels.append(widths)
    return kernels


def _resolve_sigmas(value, kinds, default, name, n):
    is_def = [kinds[i] == "deformable" for i in range(n)]
    if value is None:
        tokens = [str(default) if is_def[i] else "na" for i in range(n)]
    else:
        tokens = _split_stages(value, n, name)
    sigmas = []
    for i in range(n):
        if not is_def[i]:
            sigmas.append(None)
            continue
        if tokens[i] == "na":
            raise ValueError(
                f"{name}: stage {i} is deformable; provide a value, not 'na'."
            )
        sigma = float(tokens[i])
        if sigma < 0:
            raise ValueError(f"{name}: values must be non-negative.")
        sigmas.append(sigma)
    return sigmas


def build_stages(args):
    """Resolve the per-stage registration specs from parsed arguments."""
    kinds = [k.strip() for k in args.transform.split(",") if k.strip()]
    if not kinds:
        raise ValueError("--transform: at least one stage is required.")
    for kind in kinds:
        if kind not in TRANSFORM_RANK:
            raise ValueError(
                f"--transform: unknown stage {kind!r}; choose from "
                "{rigid,affine,deformable}."
            )
    ranks = [TRANSFORM_RANK[k] for k in kinds]
    if any(b < a for a, b in zip(ranks, ranks[1:])):
        raise ValueError(
            "--transform: stages must be ordered rigid <= affine <= deformable, "
            f"got {kinds}."
        )
    n = len(kinds)

    if args.loss is None:
        losses = [None] * n
    else:
        losses = _split_stages(args.loss, n, "--loss")
        for loss in losses:
            if loss not in VALID_LOSSES:
                raise ValueError(
                    f"--loss: unknown loss {loss!r}; choose from "
                    f"{sorted(VALID_LOSSES)}."
                )

    steps = _resolve_step_sizes(args.step_size, kinds, n)
    shrinks = _resolve_shrink(args.shrink_factors, n)
    iters = _resolve_iterations(args.iterations, n, shrinks)
    cc_kernels = _resolve_cc_kernels(args.cc_kernel_widths, losses, shrinks, n)
    grad_sigmas = _resolve_sigmas(
        args.smooth_grad_sigma, kinds, 1.0, "--smooth-grad-sigma", n
    )
    warp_sigmas = _resolve_sigmas(
        args.smooth_warp_sigma, kinds, 0.5, "--smooth-warp-sigma", n
    )

    return [
        {
            "kind": kinds[i], "loss": losses[i], "step": steps[i],
            "shrink": shrinks[i], "iters": iters[i], "cc_kernel": cc_kernels[i],
            "smooth_grad": grad_sigmas[i], "smooth_warp": warp_sigmas[i],
        }
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Feature / backbone / input resolution
# --------------------------------------------------------------------------- #
def parse_sliding_window(args):
    parts = [p.strip() for p in args.sliding_window_params.split(",")]
    if len(parts) != 5:
        raise ValueError(
            "--sliding-window-params: expected window,sw_batch,overlap,mode,"
            f"sigma (5 values), got {len(parts)}."
        )
    args.sw_window = int(parts[0])
    args.sw_batch = int(parts[1])
    args.sw_overlap = float(parts[2])
    args.sw_mode = parts[3]
    args.sw_sigma = float(parts[4])
    if args.sw_window <= 0 or args.sw_batch <= 0:
        raise ValueError("--sliding-window-params: window/sw_batch must be > 0.")
    if not 0.0 <= args.sw_overlap < 1.0:
        raise ValueError("--sliding-window-params: overlap must be in [0, 1).")
    if args.sw_mode not in ("constant", "gaussian"):
        raise ValueError(
            "--sliding-window-params: mode must be 'constant' or 'gaussian'."
        )


def parse_mindssc(args):
    parts = [p.strip() for p in args.mindssc_params.split(",")]
    if len(parts) != 2:
        raise ValueError("--mindssc-params: expected radius,dilation.")
    args.mindssc_radius = int(parts[0])
    args.mindssc_dilation = int(parts[1])
    if args.mindssc_radius <= 0 or args.mindssc_dilation <= 0:
        raise ValueError("--mindssc-params: radius/dilation must be > 0.")


def build_custom_kwargs(args):
    args.unet_kwargs = None
    args.vit_kwargs = None
    if args.backbone != "custom":
        return
    if not args.custom_weights:
        raise ValueError("--backbone custom requires --custom-weights.")
    if args.custom_arch == "unet":
        args.unet_kwargs = dict(
            output_nc=args.unet_output_nc, num_downs=args.unet_num_downs,
            ngf=args.unet_ngf, norm=args.unet_norm,
            final_act=args.unet_final_act, activation=args.unet_activation,
            pad_type=args.unet_pad_type, doubleconv=args.unet_doubleconv,
            residual_connection=args.unet_residual_connection,
            pooling=args.unet_pooling, interp=args.unet_interp,
            use_skip_connection=args.unet_use_skip_connection,
            norm_eps=args.unet_norm_eps,
        )
    elif args.custom_arch == "vit":
        args.vit_kwargs = dict(
            num_classes=args.vit_num_classes, embed_dim=args.vit_embed_dim,
            eva_depth=args.vit_eva_depth, eva_numheads=args.vit_eva_numheads,
            patch_embed_size=_axtuple(args.vit_patch_embed_size),
            input_shape=_axtuple(args.vit_input_shape),
            num_register_tokens=args.vit_num_register_tokens,
            init_values=args.vit_init_values,
            scale_attn_inner=args.vit_scale_attn_inner,
            qk_norm=args.vit_qk_norm, out_norm=args.vit_out_norm,
            out_norm_eps=args.vit_out_norm_eps,
            register_init_std=args.vit_register_init_std,
            in_eps=args.vit_in_eps,
        )
    else:
        raise ValueError(
            "--backbone custom requires --custom-arch {unet,vit}."
        )


def _list_nifti(directory, role):
    if not os.path.isdir(directory):
        raise ValueError(f"{role}: not a directory: {directory}")
    files = sorted(f for f in os.listdir(directory) if f.endswith(NIFTI_EXTS))
    if not files:
        raise ValueError(f"{role}: no .nii/.nii.gz files in {directory}")
    return [os.path.abspath(os.path.join(directory, f)) for f in files]


def _abspath(value):
    return os.path.abspath(value) if value else None


def _single_pair(args):
    columns = ["fixed", "moving"]
    pair = {"fixed": _abspath(args.fixed), "moving": _abspath(args.moving)}
    if bool(args.fixed_mask) != bool(args.moving_mask):
        raise ValueError("Provide both --fixed-mask and --moving-mask, or neither.")
    if args.fixed_mask:
        pair["fixed_mask"] = _abspath(args.fixed_mask)
        pair["moving_mask"] = _abspath(args.moving_mask)
        columns += ["fixed_mask", "moving_mask"]
    if args.fixed_seg and not args.moving_seg:
        raise ValueError("--fixed-seg requires --moving-seg.")
    if args.moving_seg:
        if args.fixed_seg:
            pair["fixed_seg"] = _abspath(args.fixed_seg)
            columns.append("fixed_seg")
        pair["moving_seg"] = _abspath(args.moving_seg)
        columns.append("moving_seg")
    return [pair], columns


def _dir_pairs(args):
    fixed = _list_nifti(args.fixed_dir, "--fixed-dir")
    moving = _list_nifti(args.moving_dir, "--moving-dir")
    if len(fixed) != len(moving):
        raise ValueError(
            f"--fixed-dir ({len(fixed)}) and --moving-dir ({len(moving)}) must "
            "have equal file counts."
        )
    columns = ["fixed", "moving"]
    data = {"fixed": fixed, "moving": moving}

    if bool(args.fixed_mask_dir) != bool(args.moving_mask_dir):
        raise ValueError(
            "Provide both --fixed-mask-dir and --moving-mask-dir, or neither."
        )
    if args.fixed_mask_dir:
        for role, directory, col in [
            ("--fixed-mask-dir", args.fixed_mask_dir, "fixed_mask"),
            ("--moving-mask-dir", args.moving_mask_dir, "moving_mask"),
        ]:
            files = _list_nifti(directory, role)
            if len(files) != len(fixed):
                raise ValueError(f"{role}: count must match the image count.")
            data[col] = files
            columns.append(col)

    if args.fixed_seg_dir and not args.moving_seg_dir:
        raise ValueError("--fixed-seg-dir requires --moving-seg-dir.")
    if args.moving_seg_dir:
        if args.fixed_seg_dir:
            files = _list_nifti(args.fixed_seg_dir, "--fixed-seg-dir")
            if len(files) != len(fixed):
                raise ValueError("--fixed-seg-dir: count must match images.")
            data["fixed_seg"] = files
            columns.append("fixed_seg")
        files = _list_nifti(args.moving_seg_dir, "--moving-seg-dir")
        if len(files) != len(fixed):
            raise ValueError("--moving-seg-dir: count must match images.")
        data["moving_seg"] = files
        columns.append("moving_seg")

    pairs = [
        {col: data[col][i] for col in columns} for i in range(len(fixed))
    ]
    return pairs, columns


def resolve_inputs(args):
    """Resolve the input mode into (pairs, input_columns)."""
    has_single = bool(args.fixed) or bool(args.moving)
    has_dir = bool(args.fixed_dir) or bool(args.moving_dir)
    has_csv = bool(args.registration_pairs_csv)
    if sum([has_single, has_dir, has_csv]) != 1:
        raise ValueError(
            "Provide exactly one input mode: --fixed/--moving, "
            "--fixed-dir/--moving-dir, or --registration-pairs-csv."
        )
    if has_single:
        if not (args.fixed and args.moving):
            raise ValueError("Single-pair mode needs both --fixed and --moving.")
        return _single_pair(args)
    if has_dir:
        if not (args.fixed_dir and args.moving_dir):
            raise ValueError("Batch dir mode needs --fixed-dir and --moving-dir.")
        return _dir_pairs(args)
    columns, rows = read_pairs_csv(args.registration_pairs_csv)
    return rows, columns


def _validate_volume(path, role):
    if not os.path.isfile(path):
        raise ValueError(f"{role}: file not found: {path}")
    if not path.endswith(NIFTI_EXTS):
        raise ValueError(f"{role}: expected a .nii/.nii.gz file: {path}")
    shape = nib.load(path).header.get_data_shape()
    if len(shape) < 3:
        raise ValueError(f"{role}: expected a 3D volume, got shape {tuple(shape)}.")
    if len(shape) > 3 and any(d != 1 for d in shape[3:]):
        raise ValueError(
            f"{role}: expected a single-channel 3D volume, got {tuple(shape)}."
        )


def validate_volumes(pairs):
    for index, pair in enumerate(pairs):
        for col, path in pair.items():
            if path is not None:
                _validate_volume(path, f"pair {index} [{col}]")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None):
    """Parse arguments, validate, and run the registration pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        parse_sliding_window(args)
        parse_mindssc(args)
        if args.backbone == "anatomix-dev-vit" and args.sw_window != 128:
            raise ValueError(
                "anatomix-dev-vit requires a 128-voxel sliding window "
                "(--sliding-window-params window=128)."
            )
        build_custom_kwargs(args)
        stages = build_stages(args)
        pairs, input_columns = resolve_inputs(args)
        validate_volumes(pairs)
    except ValueError as error:
        parser.error(str(error))

    args.output_dir = os.path.abspath(args.output_dir)

    # Import here so --help and validation work without the FireANTs backend.
    from .pipeline import run

    run(args, pairs, input_columns, stages)


if __name__ == "__main__":
    main()
