"""Argument parsing and validation for ``anatomix-register.py``.

This module defines the full command-line interface and resolves it into the
inputs the pipeline consumes: a list of fixed/moving (and optional mask/seg)
pairs, the per-stage registration specs, and the feature/output settings. It
also holds every preflight check: paths, each mask/segmentation against the
geometry of its own image, and the stage schedules against the image sizes.
FireANTs is imported only lazily (inside :func:`main`, after all validation),
so ``--help`` and every argument/validation error work without the backend
installed.
"""
import argparse
import os

import nibabel as nib
import numpy as np

from .io_utils import (
    KEYPOINT_COLUMNS,
    KEYPOINT_CONVENTIONS,
    KEYPOINT_EXT,
    METRIC_COLUMNS,
    NIFTI_EXTS,
    VOLUME_COLUMNS,
    read_keypoints,
    read_pairs_csv,
)

FEATURE_CHOICES = ("anatomix+mindssc", "anatomix", "mindssc", "intensity")
# Feature families that need the anatomix backbone loaded.
MODEL_FEATURES = ("anatomix+mindssc", "anatomix")
TRANSFORM_RANK = {"rigid": 0, "affine": 1, "deformable": 2}
VALID_LOSSES = {"cc", "mi", "mse", "masked_cc", "masked_mi", "masked_mse"}
# Losses that consume a CC kernel schedule (auto default resolves to a CC loss).
CC_LOSSES = {None, "cc", "masked_cc"}
# FireANTs floors every pyramid level at this many voxels per axis, as
# ``max(int(size / shrink), MIN_IMG_SIZE)`` (fireants/utils/globals.py).
MIN_IMG_SIZE = 32
# Tolerance for a mask/segmentation's affine against its image's: below any
# physically meaningful difference (0.1 um of origin, 6e-3 degrees of
# direction), above the noise of a header round-tripped through another tool.
GEOMETRY_ATOL = 1e-4


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
            "features, MIND-SSC descriptors, or the raw intensities. Supports "
            "rigid/affine/deformable stages, masked and unmasked losses, "
            "label warping, keypoint warping with target-registration error, "
            "transform export, Dice, and fold counting, in single-pair and "
            "batch modes."
        ),
    )

    mode = parser.add_argument_group("input mode (choose exactly one)")
    mode.add_argument(
        "--fixed", help="Single-pair fixed image (.nii/.nii.gz).")
    mode.add_argument(
        "--moving", help="Single-pair moving image (.nii/.nii.gz).")
    mode.add_argument(
        "--fixed-dir",
        help="Batch: directory of fixed images. Fixed and moving directories "
        "are paired by independent lexicographic sort (equal counts required); "
        "use --registration-pairs-csv when filenames do not correspond by sort "
        "order.",
    )
    mode.add_argument(
        "--moving-dir", help="Batch: directory of moving images.")
    mode.add_argument(
        "--registration-pairs-csv",
        help="Batch: CSV with a header and columns fixed,moving"
        "[,fixed_mask,moving_mask,fixed_seg,moving_seg"
        ",fixed_keypoints,moving_keypoints].",
    )

    aux = parser.add_argument_group("masks, segmentations and keypoints")
    aux.add_argument("--fixed-mask", help="Fixed registration mask.")
    aux.add_argument("--moving-mask", help="Moving registration mask.")
    aux.add_argument("--fixed-seg", help="Fixed segmentation (enables Dice).")
    aux.add_argument(
        "--moving-seg", help="Moving segmentation (warped to fixed).")
    aux.add_argument(
        "--fixed-keypoints",
        help="Fixed-image keypoint CSV (warped into moving space).",
    )
    aux.add_argument(
        "--moving-keypoints",
        help="Corresponding moving-image keypoint CSV (enables TRE).",
    )
    aux.add_argument("--fixed-mask-dir",
                     help="Batch directory of fixed masks.")
    aux.add_argument("--moving-mask-dir",
                     help="Batch directory of moving masks.")
    aux.add_argument("--fixed-seg-dir", help="Batch directory of fixed segs.")
    aux.add_argument("--moving-seg-dir",
                     help="Batch directory of moving segs.")
    aux.add_argument("--fixed-keypoints-dir",
                     help="Batch directory of fixed keypoint CSVs.")
    aux.add_argument("--moving-keypoints-dir",
                     help="Batch directory of moving keypoint CSVs.")
    aux.add_argument(
        "--keypoint-convention", choices=list(KEYPOINT_CONVENTIONS),
        default="lps",
        help="Keypoint CSV coordinate convention: lps/ras world mm, or "
        "voxel for the ITK continuous index (i,j,k).",
    )

    clip = parser.add_argument_group(
        "intensity clipping (reused across a batch)")
    clip.add_argument("--fixed-minclip", type=float, help="Fixed lower clip.")
    clip.add_argument("--fixed-maxclip", type=float, help="Fixed upper clip.")
    clip.add_argument("--moving-minclip", type=float,
                      help="Moving lower clip.")
    clip.add_argument("--moving-maxclip", type=float,
                      help="Moving upper clip.")

    tf = parser.add_argument_group(
        "transform chain (per-stage lists are comma-separated, one entry per "
        "--transform stage)"
    )
    tf.add_argument(
        "--initialization", choices=["none", "center-of-mass", "moments"],
        default="none",
        help="Closed-form moment initialization run before the stage chain, "
        "computed from the normalized intensity images (restricted to the "
        "masks when given), not from the features.",
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
        "0.01 for rigid/affine stages.",
    )
    tf.add_argument(
        "--shrink-factors", default=None,
        help="Per-stage 'AxBx...' resolution schedule, strictly decreasing. "
        "Every level is floored at 32 voxels per axis by the backend, so a "
        "multi-resolution schedule needs > 33 voxels on every axis. "
        "Default 6x4x2x1.",
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
        "--features", choices=list(FEATURE_CHOICES),
        default="anatomix+mindssc",
        help="Which channel families to register. 'mindssc' and 'intensity' "
        "load no backbone; 'intensity' registers the clipped, min-max "
        "normalized image itself (the classic MI/CC baseline).",
    )
    feat.add_argument(
        "--backbone",
        choices=["anatomix", "anatomix-dev", "anatomix-dev-vit", "custom"],
        default="anatomix-dev-vit",
        help="anatomix feature extractor (custom: see custom-backbone flags). "
        "Only used when --features includes anatomix.",
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
    misc.add_argument("--tolerance", type=float, default=1e-6,
                      help="FireANTs convergence tolerance (loss slope over the last 10 iterations); use inf to disable early stopping.")
    misc.add_argument(
        "--device", default="auto",
        help="Compute device: 'auto' (pick the visible CUDA device with the "
        "most free memory), 'cpu', 'cuda', or 'cuda:N'. Honors "
        "CUDA_VISIBLE_DEVICES, which also restricts 'auto'. Note that N is a "
        "CUDA index, which matches nvidia-smi's only when "
        "CUDA_DEVICE_ORDER=PCI_BUS_ID is set.",
    )
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
        raise ValueError(
            f"{name}: expected 'AxBx...' integers, got {token!r}.")


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

    The default is stage-aware: 1.0 for deformable stages, and 0.01 for the far
    more sensitive linear ones, which diverge at a deformable-scale rate.
    """
    if value is None:
        return [1.0 if kinds[i] == "deformable" else 0.01 for i in range(n)]
    values = [float(p) for p in _split_stages(value, n, "--step-size")]
    if any(v <= 0 for v in values):
        raise ValueError("--step-size: values must be positive.")
    return values


def _resolve_shrink(value, n):
    """Per-stage resolution schedule.

    Levels must be *strictly* decreasing: FireANTs rejects a repeated level,
    but only once the registration object is built, long after the backbone has
    loaded.
    """
    # 6 is the coarsest level the backend actually delivers on the 192x160x192
    # reference data: anything coarser is clamped back to MIN_IMG_SIZE.
    tokens = ["6x4x2x1"] * n if value is None else _split_stages(
        value, n, "--shrink-factors"
    )
    schedules = [_int_schedule(t, "--shrink-factors") for t in tokens]
    for sched in schedules:
        if any(s <= 0 for s in sched):
            raise ValueError("--shrink-factors: values must be positive.")
        if any(b >= a for a, b in zip(sched, sched[1:])):
            raise ValueError(
                "--shrink-factors: each schedule must be strictly decreasing, "
                f"got {sched}."
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
        # No schedule requested: leave each stage on FireANTs' own default
        # kernel size. Good schedules vary by dataset and pyramid.
        return [None] * n
    tokens = _split_stages(value, n, "--cc-kernel-widths")
    kernels = []
    for i in range(n):
        if not is_cc[i]:
            if tokens[i] != "na":
                raise ValueError(
                    f"--cc-kernel-widths: stage {i} does not use a CC loss; use "
                    "'na' (CC kernel widths apply only to cc/masked_cc stages)."
                )
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
            if tokens[i] != "na":
                raise ValueError(
                    f"{name}: stage {i} is not deformable; use 'na' (smoothing "
                    "sigmas apply only to deformable stages)."
                )
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
            "tolerance": args.tolerance,
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
        raise ValueError(
            "--sliding-window-params: window/sw_batch must be > 0.")
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


def reject_custom_backbone_flags(args, because):
    """Reject ``--custom-*`` flags that cannot take effect.

    They would otherwise be ignored and the run would quietly complete against
    something other than the requested weights.
    """
    foreign = [
        flag for flag, value in (
            ("--custom-arch", args.custom_arch),
            ("--custom-weights", args.custom_weights),
        ) if value
    ]
    if foreign:
        raise ValueError(
            f"{', '.join(foreign)}: {because}, and would otherwise be ignored."
        )


def build_custom_kwargs(args):
    args.unet_kwargs = None
    args.vit_kwargs = None
    if args.backbone != "custom":
        reject_custom_backbone_flags(
            args,
            f"only used with --backbone custom (got --backbone "
            f"{args.backbone})",
        )
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


def _list_dir(directory, role, extensions, description):
    if not os.path.isdir(directory):
        raise ValueError(f"{role}: not a directory: {directory}")
    files = sorted(f for f in os.listdir(directory) if f.endswith(extensions))
    if not files:
        raise ValueError(f"{role}: no {description} files in {directory}")
    return [os.path.abspath(os.path.join(directory, f)) for f in files]


def _list_nifti(directory, role):
    return _list_dir(directory, role, NIFTI_EXTS, ".nii/.nii.gz")


def _list_keypoints(directory, role):
    return _list_dir(directory, role, (KEYPOINT_EXT,), ".csv")


def _abspath(value):
    return os.path.abspath(value) if value else None


def _single_pair(args):
    columns = ["fixed", "moving"]
    pair = {"fixed": _abspath(args.fixed), "moving": _abspath(args.moving)}
    if bool(args.fixed_mask) != bool(args.moving_mask):
        raise ValueError(
            "Provide both --fixed-mask and --moving-mask, or neither.")
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
    # Keypoints mirror segmentations with the roles swapped: the transform maps
    # fixed points into moving space, so the fixed set is the one warped.
    if args.moving_keypoints and not args.fixed_keypoints:
        raise ValueError("--moving-keypoints requires --fixed-keypoints.")
    if args.fixed_keypoints:
        pair["fixed_keypoints"] = _abspath(args.fixed_keypoints)
        columns.append("fixed_keypoints")
        if args.moving_keypoints:
            pair["moving_keypoints"] = _abspath(args.moving_keypoints)
            columns.append("moving_keypoints")
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

    if args.moving_keypoints_dir and not args.fixed_keypoints_dir:
        raise ValueError(
            "--moving-keypoints-dir requires --fixed-keypoints-dir.")
    if args.fixed_keypoints_dir:
        for role, directory, col in [
            ("--fixed-keypoints-dir", args.fixed_keypoints_dir,
             "fixed_keypoints"),
            ("--moving-keypoints-dir", args.moving_keypoints_dir,
             "moving_keypoints"),
        ]:
            if not directory:
                continue
            files = _list_keypoints(directory, role)
            if len(files) != len(fixed):
                raise ValueError(f"{role}: count must match the image count.")
            data[col] = files
            columns.append(col)

    pairs = [
        {col: data[col][i] for col in columns} for i in range(len(fixed))
    ]
    return pairs, columns


_SINGLE_AUX = (
    "fixed_mask", "moving_mask", "fixed_seg", "moving_seg",
    "fixed_keypoints", "moving_keypoints",
)
_DIR_AUX = (
    "fixed_mask_dir", "moving_mask_dir", "fixed_seg_dir", "moving_seg_dir",
    "fixed_keypoints_dir", "moving_keypoints_dir",
)


def _reject_foreign_aux_flags(args, keep):
    """Error if aux flags belonging to a different input mode are set.

    ``keep`` is the tuple of aux flag names valid for the active mode; any other
    aux flag that is set is a silent no-op and therefore rejected.
    """
    foreign = [f for f in _SINGLE_AUX + _DIR_AUX
               if f not in keep and getattr(args, f)]
    if foreign:
        flags = ", ".join("--" + f.replace("_", "-") for f in foreign)
        raise ValueError(
            f"{flags}: not valid for the chosen input mode and would be "
            "ignored. Use the mask/seg/keypoint flags (or CSV columns) that "
            "match the input mode."
        )


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
            raise ValueError(
                "Single-pair mode needs both --fixed and --moving.")
        _reject_foreign_aux_flags(args, _SINGLE_AUX)
        return _single_pair(args)
    if has_dir:
        if not (args.fixed_dir and args.moving_dir):
            raise ValueError(
                "Batch dir mode needs --fixed-dir and --moving-dir.")
        _reject_foreign_aux_flags(args, _DIR_AUX)
        return _dir_pairs(args)
    _reject_foreign_aux_flags(args, ())
    columns, rows = read_pairs_csv(args.registration_pairs_csv)
    # A passthrough column named after a metric would be overwritten by it and
    # duplicated in the metrics-CSV header.
    reserved = [col for col in columns if col in METRIC_COLUMNS]
    if reserved:
        raise ValueError(
            f"{args.registration_pairs_csv}: column(s) "
            f"{', '.join(reserved)} collide with the metrics the run writes "
            f"({', '.join(METRIC_COLUMNS)}); rename them."
        )
    return rows, columns


def _volume_geometry(path, role):
    """Validate one volume path and return its ``(spatial_shape, affine)``.

    Header-only (nibabel loads lazily), so preflight stays cheap even for a
    large batch and needs neither the GPU nor the FireANTs backend.
    """
    if not os.path.isfile(path):
        raise ValueError(f"{role}: file not found: {path}")
    if not path.endswith(NIFTI_EXTS):
        raise ValueError(f"{role}: expected a .nii/.nii.gz file: {path}")
    image = nib.load(path)
    shape = image.header.get_data_shape()
    if len(shape) < 3:
        raise ValueError(
            f"{role}: expected a 3D volume, got shape {tuple(shape)}.")
    if len(shape) > 3 and any(d != 1 for d in shape[3:]):
        raise ValueError(
            f"{role}: expected a single-channel 3D volume, got {tuple(shape)}."
        )
    return tuple(int(d) for d in shape[:3]), image.affine


def validate_volumes(pairs):
    """Validate every volume path and collect the geometries.

    Returns
    -------
    list of dict
        One dict per pair, mapping each present volume column to its
        ``(spatial_shape, affine)``; consumed by :func:`validate_geometry` and
        :func:`validate_pyramid`.
    """
    geometries = []
    for index, pair in enumerate(pairs):
        geometry = {}
        for col in VOLUME_COLUMNS:
            path = pair.get(col)
            if path is not None:
                geometry[col] = _volume_geometry(path, f"pair {index} [{col}]")
        geometries.append(geometry)
    return geometries


def validate_keypoints(pairs):
    """Validate every keypoint CSV and check the fixed/moving counts match.

    Keypoints are absent from :func:`validate_volumes` and
    :func:`validate_geometry` on purpose: they carry no voxel grid to check.
    """
    for index, pair in enumerate(pairs):
        counts = {}
        for col in KEYPOINT_COLUMNS:
            path = pair.get(col)
            if path is None:
                continue
            role = f"pair {index} [{col}]"
            if not os.path.isfile(path):
                raise ValueError(f"{role}: file not found: {path}")
            if not path.endswith(KEYPOINT_EXT):
                raise ValueError(f"{role}: expected a .csv file: {path}")
            _, _, coordinates = read_keypoints(path)
            counts[col] = len(coordinates)
        if len(counts) == 2 and len(set(counts.values())) != 1:
            raise ValueError(
                f"pair {index}: fixed_keypoints has "
                f"{counts['fixed_keypoints']} points but moving_keypoints "
                f"has {counts['moving_keypoints']}; "
                "they must correspond row by row."
            )


# Each auxiliary volume is consumed on the grid of exactly one image.
# Keypoints are excluded on purpose (see validate_keypoints).
_AUX_IMAGE = {
    "fixed_mask": "fixed", "fixed_seg": "fixed",
    "moving_mask": "moving", "moving_seg": "moving",
}


def validate_geometry(geometries):
    """Check that every mask/segmentation sits on the grid of its own image.

    Masks and segmentations are consumed voxelwise and never resampled: a mask
    gates the feature volume, and a segmentation is warped by a grid expressed
    in *its image's* normalized coordinates. A differing grid is therefore
    reinterpreted rather than resampled, putting the labels at the wrong
    physical locations.
    """
    for index, geometry in enumerate(geometries):
        for aux, image in _AUX_IMAGE.items():
            if aux not in geometry or image not in geometry:
                continue
            aux_shape, aux_affine = geometry[aux]
            image_shape, image_affine = geometry[image]
            if aux_shape != image_shape:
                raise ValueError(
                    f"pair {index} [{aux}]: shape {aux_shape} does not match "
                    f"[{image}] {image_shape}; resample it onto the {image} "
                    "image's grid first."
                )
            if not np.allclose(
                aux_affine, image_affine, rtol=0.0, atol=GEOMETRY_ATOL,
            ):
                worst = float(np.max(np.abs(aux_affine - image_affine)))
                raise ValueError(
                    f"pair {index} [{aux}]: voxel-to-world affine differs from "
                    f"[{image}] by up to {worst:.3g} (tolerance "
                    f"{GEOMETRY_ATOL:g}), so it would be applied at the wrong "
                    f"physical locations; resample it onto the {image} image's "
                    "grid first."
                )


def _check_stage_size(role, shape, index, stage_index, stage):
    """Check one image's size against one stage's pyramid and CC kernel."""
    smallest = min(shape)
    axis = shape.index(smallest)
    where = f"axis {axis} of {shape} has {smallest} voxels"
    schedule = "x".join(str(s) for s in stage["shrink"])

    if any(s > 1 for s in stage["shrink"]):
        # Levels with shrink > 1 are resampled by an FFT crop, which needs one
        # voxel of margin on each side of the floored target size. Below that
        # the backend raises an opaque error, or silently truncates the axis to
        # a handful of voxels and registers that. Pyramid depth is irrelevant:
        # every level below the floor becomes the floor.
        if smallest < MIN_IMG_SIZE + 2:
            # Dropping to one resolution rescues the stage unless it is
            # deformable and below the floor, where the warp field is the
            # problem (the branch below) and only more voxels can help.
            single_res_works = (
                stage["kind"] != "deformable" or smallest >= MIN_IMG_SIZE
            )
            hint = (
                "Use --shrink-factors 1 for this stage, or resample the volume."
                if single_res_works else "Resample or pad the volume."
            )
            raise ValueError(
                f"pair {index} [{role}]: {where}, but stage {stage_index} "
                f"({stage['kind']}, --shrink-factors {schedule}) is "
                f"multi-resolution and the backend floors every level at "
                f"{MIN_IMG_SIZE} voxels per axis, so it needs at least "
                f"{MIN_IMG_SIZE + 2}. {hint}"
            )
    elif stage["kind"] == "deformable":
        # Nothing is resampled at shrink 1, but the warp field is still
        # allocated at max(size, MIN_IMG_SIZE) and then mismatches the image.
        if smallest < MIN_IMG_SIZE:
            raise ValueError(
                f"pair {index} [{role}]: {where}, but stage {stage_index} "
                f"(deformable) allocates its warp field at no fewer than "
                f"{MIN_IMG_SIZE} voxels per axis, so it needs at least "
                f"{MIN_IMG_SIZE}. Resample or pad the volume."
            )

    if stage["cc_kernel"] is None:
        return
    for level, (scale, width) in enumerate(
        zip(stage["shrink"], stage["cc_kernel"])
    ):
        level_size = max(smallest // scale, MIN_IMG_SIZE)
        if width > level_size:
            raise ValueError(
                f"pair {index} [{role}]: --cc-kernel-widths stage "
                f"{stage_index} level {level} is {width} voxels, but at shrink "
                f"{scale} the smallest axis (axis {axis} of {shape}) is only "
                f"{level_size}. The CC window must fit inside the image."
            )


def validate_pyramid(geometries, stages):
    """Check every image size against every stage's pyramid and CC kernel.

    FireANTs resamples each level to ``max(int(size / shrink), MIN_IMG_SIZE)``
    per axis, so an axis at or below the floor is replaced rather than
    downsampled. Both images of a pair are checked: they are floored
    independently and need not be the same size.
    """
    for index, geometry in enumerate(geometries):
        for role in ("fixed", "moving"):
            shape = geometry[role][0]
            for stage_index, stage in enumerate(stages):
                _check_stage_size(role, shape, index, stage_index, stage)


def validate_pairs(pairs, stages):
    """Per-pair checks that must run before the backbone loads.

    A missing required ``fixed``/``moving`` path, a lone mask, a fixed
    segmentation without a moving one, moving keypoints without fixed ones, or
    an explicit masked loss on a pair with no masks -- each would otherwise
    surface mid-batch.
    """
    explicit_masked = any(
        s["loss"] is not None and s["loss"].startswith("masked_") for s in stages
    )
    for index, pair in enumerate(pairs):
        if not pair.get("fixed"):
            raise ValueError(f"pair {index}: missing required 'fixed' path.")
        if not pair.get("moving"):
            raise ValueError(f"pair {index}: missing required 'moving' path.")
        has_fixed_mask = pair.get("fixed_mask") is not None
        has_moving_mask = pair.get("moving_mask") is not None
        if has_fixed_mask != has_moving_mask:
            raise ValueError(
                f"pair {index}: provide both fixed and moving masks, or neither."
            )
        if pair.get("fixed_seg") is not None and pair.get("moving_seg") is None:
            raise ValueError(
                f"pair {index}: a fixed segmentation requires a moving one."
            )
        if (pair.get("moving_keypoints") is not None
                and pair.get("fixed_keypoints") is None):
            raise ValueError(
                f"pair {index}: moving keypoints require fixed keypoints (the "
                "transform maps fixed points into moving space)."
            )
        if explicit_masked and not (has_fixed_mask and has_moving_mask):
            raise ValueError(
                f"pair {index}: a masked loss was requested but the pair has no "
                "masks."
            )


def validate_device(spec):
    """Validate the ``--device`` string (resolved to a torch device later)."""
    if spec in ("auto", "cpu", "cuda"):
        return
    if spec.startswith("cuda:") and spec[5:].isdigit():
        return
    raise ValueError(
        f"--device: expected 'auto', 'cpu', 'cuda', or 'cuda:N', got {spec!r}."
    )


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
        # The backbone settings are inert unless a network is actually loaded.
        if args.features in MODEL_FEATURES:
            if args.backbone == "anatomix-dev-vit" and args.sw_window != 128:
                raise ValueError(
                    "anatomix-dev-vit requires a 128-voxel sliding window "
                    "(--sliding-window-params window=128)."
                )
            build_custom_kwargs(args)
        else:
            reject_custom_backbone_flags(
                args, f"--features {args.features} loads no backbone")
            args.unet_kwargs = args.vit_kwargs = None
        stages = build_stages(args)
        validate_device(args.device)
        pairs, input_columns = resolve_inputs(args)
        geometries = validate_volumes(pairs)
        validate_pairs(pairs, stages)
        validate_keypoints(pairs)
        validate_geometry(geometries)
        validate_pyramid(geometries, stages)
    except ValueError as error:
        parser.error(str(error))

    args.output_dir = os.path.abspath(args.output_dir)

    # Import here so --help and validation work without the FireANTs backend.
    from .pipeline import run

    run(args, pairs, input_columns, stages)


if __name__ == "__main__":
    main()
