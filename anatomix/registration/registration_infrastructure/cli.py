"""Argument parsing and validation for ``anatomix-register.py``.

This module defines the full command-line interface and resolves it into the
inputs the pipeline consumes: a list of fixed/moving (and optional mask/seg)
pairs, the per-stage registration specs, and the feature/output settings. It
also holds every preflight check -- paths, each mask/segmentation against the
geometry of its own image, and the stage schedules against the image sizes --
so an input the backend cannot handle is rejected with a message naming the
offending flag instead of failing opaquely after the model load. FireANTs is
imported only lazily (inside :func:`main`, after all validation), so ``--help``
and every argument/validation error work without the backend installed.
"""
import argparse
import os

import nibabel as nib
import numpy as np

from .io_utils import NIFTI_EXTS, VOLUME_COLUMNS, read_pairs_csv

TRANSFORM_RANK = {"rigid": 0, "affine": 1, "deformable": 2}
VALID_LOSSES = {"cc", "mi", "mse", "masked_cc", "masked_mi", "masked_mse"}
# Losses that consume a CC kernel schedule (auto default resolves to a CC loss).
CC_LOSSES = {None, "cc", "masked_cc"}
# FireANTs floors every pyramid level at this many voxels per axis
# (``fireants/utils/globals.py::MIN_IMG_SIZE``, applied as
# ``max(int(size / shrink), MIN_IMG_SIZE)``). Mirrored here so preflight can
# check image sizes without importing the backend.
MIN_IMG_SIZE = 32
# Tolerance for comparing a mask/segmentation's voxel-to-world affine against
# its image's: 1e-4 is far below any physically meaningful header difference
# (0.1 um of origin, 6e-3 degrees of direction) and far above the round-trip
# noise of writing the same geometry through a different tool.
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
            "features (and/or MIND-SSC descriptors). Supports rigid/affine/"
            "deformable stages, masked and unmasked losses, label warping, "
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
        "[,fixed_mask,moving_mask,fixed_seg,moving_seg].",
    )

    aux = parser.add_argument_group("masks and segmentations")
    aux.add_argument("--fixed-mask", help="Fixed registration mask.")
    aux.add_argument("--moving-mask", help="Moving registration mask.")
    aux.add_argument("--fixed-seg", help="Fixed segmentation (enables Dice).")
    aux.add_argument(
        "--moving-seg", help="Moving segmentation (warped to fixed).")
    aux.add_argument("--fixed-mask-dir",
                     help="Batch directory of fixed masks.")
    aux.add_argument("--moving-mask-dir",
                     help="Batch directory of moving masks.")
    aux.add_argument("--fixed-seg-dir", help="Batch directory of fixed segs.")
    aux.add_argument("--moving-seg-dir",
                     help="Batch directory of moving segs.")

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
        "0.1 for rigid/affine stages.",
    )
    tf.add_argument(
        "--shrink-factors", default=None,
        help="Per-stage 'AxBx...' resolution schedule, strictly decreasing. "
        "Every level is floored at 32 voxels per axis by the backend, so a "
        "multi-resolution schedule needs > 33 voxels on every axis. "
        "Default 8x4x2x1.",
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
    """Per-stage resolution schedule, validated as FireANTs requires it.

    FireANTs' own ``_assert_check_scales_decreasing`` rejects a repeated level
    (its test is ``scales[i] <= scales[i + 1]``), and it only runs once the
    registration object is constructed -- i.e. after the backbone has loaded --
    so the same *strictly* decreasing rule is enforced here instead.
    """
    tokens = ["8x4x2x1"] * n if value is None else _split_stages(
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
        # No schedule requested: leave each stage to FireANTs' own default CC
        # kernel size. Task-appropriate schedules (which vary by dataset and
        # pyramid) are passed explicitly via --cc-kernel-widths.
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


_SINGLE_AUX = ("fixed_mask", "moving_mask", "fixed_seg", "moving_seg")
_DIR_AUX = (
    "fixed_mask_dir", "moving_mask_dir", "fixed_seg_dir", "moving_seg_dir",
)


def _reject_foreign_aux_flags(args, keep):
    """Error if mask/seg flags belonging to a different input mode are set.

    ``keep`` is the tuple of aux flag names valid for the active mode; any other
    aux flag that is set is a silent no-op and therefore rejected.
    """
    foreign = [f for f in _SINGLE_AUX + _DIR_AUX
               if f not in keep and getattr(args, f)]
    if foreign:
        flags = ", ".join("--" + f.replace("_", "-") for f in foreign)
        raise ValueError(
            f"{flags}: not valid for the chosen input mode and would be "
            "ignored. Use the mask/seg flags (or CSV columns) that match the "
            "input mode."
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


# Each auxiliary volume is consumed on the grid of exactly one image.
_AUX_IMAGE = {
    "fixed_mask": "fixed", "fixed_seg": "fixed",
    "moving_mask": "moving", "moving_seg": "moving",
}


def validate_geometry(geometries):
    """Check that every mask/segmentation sits on the grid of its own image.

    Masks and segmentations are consumed as bare arrays on their image's grid:
    a mask multiplies (or is appended to) the feature volume voxel for voxel,
    and a segmentation is resampled by a grid expressed in *its image's*
    normalized coordinates. A differing grid is therefore never resampled, only
    reinterpreted -- so a mismatched moving segmentation is warped from the
    wrong physical locations and its Dice is silently meaningless, while a
    mismatched mask fails much later inside FireANTs (and only when the loss is
    masked), after the backbone and both feature extractions have run.
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
        # A level is resampled by FFT crop, which needs one voxel of margin on
        # each side of the (floored) target size. Measured boundary: 34 voxels
        # works, 33 and below either raise "Invalid number of data points (0)"
        # or -- for linear stages -- silently return a truncated axis (a
        # 24-voxel axis came back with 3) and register garbage. The pyramid
        # *depth* does not matter: every level below the floor becomes the
        # floor, which is why the shipped 8x4x2x1 default is fine on volumes
        # far smaller than 8 * 32 voxels.
        if smallest < MIN_IMG_SIZE + 2:
            # Dropping to a single resolution rescues the stage unless it is
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
        # Single-resolution deformable: no downsampling happens, but the warp
        # field is still allocated at max(size, MIN_IMG_SIZE) and then does not
        # match the image.
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

    FireANTs resamples each pyramid level to ``max(int(size / shrink),
    MIN_IMG_SIZE)`` per axis, so an axis at or below that floor is not
    downsampled but *replaced*, and the FFT resampler it uses then needs at
    least two voxels of margin. Below that the backend fails with an error
    naming neither the flag nor the cause (``Invalid number of data points
    (0)``, or a fixed/moved shape mismatch) or, for linear stages, silently
    truncates the volume. Both images of every pair are checked, since they are
    floored independently and need not be the same size.
    """
    for index, geometry in enumerate(geometries):
        for role in ("fixed", "moving"):
            shape = geometry[role][0]
            for stage_index, stage in enumerate(stages):
                _check_stage_size(role, shape, index, stage_index, stage)


def validate_pairs(pairs, stages):
    """Pre-flight per-pair checks that must run before the backbone loads.

    Catches the cases that would otherwise fail deep in the pipeline (after the
    expensive model load, mid-batch): a missing required ``fixed``/``moving``
    path, a lone mask, a fixed segmentation without a moving one, and an explicit
    masked loss on a pair that has no masks.
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
        if args.backbone == "anatomix-dev-vit" and args.sw_window != 128:
            raise ValueError(
                "anatomix-dev-vit requires a 128-voxel sliding window "
                "(--sliding-window-params window=128)."
            )
        build_custom_kwargs(args)
        stages = build_stages(args)
        validate_device(args.device)
        pairs, input_columns = resolve_inputs(args)
        geometries = validate_volumes(pairs)
        validate_pairs(pairs, stages)
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
