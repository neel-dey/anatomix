"""Primus (3D ViT) command-line arguments.

Split out of ``base_options.py`` so the core CLI stays readable; every option
here is only consulted when ``--netG primus``. ``add_primus_arguments`` is
called once from ``BaseOptions.initialize``.
"""
import argparse

from util import util


def primus_out_norm_mode(v):
    """Parse ``--primus_out_norm`` to none|instance|demean|layernorm|layernorm_affine
    (see anatomix/model/vit3d build_out_norm). Bools accepted: true->instance,
    false->none."""
    s = str(v).strip().lower()
    if s in ("instance", "instancenorm", "in"):
        return "instance"
    if s in ("demean", "center"):
        return "demean"
    if s in ("layernorm", "layer", "ln"):
        return "layernorm"
    if s in ("layernorm_affine", "layernorm-affine", "ln_affine"):
        return "layernorm_affine"
    if s in ("none", "identity", "off"):
        return "none"
    if s in ("true", "t", "yes", "y", "1"):
        return "instance"
    if s in ("false", "f", "no", "n", "0"):
        return "none"
    raise argparse.ArgumentTypeError(
        "expected one of none|instance|demean|layernorm|layernorm_affine (or a bool); "
        "got %r" % v
    )


def add_primus_arguments(parser):
    """Register the ``--primus_*`` 3D ViT options on ``parser`` and return it.
    Only used when ``--netG primus``."""
    parser.add_argument(
        "--primus_version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="v1 (single-conv patch embed) or v2 (deeper residual patch embed; "
        "v2 requires --primus_patch_size 8).",
    )
    parser.add_argument(
        "--primus_config",
        type=str,
        default="B",
        choices=["S", "B", "M", "L"],
        help="ViT scale: depth/heads/embed_dim.",
    )
    parser.add_argument(
        "--primus_patch_size",
        type=int,
        default=8,
        help="Tokenizer patch size (isotropic); crop_size must be divisible by "
        "it. v2 requires 8.",
    )
    parser.add_argument(
        "--primus_drop_path_rate",
        type=float,
        default=0.0,
        help="Stochastic-depth (DropPath) rate.",
    )
    parser.add_argument(
        "--primus_num_register_tokens",
        type=int,
        default=0,
        help="Number of ViT register tokens; 0 (default) disables them. Only "
        "used when --netG primus.",
    )
    parser.add_argument(
        "--primus_v2_in_eps",
        type=float,
        default=1e-5,
        help="InstanceNorm3d eps in the v2 deeper tokenizer (v2 only; default "
        "1e-5 matches upstream). Also used as the --primus_out_norm eps.",
    )
    parser.add_argument(
        "--primus_qk_norm",
        type=util.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable QK-norm (per-head LayerNorm on q,k) in the ViT attention. "
        "Adds eva.blocks.*.attn.{q,k}_norm params (changes the state_dict).",
    )
    parser.add_argument(
        "--primus_out_norm",
        type=primus_out_norm_mode,
        nargs="?",
        const="instance",
        default="none",
        choices=["none", "instance", "demean", "layernorm", "layernorm_affine"],
        help="Output spatial norm: none|instance|demean|layernorm|layernorm_affine "
        "(see anatomix/model/vit3d build_out_norm; eps = --primus_v2_in_eps). Only "
        "layernorm_affine adds params. Bare --primus_out_norm = instance; bools "
        "accepted (true->instance, false->none).",
    )
    parser.add_argument(
        "--primus_register_init_std",
        type=float,
        default=0.02,
        help="Init std for register tokens (upstream 1e-6). Only used when "
        "--primus_num_register_tokens > 0.",
    )
    return parser
