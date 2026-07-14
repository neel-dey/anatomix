"""Command-line options for Primus models."""
import argparse

from util import util


def primus_out_norm_mode(v):
    """Parse an output-normalization name or legacy boolean value.

    Parameters
    ----------
    v : object
        Value supplied by argparse.

    Returns
    -------
    str
        Canonical normalization mode.
    """
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
    """Add Primus options to an argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.

    Returns
    -------
    argparse.ArgumentParser
        The same parser with Primus arguments registered.
    """
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
        help="ViT scale as depth/heads/embedding width: S=12/6/396, "
        "B=12/12/792, M=16/12/864, or L=24/16/1056.",
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
        help="Maximum probability of skipping a transformer block during "
        "training; 0 disables stochastic depth.",
    )
    parser.add_argument(
        "--primus_num_register_tokens",
        type=int,
        default=0,
        help="Number of learned tokens included in attention but discarded "
        "before decoding; 0 disables them.",
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
        help="Decoded-volume norm: instance/demean operate spatially per "
        "channel; layernorm modes operate across channels per voxel; none "
        "disables. Only layernorm_affine adds parameters. A bare flag or true "
        "selects instance; false selects none. Uses --primus_v2_in_eps.",
    )
    parser.add_argument(
        "--primus_register_init_std",
        type=float,
        default=0.02,
        help="Init std for register tokens (upstream 1e-6). Only used when "
        "--primus_num_register_tokens > 0.",
    )
    return parser
