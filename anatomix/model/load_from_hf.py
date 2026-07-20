"""Load anatomix model weights from the HuggingFace Hub."""
import torch
from huggingface_hub import hf_hub_download

from anatomix.model.network import Unet


DEFAULT_REPO = "neeldey/anatomix"

# Variant name -> model constructor kwargs and output width.
ANATOMIX_VARIANTS = {
    "anatomix": {
        "unet_kwargs": dict(
            dimension=3, input_nc=1, output_nc=16, num_downs=4, ngf=16,
        ),
        "output_channels": 16,
    },
    "anatomix-dev": {
        "unet_kwargs": dict(
            dimension=3, input_nc=1, output_nc=32, num_downs=5, ngf=32,
            norm="instance", pooling="Avg", interp="trilinear", norm_eps=1e-2,
        ),
        "output_channels": 32,
    },
    "anatomix-dev-vit": {
        "vit_kwargs": dict(
            input_channels=1, num_classes=32, embed_dim=396, eva_depth=12,
            eva_numheads=6, patch_embed_size=(8, 8, 8),
            input_shape=(128, 128, 128), num_register_tokens=8,
            init_values=0.1, scale_attn_inner=True, qk_norm=True,
            out_norm="demean", out_norm_eps=1e-2,
            register_init_std=0.02, in_eps=1e-2,
        ),
        "output_channels": 32,
    },
}


def _load_handling_compile(model, state_dict):
    """Load `state_dict` into `model`, transparently handling weights saved
    from a `torch.compile()`-wrapped model (whose keys carry an `_orig_mod.`
    prefix)."""
    if state_dict and next(iter(state_dict)).startswith("_orig_mod."):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    return model


def load_from_hf(
    variant,
    repo_id=DEFAULT_REPO,
    revision=None,
    map_location="cpu",
):
    """Download `<variant>.pth` from `repo_id` on the HuggingFace Hub and
    return the registered model with its weights loaded.

    Checkpoints saved from ``torch.compile`` are returned as ordinary modules
    so architecture-specific forward arguments remain available.
    """
    if variant not in ANATOMIX_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. "
            f"Known: {sorted(ANATOMIX_VARIANTS)}"
        )
    weights_path = hf_hub_download(
        repo_id, f"{variant}.pth", revision=revision,
    )
    state_dict = torch.load(weights_path, map_location=map_location)
    config = ANATOMIX_VARIANTS[variant]
    if "vit_kwargs" in config:
        from anatomix.model.vit3d import PrimusV2
        model = PrimusV2(**config["vit_kwargs"])
    else:
        model = Unet(**config["unet_kwargs"])
    return _load_handling_compile(model, state_dict)
