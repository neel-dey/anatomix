"""Load anatomix Unet weights from the HuggingFace Hub."""
import torch
from huggingface_hub import hf_hub_download

from anatomix.model.network import Unet


DEFAULT_REPO = "neeldey/anatomix"

# Variant name -> kwargs passed to Unet.__init__. Add new entries here when a
# new variant is published to the Hub. Any kwarg accepted by Unet is allowed.
ANATOMIX_VARIANTS = {
    "anatomix": {
        "unet_kwargs": dict(
            dimension=3, input_nc=1, output_nc=16, num_downs=4, ngf=16,
        ),
    },
    "anatomix+brains": {
        "unet_kwargs": dict(
            dimension=3, input_nc=1, output_nc=16, num_downs=4, ngf=16,
        ),
    },
    "anatomix-dev": {
        "unet_kwargs": dict(
            dimension=3, input_nc=1, output_nc=16, num_downs=5, ngf=20,
            norm="instance", pooling="Avg", interp="trilinear",
        ),
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
    return a Unet instantiated with the registered kwargs and weights loaded.
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
    model = Unet(**ANATOMIX_VARIANTS[variant]["unet_kwargs"])
    return _load_handling_compile(model, state_dict)
