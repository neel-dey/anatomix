"""Primus extensions used by Anatomix pretraining.

The base models come from ``dynamic-network-architectures``. These subclasses
only expose the normalization and register settings used by this project.
"""

import torch
from torch import nn

from dynamic_network_architectures.architectures.primus import (
    Primus as _Primus,
    PrimusV2 as _PrimusV2,
)
from dynamic_network_architectures.building_blocks.patch_encode_decode import (
    LayerNormNd,
)


PRIMUS_CONFIGS = {
    "S": {"eva_depth": 12, "eva_numheads": 6, "embed_dim": 396},
    "B": {"eva_depth": 12, "eva_numheads": 12, "embed_dim": 792},
    "M": {"eva_depth": 16, "eva_numheads": 12, "embed_dim": 864},
    "L": {"eva_depth": 24, "eva_numheads": 16, "embed_dim": 1056},
}


class ChannelDemean(nn.Module):
    """Subtract each channel's spatial mean."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=(2, 3, 4), keepdim=True)


class ChannelLayerNorm(nn.Module):
    """Apply affine-free layer normalization across channels."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)


def build_out_norm(mode, num_classes, eps):
    """Build the requested output normalization layer."""
    if isinstance(mode, bool):
        mode = "instance" if mode else "none"
    mode = (mode or "none").lower()
    if mode in ("none", "identity", "off"):
        return nn.Identity()
    if mode in ("instance", "instancenorm", "in"):
        return nn.InstanceNorm3d(num_classes, eps=eps, affine=False)
    if mode in ("demean", "center"):
        return ChannelDemean()
    if mode in ("layernorm", "layer", "ln"):
        return ChannelLayerNorm(eps=eps)
    if mode in ("layernorm_affine", "layernorm-affine", "ln_affine"):
        return LayerNormNd(num_classes, eps=eps)
    raise ValueError(f"unsupported output normalization: {mode!r}")


class _PrimusExtensions:
    def _configure_extensions(
        self, qk_norm, out_norm, out_norm_eps, register_init_std
    ):
        if qk_norm:
            for block in self.eva.blocks:
                attention = block.attn
                head_dim = getattr(attention, "head_dim", None) or (
                    attention.q_proj.out_features // attention.num_heads
                )
                attention.q_norm = nn.LayerNorm(head_dim)
                attention.k_norm = nn.LayerNorm(head_dim)

        if self.register_tokens is not None:
            # Upstream initializes registers with std=1e-6.
            with torch.no_grad():
                self.register_tokens.mul_(register_init_std / 1e-6)

        self.out_norm = build_out_norm(
            out_norm, self.up_projection.decode[-1].out_channels, out_norm_eps
        )

    def forward(
        self, x, layers=None, encode_only=False, verbose=False, ret_mask=False
    ):
        # Preserve Primus' positional ``ret_mask`` argument.
        if isinstance(layers, bool):
            ret_mask = layers
            layers = None
        result = super().forward(x, ret_mask=ret_mask)
        if ret_mask:
            output, mask = result
            return self.out_norm(output), mask
        output = self.out_norm(result)
        if layers:
            features = [output]
            return features if encode_only else (output, features)
        return output


class Primus(_PrimusExtensions, _Primus):
    """Primus with optional QK and output normalization."""

    def __init__(
        self,
        *args,
        qk_norm=False,
        out_norm="none",
        out_norm_eps=1e-5,
        register_init_std=1e-6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._configure_extensions(
            qk_norm, out_norm, out_norm_eps, register_init_std
        )


class PrimusV2(_PrimusExtensions, _PrimusV2):
    """PrimusV2 with configurable tokenizer epsilon and Primus extensions."""

    def __init__(
        self,
        *args,
        qk_norm=False,
        out_norm="none",
        out_norm_eps=1e-5,
        register_init_std=1e-6,
        in_eps=1e-5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for module in self.down_projection.modules():
            if isinstance(module, nn.InstanceNorm3d):
                module.eps = in_eps
        self._configure_extensions(
            qk_norm, out_norm, out_norm_eps, register_init_std
        )
