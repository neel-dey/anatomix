"""Compatibility wrapper for the configurable PrimusV2 tokenizer."""

from torch import nn

from dynamic_network_architectures.building_blocks.patch_encode_decode import (
    PatchEmbed_deeper,
    block_style,
    block_type,
)


class PatchEmbedDeeper(PatchEmbed_deeper):
    def __init__(
        self,
        input_channels=3,
        embed_dim=864,
        base_features=32,
        depth_per_level=(1, 1, 1),
        embed_proj_3x3x3=False,
        embed_block_type="basic",
        embed_block_style="residual",
        in_eps=1e-5,
    ):
        super().__init__(
            input_channels,
            embed_dim,
            base_features,
            depth_per_level,
            embed_proj_3x3x3,
            embed_block_type,
            embed_block_style,
        )
        for module in self.modules():
            if isinstance(module, nn.InstanceNorm3d):
                module.eps = in_eps


__all__ = ["PatchEmbedDeeper", "block_style", "block_type"]
