"""Compatibility wrapper for the configurable PrimusV2 tokenizer."""

from torch import nn

from dynamic_network_architectures.building_blocks.patch_encode_decode import (
    PatchEmbed_deeper,
    block_style,
    block_type,
)


class PatchEmbedDeeper(PatchEmbed_deeper):
    """Downsample a 3D volume into patch embeddings with convolutional stages.

    This upstream-compatible PrimusV2 tokenizer only adds configurable
    InstanceNorm epsilon.

    Parameters
    ----------
    input_channels : int, optional
        Number of channels in the input volume.
    embed_dim : int, optional
        Number of channels in the output embedding grid.
    base_features : int, optional
        Channel count in the stem and first downsampling stage.
    depth_per_level : tuple[int, ...], optional
        Block counts in successive stride-2 stages; its length determines the
        total downsampling factor, ``2 ** len(depth_per_level)``.
    embed_proj_3x3x3 : bool, optional
        Use a 3x3x3 rather than 1x1x1 final embedding projection.
    embed_block_type : {"basic", "bottleneck"}, optional
        Residual block type; used only with residual-style stages.
    embed_block_style : {"residual", "conv"}, optional
        Build each stage from residual or plain convolutional blocks.
    in_eps : float, optional
        Epsilon used by every InstanceNorm3d layer.

    Notes
    -----
    The inherited forward maps ``(B, C_in, D, H, W)`` to
    ``(B, embed_dim, D/s, H/s, W/s)``, where ``s`` is the downsampling factor.
    """

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
