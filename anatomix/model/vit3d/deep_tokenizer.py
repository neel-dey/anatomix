"""
Local copy of ``PatchEmbed_deeper`` (PrimusV2's deeper residual tokenizer) from
MIC-DKFZ ``dynamic-network-architectures`` (see architectures.py for the full
citation). Copied rather than imported for a single reason: to make the
InstanceNorm3d eps (``in_eps``) a constructor arg instead of the upstream
hardcoded 1e-5. The submodule tree matches upstream so state_dicts stay
interchangeable.
"""
from typing import Literal

import torch
from torch import nn

from dynamic_network_architectures.building_blocks.residual import (
    BasicBlockD,
    BottleneckD,
    StackedResidualBlocks,
)
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    StackedConvBlocks,
)

block_type = Literal["basic", "bottleneck"]
block_style = Literal["residual", "conv"]


class PatchEmbedDeeper(nn.Module):
    """ResNet-style patch embedding with progressive downsampling.

    Identical to upstream ``PatchEmbed_deeper`` except that the InstanceNorm3d
    epsilon (``in_eps``) is configurable instead of hardcoded to 1e-5.
    """

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 864,
        base_features: int = 32,
        depth_per_level: tuple = (1, 1, 1),
        embed_proj_3x3x3: bool = False,
        embed_block_type: block_type = "basic",
        embed_block_style: block_style = "residual",
        in_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        norm_op = nn.InstanceNorm3d
        block = BottleneckD if embed_block_type == "bottleneck" else BasicBlockD
        nonlin = nn.LeakyReLU if embed_block_type == "bottleneck" else nn.ReLU
        norm_op_kwargs = {"eps": in_eps, "affine": True}  # <-- in_eps is the only change vs upstream
        nonlin_kwargs = {"inplace": True}

        if embed_block_type == "bottleneck":
            bottleneck_channels = base_features // 4
        else:
            bottleneck_channels = None

        if embed_block_style == "residual":
            self.stem = StackedResidualBlocks(
                1,
                nn.Conv3d,
                input_channels,
                base_features,
                [3, 3, 3],
                1,
                True,
                norm_op,
                norm_op_kwargs,
                None,
                None,
                nonlin,
                nonlin_kwargs,
                block=block,
            )
        elif embed_block_style == "conv":
            self.stem = StackedConvBlocks(
                1,
                nn.Conv3d,
                input_channels,
                base_features,
                [3, 3, 3],
                1,
                True,
                norm_op,
                norm_op_kwargs,
                None,
                None,
                nonlin,
                nonlin_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown embed_block_style: {embed_block_style}. "
                "Must be 'residual' or 'conv'."
            )

        levels_needed = len(depth_per_level)

        # Build encoder stages
        self.stages = nn.ModuleList()
        input_channels = base_features

        for i in range(levels_needed):
            # First block in each stage handles downsampling and channel increase
            stride = 2
            output_channels = base_features * (2**i)
            if embed_block_style == "residual":
                if embed_block_type == "bottleneck":
                    bottleneck_channels = output_channels // 4
                else:
                    bottleneck_channels = None
                stage = StackedResidualBlocks(
                    n_blocks=depth_per_level[i],
                    conv_op=nn.Conv3d,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    kernel_size=3,
                    initial_stride=stride,
                    conv_bias=False,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    block=block,
                    bottleneck_channels=bottleneck_channels,
                )
            elif embed_block_style == "conv":
                stage = StackedConvBlocks(
                    num_convs=depth_per_level[i],
                    conv_op=nn.Conv3d,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    kernel_size=3,
                    initial_stride=stride,
                    conv_bias=False,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            self.stages.append(stage)
            input_channels = output_channels

        final_proj_kernel = [3, 3, 3] if embed_proj_3x3x3 else [1, 1, 1]
        final_pad = [1, 1, 1] if embed_proj_3x3x3 else [0, 0, 0]
        self.final_proj = nn.Conv3d(
            input_channels, embed_dim, kernel_size=final_proj_kernel, stride=[1, 1, 1], padding=final_pad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_proj(x)
        return x
