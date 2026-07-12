"""
Vendored Primus / PrimusV2 (v1 / v2) 3D ViT, adapted from
``dynamic_network_architectures.architectures.primus`` to expose two tokenizer
knobs as constructor args: ``num_register_tokens`` (ViT register tokens) and
``in_eps`` (PrimusV2's deeper-tokenizer InstanceNorm3d eps, hardcoded to 1e-5
upstream). Only the model-definition classes are vendored; heavy blocks (``Eva``,
``PatchDecode``, RoPE, weight init) come from the installed
``dynamic_network_architectures`` / ``timm``. The submodule tree matches upstream,
so checkpoints are interchangeable with the official Primus / PrimusV2.
"""
from typing import Tuple

import torch
from torch import nn

from timm.layers import RotaryEmbeddingCat
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.building_blocks.patch_encode_decode import (
    LayerNormNd,
    PatchDecode,
    PatchEmbed,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from einops import rearrange

from .deep_tokenizer import PatchEmbedDeeper


# Per-config depth / heads / embed_dim.
PRIMUS_CONFIGS = {
    "S": {"eva_depth": 12, "eva_numheads": 6, "embed_dim": 396},
    "B": {"eva_depth": 12, "eva_numheads": 12, "embed_dim": 792},
    "M": {"eva_depth": 16, "eva_numheads": 12, "embed_dim": 864},
    "L": {"eva_depth": 24, "eva_numheads": 16, "embed_dim": 1056},
}


class ChannelDemean(nn.Module):
    """Subtract each channel's per-sample spatial mean (param-free demean; no
    variance rescale)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=(2, 3, 4), keepdim=True)


class ChannelLayerNorm(nn.Module):
    """Affine-free LayerNorm across channels per voxel (zero-mean / unit-std).
    Param-free; the forward argmax-over-channels is unchanged."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps)


def build_out_norm(mode, num_classes, eps):
    """Resolve ``--primus_out_norm`` to a param-free output-norm module. Modes:
    ``none`` -> Identity (default; matches the official state_dict);
    ``instance`` -> affine-free InstanceNorm3d (per-channel demean + unit-variance over space);
    ``demean`` -> per-channel demean only (ChannelDemean);
    ``layernorm`` -> affine-free per-voxel LayerNorm across channels (ChannelLayerNorm);
    ``layernorm_affine`` -> same, with a learned per-channel affine (``LayerNormNd``).
    Bools accepted for back-compat (True -> instance, False -> none). Only
    ``layernorm_affine`` adds params (``out_norm.{weight,bias}``); other modes leave the
    state_dict unchanged, so existing checkpoints load.
    """
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
    raise ValueError(
        "out_norm must be one of 'none' | 'instance' | 'demean' | 'layernorm' | "
        "'layernorm_affine' (or a bool); got %r" % (mode,)
    )


class Primus(AbstractDynamicNetworkArchitectures):
    """Primus v1: single-conv ``PatchEmbed`` tokenizer + EVA ViT + PatchDecode."""

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        num_classes: int,
        eva_depth: int = 24,
        eva_numheads: int = 16,
        input_shape: Tuple[int, ...] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
        qk_norm=False,
        out_norm="none",
        out_norm_eps=1e-5,
        register_init_std=1e-6,
    ):
        assert input_shape is not None
        assert len(input_shape) == 3, "Currently only 3d is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()
        self.embed_dim = embed_dim
        self.key_to_encoder = "eva"
        self.key_to_stem = "down_projection"
        self.keys_to_in_proj = ("down_projection.proj",)
        self.key_to_lpe = "eva.pos_embed"

        self.down_projection = PatchEmbed(patch_embed_size, input_channels, embed_dim)
        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.eva = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_embed,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )

        # QK-norm: the vendored ``Eva`` builder doesn't expose timm's qk_norm, so enable
        # it post-hoc by swapping per-head LayerNorm(head_dim) in. Bounds the pre-softmax
        # logits, preventing the attention-sink that injects a spatially-uniform (DC)
        # component -> the per-channel output bias that hijacks argmax. Adds
        # eva.blocks.*.attn.{q,k}_norm.{weight,bias} to the state_dict.
        if qk_norm:
            for blk in self.eva.blocks:
                attn = blk.attn
                head_dim = getattr(attn, "head_dim", None) or (
                    attn.q_proj.out_features // attn.num_heads
                )
                attn.q_norm = nn.LayerNorm(head_dim)
                attn.k_norm = nn.LayerNorm(head_dim)

        # Optional param-free spatial norm on the dense output (see build_out_norm);
        # counters the per-channel DC/scale offset the ViT's channel-wise LayerNorm
        # leaves unchecked. eps reuses ``--primus_v2_in_eps``.
        self.out_norm = build_out_norm(out_norm, num_classes, out_norm_eps)

        self.mask_token: torch.Tensor
        self.register_buffer("mask_token", torch.zeros(1, 1, embed_dim))

        if num_register_tokens > 0:
            self.register_tokens = (
                nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
            )
            # Upstream std 1e-6 leaves registers inert (never attended); a token-scale
            # init (~0.02, like pos_embed) makes them recruitable as the attention sink.
            nn.init.normal_(self.register_tokens, std=register_init_std)
        else:
            self.register_tokens = None

        self.down_projection.apply(InitWeights_He(1e-2))
        self.up_projection.apply(InitWeights_He(1e-2))
        # eva has its own initialization

    def restore_full_sequence(self, x, keep_indices, num_patches):
        """
        Restore the full sequence by filling blanks with mask tokens and reordering.
        """
        if keep_indices is None:
            return x, None
        B, num_kept, C = x.shape
        device = x.device

        # Create mask tokens for missing patches
        num_masked = num_patches - num_kept
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)

        # Prepare an empty tensor for the restored sequence
        restored = torch.zeros(B, num_patches, C, device=device)
        restored_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)

        # Assign the kept patches and mask tokens in the correct positions
        for i in range(B):
            kept_pos = keep_indices[i]
            all_indices = torch.arange(num_patches, device=device)  # Create tensor of all indices
            mask = torch.ones(num_patches, device=device, dtype=torch.bool)  # Start with all True
            mask[kept_pos] = False  # Set kept positions to False
            masked_pos = all_indices[mask]  # Extract indices not in kept_pos

            restored[i, kept_pos] = x[i]
            restored[i, masked_pos] = mask_tokens[i, : len(masked_pos)]
            restored_mask[i, kept_pos] = True

        return (restored, restored_mask)

    def forward(self, x, ret_mask=False):
        FW, FH, FD = x.shape[2:]  # Full W , ...
        x = self.down_projection(x)
        # last output of the encoder is the input to EVA
        B, C, W, H, D = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c w h d -> b (w h d) c")
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        x, keep_indices = self.eva(x)

        if self.register_tokens is not None:
            x = x[:, self.register_tokens.shape[1] :]  # Removes the register tokens
        # In-fill in-active patches with empty tokens
        restored_x, restoration_mask = self.restore_full_sequence(x, keep_indices, num_patches)
        x = rearrange(restored_x, "b (w h d) c -> b c w h d", h=H, w=W, d=D)
        if restoration_mask is not None:
            mask = rearrange(restoration_mask, "b (w h d) -> b w h d", h=H, w=W, d=D)
            full_mask = (
                mask.repeat_interleave(FW // W, dim=1)
                .repeat_interleave(FH // H, dim=2)
                .repeat_interleave(FD // D, dim=3)
            )
            full_mask = full_mask[:, None, ...]  # Add channel dimension  # [B, 1, W, H, D]
        else:
            full_mask = None

        dec_out = self.out_norm(self.up_projection(x))
        if ret_mask:
            return dec_out, full_mask
        else:
            return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


class PrimusV2(Primus):
    """Primus v2: same as v1 but with the deeper residual tokenizer
    (``PatchEmbedDeeper``), whose InstanceNorm3d eps is configurable via
    ``in_eps``. The deeper tokenizer is hardwired to an 8x stride."""

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        num_classes: int,
        eva_depth: int = 24,
        eva_numheads: int = 16,
        input_shape: Tuple[int, ...] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
        qk_norm=False,
        out_norm="none",
        out_norm_eps=1e-5,
        register_init_std=1e-6,
        in_eps: float = 1e-5,
    ):
        super().__init__(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            num_classes=num_classes,
            eva_depth=eva_depth,
            eva_numheads=eva_numheads,
            input_shape=input_shape,
            decoder_norm=decoder_norm,
            decoder_act=decoder_act,
            num_register_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_embed=use_abs_pos_embed,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
            qk_norm=qk_norm,
            out_norm=out_norm,
            out_norm_eps=out_norm_eps,
            register_init_std=register_init_std,
        )
        self.keys_to_in_proj = (
            "down_projection.stem.blocks.0.conv1.conv",
            "down_projection.stem.blocks.0.conv1.all_modules.0",
        )
        self.down_projection = PatchEmbedDeeper(
            input_channels=input_channels,
            embed_dim=embed_dim,
            base_features=32,
            depth_per_level=(1, 1, 1),
            embed_proj_3x3x3=False,
            embed_block_style="residual",
            embed_block_type="basic",  # "basic" or "bottleneck" (if "residual" style)
            in_eps=in_eps,
        )
        self.down_projection.apply(InitWeights_He(1e-2))
