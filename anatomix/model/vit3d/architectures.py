"""This file imports 3D ViT backbones and exposes some configurables that were
hardcoded in the upstream but might be modified in anatomix pretraining, such
as output layer normalization and register (ViT) initializations. This also  
adds QK normalization to the upstream. The base models come from 
``dynamic-network-architectures`` on PyPI.
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
        """Center a ``(B, C, D, H, W)`` tensor, preserving its shape."""
        return x - x.mean(dim=(2, 3, 4), keepdim=True)


class ChannelLayerNorm(nn.Module):
    """Pointwise standardize channels independently, without affine renorms.

    Parameters
    ----------
    eps : float, optional
        eps added to the channel variance to prevent explosions.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)


def build_out_norm(mode, num_classes, eps):
    """Build an output normalization layer.

    Parameters
    ----------
    mode : str or bool
        ``none``, ``instance``, ``demean``, ``layernorm``, or
        ``layernorm_affine``. Booleans map to instance norm or identity.
    num_classes : int
        Number of output channels.
    eps : float
        Numerical-stability epsilon.

    Returns
    -------
    nn.Module
        The requested normalization layer.
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
    raise ValueError(f"unsupported output normalization: {mode!r}")


class _PrimusExtensions:
    """Add normalization options to upstream models."""

    def _configure_extensions(
        self, qk_norm, out_norm, out_norm_eps, register_init_std
    ):
        """Configure the local extensions after the upstream model is built.

        Parameters
        ----------
        qk_norm : bool
            Add per-head LayerNorm to queries and keys in every EVA block.
        out_norm : str or bool
            Normalization applied to the decoded feature volume.
        out_norm_eps : float
            Epsilon used by the output normalization.
        register_init_std : float
            Standard deviation for initialized register tokens.
        """
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
        """Run a 3D ViT and adapt its output for anatomix pretraining.

        Parameters
        ----------
        x : torch.Tensor of shape (B, C, D, H, W)
            Input volume.
        layers : sequence or bool, optional
            A nonempty sequence requests the final volume as the sole NCE
            feature. A boolean is treated as the upstream positional
            ``ret_mask`` argument.
        encode_only : bool, optional
            With ``layers``, return only the feature list instead of both the
            decoded volume and feature list.
        verbose : bool, optional
            Accepted for compatibility with the UNet pretraining interface.
        ret_mask : bool, optional
            Also return the spatial mask produced when patch dropout is active.

        Returns
        -------
        torch.Tensor, list[torch.Tensor], or tuple
            A ``(B, C_out, D, H, W)`` volume, requested feature result, or
            ``(volume, mask)`` when ``ret_mask`` is true.
        """
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
    """Encode 3D patches with an EVA ViT and decode them to a dense volume.

    This wrapper adds query/key and output normalization controls to upstream
    Primus while preserving its input and output shapes.

    Parameters
    ----------
    input_channels, num_classes : int
        Numbers of channels in the input and decoded output volumes.
    embed_dim : int
        Width of each patch token and transformer block.
    patch_embed_size : tuple[int, int, int]
        Non-overlapping patch size; each axis must divide ``input_shape``.
    eva_depth, eva_numheads : int, optional
        Number of transformer blocks and attention heads. ``embed_dim`` must
        be divisible by ``eva_numheads``.
    input_shape : tuple[int, int, int]
        Spatial training crop shape used to size positional embeddings.
    decoder_norm, decoder_act : type, optional
        Normalization and activation classes used by the patch decoder.
    num_register_tokens : int, optional
        Extra learned tokens prepended during attention and removed before
        decoding; zero disables them.
    use_rot_pos_emb, use_abs_pos_embed : bool, optional
        Enable rotary and learned absolute positional embeddings.
    mlp_ratio : float, optional
        Hidden width of each transformer MLP relative to ``embed_dim``.
    drop_path_rate : float, optional
        Maximum stochastic-depth probability across transformer blocks.
    patch_drop_rate, proj_drop_rate, attn_drop_rate : float, optional
        Dropout probabilities for input patches, projections, and attention.
    rope_impl, rope_kwargs : object, dict, optional
        Rotary-position implementation and its constructor arguments.
    init_values : float or None, optional
        Initial LayerScale value; ``None`` disables LayerScale.
    scale_attn_inner : bool, optional
        Apply an additional normalization inside each attention block.
    qk_norm : bool, optional
        Add per-head LayerNorm to attention queries and keys.
    out_norm : str or bool, optional
        Decoded-volume normalization; see :func:`build_out_norm` for modes.
    out_norm_eps : float, optional
        Numerical-stability epsilon for ``out_norm``.
    register_init_std : float, optional
        Initialization standard deviation for register tokens.
    """

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
    """The 3D ViT Primus V2 uses a residual convolutional tokenizer before
    the transformer blocks.

    Common parameters are documented by :class:`Primus`. The three stride-2
    tokenizer stages require a patch size of ``(8, 8, 8)``.

    Parameters
    ----------
    in_eps : float, optional
        Numerical-stability epsilon for every tokenizer InstanceNorm layer.
    """

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
