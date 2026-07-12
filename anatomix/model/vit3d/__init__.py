"""Vendored 3D ViT (Primus / PrimusV2) classes + PRIMUS_CONFIGS; heavy blocks
come from dynamic_network_architectures / timm. See architectures.py."""
from .architectures import PRIMUS_CONFIGS, Primus, PrimusV2

__all__ = ["Primus", "PrimusV2", "PRIMUS_CONFIGS"]
