"""Local copy of the 3D ViT (Primus / PrimusV2) model definitions + PRIMUS_CONFIGS;
heavy blocks come from dynamic_network_architectures / timm. See architectures.py
for why these are copied and for the upstream citation."""
from .architectures import PRIMUS_CONFIGS, Primus, PrimusV2

__all__ = ["Primus", "PrimusV2", "PRIMUS_CONFIGS"]
