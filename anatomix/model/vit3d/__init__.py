"""3D ViT (Primus; https://arxiv.org/pdf/2503.01835) model backbones."""
from .architectures import PRIMUS_CONFIGS, Primus, PrimusV2

__all__ = ["Primus", "PrimusV2", "PRIMUS_CONFIGS"]
