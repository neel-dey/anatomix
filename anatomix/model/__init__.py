from .network import Unet

__all__ = ["Unet", "Primus", "PrimusV2", "PRIMUS_CONFIGS"]


def __getattr__(name):
    """Lazily import optional 3D ViT (Primus) symbols."""
    if name in ("Primus", "PrimusV2", "PRIMUS_CONFIGS"):
        from . import vit3d

        return getattr(vit3d, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
