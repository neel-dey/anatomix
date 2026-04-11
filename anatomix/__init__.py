__all__ = [
    "network",
    "registration",
    "segmentation",
]

def __getattr__(name):
    if name == "network":
        from .model import network
        return network
    if name == "registration":
        from . import registration
        return registration
    if name == "segmentation":
        from . import segmentation
        return segmentation
    raise AttributeError(f"module 'anatomix' has no attribute {name!r}")

