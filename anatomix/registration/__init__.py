"""anatomix 3D registration.

- ``anatomix-register.py``: the FireANTs command-line entry point.
- :mod:`.registration_infrastructure`: the pipeline behind it.
- :mod:`.registration_backend`: the retained ConvexAdam backend, and the
  gitignored FireANTs clone that ``registration_backend/install_fireants.sh``
  creates.

Subpackages import lazily, so importing this package never imports FireANTs.
"""

__all__ = ["registration_infrastructure", "registration_backend"]


def __getattr__(name):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
