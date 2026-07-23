"""anatomix 3D registration.

The registration tools live in subpackages so that only ``anatomix-register.py``
(the FireANTs command-line entry point) and ``README.md`` sit at the top of
``anatomix/registration/``:

- :mod:`anatomix.registration.registration_infrastructure` -- the FireANTs
  feature-registration pipeline that backs ``anatomix-register.py``.
- :mod:`anatomix.registration.registration_backend.convexadam` -- the retained
  ConvexAdam backend used for the ICLR'25 results.
- ``registration_backend/fireants`` -- a gitignored, editable FireANTs clone,
  installed separately via ``registration_backend/install_fireants.sh``.

Submodules are imported lazily so that ``import anatomix.registration`` stays
cheap and never eagerly imports FireANTs (an optional, separately installed
backend).
"""

__all__ = ["registration_infrastructure", "registration_backend"]


def __getattr__(name):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
