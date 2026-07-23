"""Retained ConvexAdam registration backend (anatomix ICLR'25 results).

This is the original network-feature + MIND-SSC ConvexAdam registration code,
relocated unchanged from ``anatomix/registration/`` so that only
``anatomix-register.py`` and ``README.md`` remain at the top of the registration
folder. It is self-contained (it keeps its own bundled MIND-SSC) and is not
wired into the new FireANTs command-line interface. Use it directly through
:func:`convex_adam` or the accompanying ``anatomix_registration_convexadam``
notebook; see the registration ``README.md`` for how to reproduce the ICLR'25
numbers.
"""
from .convex_adam_utils import (
    extract_features,
    load_model,
    diffusion_regularizer,
    apply_avg_pool3d,
    correlate,
    coupled_convex,
    inverse_consistency,
    MINDSSC,
)
from .instance_optimization import (
    run_stage1_registration,
    run_instance_opt,
    merge_features,
)
from .run_convex_adam_with_network_feats import convex_adam

__all__ = [
    "convex_adam",
    "extract_features",
    "load_model",
    "diffusion_regularizer",
    "apply_avg_pool3d",
    "correlate",
    "coupled_convex",
    "inverse_consistency",
    "MINDSSC",
    "run_stage1_registration",
    "run_instance_opt",
    "merge_features",
]
