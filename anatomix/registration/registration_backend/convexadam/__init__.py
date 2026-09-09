"""ConvexAdam registration backend of the anatomix ICLR'25 paper.

Use it through :func:`convex_adam`, its command-line script
``run_convex_adam_with_network_feats.py``, or the
``tutorials/anatomix_registration_convexadam.ipynb`` notebook. The FireANTs
command-line interface does not use it.
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
