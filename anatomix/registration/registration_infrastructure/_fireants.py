"""Single import surface for the optional FireANTs backend.

Every FireANTs symbol used by the pipeline is imported here, behind one
``try``/``except`` so that a missing backend raises a single, actionable error
(pointing at ``install_fireants.sh``) instead of an opaque ``ImportError`` deep
in the pipeline. Other modules import what they need from here rather than from
``fireants`` directly, which also keeps ``import anatomix.registration`` free of
any eager FireANTs dependency.
"""
import os

_INSTALLER = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), os.pardir,
        "registration_backend", "install_fireants.sh",
    )
)

try:
    from fireants.io import Image, BatchedImages
    from fireants.io.image import FakeBatchedImages
    from fireants.io.imagemask import apply_mask_to_image
    from fireants.registration.moments import MomentsRegistration
    from fireants.registration.rigid import RigidRegistration
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.greedy import GreedyRegistration
    from fireants.registration.deformablemixin import DeformableMixin
    from fireants.interpolator import FFO_AVAILABLE
    from fireants.interpolator.grid_sample import torch_grid_sampler_3d
    from fireants.utils.imageutils import jacobian
except ImportError as exc:  # pragma: no cover - exercised only without backend
    raise ImportError(
        "The FireANTs registration backend is not installed. Install it with:\n"
        f"    bash {_INSTALLER}\n"
        f"(underlying import error: {exc})"
    ) from exc

__all__ = [
    "Image",
    "BatchedImages",
    "FakeBatchedImages",
    "apply_mask_to_image",
    "MomentsRegistration",
    "RigidRegistration",
    "AffineRegistration",
    "GreedyRegistration",
    "DeformableMixin",
    "FFO_AVAILABLE",
    "torch_grid_sampler_3d",
    "jacobian",
]
