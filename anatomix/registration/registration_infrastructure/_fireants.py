"""FireANTs imports, with an installation hint when the backend is missing."""
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
    from fireants.io.imagemask import generate_image_mask_allones
    from fireants.registration.moments import MomentsRegistration
    from fireants.registration.rigid import RigidRegistration
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.greedy import GreedyRegistration
    from fireants.registration.deformablemixin import DeformableMixin
    from fireants.interpolator import FFO_AVAILABLE
    from fireants.interpolator.grid_sample import torch_grid_sampler_3d
    from fireants.utils.imageutils import jacobian
except ImportError as exc:
    raise ImportError(
        "The FireANTs registration backend is not installed. Install it with:\n"
        f"    bash {_INSTALLER}\n"
        f"(underlying import error: {exc})"
    ) from exc

__all__ = [
    "Image",
    "BatchedImages",
    "FakeBatchedImages",
    "generate_image_mask_allones",
    "MomentsRegistration",
    "RigidRegistration",
    "AffineRegistration",
    "GreedyRegistration",
    "DeformableMixin",
    "FFO_AVAILABLE",
    "torch_grid_sampler_3d",
    "jacobian",
]
