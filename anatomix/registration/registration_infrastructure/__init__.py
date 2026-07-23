"""FireANTs-based feature registration pipeline for anatomix.

This package implements ``anatomix-register.py``: it extracts anatomix network
features (and/or MIND-SSC descriptors) from a pair of 3D volumes and registers
them with FireANTs, supporting rigid/affine/deformable stages, masked and
unmasked losses, optional label warping, transform export, Dice, and fold
counting, in both single-pair and batch modes.

FireANTs is imported lazily by the submodules that need it (through the single
shim :mod:`~anatomix.registration.registration_infrastructure._fireants`), so a
clear installation error is raised only when the pipeline is actually run
without the backend installed. Install it with
``registration_backend/install_fireants.sh``.
"""
