#!/usr/bin/env python
"""anatomix-register.py -- FireANTs registration on anatomix features.

Register arbitrary 3D volume pairs by extracting anatomix network features
(and/or MIND-SSC descriptors) and optimizing them with FireANTs. Supports
rigid/affine/deformable stages, masked and unmasked losses, optional label
warping, transform export (ants/scipy/pytorch), Dice, and fold counting, in
single-pair and batch modes.

Install the backend once with::

    bash registration_backend/install_fireants.sh

Reproduce the SOTA Learn2Reg AbdomenMRCT result (≈0.879 mean macro-Dice, ~0
folds) with a single deformable ``masked_cc`` stage. The ``21x13x11x9`` CC
kernel schedule is tuned for this dataset and must be passed explicitly (omit it
and each stage falls back to FireANTs' own default kernel, which does not reach
zero folds)::

    python anatomix-register.py \\
        --fixed CT.nii.gz --moving MR.nii.gz \\
        --fixed-mask CT_mask.nii.gz --moving-mask MR_mask.nii.gz \\
        --moving-seg MR_seg.nii.gz --fixed-seg CT_seg.nii.gz \\
        --backbone anatomix-dev-vit --step-size 1.0 \\
        --cc-kernel-widths 21x13x11x9 \\
        --fixed-minclip -450 --fixed-maxclip 450 \\
        --moving-minclip 0 --moving-maxclip 20000

Run ``python anatomix-register.py --help`` for the full interface.

Reproducing the ICLR'25 ConvexAdam results
------------------------------------------
The original ConvexAdam backend that produced the anatomix ICLR'25 registration
numbers is retained in this checkout under
``registration_backend/convexadam/`` and demonstrated in the
``tutorials/anatomix_registration_convexadam.ipynb`` notebook. It is not exposed
by this CLI; use that backend/notebook directly. No old-commit checkout is
required.
"""
from anatomix.registration.registration_infrastructure.cli import main

if __name__ == "__main__":
    main()
