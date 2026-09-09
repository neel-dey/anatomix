#!/usr/bin/env python
"""Register 3D volume pairs with FireANTs on anatomix features.

    bash registration_backend/install_fireants.sh      # once
    python anatomix-register.py --fixed F.nii.gz --moving M.nii.gz --output-dir out
    python anatomix-register.py --help

What a run does, and where each step lives:

1. ``cli.build_parser`` / ``cli.prepare``: parse the options, validate every
   input header, transform file and per-stage schedule before anything is
   loaded. Options come in groups: inputs (pair, CSV or directories, plus
   masks, segmentations, landmarks and an initial transform), intensity
   clipping, the stage chain (``--transform``, ``--initialization``,
   ``--loss``, ``--step-size``, ``--shrink-factors``, ``--iterations``,
   ``--cc-kernel-widths``, smoothing), features (``--features``,
   ``--backbone``, sliding-window and MIND-SSC settings), outputs
   (``--output-dir``, transform convention, snapshots, inverse) and device.
2. ``features``: intensity normalization, anatomix network features
   (MONAI sliding windows), MIND-SSC descriptors, mask channel.
3. ``register.run_registration``: the initial linear transform, then one
   FireANTs stage per ``--transform`` entry. Every stage resamples the
   original moving image onto the fixed grid with the transform so far,
   extracts its features there and fits a residual, so fixed and moving
   images may have different grids.
4. ``warp_io`` / ``metrics``: resample the moving image and labels once,
   map landmarks, export the transform (ANTs, SciPy or PyTorch convention)
   and its inverse, and write Dice, fold count and landmark error to
   ``metrics.csv``.

The README in this directory documents the options, recipes and output
conventions; ``tests/README.md`` holds the measured results. The ConvexAdam
backend of the ICLR'25 paper is kept under ``registration_backend/convexadam/``
and is not used here.
"""
from anatomix.registration.registration_infrastructure.cli import build_parser, prepare


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        pairs, input_columns, stages = prepare(args)
    except ValueError as error:
        parser.error(str(error))
    from anatomix.registration.registration_infrastructure.pipeline import run

    run(args, pairs, input_columns, stages)


if __name__ == "__main__":
    main()
