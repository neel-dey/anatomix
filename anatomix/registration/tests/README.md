# Verification

Measured on 2026-09-08 with the environment in
[`results/environment.json`](results/environment.json) (PyTorch 2.13+cu130,
FireANTs fork `1b39f6d`, fused ops built) on RTX PRO 6000 Blackwell GPUs
(96 GB). Each `results/<run>/` folder keeps the exact command
(`command.txt`) and `metrics.csv`; images and fields are not stored.

## Tests

```bash
# from the repository root
USE_FFO=False python -m unittest discover -s anatomix/registration/tests -v          # CPU
CUDA_VISIBLE_DEVICES=<gpu> REGISTRATION_TEST_DEVICE=cuda:0 \
    python -m unittest discover -s anatomix/registration/tests -v                    # GPU
python -m unittest discover -s anatomix/registration/registration_backend/fireants/tests \
    -p test_linear_translation.py -v                                                 # fork change
```

The 23 tests pass on both. They check physical-coordinate contracts of grids,
landmarks (all three conventions) and exports against SimpleITK; linear and
dense composition; fold counting under reflected headers; damped grid
inversion (linear and dense); ITK `.mat` and FreeSurfer `.lta` (RAS-to-RAS and
vox-to-vox, both `src` directions) read back as initial transforms;
center-of-mass initialization on signed (CT-like) intensities; the six losses,
chained stages, batch naming, one-sided masks and image-center initialization
through the CLI; input validation.

Scripts: [`reproduce_abdomen.py`](reproduce_abdomen.py) (acceptance run),
[`probe_linear.py`](probe_linear.py) (recovery of known rigid/affine
transforms on native images) and [`run_recipes.py`](run_recipes.py) (a small
manifest through the README examples).

## AbdomenMRCT acceptance

The eight paired training cases (CT fixed, MR moving, both 192×160×192 at
2 mm) with the README example:

| Pair | 0001 | 0002 | 0003 | 0004 | 0005 | 0006 | 0007 | 0008 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dice | 0.8568 | 0.8146 | 0.8400 | 0.9137 | 0.9183 | 0.8980 | 0.8859 | 0.9050 | **0.8790** |
| folds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Reference: 0.879067 ± 0.001, zero folds ([`results/abdomen_repro`](results/abdomen_repro)).
Peak GPU memory 24 GB, about 30 s per pair.

## Transforms and exports

- **ANTs.** `antsApplyTransforms` on every exported `warp-*.nii.gz` reproduces
  the CLI's moved image to a relative error of about 2e-7 (mean) and 4e-6
  (99.9th percentile) at fixed voxels that sample inside the moving FOV, and
  gives the same label Dice ([`ants_check.json`](results/abdomen_repro/ants_check.json)).
  Voxels that sample outside the moving FOV differ because ITK and PyTorch pad
  differently at the image border.
- **SciPy and PyTorch.** `map_coordinates` on `.npz` and `grid_sample` on `.pt`
  reproduce the moved image to 1e-7 relative error (`export_verification.json`
  in the chain runs).
- **Snapshots.** With `--collapse-output-transforms 0`, the `0-affine.mat`
  of an `affine,deformable` chain is identical to a standalone affine run, and
  the final snapshot applied with ANTs equals the collapsed output.
- **Initial transform.** Passing that `.mat` through `--initial-transform` to
  a deformable stage gives the chain's Dice (0.8333 both ways). The LTA
  reader agrees with FreeSurfer's `lta_convert` to 2e-5 for the same
  transform written as RAS-to-RAS and as vox-to-vox.
- **Inverse.** `--save-inverse` inverts the abdominal fields to a worst
  inverse-consistency residual of 0.002 mm (BraTS fields: 0.0005 mm); a
  linear chain's inverse is exact. `antsApplyTransforms` with
  `inverse-warp-*.nii.gz` reproduces `inverse-moved-*` inside the fixed FOV
  ([`inverse_verification.json`](results/abd_inverse/inverse_verification.json)).
  On the 2 mm whole-body MR→CT field below, 0.8% of body voxels keep a
  residual above 1 mm (the field is fold-free but locally near-singular); the
  verbose log reports this fraction and `metrics.csv` the maximum.

## Different headers and grids

Pair 0001 with the moving MR (and its mask and labels) re-oriented or
resampled, same settings as above. The original gives 0.8568.

| moving variant | Dice | folds |
|---|---:|---:|
| flipped x (LAS) | 0.8568 | 0 |
| axes permuted (ASR) | 0.8568 | 0 |
| LIA | 0.8569 | 0 |
| PSR | 0.8568 | 0 |
| 3×3×5 mm, 128×107×77 | 0.8661 | 1215, all in the fixed border where it samples outside the smaller moving FOV |

The result does not depend on the header orientation. The reason is the
resampling step in `register.py`: given the same re-oriented pairs directly,
FireANTs' greedy registration (intensity CC) still reaches 0.79–0.80 Dice but
with 2k–21k folds, because its warp update mixes the two images' normalized
coordinate frames.

## Chains and initializations (AbdomenMRCT pairs 0001 / 0002)

| Configuration | Dice | folds |
|---|---:|---:|
| deformable (example) | 0.857 / 0.815 | 0 / 0 |
| affine → deformable | 0.833 / 0.814 | 0 / 0 |
| rigid → affine → deformable | 0.507 / 0.831 | 2848 / 0 |
| deformable → deformable | 0.859 / 0.834 | 197 / 327 |
| center-of-mass → deformable | 0.858 / 0.839 | 7967 / 0 |
| image-centers → deformable | 0.857 / 0.815 | 0 / 0 |
| rigid only | 0.468 / 0.632 | 0 / 0 |
| affine only | 0.621 / 0.701 | 0 / 0 |
| moments only | 0.015 | 0 |

Starting overlap is 0.563 / 0.535. Every non-zero fold count above lies
outside the body mask (air) or in the two-voxel border where a stage samples
outside the moving FOV; none is inside the anatomy. A rigid stage misaligns
pair 0001, and the moments initializer returns a 150° principal-axis
rotation, so these scans are best registered by the single deformable stage.

## Loss implementations

FireANTs has a PyTorch local CC (`cc`) and a fused CUDA one (`fusedcc`).
Pairs 0001 / 0002, deformable example settings:

| loss | Dice | folds |
|---|---:|---:|
| `cc`, no masks | 0.850 / 0.666 | 0 / 0 |
| `fusedcc`, no masks | 0.851 / 0.767 | 1763 / 1894 |
| `masked_cc` | 0.857 / 0.815 | 0 / 0 |
| `masked_fusedcc` | 0.387 / 0.479 | 5166 / 0 |

`masked_fusedcc` does not work, and unmasked `fusedcc` adds folds, so the CLI
uses `cc` and `masked_cc` (the fused kernels are still used for grid
sampling, blurring and warp composition).

## BraTS-Reg (six training cases, baseline T1 fixed, follow-up FLAIR moving)

Median landmark error in mm, initial → registered; all runs fold-free.

| Case | initial | deformable | COM → deformable | affine → deformable | intensity affine (MI) |
|---|---:|---:|---:|---:|---:|
| 021 | 2.50 | 3.00 | 2.89 | 3.23 | 3.42 |
| 045 | 1.21 | 1.43 | 1.23 | 1.44 | 1.75 |
| 060 | 3.16 | 2.46 | 2.24 | 2.41 | 3.17 |
| 085 | 3.32 | 1.77 | 1.95 | 1.79 | 2.96 |
| 107 | 26.29 | 22.95 | 5.66 | 3.77 | 27.11 |
| 138 | 12.65 | 5.03 | 1.55 | 1.43 | 1.32 |

Cases that start within the annotation noise (about 2–3 mm) are not improved
by any configuration; the large offsets need an affine stage or a
center-of-mass start. Peak memory 44 GB for the feature runs (240×240×155 at
1 mm, 45 channels), 2 GB for the intensity run.

## Whole-body scans

**PSMAReg CT→CT** (Learn2Reg 2026, 192×192×288 at 2.7×2.7×3.3 mm, follow-up
onto baseline, body masks, TotalSegmentator labels), three ablation-cohort
pairs, next to the specialised PSMAReg driver's own numbers for the same
pairs (its "tuned recipe" registers a CT+PET stack; "+ features" adds the
same anatomix + MIND-SSC channels):

| pair | CLI defaults | CLI intensity, COM init | CLI features, COM init | driver: tuned recipe | driver: + features |
|---|---:|---:|---:|---:|---:|
| 0378 | 0.779 | 0.760 | 0.782 | 0.741 | 0.791 |
| 0297 | 0.000 | 0.714 | 0.750 | 0.705 | 0.765 |
| 0102 | 0.747 | 0.731 | 0.754 | 0.721 | 0.763 |

The CLI defaults (no initialization) fail on 0297, as the driver's "FireANTs
defaults" arm does; center-of-mass initialization fixes it. Peak memory
43 GB, about 70 s per pair.

**MR→CT, one subject** (native CT 512×512×531 at 0.98×0.98×2 mm in LAS,
native MR 384×312×396 at 1.3×1.3×3 mm in LPS, about 300 mm apart; 46 shared
TotalSegmentator labels):

| inputs | configuration | Dice | folds | peak GB | s |
|---|---|---:|---:|---:|---:|
| prepared 3 mm grid (shared) | rigid → deformable, `10x6x4x2` | 0.686 | 0 | 13 | 37 |
| both resampled to 2 mm, own grids | image-centers, rigid → deformable, `8x4x2` | 0.676 | 0 | 63 | 187 |
| native grids | image-centers, rigid → deformable, `8x4x2` | out of memory: 51 GB in use after feature extraction, then FireANTs' FFT downsampling of the 45-channel CT requests another 46 GB | | > 96 | |

The segmentation-supervised, polyrigid pipeline built for this subject
reaches 0.847; the general CLI reaches 0.69 with no labels in the loss.

## Memory

Peak GPU memory grows with fixed-grid voxels × channels (44 feature channels
plus the mask).

| fixed grid | channels | pyramid | peak GB |
|---|---:|---|---:|
| 192×160×192 (5.9M) | 45 | `6x4x2x1` | 24 |
| 240×240×155 (8.9M) | 45 | `4x2x1` | 44 |
| 192×192×288 (10.6M) | 45 | `6x4x2x1` | 43 |
| 251×251×531 (33M) | 45 | `8x4x2` | 63 |
| 251×251×531 (33M) | 45 | `8x4x2x1` | > 96 (out of memory at the full-resolution level) |
| 512×512×531 (139M) | 45 | any | > 96 (out of memory building the pyramid) |

## Linear stages and dimensionless translation

The fork optimizes translation as `t / R`, `R` the RMS half-extent of the
fixed FOV, in an Adam parameter group separate from the rotation/linear
part, so `--step-size 0.01` works for rigid/affine stages at any voxel size
and `--translation-step-size` can be set on its own. Recovery of known
transforms on native CT and brain images (rotation 4/-3/7°, translation
8/-5/4 mm, plus scale/shear for affine) is 0.01–0.26 mm mean error with
features or intensities, against 5–10 mm with the previous physical-unit
rate; translation rates of 0.5×, 1× and 2× the rotation rate all recover the
transforms
([`native_ct_rigid_dimensionless.json`](results/native_ct_rigid_dimensionless.json),
[`native_brain_rigid_dimensionless.json`](results/native_brain_rigid_dimensionless.json),
[`native_ct_affine_dimensionless.json`](results/native_ct_affine_dimensionless.json)).
