# 3D registration with anatomix + FireANTs

`anatomix-register.py` registers a moving 3D volume onto a fixed one with
[FireANTs](https://github.com/rohitrango/FireANTs). The images it matches are
anatomix network features, MIND-SSC descriptors, or raw intensities, so no
dataset-specific training is needed. It supports rigid, affine and deformable
stages, masks, label propagation, landmarks, initial transforms, transform
export in ANTs/SciPy/PyTorch/inverse form, and batch processing.

[Tutorial notebook](tutorials/anatomix_registration_fireants.ipynb) ·
[Measured results](tests/README.md)

## Install

In your anatomix Python environment, from this directory:

```bash
bash registration_backend/install_fireants.sh   # add --no-fused-ops to skip the CUDA build
```

This clones the [anatomix author's FireANTs fork](https://github.com/neel-dey/FireANTs)
at the tested revision into `registration_backend/fireants/` (gitignored) and
installs it. The fused CUDA kernels need a CUDA toolkit that matches your
PyTorch build; without them FireANTs runs its slower pure-PyTorch code.

## Usage

```bash
python anatomix-register.py --fixed fixed.nii.gz --moving moving.nii.gz --output-dir out
```

The interface follows the ANTs command-line tools: one flag per input, comma
separated per-stage lists, `x` separated pyramid schedules. Types: `PATH` is
a file, `DIR` a directory, `N` an integer, `F` a float, `A,B` one value per
stage, `AxB` one value per pyramid level.

```
Inputs (choose one mode)
  --fixed PATH, --moving PATH          One pair of scalar 3D NIfTI images.
  --fixed-dir DIR, --moving-dir DIR    Batch: directories paired by sorted filename.
  --registration-pairs-csv PATH        Batch: CSV with columns fixed,moving[,fixed_mask,moving_mask,
                                       fixed_seg,moving_seg,fixed_keypoints,moving_keypoints,initial_transform].

Optional inputs (pair mode)
  --fixed-mask PATH, --moving-mask PATH  Foreground masks (>0), on their own image's grid.
  --fixed-seg PATH, --moving-seg PATH    Integer label maps; moving labels are propagated, both give Dice.
  --fixed-keypoints PATH               CSV with x,y,z columns; the points are mapped into moving space.
  --moving-keypoints PATH              Corresponding points, row by row; gives landmark error (TRE).
  --keypoint-convention lps|ras|voxel  Coordinates of the keypoint CSVs (default lps = ITK mm).
  --initial-transform PATH             Linear transform applied first: ITK/ANTs .mat/.txt/.tfm/.h5 or FreeSurfer .lta.
  Directory batch mode takes the same inputs as directories: --fixed-mask-dir, --moving-mask-dir,
  --fixed-seg-dir, --moving-seg-dir, --fixed-keypoints-dir, --moving-keypoints-dir, --initial-transform-dir.
  CSV batch mode takes them as columns.
  --fixed-minclip F, --fixed-maxclip F, --moving-minclip F, --moving-maxclip F
                                       Intensity clipping before min-max normalization.

Transform chain
  --initialization none|image-centers|center-of-mass|moments   Closed-form start (default none).
  --transform A,B                      Stages from rigid, affine, deformable, in that order (default deformable).
  --loss A,B                           cc, mi, mse or masked_cc, masked_mi, masked_mse per stage
                                       (default masked_cc when a mask is given, else cc).
  --step-size A,B                      Adam learning rate per stage (default 0.01 linear, 1.0 deformable).
  --translation-step-size A,B          Translation learning rate of linear stages (default = --step-size); na for deformable.
  --shrink-factors AxB,AxB             Pyramid per stage, strictly decreasing (default 6x4x2x1).
  --iterations AxB,AxB                 Iterations per level (default 100 each).
  --cc-kernel-widths AxB,AxB           Odd local-CC window widths per level; na for mi/mse stages.
  --smooth-grad-sigma A,B, --smooth-warp-sigma A,B
                                       Deformable regularization in voxels (default 1.0 and 0.5); na for linear stages.
  --tolerance F                        Early-stopping tolerance on the loss slope (default 1e-6; inf disables).

Features
  --features anatomix+mindssc|anatomix|mindssc|intensity   What is registered (default anatomix+mindssc).
  --backbone anatomix|anatomix-dev|anatomix-dev-vit|custom  Network (default anatomix-dev-vit).
  --isotropic-features 0|1             Extract network/MIND features on an isotropic grid (default 1).
  --sliding-window-params W,B,O,M,S    MONAI sliding window: size, batch, overlap, mode, sigma (default 128,4,0.8,gaussian,0.25).
  --feature-normalization l2|standardized|none   Per-voxel normalization of network features (default l2).
  --mindssc-params R,D                 MIND-SSC radius and dilation (default 1,2).
  --custom-arch unet|vit, --custom-weights PATH, --unet-*, --vit-*   Own checkpoints (see --help).

Outputs
  --output-dir DIR                     Where everything is written (default .).
  --exp-name NAME                      Prefix for every output file.
  --output-transformation-convention ants|scipy|pytorch   Transform file format (default ants).
  --collapse-output-transforms 1|0     1: one final transform; 0: one cumulative snapshot per stage.
  --save-inverse                       Also write the moving-to-fixed transform and the fixed image on the moving grid.

Run control
  --device auto|cpu|cuda|cuda:N        auto picks the visible GPU with the most free memory.
  --seed N                             Random seed (default 12345).
  --verbose / --no-verbose             Print inputs, stage progress, metrics and peak GPU memory.
```

<details>
<summary><b>Inputs and file formats</b></summary>

Images are scalar 3D NIfTI files (`.nii`, `.nii.gz`). Fixed and moving may
differ in shape, spacing, orientation and field of view; the moving image is
resampled onto the fixed grid internally, and all transforms are defined in
physical space. Each mask or segmentation must have the shape and affine of
its own image; this is checked from the headers before anything is loaded.

Masks: positive voxels are foreground. Either mask, both, or neither may be
given. A mask multiplies the network or intensity channels, and with a
`masked_*` loss (the default when any mask is present) the loss is restricted
to the overlap of the fixed mask and the warped moving mask.

Segmentations: `--moving-seg` is propagated with nearest-neighbour sampling;
with `--fixed-seg` the mean Dice over the fixed segmentation's non-zero labels
is written to `metrics.csv`.

Keypoints: CSV files with `x,y,z` columns (other columns are kept). The
transform maps fixed points into moving space, so `--fixed-keypoints` is what
gets warped; `--moving-keypoints` are the corresponding points, row by row,
and give the landmark error. `lps` is ITK/ANTs world mm, `ras` is NIfTI/nibabel
world mm, `voxel` is the ITK continuous index.

CSV batch mode: relative paths resolve against the CSV; empty cells mean
"absent"; any other column is copied into `metrics.csv`. Directory batch mode
pairs files by sorted name, so the counts must match.
</details>

<details>
<summary><b>Stages, initialization and initial transforms</b></summary>

A run first sets a linear transform, then runs the `--transform` stages.
Every stage resamples the original moving image onto the fixed grid with the
transform so far, extracts its features there, and fits an identity-initialized
residual: linear residuals multiply onto the running matrix, deformable
residuals are composed as coordinate fields. This adds one feature
extraction per stage. The original moving image and labels are resampled only
once, by the final transform.

Initializations:

- `none`: physical identity. Correct when the scans already share a frame.
- `image-centers` (FireANTs' `cof`): translation that aligns the geometric
  centers of the two fields of view. Uses headers only.
- `center-of-mass`: translation that aligns the intensity centers of mass,
  computed from the clipped, min-max normalized images inside the masks, so
  signed intensities such as CT are fine. It depends on what the two fields
  of view contain: on the AbdomenMRCT pairs it moves already aligned scans by
  30 mm.
- `moments`: rotation and translation from second-order moments. Principal
  axes have no sign, so it can return a large rotation (150° on AbdomenMRCT).
- `--initial-transform`: a file instead. ITK/ANTs files (`.mat`, `.txt`,
  `.tfm`, `.h5`, e.g. `0GenericAffine.mat` or this tool's own `warp-*.mat`)
  map fixed to moving physical coordinates. FreeSurfer `.lta` files (RAS-to-RAS
  or vox-to-vox) map their `src` volume to their `dst` volume; the file's
  volume geometry is compared with the images to tell which one is `src`, and
  when neither matches, `src` is taken as the moving image, as `mri_coreg
  --mov moving --ref fixed` writes it. Cannot be combined with
  `--initialization`.

Linear stages optimize the rotation (or the full linear part) with
`--step-size` and the translation with `--translation-step-size` as two
separate Adam parameter groups. Translation is expressed in units of the fixed
image's physical radius (the RMS half-extent of its field of view), so the
same rate works at any voxel size; by default it equals `--step-size`.

Every pyramid level is floored at 32 voxels per axis, so a multi-resolution
stage needs at least 34 voxels along every axis of both images.
</details>

<details>
<summary><b>Features</b></summary>

`anatomix+mindssc` concatenates 32 network channels and 12 MIND-SSC channels;
`anatomix` and `mindssc` use one family; `intensity` registers the clipped,
min-max normalized image itself and loads no network. Network weights download
from Hugging Face on first use. Features are extracted on an isotropic grid at
the finest spacing and resampled back; on strongly anisotropic images this can
multiply memory, and `--isotropic-features 0` extracts on the native grid.
`anatomix-dev-vit` requires a 128-voxel window. Changing the sliding-window
overlap changes the features.
</details>

<details>
<summary><b>Losses</b></summary>

`cc` is FireANTs' local normalized cross-correlation (its PyTorch
implementation; the fused CUDA kernel is not used because its masked variant
fails and its unmasked variant adds folds, see the measured results). `mi`
is global mutual information on inputs clamped to [0,1], which is a poor fit
for signed network features; `mse` is mean squared error. The `masked_`
variants use the mask channel.
</details>

## Examples

All numbers are from [tests/README.md](tests/README.md). Choose settings with
validation labels or landmarks; a lower loss alone does not establish accuracy.

**Abdominal MR→CT (Learn2Reg AbdomenMRCT).** One deformable stage with wide
CC windows gives mean Dice 0.879 with zero folds over the eight training pairs:

```bash
python anatomix-register.py --fixed CT.nii.gz --moving MR.nii.gz \
    --fixed-mask CT_mask.nii.gz --moving-mask MR_mask.nii.gz \
    --fixed-seg CT_seg.nii.gz --moving-seg MR_seg.nii.gz \
    --transform deformable --step-size 1.0 --shrink-factors 6x4x2x1 \
    --iterations 100x100x100x100 --cc-kernel-widths 21x13x11x9 \
    --fixed-minclip -450 --fixed-maxclip 450 --moving-minclip 0 --moving-maxclip 20000 \
    --output-dir abdomen-out
```

These scans are already aligned; rigid/affine stages and center-of-mass
initialization lower the Dice.

**Longitudinal brain scans with landmarks (BraTS-Reg).** Skull-stripped scans
in one frame:

```bash
python anatomix-register.py --fixed baseline_t1.nii.gz --moving followup_flair.nii.gz \
    --fixed-mask baseline_brainmask.nii.gz --moving-mask followup_brainmask.nii.gz \
    --fixed-keypoints baseline_landmarks.csv --moving-keypoints followup_landmarks.csv \
    --transform deformable --step-size 0.1 --shrink-factors 4x2x1 \
    --iterations 200x100x50 --cc-kernel-widths 7x5x3 --output-dir brain-out
```

For cases that start tens of mm apart, add an affine stage:
`--transform affine,deformable --step-size 0.01,0.1 --shrink-factors
4x2x1,4x2x1 --iterations 100x100x100,200x100x50 --cc-kernel-widths
9x7x5,7x5x3 --smooth-grad-sigma na,1.0 --smooth-warp-sigma na,0.5`
(case 107: 26.3 mm → 3.8 mm). Compare `tre_median` with `tre_initial_median`
per case.

**Rigid or affine only.** `--transform rigid` (or `affine`) with `--step-size
0.01 --shrink-factors 4x2x1 --iterations 100x100x100 --cc-kernel-widths
9x7x5`. The intensity baseline needs no network: `--features intensity
--loss mi` (or `cc`/`mse` within one modality).

**Whole-body scans.** CT→CT (PSMAReg) and MR→CT on native grids both work
with `--initialization center-of-mass` or `image-centers`, `--transform
rigid,deformable`, and a pyramid such as `8x4x2x1` with CC windows
`15x11x9x5`. GPU memory grows with fixed-grid voxels × channels (45 with a
mask): a 2 mm whole-body grid (33M voxels) needs 63 GB with the pyramid
stopping at shrink 2 (`--shrink-factors 8x4x2`) and more than 96 GB at full
resolution; native 1 mm whole-body scans do not fit. Resample them to 2–3 mm
first, or use `--features intensity`.

**Starting from an existing transform.**
`--initial-transform previous/warp-moving.mat` (or `reg.lta` from
`mri_coreg`) followed by `--transform deformable` gives the same result as the
corresponding chain run in one go.

## Outputs

Files use the moving filename stem, prefixed by `--exp-name` when given:

| File | Contents |
|---|---|
| `moved-*.nii.gz` | Moving image on the fixed grid (trilinear). |
| `moved-seg-*.nii.gz` | Propagated labels (nearest). |
| `moved-keypoints-*.csv` | Fixed landmarks mapped into moving space. |
| `warp-*` | The transform (see below). |
| `inverse-warp-*`, `inverse-moved-*` | With `--save-inverse`: the moving-to-fixed transform on the moving grid, and the fixed image (and segmentation) resampled onto the moving grid. |
| `metrics.csv` | Input columns plus `dice`, `num_folds`, `tre_median`, `tre_mean`, `tre_initial_median`, `robustness` (fraction of landmarks improved) and `inverse_residual_mm`; one row per pair, written as each pair completes. |

All `warp-*` transforms map **fixed coordinates to moving coordinates**, the
direction that resamples the moving image onto the fixed grid:

- `ants` (default): a linear `.mat` for rigid/affine chains, otherwise an ITK
  displacement field `.nii.gz` in LPS mm. Apply with
  `antsApplyTransforms -d 3 -i moving.nii.gz -r fixed.nii.gz -t warp-moving.nii.gz -o out.nii.gz`
  (`-n NearestNeighbor` for labels).
- `pytorch`: a `.pt` sampling grid `(1,Z,Y,X,3)` of normalized moving
  coordinates for `grid_sample(..., align_corners=True)`; transpose nibabel
  arrays from `(X,Y,Z)` to `(Z,Y,X)` first (see the notebook).
- `scipy`: `.npz` with `arr_0` of shape `(X,Y,Z,3)` such that
  `moving_index = fixed_index + arr_0`, for `scipy.ndimage.map_coordinates`;
  linear-only chains store `affine`, the physical LPS-mm matrix, instead.

ANTs, SciPy and PyTorch application of these files reproduce the CLI's moved
image to about 1e-6 relative error inside the moving field of view; the
libraries differ at its border. `--collapse-output-transforms 0` writes one
cumulative snapshot per stage (`warp-*-init`, `warp-*-0-affine`, ...), which
must not be composed again. Dense inverses are computed numerically;
`inverse_residual_mm` is the largest inverse-consistency error inside the
fixed FOV (0.002 mm on fold-free abdominal fields, larger where a field is
near-singular).

`num_folds` counts interior fixed voxels with a non-positive physical Jacobian
determinant. Repeated deformable stages, initial transforms and initializations
that move part of the fixed FOV outside the moving image add clamped border
voxels to this count; those are not anatomical folds, but inspect results near
the FOV boundary.

## Not supported

Multi-channel input images (a stack of co-registered modalities; extract
features yourself with `registration_infrastructure.features`), FireANTs'
symmetric (SyN) deformation, per-stage feature families, and 2D images.

## Legacy ConvexAdam

The [ConvexAdam backend](registration_backend/convexadam/) and its
[notebook](tutorials/anatomix_registration_convexadam.ipynb) reproduce the
ICLR'25 anatomix registration results and are not used by this CLI.

## Credits and license

Registration is performed by **FireANTs**
([repository](https://github.com/rohitrango/FireANTs),
[documentation](https://fireants.readthedocs.io/en/latest/)); this project uses [the anatomix author’s fork](https://github.com/neel-dey/FireANTs).
If you use this backend in a paper, please cite anatomix and FireANTs:

```bibtex
@inproceedings{dey2025learning,
  title={Learning general-purpose biomedical volume representations using randomized synthesis},
  author={Dey, Neel and Billot, Benjamin and Wong, Hallee and Wang, Clinton and Ren, Mengwei and Grant, Ellen and Dalca, Adrian and Golland, Polina},
  booktitle={International Conference on Learning Representations},
  volume={2025},
  pages={32033--32064},
  year={2025}
}
@article{jena2024fireants,
  title={FireANTs: Adaptive Riemannian Optimization for Multi-Scale Diffeomorphic Registration},
  author={Jena, Rohit and Chaudhari, Pratik and Gee, James C},
  journal={Nature Communications},
  year={2024}
}
@inproceedings{jena2025scalable,
  title={A Scalable Distributed Framework for Multimodal GigaVoxel Image Registration},
  author={Jena, Rohit and Zope, Vedant and Chaudhari, Pratik and Gee, James C},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

MIND-SSC follows Heinrich et al., MICCAI 2013. The retained ConvexAdam backend is
a modified fork of the
[ConvexAdam repository](https://github.com/multimodallearning/convexAdam).
If you use the ConvexAdam backend in a paper, please cite ConvexAdam as well:

```bibtex
@article{siebert2024convexadam,
  title={Convexadam: Self-configuring dual-optimization-based 3d multitask medical image registration},
  author={Siebert, Hanna and Gro{\ss}br{\"o}hmer, Christoph and Hansen, Lasse and Heinrich, Mattias P},
  journal={IEEE Transactions on Medical Imaging},
  volume={44},
  number={2},
  pages={738--748},
  year={2024},
  publisher={IEEE}
}
```
