# Multi-modality 3D registration with anatomix features + FireANTs

`anatomix-register.py` registers arbitrary **3D** volume pairs by extracting
anatomix network features (and/or a hand-crafted MIND-SSC descriptor) and
optimizing them with [**FireANTs**](https://github.com/rohitrango/FireANTs), a
GPU diffeomorphic registration library. It supports rigid / affine / deformable
stages, masked and unmasked losses, optional label warping, transform export,
Dice, and fold counting, in single-pair and batch modes.

Only `anatomix-register.py` and this `README.md` sit at the top of this folder;
everything else lives in subfolders:

```
registration/
├── anatomix-register.py          # CLI entry point (this is what you run)
├── registration_infrastructure/  # the FireANTs feature-registration pipeline
├── registration_backend/
│   ├── install_fireants.sh        # installs the FireANTs backend (below)
│   ├── fireants/                  # gitignored editable FireANTs clone
│   └── convexadam/                # the retained ConvexAdam backend (see below)
└── tutorials/
    ├── anatomix_registration_fireants.ipynb
    └── anatomix_registration_convexadam.ipynb
```

## Reproducing the ICLR'25 ConvexAdam results

The ICLR'25 anatomix registration numbers came from a ConvexAdam backend, which
is **retained unchanged** in this checkout under
`registration_backend/convexadam/` and demonstrated in
[`tutorials/anatomix_registration_convexadam.ipynb`](tutorials/anatomix_registration_convexadam.ipynb).
No old-commit checkout or git tag is needed — call it directly:

```python
from anatomix.registration.registration_backend.convexadam import convex_adam
```

It is **not** exposed by the new `anatomix-register.py` CLI (which is
FireANTs-only). See that subfolder for its own CLI
(`run_convex_adam_with_network_feats.py`).

## Installing the FireANTs backend

FireANTs is installed separately (it is not an anatomix dependency) as a
gitignored editable clone of the author's fork:

```bash
bash registration_backend/install_fireants.sh          # core install
bash registration_backend/install_fireants.sh --with-fused-ops   # + fused CUDA kernels (recommended)
```

The fused-ops kernels (`--with-fused-ops`) are a **speed** optimization — the
PyTorch fallback now produces the same registration quality (an earlier fork bug,
where the non-fused multi-resolution downsampling used a different Gaussian and
caused worse, fold-heavy deformations on harder pairs, has been fixed so the FFT
downsampling path runs in pure torch when the kernels are absent). Compiling the
kernels needs a CUDA toolkit whose version matches your PyTorch build (e.g. for a
`cu130` torch, point `CUDA_HOME` at a CUDA 13.x toolkit and set
`TORCH_CUDA_ARCH_LIST` to your GPU's arch, e.g. `12.0` for Blackwell).
`scikit-learn` (for Dice) is an anatomix dependency and is already installed.

## Quick start — reproduce the SOTA Learn2Reg-AbdomenMRCT result

The reference AbdomenMRCT result is a single deformable `masked_cc`
`GreedyRegistration` over an 8→4→2→1 pyramid with 100 iters/level, CC kernels
21×13×11×9, step size 1.0, and gradient/warp smoothing 1.0/0.5. The 21×13×11×9
CC schedule is tuned for this dataset, so it is passed explicitly — omit
`--cc-kernel-widths` and each stage falls back to FireANTs' own default kernel
size. Register the 8 MR→CT pairs (fixed = CT `_0001`, moving = MR `_0000`) with
masks and labels:

```bash
python anatomix-register.py \
    --fixed  CT.nii.gz  --moving MR.nii.gz \
    --fixed-mask CT_mask.nii.gz --moving-mask MR_mask.nii.gz \
    --fixed-seg  CT_seg.nii.gz  --moving-seg  MR_seg.nii.gz \
    --backbone anatomix-dev-vit --step-size 1.0 --cc-kernel-widths 21x13x11x9 \
    --fixed-minclip -450 --fixed-maxclip 450 \
    --moving-minclip 0 --moving-maxclip 20000 \
    --output-dir out --exp-name mrct
```

Over the 8 pairs this reaches **mean macro-Dice ≈ 0.875** with near-zero folds,
matching the reference `anatomix-dev-vit` result (report ViT-S ≈ 0.87). `--loss`
defaults to `masked_cc` when both masks are present and `cc` otherwise.

## Input modes

Exactly one of:

- **Single pair:** `--fixed` / `--moving`.
- **Directory batch:** `--fixed-dir` / `--moving-dir` (equal counts, paired
  lexicographically). Optional `--fixed-mask-dir`, `--moving-mask-dir`,
  `--fixed-seg-dir`, `--moving-seg-dir`.
- **CSV batch:** `--registration-pairs-csv` — a header row with columns
  `fixed,moving[,fixed_mask,moving_mask,fixed_seg,moving_seg]`; empty cells mean
  "absent"; relative paths resolve against the CSV's directory. Use this when
  you need explicit (non-lexicographic) pairing.

Every input volume must exist, end in `.nii`/`.nii.gz`, and be a single-channel
3D volume. Run `python anatomix-register.py --help` for the full interface.

### Masks and segmentations

- Provide **both** masks or neither (one alone is an error). When present, masks
  gate the network features and — for a `masked_*` loss — become the last loss
  channel. An explicit unmasked loss is honored even when masks are present.
- A **moving** segmentation is required to warp labels; a **fixed** segmentation
  is optional. With both, Dice is reported (background label 0 excluded); with
  only the moving one, the warped label is written and Dice is left blank.

## Transform chain

`--transform` is a comma-separated list of stages from
`{rigid,affine,deformable}`, ordered `rigid ≤ affine ≤ deformable` (repeated
deformable stages are allowed). All per-stage lists are comma-separated with one
entry per stage; each pyramid schedule is `AxBx...`:

| flag | meaning | default |
|------|---------|---------|
| `--initialization` | `none` / `center-of-mass` / `moments` (closed-form, before the chain) | `none` |
| `--loss` | one of `cc,mi,mse,masked_cc,masked_mi,masked_mse` per stage | auto (masked_cc if masks else cc) |
| `--step-size` | Adam LR per stage | `1.0` deformable / `0.1` linear |
| `--shrink-factors` | resolution schedule per stage | `8x4x2x1` |
| `--iterations` | iterations per level (matches shrink) | `100` per level |
| `--cc-kernel-widths` | odd CC widths per level (`na` for non-CC stages) | FireANTs' default kernel per stage |
| `--smooth-grad-sigma` / `--smooth-warp-sigma` | deformable regularization (`na` for linear) | `1.0` / `0.5` |

Examples:

```bash
# center-of-mass init, then affine + deformable (step sizes default per stage)
python anatomix-register.py --fixed CT.nii.gz --moving MR.nii.gz \
    --fixed-mask CT_mask.nii.gz --moving-mask MR_mask.nii.gz \
    --initialization center-of-mass --transform affine,deformable \
    --loss masked_cc,masked_cc \
    --shrink-factors 4x2x1,8x4x2x1 --iterations 100x100x50,100x100x100x100 \
    --cc-kernel-widths 7x5x3,21x13x11x9

# batch over a CSV
python anatomix-register.py --registration-pairs-csv pairs.csv --output-dir out

# MIND-SSC only (no network backbone is loaded/downloaded)
python anatomix-register.py --fixed CT.nii.gz --moving MR.nii.gz --use-mindssc mindssc-only
```

## Features

`--backbone {anatomix, anatomix-dev, anatomix-dev-vit (default), custom}`.
Features are (optionally) extracted on an isotropic grid (`--isotropic-features`,
finest spacing) via MONAI sliding-window inference
(`--sliding-window-params window,sw_batch,overlap,mode,sigma`, default
`128,4,0.8,gaussian,0.25`; `anatomix-dev-vit` requires `window=128`), then
network features are normalized (`--feature-normalization l2|standardized|none`)
and combined with MIND-SSC (`--use-mindssc both|feats-only|mindssc-only`,
`--mindssc-params radius,dilation`, default `1,2`). `--backbone custom` requires
`--custom-arch {unet,vit}` + `--custom-weights` and exposes every UNet/ViT
constructor argument as an explicit `--unet-*` / `--vit-*` flag (see `--help`).

## Outputs

Per pair, written to `--output-dir` with an optional `--exp-name` prefix:

- `moved-<moving_stem>.nii.gz` — moving image warped onto the fixed grid
  (trilinear).
- `moved-seg-<moving_stem>.nii.gz` — moving label warped (nearest), if a moving
  segmentation was given.
- `warp-<moving_stem>.<ext>` — the transform (see below).
- `metrics.csv` — the input columns plus `dice` (blank if N/A) and `num_folds`.

`--output-transformation-convention`:

- `ants` — FireANTs ANTs-compatible transform (deformable → an ITK vector
  displacement field `.nii.gz`; linear → `.mat`).
- `scipy` — FireANTs SciPy transform (deformable → `.npz`; linear → an `.npz`
  with the affine matrix).
- `pytorch` — the normalized fixed→moving sampling grid as a `.pt` tensor of
  shape `[1, H, W, D, 3]` (always available, exact for any chain).

`--collapse-output-transforms 1` (default) writes one composed transform;
`0` writes one **cumulative** snapshot per stage (numbered, and after a
non-`none` initialization) — these are diagnostic snapshots, not sequentially
chainable residuals. For chains with more than one deformable stage, only the
`pytorch` convention captures the exact composed transform.

Apply a saved `pytorch` transform to any volume on the moving grid:

```python
import torch, torch.nn.functional as F
grid = torch.load("warp-MR.pt")                    # [1, H, W, D, 3], fixed->moving
moved = F.grid_sample(moving[None, None].float(), grid,
                      mode="bilinear", align_corners=True)   # on the fixed grid
```

The `ants`/`scipy` files follow FireANTs' formats (apply with `antsApplyTransforms`
/ ITK, or FireANTs' loaders, respectively; see the FireANTs docs).

## Credits and license

Registration is performed by **FireANTs**
([repository](https://github.com/rohitrango/FireANTs),
[documentation](https://fireants.readthedocs.io/en/latest/)); this project uses
the [`neel-dey/FireANTs`](https://github.com/neel-dey/FireANTs) fork. Please cite
FireANTs as requested by its documentation when you use this backend. FireANTs is
installed separately under its own license (see
`registration_backend/fireants/LICENSE`). MIND-SSC follows Heinrich et al.,
MICCAI 2013. The retained ConvexAdam backend is a modified fork of the
[ConvexAdam repository](https://github.com/multimodallearning/convexAdam).
