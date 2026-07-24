# Training-free 3D multi-modality registration with anatomix + FireANTs

### [FireANTs tutorial notebook](tutorials/anatomix_registration_fireants.ipynb) | [Legacy ConvexAdam tutorial](tutorials/anatomix_registration_convexadam.ipynb)

`anatomix-register.py` registers arbitrary **3D** volume pairs with no
dataset-specific training. It extracts contrast-invariant anatomix features (and/or
a hand-crafted MIND-SSC descriptor) and aligns them with
[**FireANTs**](https://github.com/rohitrango/FireANTs), a GPU diffeomorphic
registration library. You get rigid / affine / deformable stages, masked and
unmasked losses, optional label warping with Dice, ANTs / SciPy / PyTorch
transform export, and fold counting — in single-pair and batch modes.

> This is the general registration backend for anatomix. The ICLR'25 ConvexAdam
> path is retained unchanged (see [below](#reproducing-the-iclr25-convexadam-results)).

Only `anatomix-register.py` and this `README.md` sit at the top of this folder;
everything else lives in subfolders (`registration_infrastructure/` = the
pipeline, `registration_backend/` = FireANTs + ConvexAdam, `tutorials/`).

## Install the FireANTs backend

FireANTs is not an anatomix dependency; install it once as a gitignored editable
clone of the author's fork:

```bash
bash registration_backend/install_fireants.sh                 # full install, WITH fused-ops (recommended)
bash registration_backend/install_fireants.sh --no-fused-ops  # skip fused-ops (degraded fallback)
```

The fused-ops CUDA kernels are built **by default** and are **required for now**
for correct results — on the current fork the pure-PyTorch fallback has a
multi-resolution downsampling bug that lowers Dice and adds folds (the CLI warns
at run time if the kernels aren't active). Compiling them needs a CUDA toolkit
matching your PyTorch build (e.g. for a `cu130` torch, point `CUDA_HOME` at a CUDA
13.x toolkit and set `TORCH_CUDA_ARCH_LIST` to your GPU arch, e.g. `12.0` for
Blackwell). `scikit-learn` (for Dice) is already an anatomix dependency.

## Reproduce the SOTA Learn2Reg-AbdomenMRCT result

One command registers an MR→CT pair with the reference configuration
(`anatomix-dev-vit` features, a single deformable `masked_cc` stage over an
8→4→2→1 pyramid, dataset-tuned `21x13x11x9` CC kernels, masks, and labels):

```bash
python anatomix-register.py \
    --fixed CT.nii.gz --moving MR.nii.gz \
    --fixed-mask CT_mask.nii.gz --moving-mask MR_mask.nii.gz \
    --fixed-seg CT_seg.nii.gz --moving-seg MR_seg.nii.gz \
    --backbone anatomix-dev-vit --step-size 1.0 --cc-kernel-widths 21x13x11x9 \
    --fixed-minclip -450 --fixed-maxclip 450 --moving-minclip 0 --moving-maxclip 20000 \
    --output-dir out --exp-name mrct
```

Run it over all 8 AbdomenMRCT pairs by swapping the single-pair inputs for a CSV
(`--registration-pairs-csv pairs.csv`, columns
`fixed,moving,fixed_mask,moving_mask,fixed_seg,moving_seg`, one row per pair) —
this reaches **mean macro-Dice ≈ 0.879 with ~0 folds**, matching the reference
`anatomix-dev-vit` result. The dataset-specific `21x13x11x9` CC schedule is passed
explicitly; omit it and each stage falls back to FireANTs' default kernel (which
does not reach zero folds).

## The CLI, in brief

`python anatomix-register.py --help` prints the complete interface (including the
`--custom-*` backbone flags). The parts you'll usually touch:

**Input modes** — exactly one of:
- **Single pair:** `--fixed` / `--moving`.
- **Directory batch:** `--fixed-dir` / `--moving-dir` (equal counts, paired by
  filename sort; optional `--fixed-mask-dir`, `--moving-mask-dir`,
  `--fixed-seg-dir`, `--moving-seg-dir`). Use CSV mode if filenames don't
  correspond by sort order.
- **CSV batch:** `--registration-pairs-csv` — header `fixed,moving` plus optional
  `fixed_mask,moving_mask,fixed_seg,moving_seg`; empty cells mean "absent",
  relative paths resolve against the CSV.

**Masks and labels** — provide **both** masks or neither. Masks gate the network
features and, for a `masked_*` loss, become FireANTs' loss mask. A **moving**
segmentation is required to warp labels; add a **fixed** segmentation to also get
Dice (background label 0 excluded).

**The transform chain** — `--transform` is a comma-separated list of stages from
`{rigid,affine,deformable}`, ordered `rigid ≤ affine ≤ deformable` (repeated
`deformable` allowed). Every per-stage list has one entry per stage; pyramid
schedules are `AxBx...`. Defaults reproduce the SOTA single-deformable setup:

| flag | meaning | default |
|------|---------|---------|
| `--transform` | the stage chain | `deformable` |
| `--initialization` | closed-form `center-of-mass` / `moments` before the chain | `none` |
| `--loss` | `cc,mi,mse,masked_cc,masked_mi,masked_mse` per stage | `masked_cc` if masks else `cc` |
| `--step-size` | Adam LR per stage | `1.0` deformable / `0.1` linear |
| `--shrink-factors` | resolution schedule per stage | `8x4x2x1` |
| `--iterations` | iters per level (matches shrink) | `100` per level |
| `--cc-kernel-widths` | odd CC widths per level (`na` for non-CC stages) | FireANTs' default kernel |
| `--smooth-grad-sigma` / `--smooth-warp-sigma` | deformable regularization (`na` for linear) | `1.0` / `0.5` |

**Features** — `--backbone {anatomix, anatomix-dev, anatomix-dev-vit (default),
custom}`. Features are (by default) extracted on an isotropic grid
(`--isotropic-features`) via MONAI sliding-window inference
(`--sliding-window-params window,sw_batch,overlap,mode,sigma`, default
`128,4,0.8,gaussian,0.25`; `anatomix-dev-vit` needs `window=128`), voxelwise
normalized (`--feature-normalization l2|standardized|none`), and combined with
MIND-SSC (`--use-mindssc both|feats-only|mindssc-only`, `--mindssc-params
radius,dilation`). `--backbone custom` takes `--custom-arch {unet,vit}` +
`--custom-weights` and exposes every constructor argument as a `--unet-*` /
`--vit-*` flag.

**Device** — `--device auto` (default) picks the visible CUDA GPU with the most
free memory (so it avoids a busy one); pin explicitly with
`CUDA_VISIBLE_DEVICES` and/or `--device {auto,cpu,cuda,cuda:N}`.

Some more examples:

```bash
# center-of-mass init, then affine + deformable
python anatomix-register.py --fixed CT.nii.gz --moving MR.nii.gz \
    --fixed-mask CT_mask.nii.gz --moving-mask MR_mask.nii.gz \
    --initialization center-of-mass --transform affine,deformable \
    --shrink-factors 4x2x1,8x4x2x1 --iterations 100x100x50,100x100x100x100 \
    --cc-kernel-widths 7x5x3,21x13x11x9

# batch over a CSV
python anatomix-register.py --registration-pairs-csv pairs.csv --output-dir out

# MIND-SSC only (no network backbone is loaded/downloaded)
python anatomix-register.py --fixed CT.nii.gz --moving MR.nii.gz --use-mindssc mindssc-only
```

## Outputs and transforms

Per pair, in `--output-dir` (with an optional `--exp-name` prefix):

- `moved-<stem>.nii.gz` — moving image warped onto the fixed grid (trilinear).
- `moved-seg-<stem>.nii.gz` — moving label warped (nearest), if a moving seg was given.
- `warp-<stem>.<ext>` — the transform.
- `metrics.csv` — input columns plus `dice` (blank if N/A) and `num_folds`
  (written incrementally, one row per completed pair).

`--output-transformation-convention {ants,scipy,pytorch}` all represent the
**full cumulative transform** for every chain (including composed
rigid/affine→deformable and repeated-deformable): `ants` → an ITK vector
displacement field `.nii.gz` (linear-only → `.mat`); `scipy` → a Learn2Reg-format
`.npz`; `pytorch` → the normalized fixed→moving sampling grid `.pt`
(`[1,H,W,D,3]`). `--collapse-output-transforms 1` (default) writes one composed
transform; `0` writes one cumulative snapshot per stage.

**The `ants` outputs are directly compatible with the original ANTs library.**
Applying them with `antsApplyTransforms` reproduces our warped labels bit-for-bit
and our Dice exactly (verified across single-deformable, composed
rigid/affine→deformable, and linear `.mat` outputs):

```bash
antsApplyTransforms -d 3 -i MR.nii.gz  -r CT.nii.gz -t warp-MR.nii.gz -o moved-MR.nii.gz
antsApplyTransforms -d 3 -i MR_seg.nii.gz -r CT.nii.gz -t warp-MR.nii.gz -n NearestNeighbor -o moved-seg-MR.nii.gz
```

To apply a saved `pytorch` grid yourself, note it is in FireANTs' SimpleITK array
axis order `(z,y,x)` — the reverse of nibabel's `(x,y,z)` — so transpose a
nibabel-loaded volume first (otherwise the warp is silently axis-scrambled on
non-cubic volumes):

```python
import torch, torch.nn.functional as F, nibabel as nib
grid = torch.load("out/mrct-warp-MR.pt")              # [1, H, W, D, 3], fixed->moving
mov = nib.load("MR.nii.gz").get_fdata()               # (x, y, z)
mov = torch.tensor(mov).permute(2, 1, 0)[None, None]  # -> (z, y, x) = FireANTs order
moved = F.grid_sample(mov.float(), grid, mode="bilinear", align_corners=True)
```

## Reproducing the ICLR'25 ConvexAdam results

The ICLR'25 anatomix registration numbers came from a ConvexAdam backend, kept
unchanged under `registration_backend/convexadam/` and demonstrated in
[`tutorials/anatomix_registration_convexadam.ipynb`](tutorials/anatomix_registration_convexadam.ipynb).

This backend is no longer maintained and kept for legacy purposes. It will be removed in a future commit.

Import it directly
(`from anatomix.registration.registration_backend.convexadam import convex_adam`)
or use its own `run_convex_adam_with_network_feats.py`. It is not exposed by the
FireANTs-only `anatomix-register.py`.

## Credits and license

Registration is performed by **FireANTs**
([repository](https://github.com/rohitrango/FireANTs),
[documentation](https://fireants.readthedocs.io/en/latest/)); this project uses [my fork](https://github.com/neel-dey/FireANTs) fork. 
If you use this backend in a paper, please cite FireANTs as well:

```bibtex
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
