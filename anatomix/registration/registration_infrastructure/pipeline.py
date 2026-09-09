"""Building blocks of ``anatomix-register.py``: loading a pair, extracting its
features, and writing the images, labels, keypoints and metrics of a result."""
import csv
import math
import os
import random
from collections import Counter
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import torch

from ._fireants import FFO_AVAILABLE, FakeBatchedImages, generate_image_mask_allones
from .features import (
    combine_feature_channels,
    minmax_normalize,
    prepare_feature_channels,
)
from .io_utils import (
    METRIC_COLUMNS,
    read_keypoints,
    strip_nifti_ext,
    write_keypoints,
)
from .metrics import dice_score, keypoint_metrics
from .warp_io import (
    array_spacing,
    as_batch,
    invert_grid,
    keypoints_from_physical,
    keypoints_to_physical,
    load_image,
    read_linear_transform,
    save_inverse_transform,
    warp_keypoints,
    warp_volume,
    write_on_geometry,
)


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch, with deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(spec):
    """Resolve cpu/cuda/cuda:N; auto chooses the visible GPU with most free memory."""
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        best, best_free = 0, -1
        for index in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(index)
            if free > best_free:
                best_free, best = free, index
        return torch.device(f"cuda:{best}")
    device = torch.device(spec)
    if device.type == "cuda":
        torch.cuda.set_device(device)  # FireANTs allocates on the default device
    return device


def warn_if_no_fused_ops():
    if not FFO_AVAILABLE:
        print(
            "[note] fireants_fused_ops is not available; FireANTs is using its "
            "pure-PyTorch fallback. "
            "Build the kernels with registration_backend/install_fireants.sh.",
            flush=True,
        )


def resolve_stage_losses(stages, has_masks):
    """Fill in default losses: masked_cc when the pair has a mask, otherwise cc."""
    resolved = []
    for stage in stages:
        spec = dict(stage)
        if spec["loss"] is None:
            spec["loss"] = "masked_cc" if has_masks else "cc"
        resolved.append(spec)
    if any(s["loss"].startswith("masked_") for s in resolved) and not has_masks:
        raise ValueError("A masked loss was requested but the pair has no masks.")
    return resolved


# Inputs
@dataclass
class PairInputs:
    """One loaded pair: FireANTs images (geometry), raw and normalized intensities, masks."""

    fixed: object            # fireants Image; its array is replaced by features later
    moving: object
    fixed_raw: torch.Tensor  # (1,1,Z,Y,X) original intensities, resampled once at the end
    moving_raw: torch.Tensor
    fixed_norm: torch.Tensor  # clipped, min-max normalized
    moving_norm: torch.Tensor
    fixed_spacing: tuple      # (z,y,x) mm
    fixed_mask: torch.Tensor = None   # binary (1,1,Z,Y,X) or None
    moving_mask: torch.Tensor = None

    @property
    def has_masks(self):
        return self.fixed_mask is not None


def _load_mask(path, image, device, label):
    mask = load_image(path, device) if path else generate_image_mask_allones(image)
    if not torch.isfinite(mask.array).all():
        raise ValueError(f"{label}: mask values must be finite.")
    binary = (mask.array > 0).to(mask.array.dtype)
    if not binary.any():
        raise ValueError(f"{label}: mask has no foreground.")
    return binary


def load_pair(pair, args, device, label="pair"):
    """Load the images of one pair, normalize their intensities and binarize the masks.

    ``pair`` maps the CSV column names (fixed, moving, fixed_mask, ...) to paths.
    A pair with one mask gets an all-ones mask for the other image."""
    if pair.get("fixed_seg") is not None and pair.get("moving_seg") is None:
        raise ValueError(f"{label}: a fixed segmentation requires a moving segmentation.")
    if pair.get("moving_keypoints") is not None and pair.get("fixed_keypoints") is None:
        raise ValueError(f"{label}: moving keypoints require fixed keypoints.")

    fixed = load_image(pair["fixed"], device)
    moving = load_image(pair["moving"], device)
    inputs = PairInputs(
        fixed=fixed, moving=moving,
        fixed_raw=fixed.array.clone(), moving_raw=moving.array.clone(),
        fixed_norm=minmax_normalize(
            fixed.array, args.fixed_minclip, args.fixed_maxclip, name=f"{label} [fixed]"),
        moving_norm=minmax_normalize(
            moving.array, args.moving_minclip, args.moving_maxclip, name=f"{label} [moving]"),
        fixed_spacing=array_spacing(fixed),
    )
    if pair.get("fixed_mask") or pair.get("moving_mask"):
        inputs.fixed_mask = _load_mask(
            pair.get("fixed_mask"), fixed, device, f"{label} [fixed mask]")
        inputs.moving_mask = _load_mask(
            pair.get("moving_mask"), moving, device, f"{label} [moving mask]")
    return inputs


def load_initial_transform(pair):
    """The pair's ``initial_transform`` file as a physical fixed-to-moving matrix, or None."""
    if pair.get("initial_transform") is None:
        return None

    def geometry(path):
        image = nib.load(path)
        return tuple(int(n) for n in image.shape[:3]), image.affine

    return read_linear_transform(
        pair["initial_transform"],
        fixed=geometry(pair["fixed"]), moving=geometry(pair["moving"]),
    )


def moment_images(inputs, fixed_batch, moving_batch):
    """Normalized intensities inside the masks, for center-of-mass and moments initialization."""
    def mass(normalized, mask):
        return normalized if mask is None else normalized * mask.to(normalized.dtype)

    return (
        FakeBatchedImages(mass(inputs.fixed_norm, inputs.fixed_mask), fixed_batch),
        FakeBatchedImages(mass(inputs.moving_norm, inputs.moving_mask), moving_batch),
    )


# Features
class FeatureExtractor:
    """Turn a normalized intensity volume into the channels that FireANTs registers.

    The channels are anatomix network features (masked), MIND-SSC descriptors
    and, for masked losses, the binary mask as the last channel."""

    def __init__(self, args, model):
        self.features = args.features
        self.model = model
        self.config = dict(
            features=args.features,
            isotropic=bool(args.isotropic_features),
            window=args.sw_window,
            sw_batch=args.sw_batch,
            overlap=args.sw_overlap,
            sw_mode=args.sw_mode,
            sigma=args.sw_sigma,
            feature_normalization=args.feature_normalization,
            mindssc_radius=args.mindssc_radius,
            mindssc_dilation=args.mindssc_dilation,
            verbose=args.verbose,
        )

    def __call__(self, image, normalized, spacing, mask, append_mask):
        """Replace ``image``'s array by its feature channels and return it as a FireANTs batch."""
        primary, mind = prepare_feature_channels(
            normalized, spacing, self.model, **self.config)
        image.array = combine_feature_channels(
            primary, mind, mask, self.features, append_mask=append_mask)
        image.channels = image.array.shape[1]
        return as_batch(image)


# Outputs
class OutputPaths:
    """Output file naming: ``<exp-name>-<kind>-<moving stem><ext>`` inside the output directory.

    Moving stems that repeat within a batch are prefixed by the pair index."""

    def __init__(self, output_dir, exp_name, pairs):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.prefix = f"{exp_name}-" if exp_name else ""
        raw_stems = [strip_nifti_ext(pair["moving"]) for pair in pairs]
        counts = Counter(raw_stems)
        width = max(1, len(str(len(pairs) - 1)))
        self.stems = [
            f"{index:0{width}d}-{stem}" if counts[stem] > 1 else stem
            for index, stem in enumerate(raw_stems)
        ]

    def path(self, index, kind, ext):
        return os.path.join(
            self.output_dir, f"{self.prefix}{kind}-{self.stems[index]}{ext}")

    @property
    def metrics_csv(self):
        return os.path.join(self.output_dir, f"{self.prefix}metrics.csv")


class MetricsWriter:
    """``metrics.csv``: the pair's input columns, then METRIC_COLUMNS; one row per pair,
    flushed as soon as the pair finishes."""

    def __init__(self, path, input_columns):
        self.input_columns = list(input_columns)
        self.fieldnames = self.input_columns + list(METRIC_COLUMNS)
        self.handle = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.handle.flush()
        self.rows = []

    def write(self, pair, metrics):
        row = {col: (pair.get(col) or "") for col in self.input_columns}
        row.update(metrics)
        self.writer.writerow({key: row.get(key, "") for key in self.fieldnames})
        self.handle.flush()
        self.rows.append(row)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.close()


def empty_metrics():
    return {col: "" for col in METRIC_COLUMNS}


def propagate_labels(pair, grid, fixed_batch, device, out_path):
    """Warp the moving segmentation onto the fixed grid; return Dice against the fixed one, or ""."""
    moving_seg = load_image(pair["moving_seg"], device, is_segmentation=False).array.float()
    moved_seg = warp_volume(moving_seg, grid, "nearest").round().to(torch.int32)
    write_on_geometry(moved_seg, fixed_batch, out_path)
    if pair.get("fixed_seg") is None:
        return ""
    fixed_seg = load_image(pair["fixed_seg"], device).array
    dice = dice_score(fixed_seg.cpu().numpy(), moved_seg.cpu().numpy())
    return "" if math.isnan(dice) else dice


def write_inverse(pair, result, inputs, fixed_batch, moving_batch, args, paths, index):
    """Write the moving-to-fixed transform and the fixed image (and labels) on the moving grid.

    Returns the largest inverse-consistency residual in mm inside the fixed FOV."""
    grid = result.warped_coordinates
    device = grid.device
    grid_inv, residual = invert_grid(grid, fixed_batch, moving_batch)
    save_inverse_transform(
        result, grid_inv, args.output_transformation_convention,
        paths.output_dir, paths.prefix, paths.stems[index],
    )
    write_on_geometry(
        warp_volume(inputs.fixed_raw, grid_inv, "bilinear"), moving_batch,
        paths.path(index, "inverse-moved", ".nii.gz"),
    )
    if pair.get("fixed_seg") is not None:
        fixed_seg = load_image(pair["fixed_seg"], device).array.float()
        write_on_geometry(
            warp_volume(fixed_seg, grid_inv, "nearest").round().to(torch.int32),
            moving_batch, paths.path(index, "inverse-moved-seg", ".nii.gz"),
        )
    finite = residual[torch.isfinite(residual)]
    if not finite.numel():
        return float("nan")
    if args.verbose:
        print(
            f"  -> inverse: residual median={float(finite.median()):.4f} "
            f"max={float(finite.max()):.4f} mm; "
            f"{float((finite > 1).float().mean()):.2%} of {finite.numel()} "
            "moving voxels inside the fixed FOV exceed 1 mm",
            flush=True,
        )
    return float(finite.max())


def map_keypoints(pair, grid, inputs, convention, out_path):
    """Map the fixed keypoints into moving space and write them.

    With moving keypoints, also returns the landmark error metrics."""
    device = grid.device
    columns, rows, coords = read_keypoints(pair["fixed_keypoints"])
    source = keypoints_to_physical(
        torch.tensor(coords, device=device, dtype=torch.float32),
        convention, inputs.fixed,
    )
    warped = warp_keypoints(source, grid, inputs.fixed, inputs.moving)
    write_keypoints(
        out_path, columns, rows,
        keypoints_from_physical(warped, convention, inputs.moving).cpu().numpy(),
    )
    if pair.get("moving_keypoints") is None:
        return {}
    _, _, target_coords = read_keypoints(pair["moving_keypoints"])
    target = keypoints_to_physical(
        torch.tensor(target_coords, device=device, dtype=torch.float32),
        convention, inputs.moving,
    )
    return keypoint_metrics(
        warped.cpu().numpy(), target.cpu().numpy(), source.cpu().numpy())
