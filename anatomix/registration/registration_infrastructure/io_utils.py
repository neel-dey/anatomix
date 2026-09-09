"""Read pair and landmark CSVs without importing FireANTs."""
import csv
import math
import os

NIFTI_EXTS = (".nii", ".nii.gz")
KEYPOINT_EXT = ".csv"
# CSV/pipeline columns that hold volume paths.
VOLUME_COLUMNS = (
    "fixed", "moving", "fixed_mask", "moving_mask", "fixed_seg", "moving_seg",
)
# Columns that hold keypoint CSVs; separate from VOLUME_COLUMNS because they
# carry no voxel grid to validate against.
KEYPOINT_COLUMNS = ("fixed_keypoints", "moving_keypoints")
# Optional ITK/ANTs linear transform (.mat/.txt) applied before the first stage.
TRANSFORM_COLUMNS = ("initial_transform",)
# Everything path-like; any other column is passthrough metadata.
PATH_COLUMNS = VOLUME_COLUMNS + KEYPOINT_COLUMNS + TRANSFORM_COLUMNS
# Columns the metrics CSV appends after the input columns, and therefore
# reserved as input column names.
METRIC_COLUMNS = (
    "dice", "num_folds",
    "tre_median", "tre_mean", "tre_initial_median", "robustness",
    "inverse_residual_mm",
)
# Keypoint CSV coordinate columns (matched case-insensitively).
_XYZ = ("x", "y", "z")
# 'lps'/'ras' world millimetres, or 'voxel' for the ITK index (i, j, k).
KEYPOINT_CONVENTIONS = ("lps", "ras", "voxel")


def strip_nifti_ext(path):
    """Return the file stem of ``path`` with a ``.nii``/``.nii.gz`` suffix removed."""
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def read_pairs_csv(path):
    """Read pair paths relative to the CSV, preserving extra columns as metadata.

    Requires fixed,moving headers. Empty optional path cells become None.
    Returns (columns, rows)."""
    base = os.path.dirname(os.path.abspath(path))
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if "fixed" not in columns or "moving" not in columns:
            raise ValueError(
                "registration-pairs CSV must have 'fixed' and 'moving' columns; "
                f"found {columns}."
            )
        rows = []
        for record in reader:
            resolved = {}
            for key in columns:
                value = (record.get(key) or "").strip()
                if key not in PATH_COLUMNS:
                    resolved[key] = value  # opaque metadata, passthrough
                elif not value:
                    resolved[key] = None
                elif os.path.isabs(value):
                    resolved[key] = value
                else:
                    resolved[key] = os.path.join(base, value)
            rows.append(resolved)
    return columns, rows


def read_keypoints(path):
    """Read (columns, rows, coordinates) from a landmark CSV.

    Requires case-insensitive x,y,z columns; extra columns are preserved."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        lookup = {name.strip().lower(): name for name in columns}
        missing = [axis for axis in _XYZ if axis not in lookup]
        if missing:
            raise ValueError(
                f"{path}: keypoint CSV needs {', '.join(_XYZ)} columns "
                f"(case-insensitive); missing {missing}, found {columns}."
            )
        rows = list(reader)
    coordinates = []
    for index, record in enumerate(rows):
        if record.get(None) is not None:
            # DictReader files surplus fields under the None key, which the
            # writer cannot emit; caught here so a ragged file fails preflight.
            raise ValueError(
                f"{path}: row {index} has more fields than the "
                f"{len(columns)}-column header."
            )
        try:
            coordinates.append(
                tuple(float(record[lookup[axis]]) for axis in _XYZ)
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"{path}: row {index} has a non-numeric coordinate."
            ) from None
    if not coordinates:
        raise ValueError(f"{path}: keypoint CSV has no rows.")
    if not all(math.isfinite(value) for point in coordinates for value in point):
        raise ValueError(f"{path}: keypoint coordinates must be finite.")
    return columns, rows, coordinates


def write_keypoints(path, columns, rows, coordinates):
    """Write a keypoint CSV, replacing the coordinates of ``rows``.

    ``columns`` and ``rows`` come from :func:`read_keypoints`, so every
    non-coordinate column of the input file is preserved verbatim.
    """
    lookup = {name.strip().lower(): name for name in columns}
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record, point in zip(rows, coordinates):
            out = dict(record)
            for axis, value in zip(_XYZ, point):
                out[lookup[axis]] = f"{float(value):.4f}"
            writer.writerow(out)
