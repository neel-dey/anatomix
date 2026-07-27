"""FireANTs-free filesystem helpers (CSV pairs, keypoints, NIfTI naming).

These live apart from :mod:`warp_io` so the CLI can parse arguments, read the
pairs CSV, and validate inputs without importing the optional FireANTs backend.
"""
import csv
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
# Everything path-like; any other column is passthrough metadata.
PATH_COLUMNS = VOLUME_COLUMNS + KEYPOINT_COLUMNS
# Columns the metrics CSV appends after the input columns, and therefore
# reserved as input column names.
METRIC_COLUMNS = (
    "dice", "num_folds",
    "tre_median", "tre_mean", "tre_initial_median", "robustness",
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
    """Read a registration-pairs CSV with a header row.

    The header must contain ``fixed`` and ``moving``; the optional columns are
    those in :data:`VOLUME_COLUMNS` and :data:`KEYPOINT_COLUMNS`. Empty cells
    there mean "absent" and relative paths resolve against the CSV's parent
    directory. Any other column is preserved verbatim as opaque metadata (not
    resolved as a path or validated) and carried through to the metrics CSV.

    Returns
    -------
    columns : list of str
        The CSV header, in order.
    rows : list of dict
        One dict per row mapping each path column to an absolute path or
        ``None``, and each metadata column to its raw string value.
    """
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
    """Read a keypoint CSV.

    The header must contain ``x``, ``y`` and ``z`` columns (case-insensitive);
    any other column -- a landmark id, a label -- is carried through unchanged.

    Returns
    -------
    columns : list of str
        The CSV header, in order.
    rows : list of dict
        One dict per row, values as raw strings.
    coordinates : list of tuple of float
        Each row's ``(x, y, z)``, in the file's own coordinate convention.
    """
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
