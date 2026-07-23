"""FireANTs-free filesystem helpers (CSV pairs, metrics, NIfTI naming).

These live apart from :mod:`warp_io` so the CLI can parse arguments, read the
pairs CSV, and validate inputs without importing the optional FireANTs backend.
"""
import csv
import os

NIFTI_EXTS = (".nii", ".nii.gz")


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

    The header must contain ``fixed`` and ``moving``; optional columns are
    ``fixed_mask``, ``moving_mask``, ``fixed_seg``, ``moving_seg``. Empty cells
    mean "absent". Relative paths are resolved against the CSV's parent
    directory.

    Returns
    -------
    columns : list of str
        The CSV header, in order.
    rows : list of dict
        One dict per row mapping each column to an absolute path or ``None``.
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
                if not value:
                    resolved[key] = None
                elif os.path.isabs(value):
                    resolved[key] = value
                else:
                    resolved[key] = os.path.join(base, value)
            rows.append(resolved)
    return columns, rows


def write_metrics_csv(out_path, input_columns, rows):
    """Write per-pair metrics, preserving input columns and appending metrics.

    Parameters
    ----------
    out_path : str
        Output CSV path.
    input_columns : list of str
        Input columns to preserve, in order.
    rows : list of dict
        One dict per pair, containing the input columns plus ``dice`` and
        ``num_folds``.
    """
    fieldnames = list(input_columns) + ["dice", "num_folds"]
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
