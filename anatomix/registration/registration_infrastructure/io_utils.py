"""FireANTs-free filesystem helpers (CSV pairs, metrics, NIfTI naming).

These live apart from :mod:`warp_io` so the CLI can parse arguments, read the
pairs CSV, and validate inputs without importing the optional FireANTs backend.
"""
import csv
import os

NIFTI_EXTS = (".nii", ".nii.gz")
# CSV/pipeline columns that hold volume paths (everything else is passthrough
# metadata, not resolved as a path or validated as a NIfTI).
VOLUME_COLUMNS = (
    "fixed", "moving", "fixed_mask", "moving_mask", "fixed_seg", "moving_seg",
)


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

    The header must contain ``fixed`` and ``moving``; optional volume columns are
    ``fixed_mask``, ``moving_mask``, ``fixed_seg``, ``moving_seg`` (see
    :data:`VOLUME_COLUMNS`). Empty volume cells mean "absent". Relative paths in
    volume columns are resolved against the CSV's parent directory. Any other
    column is preserved verbatim as opaque metadata (not resolved as a path or
    validated as a volume) and carried through to the metrics CSV.

    Returns
    -------
    columns : list of str
        The CSV header, in order.
    rows : list of dict
        One dict per row mapping each volume column to an absolute path or
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
                if key not in VOLUME_COLUMNS:
                    resolved[key] = value  # opaque metadata, passthrough
                elif not value:
                    resolved[key] = None
                elif os.path.isabs(value):
                    resolved[key] = value
                else:
                    resolved[key] = os.path.join(base, value)
            rows.append(resolved)
    return columns, rows
