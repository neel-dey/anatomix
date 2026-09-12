"""Read one scalar 3D OME-Zarr level with explicit, millimetre ITK geometry.

Input selectors are URL fragments, also supported on local paths and in CSVs:
``flower.zarr#level=2&channel=0&time=0``. No pixels are read by
``open_ome_zarr``. Only the selected level/channel/time is materialized by
``read_image``. NGFF 0.1--0.5 and Zarr 2/3 containers are supported; legacy
stores without calibration require selected-level spacing in millimetres.
"""
from dataclasses import dataclass
import os
from urllib.parse import parse_qsl, urlsplit, urlunsplit
import warnings

import numpy as np

_UNITS_MM = {
    "angstrom": 1e-7, "meter": 1e3, "millimeter": 1.,
    "micrometer": 1e-3, "nanometer": 1e-6, "picometer": 1e-9,
    "inch": 25.4, "foot": 304.8, "yard": 914.4, "mile": 1609344.,
    "parsec": 3.085677581491367e19,
}
for _prefix, _power in {
    "yocto": -24, "zepto": -21, "atto": -18, "femto": -15,
    "centi": -2, "deci": -1, "hecto": 2, "kilo": 3, "mega": 6,
    "giga": 9, "tera": 12, "peta": 15, "exa": 18, "zetta": 21, "yotta": 24,
}.items():
    _UNITS_MM[_prefix + "meter"] = 10. ** (_power + 3)

# Accept unambiguous unit spellings used by microscopy exporters in practice.
for _alias, _canonical in {"um": "micrometer", "µm": "micrometer", "μm": "micrometer",
                           "mm": "millimeter", "nm": "nanometer", "cm": "centimeter",
                           "m": "meter"}.items():
    _UNITS_MM[_alias] = _UNITS_MM[_canonical]

_OPTIONS = {"level", "channel", "time", "group", "multiscale", "unit",
            "spacing", "origin", "direction", "anon"}


def is_ome_zarr(path):
    location = os.fspath(path).split("#", 1)[0].rstrip("/")
    return (
        "://" in location or location.lower().endswith(".zarr")
        or any(os.path.isfile(os.path.join(location, name))
               for name in (".zgroup", "zarr.json"))
    )


@dataclass
class OMEZarrVolume:
    """Selected lazy array and geometry; XYZ physical coordinates are in mm.

    NGFF spatial XYZ is embedded directly in ITK XYZ, with identity direction
    unless explicitly supplied. This does not imply anatomical LPS orientation.
    ``direction`` maps the NGFF coordinate basis into the ITK world basis.
    ``origin`` is already in that ITK world basis.
    """
    array: object
    selection: tuple
    transpose: tuple
    shape_xyz: tuple
    spacing: np.ndarray
    origin: np.ndarray
    direction: np.ndarray
    level_path: str
    version: str

    @property
    def affine_ras(self):
        affine = np.eye(4)
        affine[:3, :3] = self.direction * self.spacing
        affine[:3, 3] = self.origin
        return np.diag([-1., -1., 1., 1.]) @ affine

    def read_image(self):
        """Read selected chunks, reorder to ZYX, and construct SimpleITK image."""
        import SimpleITK as sitk
        data = np.asarray(self.array[self.selection]).transpose(self.transpose)
        # Legacy IDR arrays are big-endian; SimpleITK requires native byte order.
        data = np.ascontiguousarray(data, dtype=data.dtype.newbyteorder("="))
        if data.dtype == np.bool_:
            data = data.astype(np.uint8)
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        image = sitk.GetImageFromArray(data, isVector=False)
        image.SetSpacing(tuple(self.spacing))
        image.SetOrigin(tuple(self.origin))
        image.SetDirection(tuple(self.direction.ravel()))
        return image


def open_ome_zarr(spec, *, storage_options=None):
    """Inspect one scalar 3D level without reading pixels.

    Selectors follow the path: ``image.zarr#level=2&channel=0&time=0``.
    Geometry overrides use selected-level XYZ spacing and world origin in mm;
    ``direction`` contains nine row-major orthonormal direction cosines.
    ``unit`` supplies missing spatial units. ``storage_options`` is passed to
    fsspec for remote access. Call ``read_image()`` to materialize the selection.
    """
    location, options = _parse_input(spec)
    root = _open_group(location, options, storage_options)
    metadata, multiscale, version = _select_multiscale(dict(root.attrs), options)
    array, dataset, level = _select_level(root, multiscale, options)
    axes = _read_axes(array, multiscale, version)
    selection, shape_xyz, transpose = _select_volume(array.shape, axes, metadata, options)
    spacing, origin, direction = _physical_geometry(axes, dataset, multiscale, options)

    for key in ("spacing", "origin", "direction", "unit"):
        if key in options:
            warnings.warn(
                f"OME-Zarr {dataset['path']}: explicit {key}={options[key]}; "
                f"calibration applies to selected level {level}.", stacklevel=2)
    return OMEZarrVolume(
        array=array, selection=selection, transpose=transpose, shape_xyz=shape_xyz,
        spacing=spacing, origin=origin, direction=direction,
        level_path=dataset["path"], version=version,
    )


def _parse_input(spec):
    """Split URL selectors and resolve child paths relative to the store root."""
    location, _, fragment = os.fspath(spec).partition("#")
    try:
        entries = parse_qsl(fragment.replace("+", "%2B"), keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise ValueError(f"Invalid OME-Zarr selectors: {exc}") from None
    options = dict(entries)
    if len(options) != len(entries) or set(options) - _OPTIONS:
        raise ValueError(f"Duplicate or unknown OME-Zarr selectors; allowed: {sorted(_OPTIONS)}.")
    # Normalize a direct child path to store-root + group selection so known
    # ancestors receive the same coordinate-safety checks as #group=....
    parsed = urlsplit(location)
    parts = parsed.path.split("/")
    for index, part in enumerate(parts):
        if part.lower().endswith(".zarr") and any(parts[index + 1:]):
            child = "/".join(parts[index + 1:]).rstrip("/")
            options["group"] = child + ("/" + options["group"] if "group" in options else "")
            location = urlunsplit((parsed.scheme, parsed.netloc, "/".join(parts[:index + 1]), parsed.query, ""))
            break
    return location, options


def _open_group(location, options, storage_options):
    """Open a Zarr 2/3 group and check parent coordinates while descending."""
    try:
        import zarr
    except ImportError as exc:
        raise ValueError("OME-Zarr input needs optional dependencies: pip install 'anatomix[zarr]'.") from exc
    access = dict(storage_options or {})
    if "anon" in options:
        if options["anon"] not in ("true", "false") or urlsplit(location).scheme != "s3":
            raise ValueError("anon must be true/false and is only valid for s3:// inputs.")
        anon = options["anon"] == "true"
        if "anon" in access and access["anon"] != anon:
            raise ValueError("Conflicting anonymous S3 options.")
        access["anon"] = anon
    store = location.rstrip("/")
    if "://" not in location and access:
        raise ValueError("storage_options are only valid for remote stores.")
    try:
        kwargs = {"mode": "r"}
        if int(zarr.__version__.split(".")[0]) >= 3:
            kwargs["use_consolidated"] = False
            if access:
                kwargs["storage_options"] = access
        elif "://" in location:
            import fsspec
            store = fsspec.get_mapper(store, **access)
        root = zarr.open_group(store, **kwargs)
        if "group" in options:
            # Ancestor coordinate graphs can affect a child image. Never skip
            # them silently while descending through a plate/series container.
            for part in _relative_path(options["group"]).split("/"):
                parent = dict(root.attrs)
                parent = parent.get("ome", parent)
                if not isinstance(parent, dict):
                    raise ValueError("OME-Zarr ancestor metadata must be an object.")
                _reject_extensions(parent)
                if "coordinateTransformations" in parent:
                    raise ValueError("Unsupported ancestor coordinateTransformations; resolve the parent coordinate frame before selecting its image.")
                root = root[part]
        return root
    except Exception as exc:
        raise ValueError(f"Cannot open OME-Zarr group {location}: {exc}") from exc


def _select_multiscale(attributes, options):
    """Select one image and reject coordinate metadata we cannot interpret."""
    metadata = attributes.get("ome", attributes)
    if not isinstance(metadata, dict):
        raise ValueError("OME-Zarr ome metadata must be an object.")
    _reject_extensions(metadata)
    if "coordinateTransformations" in metadata:
        raise ValueError("Unsupported image-root coordinateTransformations; resolve the coordinate frame before registration.")
    multiscales = metadata.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError("Expected an OME-Zarr image group with multiscales metadata; select group=... for nested images.")
    if any(not isinstance(m, dict) for m in multiscales):
        raise ValueError("Each multiscale image must be an object.")
    matches = multiscales
    if "multiscale" in options:
        matches = [m for m in multiscales if m.get("name") == options["multiscale"]]
    if len(matches) != 1:
        raise ValueError("Select exactly one multiscale image by name with multiscale=... .")
    multiscale = matches[0]
    _reject_extensions(multiscale)
    return metadata, multiscale, _metadata_version(attributes, metadata, multiscale)


def _metadata_version(attributes, metadata, multiscale):
    """Read the declared NGFF version, allowing omission for recognizable 0.4 metadata."""
    version = metadata.get("version") if "ome" in attributes else multiscale.get("version")
    if "ome" in attributes and multiscale.get("version", version) != version:
        raise ValueError("Conflicting OME-Zarr namespace and multiscale versions.")
    if version is None and "ome" not in attributes:
        # Version is SHOULD, not MUST, in NGFF 0.4. Only accept its recognizable
        # schema; do not guess the axes or spacing of older unversioned stores.
        candidate_axes = multiscale.get("axes")
        candidate_datasets = multiscale.get("datasets")
        structured_axes = (isinstance(candidate_axes, list) and candidate_axes
                           and all(isinstance(axis, dict) for axis in candidate_axes))
        calibrated_levels = (isinstance(candidate_datasets, list) and candidate_datasets
                             and all(isinstance(level, dict) and "coordinateTransformations" in level
                                     for level in candidate_datasets))
        if structured_axes and calibrated_levels:
            version = "0.4"
            warnings.warn("OME-Zarr version omitted; interpreting structured axes and transforms as NGFF 0.4.", UserWarning, stacklevel=3)
    if version not in ("0.1", "0.2", "0.3", "0.4", "0.5"):
        raise ValueError(f"Unsupported or missing OME-Zarr metadata version {version!r}.")
    return version


def _select_level(root, multiscale, options):
    """Resolve the selected pyramid entry without loading pixel data."""
    datasets = multiscale.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("OME-Zarr multiscale has no datasets.")
    level = _index(options.get("level", "0"), len(datasets), "level")
    dataset = datasets[level]
    if not isinstance(dataset, dict):
        raise ValueError("Each dataset must be an object.")
    _reject_extensions(dataset)
    path = _relative_path(dataset.get("path"))
    try:
        array = root[path]
        _ = array.ndim, array.shape  # Reject group paths; a level must be an array.
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"OME-Zarr dataset {path!r} is not an array.") from exc
    if np.dtype(array.dtype).kind not in "buif":
        raise ValueError(f"Unsupported OME-Zarr scalar dtype {array.dtype}.")
    return array, dataset, level


def _read_axes(array, multiscale, version):
    """Normalize spatial, time and channel axes, preserving array order."""
    ndim = array.ndim
    axes = multiscale.get("axes")
    if axes is None and version in ("0.1", "0.2") and ndim == 5:
        axes = list("tczyx")  # fixed axis order mandated by legacy NGFF
    if not isinstance(axes, list) or len(axes) != ndim:
        raise ValueError("OME-Zarr axes must describe every array dimension.")
    axes = [{"name": a} if isinstance(a, str) else a for a in axes]
    if any(not isinstance(a, dict) for a in axes):
        raise ValueError("Invalid OME-Zarr axes metadata.")
    declared_names = [a.get("name") for a in axes]
    if any(not isinstance(n, str) for n in declared_names) or len(set(declared_names)) != ndim:
        raise ValueError("Axis names must be unique strings.")
    array_metadata = getattr(array, "metadata", None)
    if version == "0.5" and tuple(getattr(array_metadata, "dimension_names", ()) or ()) != tuple(declared_names):
        raise ValueError("Zarr 3 dimension_names must match OME-Zarr axes.")
    if not set("xyz").issubset(declared_names):
        raise ValueError("OME-Zarr inputs need spatial axes named x, y and z.")
    normalized_axes = []
    for axis in axes:
        if "orientation" in axis or "anatomicalOrientation" in axis:
            raise ValueError("Axis orientation metadata is not supported; convert anatomical orientation to explicit ITK geometry first.")
        name, kind = axis["name"], axis.get("type")
        if name in ("x", "y", "z"):
            logical, expected = name, "space"
        elif kind == "time" or (kind is None and name == "t"):
            logical, expected = "t", "time"
        elif kind == "channel" or (kind is None and name == "c"):
            logical, expected = "c", "channel"
        else:
            raise ValueError(f"Unsupported axis {name!r} of type {kind!r}; select one scalar 3D image.")
        if kind is not None and kind != expected:
            raise ValueError(f"Unsupported axis type for {name!r}.")
        normalized_axes.append({**axis, "name": logical})
    axes = normalized_axes
    names = [a["name"] for a in axes]
    if len(set(names)) != ndim:
        raise ValueError("At most one time and one channel axis are supported.")
    return axes


def _select_volume(shape, axes, metadata, options):
    """Slice channel/time axes and describe the remaining XYZ volume."""
    names = [axis["name"] for axis in axes]
    shape_xyz = tuple(int(shape[names.index(a)]) for a in "xyz")
    if min(shape_xyz) < 2:
        raise ValueError(f"Expected a true 3D volume with at least 2 voxels per spatial axis, got XYZ {shape_xyz}; projections/singleton Z cannot drive 3D registration.")
    selection = []
    for name, size in zip(names, shape):
        if name in "xyz":
            selection.append(slice(None))
            continue
        selector = {"c": "channel", "t": "time"}[name]
        if size != 1 and selector not in options:
            raise ValueError(f"Axis {name} has {size} entries; select {selector}=... explicitly.")
        value = options.get(selector, "0")
        if selector == "channel":
            value = _channel_index(value, metadata)
        selection.append(_index(value, size, selector))
    for name, selector in (("c", "channel"), ("t", "time")):
        if name not in names and selector in options:
            raise ValueError(f"{selector} selector supplied but axis {name} is absent.")
    spatial_names = [name for name in names if name in "xyz"]
    transpose = tuple(spatial_names.index(name) for name in "zyx")
    return tuple(selection), shape_xyz, transpose


def _channel_index(value, metadata):
    """Accept either a numeric channel index or a unique OMERO channel label."""
    if value.lstrip("-").isdigit():
        return value
    omero = metadata.get("omero", {})
    channels = omero.get("channels", []) if isinstance(omero, dict) else []
    matches = [i for i, channel in enumerate(channels)
               if isinstance(channel, dict) and channel.get("label") == value]
    if len(matches) != 1:
        raise ValueError(f"Channel label {value!r} must match exactly one omero.channels entry.")
    return str(matches[0])


def _physical_geometry(axes, dataset, multiscale, options):
    """Compose dataset then group transforms, and express XYZ geometry in mm."""
    names = [axis["name"] for axis in axes]
    ndim = len(axes)
    dataset_transforms = dataset.get("coordinateTransformations", [])
    group_transforms = multiscale.get("coordinateTransformations", [])
    scale, offset = _transforms(dataset_transforms, ndim)
    group_scale, group_offset = _transforms(group_transforms, ndim)
    scale = scale * group_scale
    offset = offset * group_scale + group_offset
    if "spacing" not in options and not any(t.get("type") == "scale" for t in dataset_transforms):
        raise ValueError("OME-Zarr has no selected-level scale calibration; supply spacing=x,y,z in mm at this level. Pyramid shape ratios do not establish physical spacing.")

    units = _spatial_units(axes, names, options)
    xyz = [names.index(a) for a in "xyz"]
    if "spacing" in options:
        spacing = _vector(options["spacing"].split(","), 3, "spacing")
    elif None in units:
        raise ValueError("OME-Zarr spatial units are missing; supply unit=micrometer (if known) or spacing=x,y,z in mm. Unitless coordinates are not millimetres.")
    else:
        spacing = scale[xyz] * units
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing must be positive.")

    direction = np.eye(3)
    if "direction" in options:
        direction = _vector(options["direction"].split(","), 9, "direction").reshape(3, 3)
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6, rtol=0):
            raise ValueError("direction must be orthonormal; shear cannot be represented as ITK direction cosines.")

    if "origin" in options:
        origin = _vector(options["origin"].split(","), 3, "origin")
    else:
        if any(u is None and o != 0 for u, o in zip(units, offset[xyz])):
            raise ValueError("Translation units are unknown; supply unit=... or origin=x,y,z in mm.")
        origin = direction @ (offset[xyz] * [u if u is not None else 1 for u in units])
    if not np.isfinite(origin).all():
        raise ValueError("Composed origin must be finite.")
    return spacing, origin, direction


def _spatial_units(axes, names, options):
    """Return XYZ conversion factors to mm; absent units remain unknown."""
    units = []
    for name in "xyz":
        unit = axes[names.index(name)].get("unit")
        if unit is None:
            unit = options.get("unit")
        if unit is not None and (not isinstance(unit, str) or unit not in _UNITS_MM):
            raise ValueError(f"Unsupported spatial unit {unit!r}.")
        if "unit" in options and _UNITS_MM.get(options["unit"]) != _UNITS_MM.get(unit):
            raise ValueError("unit selector conflicts with declared spatial units.")
        units.append(None if unit is None else _UNITS_MM[unit])
    return units


def _vector(value, length, name):
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain {length} finite numbers.") from None
    if out.shape != (length,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain {length} finite numbers.")
    return out


def _index(value, size, name):
    try:
        index = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a nonnegative integer.") from None
    if str(index) != str(value) or not 0 <= index < size:
        raise ValueError(f"{name}={value} is outside [0, {size}).")
    return index


def _relative_path(path):
    if (not isinstance(path, str) or not path or path.startswith("/")
            or any(p in ("", ".", "..") for p in path.split("/"))):
        raise ValueError(f"Expected a relative OME-Zarr group/array path, got {path!r}.")
    return path


def _reject_extensions(metadata):
    for key in ("coordinateSystems", "coordinateSystem", "transformations", "transformation"):
        if key in metadata:
            raise ValueError(
                f"Unsupported OME-Zarr {key}; named coordinate systems/RFC "
                "transforms require conversion before registration."
            )


def _transforms(transforms, ndim):
    """Compose in listed order: physical = scale * index + offset."""
    scale, offset = np.ones(ndim), np.zeros(ndim)
    if not isinstance(transforms, list):
        raise ValueError("coordinateTransformations must be a list.")
    for transform in transforms:
        if not isinstance(transform, dict):
            raise ValueError("Each coordinate transformation must be an object.")
        kind = transform.get("type")
        if kind not in ("scale", "translation", "identity") or "path" in transform:
            raise ValueError(
                f"Unsupported OME-Zarr coordinate transformation {transform!r}; "
                "only inline scale/translation/identity can be mapped to ITK."
            )
        if kind == "scale":
            values = _vector(transform.get(kind), ndim, kind)
            if np.any(values <= 0):
                raise ValueError("OME-Zarr scale must be positive; use direction for reflections.")
            scale *= values
            offset *= values
        elif kind == "translation":
            offset += _vector(transform.get(kind), ndim, kind)
    return scale, offset
