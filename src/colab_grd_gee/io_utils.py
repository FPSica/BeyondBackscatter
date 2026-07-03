"""Input/output helpers for geospatial Back2Coh Colab products."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_single_band_geotiff(path: str | Path, fill_value: float = 0.0):
    """Read a single-band GeoTIFF as float32 array, rasterio profile, and nodata mask."""

    import rasterio

    with rasterio.open(path) as dataset:
        band = dataset.read(1, masked=True)
        profile = dataset.profile.copy()
        bounds = dataset.bounds
    mask = np.ma.getmaskarray(band)
    array = band.filled(fill_value).astype(np.float32)
    array[~np.isfinite(array)] = fill_value
    return array, profile, mask, bounds


def assert_aligned(profile_a: dict[str, Any], profile_b: dict[str, Any], shape_a, shape_b):
    """Validate that two rasters have matching shape, CRS, and affine transform."""

    if tuple(shape_a) != tuple(shape_b):
        raise ValueError(f"Raster shape mismatch: {shape_a} vs {shape_b}.")
    if profile_a.get("crs") != profile_b.get("crs"):
        raise ValueError(f"Raster CRS mismatch: {profile_a.get('crs')} vs {profile_b.get('crs')}.")
    if profile_a.get("transform") != profile_b.get("transform"):
        raise ValueError("Raster transform mismatch. Download both images with the same ROI, CRS, and scale.")
    return True


def save_metadata_json(path: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)

    def default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=default), encoding="utf-8")
    return path


def save_single_band_geotiff(
    path: str | Path,
    array: np.ndarray,
    reference_profile: dict[str, Any],
    dtype: str = "float32",
    nodata: float | int | None = -9999.0,
) -> Path:
    """Save one georeferenced raster band using a reference raster profile."""

    import rasterio

    path = Path(path)
    ensure_dir(path.parent)
    output = np.asarray(array)
    profile = reference_profile.copy()
    profile.update(driver="GTiff", count=1, dtype=dtype, compress="deflate")
    if nodata is not None:
        profile["nodata"] = nodata
        output = np.where(np.isfinite(output), output, nodata)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(output.astype(dtype), 1)
    return path


def save_rgb_geotiff(path: str | Path, rgb: np.ndarray, reference_profile: dict[str, Any]) -> Path:
    """Save a float RGB image in [0, 1] as a georeferenced uint8 GeoTIFF."""

    import rasterio

    path = Path(path)
    ensure_dir(path.parent)
    rgb_uint8 = np.clip(np.asarray(rgb, dtype=np.float32) * 255.0, 0, 255).astype("uint8")
    profile = reference_profile.copy()
    profile.update(driver="GTiff", count=3, dtype="uint8", compress="deflate", nodata=None)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(np.moveaxis(rgb_uint8, -1, 0))
    return path


def save_png(path: str | Path, image: np.ndarray, cmap: str | None = None, vmin=None, vmax=None) -> Path:
    """Save an array as a PNG without axes."""

    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    plt.imsave(path, image, cmap=cmap, vmin=vmin, vmax=vmax)
    return path


def zip_outputs(output_dir: str | Path, zip_path: str | Path) -> Path:
    """Zip all files below output_dir into zip_path."""

    output_dir = Path(output_dir)
    zip_path = Path(zip_path)
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file() and file_path.resolve() != zip_path.resolve():
                archive.write(file_path, file_path.relative_to(output_dir))
    return zip_path
