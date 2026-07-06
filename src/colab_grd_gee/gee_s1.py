"""Google Earth Engine Sentinel-1 GRD utilities for the public Colab notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

S1_GRD_COLLECTION = "COPERNICUS/S1_GRD"
DEFAULT_MAX_DIRECT_DOWNLOAD_BYTES = 48 * 1024 * 1024


def require_ee():
    import ee

    return ee


def require_geemap():
    import geemap

    return geemap


def authenticate_and_initialize(project_id: str):
    """Authenticate and initialize Earth Engine with helpful public-notebook errors."""

    if not project_id or project_id == "your-gee-project-id":
        raise ValueError(
            "Set GEE_PROJECT_ID to a Google Cloud project enabled for Earth Engine before running this cell."
        )
    ee = require_ee()
    try:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine authentication or initialization failed. Confirm that your Google account has "
            "Earth Engine access and that GEE_PROJECT_ID is a valid Cloud project ID."
        ) from exc
    return ee


def rectangle_from_bounds(bounds: Iterable[float]):
    """Create an EPSG:4326 rectangle from west, south, east, north bounds."""

    ee = require_ee()
    west, south, east, north = [float(v) for v in bounds]
    return ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)


def polygon_from_lonlat(coordinates: Iterable[Iterable[float]]):
    """Create an Earth Engine polygon from lon/lat coordinate pairs."""

    ee = require_ee()
    coords = [[float(lon), float(lat)] for lon, lat in coordinates]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return ee.Geometry.Polygon([coords], proj="EPSG:4326", geodesic=False)


def geometry_from_geojson(geojson: dict):
    """Create an Earth Engine geometry from a GeoJSON geometry or feature."""

    ee = require_ee()
    if geojson.get("type") == "Feature":
        geojson = geojson["geometry"]
    return ee.Geometry(geojson)


def get_drawn_roi(map_obj, fallback_bounds: Iterable[float] | None = None):
    """Return a geemap-drawn ROI, falling back to an EPSG:4326 rectangle."""

    ee = require_ee()
    user_roi = getattr(map_obj, "user_roi", None)
    if user_roi is not None:
        return user_roi
    draw_last_feature = getattr(map_obj, "draw_last_feature", None)
    if draw_last_feature is not None:
        return ee.Feature(draw_last_feature).geometry()
    draw_features = getattr(map_obj, "draw_features", None)
    if draw_features:
        return ee.Feature(draw_features[-1]).geometry()
    if fallback_bounds is not None:
        return rectangle_from_bounds(fallback_bounds)
    raise ValueError("Draw an ROI on the map or provide manual bounds/GeoJSON.")


def s1_grd_collection(
    roi,
    start_date: str,
    end_date: str,
    polarization: str = "VV",
    orbit_pass: str = "ASCENDING",
    relative_orbit: int | str | None = None,
    instrument_mode: str = "IW",
):
    """Build the Sentinel-1 GRD collection used by the GRD/GEE workflow."""

    ee = require_ee()
    collection = (
        ee.ImageCollection(S1_GRD_COLLECTION)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", instrument_mode))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", polarization))
    )
    if relative_orbit not in (None, ""):
        collection = collection.filter(ee.Filter.eq("relativeOrbitNumber_start", int(relative_orbit)))
    return collection


def list_acquisitions(
    roi,
    start_date: str,
    end_date: str,
    polarization: str = "VV",
    orbit_pass: str = "ASCENDING",
    relative_orbit: int | str | None = None,
    instrument_mode: str = "IW",
    limit: int = 100,
):
    """Return metadata records for matching Sentinel-1 GRD acquisitions."""

    from datetime import datetime

    ee = require_ee()
    collection = s1_grd_collection(
        roi=roi,
        start_date=start_date,
        end_date=end_date,
        polarization=polarization,
        orbit_pass=orbit_pass,
        relative_orbit=relative_orbit,
        instrument_mode=instrument_mode,
    ).sort("system:time_start")
    count = int(collection.size().getInfo())
    roi_area = ee.Number(roi.area(1)).max(1)

    def add_coverage(image):
        intersection = image.geometry().intersection(roi, ee.ErrorMargin(1))
        area_m2 = intersection.area(1)
        return image.set(
            {
                "roi_intersection_area_km2": area_m2.divide(1e6),
                "roi_coverage_percent": area_m2.divide(roi_area).multiply(100),
            }
        )

    collection = collection.map(add_coverage)
    features = collection.limit(limit).getInfo().get("features", [])
    records = []
    for index, feature in enumerate(features):
        props = feature.get("properties", {})
        timestamp_ms = props.get("system:time_start")
        time_utc = None
        if timestamp_ms is not None:
            time_utc = datetime.utcfromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        records.append(
            {
                "index": index,
                "id": feature.get("id"),
                "time_utc": time_utc,
                "orbit_pass": props.get("orbitProperties_pass"),
                "relative_orbit": props.get("relativeOrbitNumber_start"),
                "platform": props.get("platform_number"),
                "polarizations": props.get("transmitterReceiverPolarisation"),
                "instrument_mode": props.get("instrumentMode"),
                "roi_coverage_percent": props.get("roi_coverage_percent"),
                "roi_intersection_area_km2": props.get("roi_intersection_area_km2"),
            }
        )
    return records, count, collection


def select_acquisition(records: list[dict], selected_index: int | str, label: str = "image") -> dict:
    """Return exactly one acquisition record by displayed index."""

    if not records:
        raise ValueError(f"No Sentinel-1 acquisitions are available for {label}.")
    try:
        index = int(selected_index)
    except Exception as exc:
        raise ValueError(f"Select {label} using one of the displayed integer indexes.") from exc
    if index < 0 or index >= len(records):
        raise IndexError(f"Selected {label} index {index} is outside the available range 0..{len(records) - 1}.")
    record = records[index]
    if not record.get("id"):
        raise ValueError(f"Selected {label} has no Earth Engine image ID.")
    return record


def selected_image_summary(record: dict, polarization: str) -> dict:
    """Small public-safe metadata summary for one selected acquisition."""

    return {
        "image_id": record.get("id"),
        "acquisition_datetime_utc": record.get("time_utc"),
        "orbit_pass": record.get("orbit_pass"),
        "relative_orbit": record.get("relative_orbit"),
        "platform": record.get("platform"),
        "polarization": polarization,
        "available_polarizations": record.get("polarizations"),
        "instrument_mode": record.get("instrument_mode"),
        "roi_coverage_percent": record.get("roi_coverage_percent"),
        "roi_intersection_area_km2": record.get("roi_intersection_area_km2"),
    }


def to_linear(image, polarization: str):
    """Convert a Sentinel-1 GRD dB band to linear sigma0 and overwrite the selected band."""

    ee = require_ee()
    linear = ee.Image(10.0).pow(image.select(polarization).divide(10.0)).rename(polarization)
    return image.addBands(linear, None, True)


def common_valid_area(
    roi,
    image1_id: str,
    image2_id: str,
    polarization: str,
    scale: int = 10,
    crs: str = "EPSG:4326",
    min_valid_pixels: int = 4,
):
    """Build and validate the shared valid region for two selected S1 GRD images."""

    ee = require_ee()
    image_t1_db = ee.Image(image1_id).select(polarization)
    image_t2_db = ee.Image(image2_id).select(polarization)
    common_region = (
        roi.intersection(ee.Image(image1_id).geometry(), ee.ErrorMargin(1))
        .intersection(ee.Image(image2_id).geometry(), ee.ErrorMargin(1))
    )
    common_mask = image_t1_db.mask().multiply(image_t2_db.mask()).gt(0).rename("common_valid_mask")
    common_area_m2 = float(common_region.area(1).getInfo() or 0.0)
    if common_area_m2 <= 0:
        raise ValueError(
            "The selected images do not share a footprint over the ROI. Choose different images, "
            "enlarge or change the ROI, leave RELATIVE_ORBIT unspecified, or expand the date windows."
        )
    valid_area_info = (
        ee.Image.pixelArea()
        .updateMask(common_mask)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=common_region,
            scale=scale,
            crs=crs,
            maxPixels=1e13,
            bestEffort=True,
            tileScale=4,
        )
        .getInfo()
    )
    valid_area_m2 = float((valid_area_info or {}).get("area") or 0.0)
    min_area_m2 = max(float(min_valid_pixels) * float(scale) * float(scale), 1.0)
    if valid_area_m2 < min_area_m2:
        raise ValueError(
            "The selected images do not share enough valid coverage over the ROI. "
            "Choose different images, enlarge or change the ROI, leave RELATIVE_ORBIT unspecified, "
            "or expand the date windows."
        )
    return {
        "image_t1_db": image_t1_db,
        "image_t2_db": image_t2_db,
        "common_region": common_region,
        "common_valid_mask": common_mask,
        "common_region_area_m2": common_area_m2,
        "valid_overlap_area_m2": valid_area_m2,
        "estimated_valid_overlap_pixels": valid_area_m2 / (float(scale) * float(scale)),
        "common_region_geojson": common_region.getInfo(),
    }


def estimate_direct_download_bytes(region, scale: int = 10, bytes_per_pixel: int = 4) -> dict:
    """Estimate direct GeoTIFF download size for one single-band image."""

    geometry_area_m2 = float(region.area(1).getInfo() or 0.0)
    grid_area_m2 = float(region.bounds(1).area(1).getInfo() or geometry_area_m2)
    estimated_pixels = grid_area_m2 / (float(scale) * float(scale))
    estimated_bytes = estimated_pixels * int(bytes_per_pixel)
    return {
        "geometry_area_m2": geometry_area_m2,
        "grid_bounds_area_m2": grid_area_m2,
        "estimated_pixels": estimated_pixels,
        "estimated_bytes": estimated_bytes,
        "estimated_mebibytes": estimated_bytes / (1024 * 1024),
    }


def assert_direct_download_size(
    region,
    scale: int = 10,
    max_bytes: int = DEFAULT_MAX_DIRECT_DOWNLOAD_BYTES,
    bytes_per_pixel: int = 4,
) -> dict:
    """Raise a friendly error before Earth Engine direct-download URL limits are exceeded."""

    estimate = estimate_direct_download_bytes(region, scale=scale, bytes_per_pixel=bytes_per_pixel)
    if estimate["estimated_bytes"] > int(max_bytes):
        max_mib = int(max_bytes) / (1024 * 1024)
        raise ValueError(
            "The selected common region is too large for Earth Engine direct download. "
            f"Estimated one-band GeoTIFF size is {estimate['estimated_mebibytes']:.1f} MiB, "
            f"but the direct-download limit is about {max_mib:.1f} MiB. "
            "Use a smaller ROI, increase SCALE_METERS, or split the area into smaller runs."
        )
    return estimate


def selected_linear_sigma0_image(image_db, polarization: str, common_region, common_valid_mask):
    """Convert one selected S1 GRD image from dB to linear sigma0 and apply the shared mask."""

    ee = require_ee()
    linear = ee.Image(10.0).pow(image_db.select(polarization).divide(10.0)).rename(
        f"sigma0_{polarization}_linear"
    )
    return linear.updateMask(common_valid_mask).clip(common_region)


def download_image(
    image,
    filename: str | Path,
    roi,
    scale: int = 10,
    crs: str = "EPSG:4326",
    force_common_grid: bool = False,
) -> Path:
    """Download an Earth Engine image as a single-band GeoTIFF."""

    geemap = require_geemap()
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if force_common_grid:
        image = image.reproject(crs=crs, scale=scale)
    geemap.ee_export_image(
        image,
        filename=str(filename),
        scale=scale,
        region=roi,
        crs=crs,
        file_per_band=False,
    )
    return filename


def download_selected_sigma0_pair(
    roi,
    output_dir: str | Path,
    image1_id: str,
    image2_id: str,
    polarization: str = "VV",
    scale: int = 10,
    crs: str = "EPSG:4326",
    max_direct_download_bytes: int = DEFAULT_MAX_DIRECT_DOWNLOAD_BYTES,
):
    """Download two selected single-scene linear sigma0 GeoTIFFs over their common valid area."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = common_valid_area(
        roi=roi,
        image1_id=image1_id,
        image2_id=image2_id,
        polarization=polarization,
        scale=scale,
        crs=crs,
    )
    t1_image = selected_linear_sigma0_image(
        common["image_t1_db"], polarization, common["common_region"], common["common_valid_mask"]
    )
    t2_image = selected_linear_sigma0_image(
        common["image_t2_db"], polarization, common["common_region"], common["common_valid_mask"]
    )
    download_estimate = assert_direct_download_size(
        common["common_region"],
        scale=scale,
        max_bytes=max_direct_download_bytes,
    )
    t1_path = output_dir / f"sigma0_{polarization}_t1_linear.tif"
    t2_path = output_dir / f"sigma0_{polarization}_t2_linear.tif"
    download_image(
        t1_image,
        t1_path,
        roi=common["common_region"],
        scale=scale,
        crs=crs,
        force_common_grid=True,
    )
    download_image(
        t2_image,
        t2_path,
        roi=common["common_region"],
        scale=scale,
        crs=crs,
        force_common_grid=True,
    )
    common_summary = {
        key: value
        for key, value in common.items()
        if key not in {"image_t1_db", "image_t2_db", "common_region", "common_valid_mask"}
    }
    common_summary["direct_download_estimate"] = download_estimate
    return t1_path, t2_path, common_summary
