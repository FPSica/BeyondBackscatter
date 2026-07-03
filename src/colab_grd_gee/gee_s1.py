"""Google Earth Engine Sentinel-1 GRD utilities for the public Colab notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

S1_GRD_COLLECTION = "COPERNICUS/S1_GRD"


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
    features = collection.limit(limit).getInfo().get("features", [])
    records = []
    for feature in features:
        props = feature.get("properties", {})
        timestamp_ms = props.get("system:time_start")
        time_utc = None
        if timestamp_ms is not None:
            time_utc = datetime.utcfromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        records.append(
            {
                "id": feature.get("id"),
                "time_utc": time_utc,
                "orbit_pass": props.get("orbitProperties_pass"),
                "relative_orbit": props.get("relativeOrbitNumber_start"),
                "platform": props.get("platform_number"),
                "polarizations": props.get("transmitterReceiverPolarisation"),
                "instrument_mode": props.get("instrumentMode"),
            }
        )
    return records, count, collection


def to_linear(image, polarization: str):
    """Convert a Sentinel-1 GRD dB band to linear sigma0 and overwrite the selected band."""

    ee = require_ee()
    linear = ee.Image(10.0).pow(image.select(polarization).divide(10.0)).rename(polarization)
    return image.addBands(linear, None, True)


def median_linear_sigma0_image(
    roi,
    start_date: str,
    end_date: str,
    polarization: str = "VV",
    orbit_pass: str = "ASCENDING",
    relative_orbit: int | str | None = None,
    instrument_mode: str = "IW",
):
    """Filter Sentinel-1 GRD, median composite, convert dB to linear sigma0, and clip."""

    collection = s1_grd_collection(
        roi=roi,
        start_date=start_date,
        end_date=end_date,
        polarization=polarization,
        orbit_pass=orbit_pass,
        relative_orbit=relative_orbit,
        instrument_mode=instrument_mode,
    )
    count = int(collection.size().getInfo())
    if count == 0:
        raise ValueError(
            "No Sentinel-1 GRD images matched "
            f"{start_date} to {end_date}, pol={polarization}, pass={orbit_pass}, "
            f"relative_orbit={relative_orbit}, instrument_mode={instrument_mode}."
        )
    sigma0_db = collection.median()
    sigma0_linear = to_linear(sigma0_db, polarization)
    return sigma0_linear.select(polarization).clip(roi).rename(f"sigma0_{polarization}_linear")


def download_image(image, filename: str | Path, roi, scale: int = 10, crs: str = "EPSG:4326") -> Path:
    """Download an Earth Engine image as a single-band GeoTIFF."""

    geemap = require_geemap()
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    geemap.ee_export_image(
        image,
        filename=str(filename),
        scale=scale,
        region=roi,
        crs=crs,
        file_per_band=False,
    )
    return filename


def download_sigma0_pair(
    roi,
    output_dir: str | Path,
    date1_start: str,
    date1_end: str,
    date2_start: str,
    date2_end: str,
    polarization: str = "VV",
    orbit_pass: str = "ASCENDING",
    relative_orbit: int | str | None = None,
    instrument_mode: str = "IW",
    scale: int = 10,
    crs: str = "EPSG:4326",
):
    """Download the two linear sigma0 GeoTIFFs required by GRD/GEE inference."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t1_image = median_linear_sigma0_image(
        roi, date1_start, date1_end, polarization, orbit_pass, relative_orbit, instrument_mode
    )
    t2_image = median_linear_sigma0_image(
        roi, date2_start, date2_end, polarization, orbit_pass, relative_orbit, instrument_mode
    )
    t1_path = output_dir / f"sigma0_{polarization}_t1_linear.tif"
    t2_path = output_dir / f"sigma0_{polarization}_t2_linear.tif"
    download_image(t1_image, t1_path, roi=roi, scale=scale, crs=crs)
    download_image(t2_image, t2_path, roi=roi, scale=scale, crs=crs)
    return t1_path, t2_path
