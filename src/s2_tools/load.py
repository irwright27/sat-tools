from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from datetime import timezone
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from hashlib import sha256
from http.client import IncompleteRead
from contextlib import suppress
import os
import tempfile
import time
import warnings
import planetary_computer
import xml.etree.ElementTree as ET

import numpy as np

import odc.stac
import pystac
import xarray as xr


DEFAULT_BANDS = ("B04", "B08", "SCL")


def load_s2(
    items: Sequence[pystac.Item],
    aoi: Mapping[str, Any],
    bands: Sequence[str] = DEFAULT_BANDS,
    resolution: float = 10,
    chunks: dict[str, int] | None = None,
    *,
    reflectance: bool = False,
) -> xr.Dataset:
    """
    Load Sentinel-2 STAC items into an xarray Dataset.

    Parameters
    ----------
    items : sequence of pystac.Item
        Sentinel-2 STAC items, typically returned by search_s2().
    aoi : mapping
        AOI geometry represented as a GeoJSON-like mapping in EPSG:4326.
    bands : sequence of str, default ("B04", "B08", "SCL")
        Sentinel-2 assets to load.
    resolution : float, default 10
        Output spatial resolution in meters.
    chunks : dict or None
        Dask chunk sizes. If None, a default spatial chunking scheme
        is used.

    reflectance : bool, default False
        Opt-in physical BOA reflectance for canonical spectral bands B01-B12
        (including B8A). Default preserves the original loader exactly.
        Uses explicit STAC scale/offset, otherwise L2A product XML. Stored zero
        and declared NoData are masked before decoding. Negative reflectance
        is retained; no clipping. SCL and other nonspectral variables are unchanged.
        XML metadata may require small synchronous network reads; image arrays
        remain lazy. Metadata downloads retry transient failures up to three times
        and persist in ~/.cache/s2-tools/metadata (override the root with
        S2_TOOLS_CACHE_DIR). Known Planetary Computer Sentinel-2 metadata URLs
        are re-signed on HTTP 403. Conflicting calibration within a fused timestamp
        is rejected.

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions time, y, and x.
    """
    if not items:
        raise ValueError("No STAC items were provided.")

    if resolution <= 0:
        raise ValueError("resolution must be greater than zero.")

    if chunks is None:
        chunks = {
            "time": 1,
            "x": 1024,
            "y": 1024,
        }

    calibration = _calibration(items, bands) if reflectance else None

    ds = odc.stac.load(
        items,
        bands=list(bands),
        geopolygon=aoi,
        resolution=resolution,
        chunks=chunks,
    )

    if not reflectance:
        return ds
    return _decode(ds, calibration)


_SPECTRAL = {name: i for i, name in enumerate(
    ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
     "B8A", "B09", "B10", "B11", "B12"))}


def _timestamp(value):
    if value is None:
        raise ValueError("Calibrated loading requires item.datetime")
    if value.tzinfo is None:
        raise ValueError("Item datetime must have a timezone")
    return np.datetime64(value.astimezone(timezone.utc).replace(tzinfo=None), "ns")


def _metadata_xml(href, item_id):
    """Fetch immutable product XML with a disk cache and bounded retries."""
    url = urlsplit(href)
    # Only strip credentials for the known Planetary Computer Sentinel-2 store.
    pc_asset = (url.scheme == "https" and
                (url.hostname or "").startswith("sentinel2l2a") and
                (url.hostname or "").endswith(".blob.core.windows.net"))
    identity = urlunsplit(url._replace(query="", fragment="")) if pc_asset else href
    cache_dir = Path(os.environ.get(
        "S2_TOOLS_CACHE_DIR", str(Path.home() / ".cache" / "s2-tools"))) / "metadata"
    path = cache_dir / (sha256(identity.encode()).hexdigest() + ".xml")
    try:
        return ET.fromstring(path.read_bytes())
    except (OSError, ET.ParseError):
        pass

    failure = "unknown error"
    for attempt in range(4):
        try:
            with urlopen(href, timeout=60) as response:
                data = response.read()
            root = ET.fromstring(data)
            break
        except HTTPError as exc:
            failure = f"HTTP {exc.code}"
            exc.close()
            if exc.code == 403 and pc_asset and attempt < 3:
                # sign() preserves existing signatures: remove the old one first.
                try:
                    href = planetary_computer.sign(identity)
                except Exception:
                    raise RuntimeError(
                        f"{item_id}: could not refresh metadata credentials for {url.hostname}"
                    ) from None
            elif exc.code not in {408, 429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"{item_id}: metadata download failed from {url.hostname}: {failure}"
                ) from None
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
            failure = type(exc).__name__
        if attempt < 3:
            warnings.warn(
                f"{item_id}: metadata request failed ({failure}); retry {attempt + 1}/3",
                RuntimeWarning, stacklevel=2,
            )
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(
            f"{item_id}: metadata download failed from {url.hostname} after 4 attempts "
            f"({failure}). Successfully downloaded metadata remains cached; rerun to resume."
        ) from None

    # Atomic replacement prevents another process reading a partial cache entry.
    temporary = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        temporary.replace(path)
    except OSError:
        warnings.warn("Could not persist Sentinel-2 metadata cache", RuntimeWarning,
                      stacklevel=2)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
    return root


def _xml_encoding(item, band, cache):
    if item.collection_id != "sentinel-2-l2a":
        raise ValueError("XML fallback requires the sentinel-2-l2a collection")
    asset = item.assets.get("product-metadata")
    if asset is None:
        raise ValueError(f"{item.id}: no calibration or product-metadata XML")
    if asset.href not in cache:
        cache[asset.href] = _metadata_xml(asset.href, item.id)
    root = cache[asset.href]
    def elements(tag):
        return [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == tag]
    q = elements("BOA_QUANTIFICATION_VALUE")
    if len(q) != 1:
        raise ValueError("Expected one BOA_QUANTIFICATION_VALUE")
    q = float(q[0].text)
    offsets = elements("BOA_ADD_OFFSET")
    matches = [e for e in offsets if int(e.attrib["band_id"]) == _SPECTRAL[band]]
    baseline = elements("PROCESSING_BASELINE")
    baseline = baseline[0].text if baseline else item.properties.get("s2:processing_baseline")
    if len(matches) == 1:
        offset = float(matches[0].text)
    elif not offsets and baseline is not None and float(baseline) < 4:
        offset = 0.0
    else:
        raise ValueError(f"{item.id}/{band}: missing/ambiguous BOA_ADD_OFFSET")
    if not np.isfinite([q, offset]).all() or q <= 0:
        raise ValueError("Invalid XML calibration")
    return 1.0 / q, offset / q, "product XML"


def _calibration(items, bands):
    # Resolve metadata BEFORE ODC fuses overlapping items. A fused timestamp
    # cannot safely receive two different decoding formulas.
    spectral = [band for band in bands if band in _SPECTRAL]
    if any(band.lower() in {"red", "nir", "nir08", "green", "blue"} for band in bands):
        raise ValueError("reflectance=True requires canonical band names, e.g. B04/B08")
    result, cache = {}, {}
    for item in items:
        t = _timestamp(item.datetime)
        for band in spectral:
            if band not in item.assets:
                raise ValueError(f"{item.id}: missing requested asset {band}")
            entries = item.assets[band].extra_fields.get("raster:bands") or [{}]
            if len(entries) != 1:
                raise ValueError(f"{band}: expected a single-band asset")
            meta = entries[0]
            if "scale" in meta:  # offset alone is insufficient calibration
                scale, offset = float(meta["scale"]), float(meta.get("offset", 0))
                source = "STAC raster:bands"
            else:
                scale, offset, source = _xml_encoding(item, band, cache)
                if "offset" in meta and not np.isclose(float(meta["offset"]), offset,
                                                      rtol=0, atol=1e-12):
                    raise ValueError("STAC offset disagrees with product XML")
            if not np.isfinite([scale, offset]).all() or scale <= 0:
                raise ValueError(f"{item.id}/{band}: invalid scale/offset")
            nodata = float(meta.get("nodata", 0))
            entry = (scale, offset, nodata, source)
            key = (t, band)
            if key in result:
                old = result[key]
                if not np.allclose(old[:3], entry[:3], rtol=0, atol=0, equal_nan=True):
                    raise ValueError(f"Conflicting calibration for fused timestamp {t}/{band}")
            result[key] = entry
    return result


def _decode(ds, calibration):
    out = ds.copy()
    provenance = {}
    for band in (name for name in ds.data_vars if name in _SPECTRAL):
        entries = []
        for time in ds.time.values.astype("datetime64[ns]"):
            key = (time, band)
            if key not in calibration:
                raise ValueError(f"Cannot map loaded time {time}/{band} to source calibration")
            entries.append(calibration[key])
        scales = xr.DataArray([e[0] for e in entries], dims="time", coords={"time": ds.time})
        offsets = xr.DataArray([e[1] for e in entries], dims="time", coords={"time": ds.time})
        nodata = xr.DataArray([e[2] for e in entries], dims="time", coords={"time": ds.time})
        raw = ds[band]
        valid = np.isfinite(raw) & (raw != 0) & (raw != nodata)
        # ODC's raster reader returns raw values. Perform one lazy decoding step.
        out[band] = raw.astype("float64").where(valid) * scales + offsets
        out[band].attrs = {k: v for k, v in raw.attrs.items()
                           if k not in {"scale_factor", "add_offset", "nodata", "_FillValue", "units"}}
        out[band].attrs.update(units="1", radiometry="BOA reflectance")
        # Do not carry packed-DN encoding onto already decoded floats.
        out[band].encoding = {}
        provenance[band] = [dict(time=str(t), scale=e[0], offset=e[1], source=e[3])
                            for t, e in zip(ds.time.values, entries)]
    out.attrs = dict(ds.attrs)
    out.attrs["s2_tools_reflectance"] = True
    out.attrs["s2_tools_calibration"] = provenance
    return out