"""Transfer native-detector neighborhoods as quantitative and visual evidence."""
from io import BytesIO
import json
import math
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw

from .calibration import Plane
from .domain import DomainError, MAX_ACTIVE_PIXELS, enum, identifier, integer, list_value, usable_science_frames
from .exports import csv_bytes
from .preview import png, stretch_pixels
from .raster import make_fits
from .registration import offset_for

CUTOUT_STREAMS = {"raw", "calibrated"}


def _choose_rows(document, options):
    offered = options.get("frames")
    enabled = usable_science_frames(document, options.get("band"))
    if offered is None:
        return enabled[:24]
    ids = list_value(offered, "取证曝光列表", 24)
    index = {row["id"]: row for row in document["frames"]}
    try:
        result = [index[identifier(token, "曝光编号")] for token in ids]
    except KeyError as exc:
        raise DomainError("有一张指定的科学曝光不存在") from exc
    if len(result) != len(set(ids)):
        raise DomainError("重复选中了同一张取证曝光")
    return result


def cutcode(plane, center, size):
    cx, cy = center
    half = size // 2
    x, y = math.floor(cx + 0.5) - half, math.floor(cy + 0.5) - half
    height, width = plane.values.shape
    values = np.full((size, size), np.nan)
    valid = np.zeros((size, size), dtype=bool)
    sat = np.zeros((size, size), dtype=bool)
    dest_x0, dest_y0 = max(-x, 0), max(-y, 0)
    dest_x1, dest_y1 = min(width - x, size), min(height - y, size)
    if dest_x1 > dest_x0 and dest_y1 > dest_y0:
        source = np.s_[y + dest_y0:y + dest_y1, x + dest_x0:x + dest_x1]
        target = np.s_[dest_y0:dest_y1, dest_x0:dest_x1]
        valid[target] = plane.valid[source]
        values[target] = np.where(plane.valid[source], plane.values[source], np.nan)
        sat[target] = plane.saturated[source]
    return Plane(values, valid, sat), {
        "origin_x": x,
        "origin_y": y,
        "center_x": cx,
        "center_y": cy,
        "local_x": cx - x,
        "local_y": cy - y,
        "valid_fraction": float(np.mean(valid)),
    }


def stamp_preview(stamp, coords, settings):
    gray, bounds = stretch_pixels(stamp, settings)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[~stamp.valid] = [63, 33, 65]
    rgb[stamp.valid & stamp.saturated] = [236, 167, 76]
    multiplier = min(8, max(1, 320 // gray.shape[1]))
    art = Image.fromarray(rgb, "RGB")
    art = art.resize((art.width * multiplier, art.height * multiplier), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(art)
    x = (coords["local_x"] + 0.5) * multiplier
    y = (coords["local_y"] + 0.5) * multiplier
    arm = max(3, multiplier)
    color = (107, 220, 209)
    draw.line((x - 2 * arm, y, x - arm, y), fill=color)
    draw.line((x + arm, y, x + 2 * arm, y), fill=color)
    draw.line((x, y - 2 * arm, x, y - arm), fill=color)
    draw.line((x, y + arm, x, y + 2 * arm), fill=color)
    return png(art), bounds


def _write_frame(archive, images, doc, frame, source, size, stream):
    frame_id = frame["id"]
    key = frame["id"]
    record = {"frame": frame_id, "filename": frame["filename"], "started": frame["started"], "exposure": frame["exposure"],
              "band": frame["band"], "state": "unaligned"}
    offset = offset_for(doc, frame_id)
    if offset is None:
        return record
    center = offset.native_xy(source["x"], source["y"])
    plane = images.calibrated(frame_id) if stream == "calibrated" else images.raw(frame_id, True)
    samples, coords = cutcode(plane, center, size)
    art, bounds = stamp_preview(samples, coords, doc["stretch"])
    if coords["valid_fraction"] > 0:
        fits_header = {
            "BUNIT": "ADU",
            "XORIGIN": coords["origin_x"],
            "YORIGIN": coords["origin_y"],
            "EXPTIME": frame["exposure"],
            "FILTER": frame["band"],
            "DATE-OBS": frame["started"],
        }
        archive.writestr(f"stamps/{key}.fits", make_fits(samples.values, fits_header))
        archive.writestr(f"stamps/{key}.png", art)
        record["state"] = "measured"
        record["preview"] = f"stamps/{key}.png"
    else:
        record["state"] = "no_support"
    record.update(coords, display_low=bounds[0], display_high=bounds[1])
    return record


def cutout_zip(images, options):
    doc = images.document
    source_id = identifier(options.get("source"), "取证孔径")
    source = next((row for row in doc["apertures"] if row["id"] == source_id), None)
    if not source:
        raise DomainError("指定的取证孔径不存在")
    size = integer(options.get("size", 41), "取证画幅", 17, 161)
    if not size % 2:
        raise DomainError("取证画幅需为奇数，以一个中心像素定位")
    stream = enum(options.get("stream", "calibrated"), "取证影像", CUTOUT_STREAMS)
    chosen = _choose_rows(doc, options)
    if not chosen:
        raise DomainError("请选择至少一张科学曝光")
    if len(chosen) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("取证画幅总源像素数超过单次预算，请减少曝光")
    storage = BytesIO()
    with ZipFile(storage, "w", ZIP_DEFLATED, compresslevel=3) as archive:
        statuses = [_write_frame(archive, images, doc, frame, source, size, stream) for frame in chosen]
        summary = {
            "revision": doc["revision"],
            "reference": doc["reference"],
            "source": source,
            "stream": stream,
            "size": size,
            "pixel_origin": "zero-based native detector pixel centers",
            "frames": statuses,
        }
        archive.writestr("stamps.json", json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
        keys = ("filename", "started", "exposure", "band", "state", "origin_x", "origin_y",
                "center_x", "center_y", "local_x", "local_y", "valid_fraction", "preview")
        archive.writestr("stamps.csv", csv_bytes(keys, ([row.get(key) for key in keys] for row in statuses)))
    return storage.getvalue(), summary
