"""Exposure quality measured from native star stamps and background tiles."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import maximum_filter

from .calibration import robust_summary
from .domain import DomainError, MAX_ACTIVE_PIXELS, enum, integer, number, usable_science_frames
from .registration import offset_for

FWHM_FACTOR = 2 * math.sqrt(2 * math.log(2))


@dataclass(frozen=True)
class StarSearch:
    threshold: float = 6.0
    stamp_radius: int = 5
    cap: int = 48
    separation: int = 12

    @classmethod
    def requested(cls, settings):
        return cls(
            threshold=number(settings.get("threshold", 6), "检星阈值", 3, 30),
            stamp_radius=integer(settings.get("stamp_radius", 5), "检测窗口半径", 3, 10),
            cap=integer(settings.get("cap", 48), "星点上限", 6, 100),
            separation=integer(settings.get("separation", 12), "星点间距", 3, 30),
        )


def tile_background(plane, divisions=4):
    height, width = plane.values.shape
    y_edges = np.rint(np.linspace(0, height, divisions + 1)).astype(int)
    x_edges = np.rint(np.linspace(0, width, divisions + 1)).astype(int)
    results = []
    for line, (y0, y1) in enumerate(zip(y_edges, y_edges[1:])):
        for column, (x0, x1) in enumerate(zip(x_edges, x_edges[1:])):
            sl = np.s_[y0:y1, x0:x1]
            keep = plane.valid[sl] & ~plane.saturated[sl]
            usable = plane.values[sl][keep]
            quality = robust_summary(usable)
            results.append({
                "line": line,
                "column": column,
                "x": int(x0),
                "y": int(y0),
                "width": int(x1 - x0),
                "height": int(y1 - y0),
                "median": quality["median"],
                "noise": quality["sigma"],
                "valid_fraction": quality["count"] / keep.size,
            })
    return results


def _candidate_points(plane, search):
    values = plane.values
    usable = plane.valid & ~plane.saturated
    valid = values[usable]
    if valid.size < 49:
        return [], None
    background = robust_summary(valid)
    cutoff = background["median"] + max(search.threshold * background["sigma"], 1e-6)
    working = np.where(usable, values, -np.inf)
    peaks = working == maximum_filter(working, size=3, mode="constant", cval=-np.inf)
    radius = search.stamp_radius + 2
    peaks[:radius] = False
    peaks[-radius:] = False
    peaks[:, :radius] = False
    peaks[:, -radius:] = False
    yy, xx = np.where(peaks & (working > cutoff))
    if xx.size == 0:
        return [], background
    scores = values[yy, xx]
    max_candidates = min(search.cap * 16, scores.size)
    pool = np.argpartition(-scores, max_candidates - 1)[:max_candidates]
    pool = sorted(pool, key=lambda index: (-scores[index], yy[index], xx[index]))
    picked = []
    for index in pool:
        x, y = int(xx[index]), int(yy[index])
        if any((x - other_x)**2 + (y - other_y)**2 < search.separation**2 for other_x, other_y in picked):
            continue
        picked.append((x, y))
        if len(picked) >= search.cap:
            break
    return picked, background


def _star_stamp(plane, px, py, radius):
    window = np.s_[py - radius:py + radius + 1, px - radius:px + radius + 1]
    pixels = plane.values[window]
    valid = plane.valid[window]
    sat = plane.saturated[window]
    yy, xx = np.indices(pixels.shape)
    rho = np.hypot(xx - radius, yy - radius)
    star_region = rho <= radius - 1
    rim = valid & ~sat & (rho >= radius - 0.25)
    if np.any(sat & star_region):
        return {"x": px, "y": py, "state": "saturated", "peak": None}
    if np.count_nonzero(rim) < 10 or np.mean(valid[star_region]) < 0.85:
        return {"x": px, "y": py, "state": "masked", "peak": None}
    sky = robust_summary(pixels[rim])
    residual = pixels - sky["median"]
    weights = np.where(valid & star_region & (residual > max(sky["sigma"], 0)), residual, 0)
    total = float(np.sum(weights))
    if total <= 0:
        return {"x": px, "y": py, "state": "no_signal", "peak": None}
    x_center = float(np.sum(xx * weights) / total)
    y_center = float(np.sum(yy * weights) / total)
    gx, gy = xx - x_center, yy - y_center
    mxx = float(np.sum(gx**2 * weights) / total)
    mxy = float(np.sum(gx * gy * weights) / total)
    myy = float(np.sum(gy**2 * weights) / total)
    moment = np.array([[mxx, mxy], [mxy, myy]])
    eigen, vectors = np.linalg.eigh(moment)
    minor, major = np.sqrt(np.maximum(0, eigen))
    state = "measured"
    if minor < 0.35:
        state = "compact_pixel"
    elif major > radius * 0.72 or math.hypot(x_center - radius, y_center - radius) > radius * 0.6:
        state = "confused"
    return {
        "x": px + x_center - radius,
        "y": py + y_center - radius,
        "state": state,
        "major": float(major * FWHM_FACTOR),
        "minor": float(minor * FWHM_FACTOR),
        "fwhm": float(math.sqrt(major * minor) * FWHM_FACTOR),
        "elongation": float(major / minor) if minor else None,
        "angle": float(math.degrees(math.atan2(vectors[1, 1], vectors[0, 1])) % 180),
        "peak": float(pixels[radius, radius]),
        "net_counts": total,
    }


def analyze_stars(plane, settings):
    search = StarSearch.requested(settings)
    points, sky = _candidate_points(plane, search)
    stars = [_star_stamp(plane, x, y, search.stamp_radius) for x, y in points]
    stable = [star for star in stars if star["state"] == "measured"]
    return {
        "sky": sky,
        "stars": stars,
        "median_fwhm": robust_summary([star["fwhm"] for star in stable])["median"],
        "median_elongation": robust_summary([star["elongation"] for star in stable])["median"],
        "stars_used": len(stable),
        "candidate_count": len(stars),
    }


def frame_quality(plane, options):
    result = analyze_stars(plane, options)
    tiles = tile_background(plane, 4)
    usable = [row["median"] for row in tiles if row["valid_fraction"] > 0.3]
    result["tiles"] = tiles
    result["sky_spread"] = None if len(usable) < 2 else float(max(usable) - min(usable))
    result["masked_fraction"] = float(np.mean(~plane.valid))
    result["saturated_fraction"] = float(np.mean(plane.saturated & plane.valid))
    return result


def quality_report(images, options):
    doc = images.document
    band = options.get("band")
    stream = enum(options.get("stream", "calibrated"), "质检影像", {"raw", "calibrated"})
    frames = usable_science_frames(doc, band)
    if not frames:
        raise DomainError("没有符合所选滤镜的科学曝光")
    if len(frames) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("当前质检超过 2400 万像素，先挂起部分科学曝光")
    fwhm_limit = number(options.get("fwhm_limit", 5), "FWHM 提示阈值", 1, 25)
    elongation_limit = number(options.get("elongation_limit", 1.7), "拖尾提示阈值", 1.05, 9)
    quality_rows = []
    for frame in frames:
        frame_id = frame["id"]
        plane = images.calibrated(frame_id) if stream == "calibrated" else images.raw(frame_id, True)
        result = frame_quality(plane, options)
        offset = offset_for(doc, frame_id)
        flags = []
        if result["stars_used"] < 3:
            flags.append("few_stars")
        if result["median_fwhm"] and result["median_fwhm"] > fwhm_limit:
            flags.append("broad_stars")
        if result["median_elongation"] and result["median_elongation"] > elongation_limit:
            flags.append("elongated")
        if result["masked_fraction"] > 0.05:
            flags.append("masked")
        if offset is None:
            flags.append("unaligned")
        quality_rows.append({
            "frame": frame_id,
            "filename": frame["filename"],
            "started": frame["started"],
            "band": frame["band"],
            "flags": flags,
            "sky": result["sky"],
            "sky_spread": result["sky_spread"],
            "median_fwhm": result["median_fwhm"],
            "median_elongation": result["median_elongation"],
            "stars_used": result["stars_used"],
            "candidate_count": result["candidate_count"],
            "masked_fraction": result["masked_fraction"],
            "saturated_fraction": result["saturated_fraction"],
            "offset": [offset.dx, offset.dy] if offset else None,
        })
    return {
        "kind": "quality",
        "revision": doc["revision"],
        "stream": stream,
        "band": band,
        "thresholds": {"fwhm": fwhm_limit, "elongation": elongation_limit},
        "rows": quality_rows,
    }
