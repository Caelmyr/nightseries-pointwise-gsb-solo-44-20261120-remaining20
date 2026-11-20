"""Star-centroid checks of mount drift in a fixed reference field."""
from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from .calibration import robust_summary
from .domain import DomainError, MAX_ACTIVE_PIXELS, enum, number, usable_science_frames
from .photometry import time_midpoint
from .quality import FWHM_FACTOR
from .registration import offset_for


@dataclass(frozen=True)
class GuideSpec:
    x: float
    y: float
    search_radius: int
    fit_radius: int
    base: str
    threshold: float
    movement_limit: float

    @classmethod
    def request(cls, dimensions, options):
        x, y = dimensions.xy(options.get("x"), options.get("y"))
        search = number(options.get("search_radius", 11), "导星搜索半径", 3, 40)
        radius = number(options.get("fit_radius", 4), "光心窗口半径", 2, 15)
        if radius + 1 > search:
            raise DomainError("光心窗口需要小于搜索范围")
        return cls(
            x, y, int(round(search)), int(round(radius)),
            enum(options.get("base", "alignment"), "搜索中心", {"alignment", "fixed"}),
            number(options.get("threshold", 7), "星点最低信噪比", 3, 35),
            number(options.get("movement_limit", 2), "偏离提示像素", 0.25, 40),
        )


def _windows(plane, x, y, radius):
    h, w = plane.values.shape
    x0 = min(w, max(0, int(math.floor(x - radius))))
    x1 = min(w, max(0, int(math.ceil(x + radius + 1))))
    y0 = min(h, max(0, int(math.floor(y - radius))))
    y1 = min(h, max(0, int(math.ceil(y + radius + 1))))
    cut = np.s_[y0:y1, x0:x1]
    return cut, np.arange(x0, x1)[None, :], np.arange(y0, y1)[:, None]


def _background(data, valid, within):
    sky = data[valid & ~within]
    if sky.size < 14:
        return None
    first = robust_summary(sky)
    retained = np.abs(sky - first["median"]) <= max(first["sigma"] * 3.2, 1e-6)
    if np.count_nonzero(retained) < 10:
        return None
    return robust_summary(sky[retained])


def _search_peak(data, u, v, valid, center, spec):
    x, y = center
    search = np.hypot(u - x, v - y) <= spec.search_radius
    if not np.any(valid & search):
        return None
    masked = np.where(valid & search, data, -np.inf)
    j, i = np.unravel_index(np.argmax(masked), masked.shape)
    return float(u[0, i]), float(v[j, 0])


def guide_centroid(plane, center, spec):
    xx, yy = center
    cut, u, v = _windows(plane, xx, yy, spec.search_radius + spec.fit_radius + 3)
    data = plane.values[cut]
    valid = plane.valid[cut] & ~plane.saturated[cut]
    if data.size == 0 or np.count_nonzero(valid) < 30:
        return {"state": "no_support"}
    peak = _search_peak(plane.values[cut], u, v, plane.valid[cut], center, spec)
    if peak is None:
        return {"state": "no_support"}
    px, py = peak
    cut, u, v = _windows(plane, px, py, spec.fit_radius + 3)
    data = plane.values[cut]
    radius = np.hypot(u - px, v - py)
    valid = plane.valid[cut] & ~plane.saturated[cut]
    inside = radius <= spec.fit_radius + 0.2
    usable = valid & inside
    if np.any(plane.saturated[cut] & inside):
        return {"state": "saturated", "peak": [px, py]}
    if np.count_nonzero(usable) < 0.83 * math.pi * spec.fit_radius**2:
        return {"state": "clipped", "peak": [px, py]}
    sky = _background(data, valid, radius < spec.fit_radius + 1.0)
    if sky is None:
        return {"state": "missing_sky", "peak": [px, py]}
    flux = np.where(usable, np.maximum(data - sky["median"], 0), 0.0)
    total = float(np.sum(flux))
    if total <= 0:
        return {"state": "no_signal", "peak": [px, py]}
    peak_snr = float(np.max(flux)) / max(sky["sigma"], 0.01)
    if peak_snr < spec.threshold:
        return {"state": "weak", "peak": [px, py], "snr": peak_snr}
    u_fit = float(np.sum(u * flux) / total)
    v_fit = float(np.sum(v * flux) / total)
    var_x = float(np.sum((u - u_fit)**2 * flux) / total)
    var_y = float(np.sum((v - v_fit)**2 * flux) / total)
    covar = float(np.sum((u - u_fit) * (v - v_fit) * flux) / total)
    eigen = np.linalg.eigvalsh([[var_x, covar], [covar, var_y]])
    low, high = np.sqrt(np.maximum(0, eigen))
    if low < 0.36:
        return {"state": "compact_pixel", "peak": [px, py], "snr": peak_snr}
    if high > spec.fit_radius * 0.62:
        return {"state": "broad", "peak": [px, py], "snr": peak_snr}
    radius_used = math.hypot(u_fit - xx, v_fit - yy)
    if radius_used > spec.search_radius - 0.4:
        return {"state": "at_search_edge", "peak": [px, py], "snr": peak_snr}
    return {
        "state": "measured",
        "x": u_fit, "y": v_fit, "snr": peak_snr,
        "fwhm_x": math.sqrt(var_x) * FWHM_FACTOR,
        "fwhm_y": math.sqrt(var_y) * FWHM_FACTOR,
        "axis_ratio": float(high / low),
        "net_counts": total,
        "local_sky": sky["median"],
        "centroid_uncertainty": float(high) / max(peak_snr, 1),
    }


def _plane_check(images, frame, spec):
    offset = offset_for(images.document, frame["id"])
    if spec.base == "alignment" and offset is None:
        return {"state": "unaligned", "pred_x": None, "pred_y": None}
    predicted = offset.native_xy(spec.x, spec.y) if spec.base == "alignment" else (spec.x, spec.y)
    result = guide_centroid(images.calibrated(frame["id"]), predicted, spec)
    result.update(pred_x=predicted[0], pred_y=predicted[1])
    if result["state"] != "measured":
        return result
    result.update(
        native_dx=result["x"] - spec.x,
        native_dy=result["y"] - spec.y,
        residual_dx=result["x"] - predicted[0],
        residual_dy=result["y"] - predicted[1],
    )
    disjoint = math.hypot(result["residual_dx"], result["residual_dy"])
    result["distance"] = disjoint
    result["outside_limit"] = disjoint > spec.movement_limit
    return result


def _drift_summary(guide_rows):
    accepted = [row for row in guide_rows if row["state"] == "measured"]
    rejected = Counter(row["state"] for row in guide_rows if row["state"] != "measured")
    metrics = {name: robust_summary([row[name] for row in accepted])
               for name in ("native_dx", "native_dy", "residual_dx", "residual_dy")}
    if not accepted:
        return {**metrics, "measured": 0, "rejected": dict(rejected), "max_distance": None, "outside_limit": 0}
    max_row = max(accepted, key=lambda row: row["distance"])
    return {
        **metrics,
        "measured": len(accepted),
        "rejected": dict(rejected),
        "max_distance": max_row["distance"],
        "max_frame": max_row["frame"],
        "outside_limit": sum(row["outside_limit"] for row in accepted),
    }


def guide_report(images, options):
    spec = GuideSpec.request(images.dims, options)
    rows = usable_science_frames(images.document, band=options.get("band"))
    if not rows:
        raise DomainError("没有符合所选滤镜的科学曝光")
    if len(rows) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("导星检查超过 2400 万像素，请挂起部分曝光")
    result = []
    for row in rows:
        frame = _plane_check(images, row, spec)
        frame.update(
            frame=row["id"], filename=row["filename"], band=row["band"],
            timestamp=time_midpoint(row["started"], row["exposure"]),
        )
        result.append(frame)
    return {
        "revision": images.document["revision"],
        "source": {"x": spec.x, "y": spec.y, "base": spec.base},
        "settings": {"threshold": spec.threshold, "search_radius": spec.search_radius, "fit_radius": spec.fit_radius,
                     "movement_limit": spec.movement_limit},
        "summary": _drift_summary(result),
        "rows": result,
    }
