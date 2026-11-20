"""Native-pixel aperture photometry of fixed reference-frame sources."""
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

import numpy as np

from .calibration import robust_summary
from .domain import DomainError, MAX_ACTIVE_PIXELS, identifier, usable_science_frames
from .registration import offset_for

SUBPIXEL_GRID = (-0.375, -0.125, 0.125, 0.375)


@dataclass(frozen=True)
class SampleRegion:
    x: float
    y: float
    radius: float
    sky_inner: float
    sky_outer: float

    def cut(self, detector_shape):
        height, width = detector_shape
        x0 = max(0, math.floor(self.x - self.sky_outer - 1))
        y0 = max(0, math.floor(self.y - self.sky_outer - 1))
        x1 = min(width, math.ceil(self.x + self.sky_outer + 1))
        y1 = min(height, math.ceil(self.y + self.sky_outer + 1))
        return slice(y0, max(y0, y1)), slice(x0, max(x0, x1))

    def weights(self, cut):
        yy = np.arange(cut[0].start, cut[0].stop)[:, None] - self.y
        xx = np.arange(cut[1].start, cut[1].stop)[None, :] - self.x
        stamp = np.broadcast_shapes(xx.shape, yy.shape)
        aperture = np.zeros(stamp, dtype=np.float64)
        annulus = np.zeros(stamp, dtype=np.float64)
        for dx in SUBPIXEL_GRID:
            for dy in SUBPIXEL_GRID:
                rho2 = (xx + dx)**2 + (yy + dy)**2
                aperture += rho2 <= self.radius**2
                annulus += (rho2 <= self.sky_outer**2) & (rho2 >= self.sky_inner**2)
        divisor = len(SUBPIXEL_GRID)**2
        return aperture / divisor, annulus / divisor


def weighted_median(values, weights):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    partial = np.cumsum(weights)
    index = int(np.searchsorted(partial, partial[-1] * 0.5))
    return float(values[index])


def sky_background(values, valid, weights):
    mask = valid & (weights >= 0.25)
    effective_area = float(np.sum(weights[mask]))
    if effective_area < 12:
        return None
    values = values[mask]
    weights = weights[mask]
    center = weighted_median(values, weights)
    deviations = np.abs(values - center)
    noise = weighted_median(deviations, weights) * 1.482602218505602
    use = np.ones(values.size, dtype=bool) if noise < 1e-6 else deviations <= 3 * noise
    kept_area = float(np.sum(weights[use]))
    if kept_area < 10:
        return None
    return {
        "level": weighted_median(values[use], weights[use]),
        "noise": noise,
        "area": kept_area,
        "count": int(np.count_nonzero(use)),
        "rejected": int(np.count_nonzero(~use)),
    }


def aperture_measurement(plane, source, offset, settings, seconds):
    x, y = offset.native_xy(source["x"], source["y"])
    region = SampleRegion(x, y, settings["radius"], settings["sky_inner"], settings["sky_outer"])
    cut = region.cut(plane.values.shape)
    values = plane.values[cut]
    aperture, annulus = region.weights(cut)
    valid = plane.valid[cut]
    saturated = plane.saturated[cut]
    use = valid & ~saturated & (aperture > 0)
    actual_area = float(np.sum(aperture[use]))
    expected_area = math.pi * region.radius**2
    coverage = min(1.0, actual_area / expected_area)
    result = {
        "source": source["id"],
        "label": source["label"],
        "role": source["role"],
        "native_x": x,
        "native_y": y,
        "coverage": coverage,
        "flux": None,
        "flux_per_second": None,
        "uncertainty_per_second": None,
        "background": None,
        "peak": None,
    }
    if np.any(saturated & (aperture >= 0.25)):
        return {**result, "state": "saturated"}
    if values.size == 0 or coverage + 0.01 < settings["min_fraction"]:
        return {**result, "state": "insufficient_aperture"}
    sky = sky_background(values, valid & ~saturated, annulus)
    if sky is None:
        return {**result, "state": "insufficient_sky"}
    pixel_weights = aperture[use]
    counts = values[use]
    flux = float(np.sum(pixel_weights * (counts - sky["level"])))
    variance_sky = sky["noise"]**2 * (float(np.sum(pixel_weights**2)) + actual_area**2 / sky["area"])
    variance_source = max(0.0, flux) / settings["gain"]
    sigma = math.sqrt(variance_sky + variance_source)
    return {
        **result,
        "state": "measured" if flux > 0 else "nonpositive",
        "flux": flux,
        "flux_per_second": flux / seconds,
        "uncertainty_per_second": sigma / seconds,
        "background": sky,
        "peak": float(np.max(counts)),
    }


def measure_missing(source):
    return {
        "source": source["id"],
        "label": source["label"],
        "role": source["role"],
        "state": "unaligned",
        "native_x": None,
        "native_y": None,
        "coverage": None,
        "flux": None,
        "flux_per_second": None,
        "uncertainty_per_second": None,
        "peak": None,
        "background": None,
    }


def time_midpoint(started, exposure):
    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        middle = start + timedelta(seconds=exposure * 0.5)
    except (ValueError, OverflowError) as exc:
        raise DomainError("曝光中点时间超出范围") from exc
    return middle.isoformat(timespec="microseconds").replace("+00:00", "Z")


def per_frame_measurements(images, frames, sources):
    for frame in frames:
        offset = offset_for(images.document, frame["id"])
        if offset is None:
            measures = [measure_missing(source) for source in sources]
            calibration = []
        else:
            plane = images.calibrated(frame["id"])
            settings = images.document["photometry"]
            measures = [
                aperture_measurement(plane, source, offset, settings, frame["exposure"])
                for source in sources
            ]
            calibration = list(plane.notes)
        yield {
            "frame": frame["id"],
            "filename": frame["filename"],
            "started": frame["started"],
            "midpoint": time_midpoint(frame["started"], frame["exposure"]),
            "time_seconds": datetime.fromisoformat(frame["started"].replace("Z", "+00:00")).timestamp() + frame["exposure"] * 0.5,
            "exposure": frame["exposure"],
            "band": frame["band"],
            "alignment": offset.source if offset else None,
            "calibration": calibration,
            "measurements": measures,
        }


def ratio_error(magnitude, target, comparison):
    # Logarithmic errors assume independent count estimates.
    a = target["uncertainty_per_second"] / target["flux_per_second"]
    b = comparison["uncertainty_per_second"] / comparison["flux_per_second"]
    return magnitude, 2.5 / math.log(10) * math.hypot(a, b)


def differential(target, comparison):
    if target["state"] != "measured":
        return {"state": f"target_{target['state']}", "ratio": None, "magnitude": None, "error": None}
    if comparison["state"] != "measured":
        return {"state": f"comparison_{comparison['state']}", "ratio": None, "magnitude": None, "error": None}
    ratio = target["flux_per_second"] / comparison["flux_per_second"]
    magnitude, sigma = ratio_error(-2.5 * math.log10(ratio), target, comparison)
    return {"state": "measured", "ratio": ratio, "magnitude": magnitude, "error": sigma}


def choose_sources(doc, options):
    by_id = {source["id"]: source for source in doc["apertures"]}
    requested = {}
    for field in ("target", "comparison", "check"):
        value = options.get(field)
        if not value:
            continue
        source = by_id.get(identifier(value, "孔径"))
        if not source or source["role"] != field:
            raise DomainError(f"所选 {field} 孔径不存在或角色不符")
        requested[field] = source
    if "target" not in requested or "comparison" not in requested:
        raise DomainError("请指定一颗目标星和一颗比较星")
    return requested


def filter_photometry_frames(doc, band):
    frames = usable_science_frames(doc)
    filters = sorted({frame["band"] for frame in frames})
    if band is None:
        if len(filters) > 1:
            raise DomainError("存在多种滤镜，请选定一种滤镜再计算曲线")
        band = next(iter(filters), "")
    chosen = [frame for frame in frames if frame["band"] == band]
    if not chosen:
        raise DomainError("所选滤镜没有可用曝光")
    return band, chosen


def light_curve(images, options):
    doc = images.document
    requested = choose_sources(doc, options)
    band, frames = filter_photometry_frames(doc, options.get("band"))
    if len(frames) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("请暂时挂起部分曝光，使单次曲线图像总量降至 2400 万像素内")
    rows = []
    for row in per_frame_measurements(images, frames, requested.values()):
        by_role = {measure["role"]: measure for measure in row["measurements"]}
        row["target"] = differential(by_role["target"], by_role["comparison"])
        row["check"] = differential(by_role["check"], by_role["comparison"]) if "check" in by_role else None
        rows.append(row)
    accepted = [row for row in rows if row["target"]["state"] == "measured"]
    return {
        "kind": "photometry",
        "revision": doc["revision"],
        "band": band,
        "unit": "mag",
        "reference": doc["reference"],
        "sources": requested,
        "settings": dict(doc["photometry"]),
        "rows": rows,
        "summary": {
            "frames": len(rows),
            "measured": len(accepted),
            "unavailable": len(rows) - len(accepted),
            "target_baseline": robust_summary([row["target"]["magnitude"] for row in accepted]),
        },
    }
