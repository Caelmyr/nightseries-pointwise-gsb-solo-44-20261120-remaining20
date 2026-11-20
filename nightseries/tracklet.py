"""Measured per-frame positions of a slowly moving point source."""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np
from scipy.ndimage import maximum_filter

from .calibration import robust_summary
from .domain import (
    DomainError, MAX_ACTIVE_PIXELS, exposure_row, identifier, list_value, number,
    object_value, usable_science_frames,
)
from .photometry import time_midpoint
from .quality import FWHM_FACTOR
from .registration import offset_for

MEASURED = "measured"
REVIEW = "review"
MAX_CANDIDATES = 6


@dataclass(frozen=True)
class TrackSpec:
    search: float
    fit: float
    threshold: float
    blend: float

    @classmethod
    def request(cls, options):
        search = number(options.get("search_radius", 6), "搜索半径", 2, 24)
        fit = number(options.get("fit_radius", 4), "拟合窗口", 2, 10)
        if fit + 1 > search:
            raise DomainError("拟合窗口需要小于搜索半径")
        return cls(
            search,
            fit,
            number(options.get("threshold", 5), "检出阈值", 3, 30),
            number(options.get("blend_radius", 4), "混淆半径", 2, 12),
        )


def mid_seconds(frame):
    start = datetime.fromisoformat(frame["started"].replace("Z", "+00:00"))
    return start.timestamp() + frame["exposure"] * 0.5


def seed_points(document, dims, value):
    entries = list_value(value, "种子点", 2)
    if len(entries) != 2:
        raise DomainError("需要在两张曝光上各标一粒种子点")
    seeds = []
    for entry in entries:
        data = object_value(entry, "种子点")
        frame = exposure_row(document, identifier(data.get("frame"), "种子曝光"))
        if not frame["enabled"]:
            raise DomainError("种子曝光已挂起，请恢复参与或换一张")
        x, y = dims.xy(data.get("x"), data.get("y"))
        seeds.append({
            "frame": frame["id"], "filename": frame["filename"], "band": frame["band"],
            "x": x, "y": y, "t": mid_seconds(frame),
        })
    if seeds[0]["frame"] == seeds[1]["frame"]:
        raise DomainError("两粒种子需要在不同的曝光上")
    if seeds[0]["band"] != seeds[1]["band"]:
        raise DomainError("两粒种子曝光的滤镜不一致")
    if seeds[0]["t"] == seeds[1]["t"]:
        raise DomainError("两张种子曝光的曝光中点相同，无法估计移动")
    return sorted(seeds, key=lambda seed: seed["t"])


def plane_cache(images, limit=6):
    cached = {}

    def get(frame_id):
        if frame_id not in cached:
            if len(cached) >= limit:
                cached.pop(next(iter(cached)))
            cached[frame_id] = images.calibrated(frame_id)
        return cached[frame_id]
    return get


def _window(shape, x, y, radius):
    height, width = shape
    x0 = max(0, min(width, int(math.floor(x - radius))))
    x1 = max(0, min(width, int(math.ceil(x + radius + 1))))
    y0 = max(0, min(height, int(math.floor(y - radius))))
    y1 = max(0, min(height, int(math.ceil(y + radius + 1))))
    return np.s_[y0:y1, x0:x1], np.arange(x0, x1)[None, :], np.arange(y0, y1)[:, None]


def _sky_level(samples):
    if samples.size < 12:
        return None
    first = robust_summary(samples)
    kept = samples[np.abs(samples - first["median"]) <= max(3.2 * first["sigma"], 1e-6)]
    if kept.size < 10:
        return None
    return robust_summary(kept)


def local_peaks(plane, cx, cy, search, spec):
    cut, gx, gy = _window(plane.values.shape, cx, cy, search + spec.fit + 3)
    values = plane.values[cut]
    valid = plane.valid[cut] & ~plane.saturated[cut]
    if np.count_nonzero(valid) < 30:
        return None, []
    sky = _sky_level(values[valid])
    if sky is None:
        return None, []
    working = np.where(valid, values, -np.inf)
    tops = working == maximum_filter(working, size=3, mode="constant", cval=-np.inf)
    reach = (gx - cx)**2 + (gy - cy)**2 <= search**2
    cutoff = sky["median"] + spec.threshold * max(sky["sigma"], 0.01)
    yy, xx = np.where(tops & reach & (working >= cutoff))
    order = np.argsort(-working[yy, xx])
    picked = []
    for index in order:
        x, y = float(gx[0, xx[index]]), float(gy[yy[index], 0])
        if any((x - a)**2 + (y - b)**2 < 2.2**2 for a, b in picked):
            continue
        picked.append((x, y))
        if len(picked) >= MAX_CANDIDATES:
            break
    return sky, picked


def measure_candidate(plane, px, py, spec):
    base = {"x": px, "y": py}
    cut, gx, gy = _window(plane.values.shape, px, py, spec.fit + 3)
    data = plane.values[cut]
    valid = plane.valid[cut] & ~plane.saturated[cut]
    rho = np.hypot(gx - px, gy - py)
    inside = rho <= spec.fit + 0.2
    if np.any(plane.saturated[cut] & inside):
        return {**base, "state": "saturated"}
    if np.count_nonzero(valid & inside) < 0.8 * math.pi * spec.fit**2:
        return {**base, "state": "clipped"}
    sky = _sky_level(data[valid & (rho >= spec.fit + 1.0)])
    if sky is None:
        return {**base, "state": "missing_sky"}
    flux = np.where(valid & inside, np.maximum(data - sky["median"], 0), 0.0)
    total = float(np.sum(flux))
    if total <= 0:
        return {**base, "state": "no_signal"}
    x = float(np.sum(gx * flux) / total)
    y = float(np.sum(gy * flux) / total)
    var_x = float(np.sum((gx - x)**2 * flux) / total)
    var_y = float(np.sum((gy - y)**2 * flux) / total)
    minor, major = sorted((math.sqrt(max(var_x, 0.0)), math.sqrt(max(var_y, 0.0))))
    flags = []
    if minor < 0.36:
        flags.append("compact")
    if major > spec.fit * 0.62:
        flags.append("broad")
    return {
        **base,
        "state": "ok",
        "x": x,
        "y": y,
        "snr": float(np.max(flux)) / max(sky["sigma"], 0.01),
        "flux": total,
        "fwhm": math.sqrt(major * minor) * FWHM_FACTOR,
        "flags": flags,
    }


def find_candidates(plane, offset, pred, spec):
    px, py = offset.native_xy(*pred)
    sky, peaks = local_peaks(plane, px, py, spec.search, spec)
    found = []
    for x, y in peaks:
        result = measure_candidate(plane, x, y, spec)
        if result["state"] == "ok":
            result["ref"] = offset.reference_xy(result["x"], result["y"])
            result["distance"] = math.hypot(result["x"] - px, result["y"] - py)
        found.append(result)
    return found, sky


def static_witness(plane_for, contexts, current, cand, spec):
    ref = cand["ref"]
    avoid = spec.search + spec.fit + 2
    others = [
        ctx for ctx in contexts
        if ctx["frame"] != current["frame"] and ctx["offset"] is not None
        and math.hypot(ctx["pred"][0] - ref[0], ctx["pred"][1] - ref[1]) > avoid
    ]
    if not others:
        return "unknown"
    others.sort(key=lambda ctx: -abs(ctx["t"] - current["t"]))
    checked = 0
    for ctx in others[:3]:
        nx, ny = ctx["offset"].native_xy(*ref)
        plane = plane_for(ctx["frame"])
        height, width = plane.values.shape
        margin = spec.fit + 4
        if not (margin <= nx < width - margin and margin <= ny < height - margin):
            continue
        sky, peaks = local_peaks(plane, nx, ny, 2.5, spec)
        if sky is None:
            continue
        checked += 1
        if peaks:
            return "yes"
    return "no" if checked else "unknown"


def brief_candidate(cand, offset):
    ref = cand.get("ref") or offset.reference_xy(cand["x"], cand["y"])
    return {
        "state": cand["state"],
        "native_x": cand["x"],
        "native_y": cand["y"],
        "ref_x": ref[0],
        "ref_y": ref[1],
        "snr": cand.get("snr"),
        "flux": cand.get("flux"),
        "stationary": cand.get("stationary"),
    }


def measure_frame(plane_for, ctx, contexts, spec):
    frame = ctx["row"]
    row = {
        "frame": ctx["frame"],
        "filename": frame["filename"],
        "started": frame["started"],
        "midpoint": time_midpoint(frame["started"], frame["exposure"]),
        "time_seconds": ctx["t"],
        "exposure": frame["exposure"],
        "band": frame["band"],
        "seed": ctx["seed"],
        "extrapolated": ctx["extrapolated"],
        "alignment": ctx["offset"].source if ctx["offset"] else None,
        "pred_x": ctx["pred"][0],
        "pred_y": ctx["pred"][1],
        "state": REVIEW,
        "reason": None,
        "note": None,
        "measured": None,
        "candidates": [],
    }
    offset = ctx["offset"]
    if offset is None:
        return {**row, "reason": "unaligned"}
    found, sky = find_candidates(plane_for(ctx["frame"]), offset, ctx["pred"], spec)
    if sky is None:
        return {**row, "reason": "no_support"}
    if not found:
        return {**row, "reason": "no_detection"}
    usable = [cand for cand in found if cand["state"] == "ok"]
    if not usable:
        row["candidates"] = [brief_candidate(cand, offset) for cand in found]
        return {**row, "reason": found[0]["state"]}
    for cand in usable:
        cand["stationary"] = static_witness(plane_for, contexts, ctx, cand, spec)
    row["candidates"] = [brief_candidate(cand, offset) for cand in found]
    moving = [cand for cand in usable if cand["stationary"] != "yes"]
    if not moving:
        return {**row, "reason": "only_static" if len(usable) == 1 else "blend_star"}
    if len(moving) > 1:
        return {**row, "reason": "ambiguous"}
    chosen = moving[0]
    if chosen["distance"] > spec.search - 0.6:
        return {**row, "reason": "at_search_edge"}
    if "compact" in chosen["flags"]:
        return {**row, "reason": "compact_pixel"}
    if "broad" in chosen["flags"]:
        return {**row, "reason": "broad"}
    close = [
        other for other in usable if other is not chosen
        and math.hypot(other["x"] - chosen["x"], other["y"] - chosen["y"]) < spec.blend
    ]
    if close:
        return {**row, "reason": "blend_star"}
    note = "dismissed_static" if any(
        other is not chosen and other["stationary"] == "yes" for other in usable
    ) else None
    return {
        **row,
        "state": MEASURED,
        "reason": "measured",
        "note": note,
        "measured": {
            "native_x": chosen["x"],
            "native_y": chosen["y"],
            "ref_x": chosen["ref"][0],
            "ref_y": chosen["ref"][1],
            "snr": chosen["snr"],
            "flux": chosen["flux"],
            "fwhm": chosen["fwhm"],
            "offset_distance": chosen["distance"],
        },
    }


def frame_contexts(frames, document, seeds):
    first, last = seeds
    vx = (last["x"] - first["x"]) / (last["t"] - first["t"])
    vy = (last["y"] - first["y"]) / (last["t"] - first["t"])
    seed_ids = {seed["frame"] for seed in seeds}
    contexts = []
    for frame in frames:
        t = mid_seconds(frame)
        contexts.append({
            "frame": frame["id"],
            "row": frame,
            "t": t,
            "pred": (first["x"] + vx * (t - first["t"]), first["y"] + vy * (t - first["t"])),
            "offset": offset_for(document, frame["id"]),
            "seed": frame["id"] in seed_ids,
            "extrapolated": not first["t"] <= t <= last["t"],
        })
    return contexts, vx, vy


def tracklet_report(images, options):
    document = images.document
    dims = images.dims
    seeds = seed_points(document, dims, options.get("seeds"))
    band = options.get("band") or seeds[0]["band"]
    if any(seed["band"] != band for seed in seeds):
        raise DomainError("种子曝光与所选滤镜不一致")
    frames = usable_science_frames(document, band)
    if not frames:
        raise DomainError("所选滤镜没有可用曝光")
    spec = TrackSpec.request(options)
    if (len(frames) + 6) * dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("逐帧实测超过 2400 万像素预算，请挂起部分曝光")
    contexts, vx, vy = frame_contexts(frames, document, seeds)
    plane_for = plane_cache(images)
    rows = [measure_frame(plane_for, ctx, contexts, spec) for ctx in contexts]
    accepted = [row for row in rows if row["state"] == MEASURED]
    residuals = [row["measured"]["offset_distance"] for row in accepted]
    return {
        "kind": "tracklet",
        "revision": document["revision"],
        "band": band,
        "seeds": [
            {key: seed[key] for key in ("frame", "filename", "x", "y")} for seed in seeds
        ],
        "settings": {
            "search_radius": spec.search, "fit_radius": spec.fit,
            "threshold": spec.threshold, "blend_radius": spec.blend,
        },
        "velocity": {"x_per_second": vx, "y_per_second": vy},
        "summary": {
            "frames": len(rows),
            "measured": len(accepted),
            "review": len(rows) - len(accepted),
            "reasons": dict(Counter(row["reason"] for row in rows if row["state"] == REVIEW)),
            "velocity_px_hour": math.hypot(vx, vy) * 3600,
            "angle_deg": math.degrees(math.atan2(vy, vx)),
            "rms_residual": math.sqrt(sum(d * d for d in residuals) / len(residuals)) if residuals else None,
        },
        "rows": rows,
    }
