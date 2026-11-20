"""Image translations between native detector pixels and a fixed reference frame."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.ndimage import map_coordinates

from .calibration import Plane, robust_summary
from .domain import DomainError, MAX_ACTIVE_PIXELS, number, usable_science_frames


@dataclass(frozen=True)
class Offset:
    dx: float = 0.0
    dy: float = 0.0
    source: str = "reference"

    def native_xy(self, x, y):
        return x + self.dx, y + self.dy

    def reference_xy(self, x, y):
        return x - self.dx, y - self.dy


def offset_for(document, frame_id):
    if frame_id == document["reference"]:
        return Offset()
    entry = document["alignments"].get(frame_id)
    if entry is not None:
        return Offset(entry["dx"], entry["dy"], entry["source"])
    return None


def warp_to_reference(plane, offset):
    if offset.dx == 0 and offset.dy == 0:
        return plane
    height, width = plane.values.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    coordinates = (yy + offset.dy, xx + offset.dx)
    valid_weights = map_coordinates(
        plane.valid.astype(np.float64), coordinates, order=1, mode="constant", cval=0, prefilter=False
    )
    valid = valid_weights > 1 - 1e-7
    saturated = map_coordinates(
        plane.saturated.astype(np.float64), coordinates, order=1, mode="constant", cval=0, prefilter=False
    ) > 1e-7
    values = map_coordinates(
        np.where(plane.valid, plane.values, 0.0), coordinates,
        order=1, mode="constant", cval=0, prefilter=False,
    )
    values[~valid] = np.nan
    return Plane(values, valid, saturated, plane.notes)


def _feature_map(plane):
    valid = plane.valid & ~plane.saturated
    summary = robust_summary(plane.values[valid])
    if summary["count"] < plane.values.size * 0.35:
        raise DomainError("有效像素不足，不能自动对准")
    median = summary["median"]
    sigma = summary["sigma"]
    values = np.maximum(np.where(valid, plane.values - median, 0), 0)
    peaks = np.percentile(values[valid], [97, 99.9])
    floor = max(3 * sigma, peaks[0] * 0.22, 1e-8)
    cap = max(peaks[1], floor + 1e-7)
    features = np.clip(values - floor, 0, cap - floor)
    if np.count_nonzero(features) < 8 or float(np.linalg.norm(features)) <= 0:
        raise DomainError("没有足够稳定的星点用于对准")
    return features


def _center_offset(prev, center, nxt):
    denom = prev - 2 * center + nxt
    if denom >= -1e-12:
        return 0.0
    return float(np.clip(0.5 * (prev - nxt) / denom, -0.5, 0.5))


def _overlap_score(ref, moving, dy, dx):
    h, w = ref.shape
    fy = int(np.clip(round(dy), -h + 1, h - 1))
    fx = int(np.clip(round(dx), -w + 1, w - 1))
    y0 = max(0, -fy)
    x0 = max(0, -fx)
    y1 = min(h, h - fy)
    x1 = min(w, w - fx)
    a = ref[y0:y1, x0:x1]
    b = moving[y0 + fy:y1 + fy, x0 + fx:x1 + fx]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def estimate_translation(reference, exposure, max_shift):
    ref = _feature_map(reference)
    moving = _feature_map(exposure)
    h, w = ref.shape
    max_x = min(int(max_shift), w // 3)
    max_y = min(int(max_shift), h // 3)
    shape = (next_fast_len(2 * h - 1), next_fast_len(2 * w - 1))
    response = irfft2(rfft2(moving, s=shape) * rfft2(ref, s=shape).conj(), s=shape)
    ylags = np.arange(-max_y, max_y + 1)
    xlags = np.arange(-max_x, max_x + 1)
    inner = response[np.ix_(ylags % shape[0], xlags % shape[1])]
    peak_y, peak_x = np.unravel_index(np.argmax(inner), inner.shape)
    raw_y = int(ylags[peak_y])
    raw_x = int(xlags[peak_x])
    height, width = shape
    cy, cx = raw_y % height, raw_x % width
    peak = response[cy, cx]
    dy = raw_y + _center_offset(response[(cy - 1) % height, cx], peak, response[(cy + 1) % height, cx])
    dx = raw_x + _center_offset(response[cy, (cx - 1) % width], peak, response[cy, (cx + 1) % width])
    score = _overlap_score(ref, moving, dy, dx)
    competitor = inner.copy()
    competitor[max(0, peak_y - 3):peak_y + 4, max(0, peak_x - 3):peak_x + 4] = 0
    secondary = max(0.0, float(np.max(competitor)))
    peak_ratio = float(peak) / secondary if secondary else None
    at_edge = abs(raw_x) == max_x or abs(raw_y) == max_y
    credible = bool(score >= 0.36 and (peak_ratio is None or peak_ratio >= 1.1) and not at_edge)
    return {
        "dx": float(dx),
        "dy": float(dy),
        "score": score,
        "peak_ratio": peak_ratio,
        "credible": credible,
        "reason": "shift_limit" if at_edge else "ambiguous" if not credible else "measured",
    }


def run_registration(images, options):
    doc = images.document
    ref = doc["reference"]
    if ref is None:
        raise DomainError("请先导入一组科学曝光并选择基准")
    max_shift = number(options.get("max_shift", 30), "最大偏移", 2, 100)
    only_missing = options.get("only_missing", True)
    if not isinstance(only_missing, bool):
        raise DomainError("保留已录入偏移必须为开关")
    frames = [frame for frame in usable_science_frames(doc) if frame["id"] != ref]
    if (len(frames) + 1) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("请暂时挂起部分曝光，使单次对准图像总量降至 2400 万像素内")
    ref_plane = images.calibrated(ref)
    report = []
    accepted = {}
    for frame in frames:
        frame_id = frame["id"]
        entry = doc["alignments"].get(frame_id)
        if entry is not None and only_missing:
            report.append({"frame": frame_id, "filename": frame["filename"], "state": "retained", **entry})
            continue
        try:
            estimate = estimate_translation(ref_plane, images.calibrated(frame_id), max_shift)
        except DomainError as exc:
            report.append({"frame": frame_id, "filename": frame["filename"], "state": "unavailable", "message": str(exc)})
            continue
        credible = estimate["credible"]
        report.append({
            "frame": frame_id,
            "filename": frame["filename"],
            "state": "measured" if credible else "unreliable",
            **estimate,
        })
        if credible:
            accepted[frame_id] = {
                "dx": estimate["dx"], "dy": estimate["dy"], "source": "automatic",
                "score": estimate["score"], "peak_ratio": estimate["peak_ratio"],
            }
    return {"reference": ref, "rows": report}, accepted


def preview_stream(images, frame_id, stream):
    if stream == "raw":
        return images.raw(frame_id)
    plane = images.calibrated(frame_id)
    if stream == "calibrated":
        return plane
    offset = offset_for(images.document, frame_id)
    if offset is None:
        raise DomainError("本曝光还没有对准偏移")
    return warp_to_reference(plane, offset)
