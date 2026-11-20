"""Display stretches and detector readouts that leave the measurement plane intact."""
from io import BytesIO
import math

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from .calibration import Plane, robust_summary
from .domain import DomainError, number, stretch_value


def percentiles(plane, settings):
    samples = plane.values[plane.valid & np.isfinite(plane.values) & ~plane.saturated]
    if samples.size < 8:
        samples = plane.values[plane.valid & np.isfinite(plane.values)]
    if samples.size == 0:
        return 0.0, 1.0
    sample = samples[::max(1, samples.size // 400000)]
    low, high = np.percentile(sample, [settings["low"], settings["high"]])
    if high <= low:
        high = low + max(abs(low) * 1e-5, 1.0)
    return float(low), float(high)


def stretch_pixels(plane, settings):
    bounds = percentiles(plane, settings)
    scaled = np.clip((np.where(plane.valid, plane.values, bounds[0]) - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
    if settings["mode"] == "asinh":
        scaled = np.arcsinh(scaled * 10) / np.arcsinh(10)
    elif settings["mode"] == "log":
        scaled = np.log1p(scaled * 30) / np.log1p(30)
    if settings["invert"]:
        scaled = 1 - scaled
    return np.rint(scaled * 255).astype(np.uint8), bounds


def png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def render_plane(plane, settings, *, max_side=1000, show_flags=False):
    check = stretch_value(settings)
    gray, bounds = stretch_pixels(plane, check)
    if show_flags:
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        rgb[~plane.valid] = [57, 21, 60]
        rgb[plane.valid & plane.saturated] = [253, 167, 77]
        image = Image.fromarray(rgb, "RGB")
    else:
        gray[~plane.valid] = 12
        image = Image.fromarray(gray, "L")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return png(image), bounds


def inspect_pixel(plane, x, y):
    height, width = plane.values.shape
    x = int(round(number(x, "横坐标", 0, width - 1)))
    y = int(round(number(y, "纵坐标", 0, height - 1)))
    valid = bool(plane.valid[y, x])
    patch = np.s_[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    summary = robust_summary(plane.values[patch][plane.valid[patch]])
    return {
        "x": x,
        "y": y,
        "valid": valid,
        "saturated": bool(plane.saturated[y, x]),
        "value": float(plane.values[y, x]) if valid else None,
        "nearby": summary,
    }


def sample_line(plane, x0, y0, x1, y1):
    height, width = plane.values.shape
    x0 = number(x0, "起点 X", 0, width - 1)
    y0 = number(y0, "起点 Y", 0, height - 1)
    x1 = number(x1, "终点 X", 0, width - 1)
    y1 = number(y1, "终点 Y", 0, height - 1)
    distance = math.hypot(x1 - x0, y1 - y0)
    if distance < 1:
        raise DomainError("剖面线至少长 1 像素")
    count = min(2048, max(2, math.ceil(distance) + 1))
    line = (np.linspace(y0, y1, count), np.linspace(x0, x1, count))
    support = map_coordinates(plane.valid.astype(float), line, order=1, mode="constant", cval=0, prefilter=False)
    measured = map_coordinates(np.where(plane.valid, plane.values, 0), line, order=1, mode="constant", cval=0, prefilter=False)
    saturated = map_coordinates(plane.saturated.astype(float), line, order=1, mode="constant", cval=0, prefilter=False)
    rows = []
    for index, (good, sample) in enumerate(zip(support, measured)):
        rows.append({
            "distance": index * distance / (count - 1),
            "x": float(line[1][index]),
            "y": float(line[0][index]),
            "value": float(sample) if good > 0.999999 else None,
            "saturated": bool(saturated[index] > 1e-7),
        })
    return {"length": distance, "rows": rows}


def raster_histogram(plane, samples=64):
    valid = plane.valid & ~plane.saturated
    good_data = plane.values[valid]
    if good_data.size == 0:
        return {"edges": [], "counts": [], "valid": 0, "underflow": 0, "overflow": 0}
    bounds = np.percentile(good_data, [0.1, 99.9])
    low, high = float(bounds[0]), float(bounds[1])
    if high <= low:
        high = low + 1
    hist, edges = np.histogram(good_data, bins=samples, range=(low, high))
    return {
        "edges": edges.tolist(),
        "counts": hist.tolist(),
        "valid": int(good_data.size),
        "underflow": int(np.count_nonzero(good_data < low)),
        "overflow": int(np.count_nonzero(good_data > high)),
        "saturated": int(np.count_nonzero(plane.valid & plane.saturated)),
        "masked": int(np.count_nonzero(~plane.valid)),
    }


def difference_plane(a, b, normalizer="median"):
    minimum = max(20, a.values.size * 0.15)
    valid = a.valid & b.valid & ~a.saturated & ~b.saturated
    if np.count_nonzero(valid) < minimum:
        raise DomainError("两帧的共同有效像素不足以显示差分")
    av, bv = a.values[valid], b.values[valid]
    shift = float(np.median(av - bv)) if normalizer == "median" else 0.0
    values = np.where(valid, a.values - b.values - shift, np.nan)
    return Plane(values, valid, np.zeros(a.values.shape, dtype=bool)), shift


def render_difference(plane):
    samples = plane.values[plane.valid]
    amplitude = float(np.percentile(np.abs(samples), 99.5))
    amplitude = max(amplitude, 0.1)
    scaled = np.clip(np.where(plane.valid, plane.values, 0.0) / amplitude, -1, 1)
    base = np.repeat((18 + 45 * np.abs(scaled))[:, :, None], 3, axis=2)
    pos = np.maximum(scaled, 0)
    neg = -np.minimum(scaled, 0)
    rgb = base + np.stack((185 * pos + 18 * neg, 80 * pos + 75 * neg, 19 * pos + 180 * neg), axis=2)
    rgb[~plane.valid] = [38, 24, 47]
    image = Image.fromarray(np.rint(rgb).astype(np.uint8), "RGB")
    image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    return {"image": png(image), "limit": amplitude, "valid": int(samples.size)}


def difference_preview(a, b, normalizer):
    plane, shift = difference_plane(a, b, normalizer)
    return {"offset": shift, **render_difference(plane)}
