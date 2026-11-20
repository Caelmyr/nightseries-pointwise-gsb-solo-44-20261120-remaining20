"""星像匹配差分：用两图共同的未饱和星点估计星象，互卷到共同星象后做差。"""
import base64
from io import BytesIO
import math

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, maximum_filter
from scipy.signal import fftconvolve

from .calibration import Plane, robust_summary
from .domain import DomainError
from .preview import render_difference
from .quality import FWHM_FACTOR

STAMP_RADIUS = 10
MIN_PAIRS = 6
MAX_PAIRS = 36
PEAK_THRESHOLD = 5.0
PEAK_SEPARATION = 14
POSITION_TOLERANCE = 1.6
CENTROID_TOLERANCE = 1.2
TRIM_FRACTION = 2e-3


def _detect_peaks(plane, cap=140):
    usable = plane.valid & ~plane.saturated & np.isfinite(plane.values)
    samples = plane.values[usable]
    if samples.size < 400:
        return []
    sky = robust_summary(samples)
    cutoff = sky["median"] + max(PEAK_THRESHOLD * (sky["sigma"] or 0.0), 1e-6)
    working = np.where(usable, plane.values, -np.inf)
    peaks = working == maximum_filter(working, size=3, mode="constant", cval=-np.inf)
    margin = STAMP_RADIUS + 2
    peaks[:margin] = False
    peaks[-margin:] = False
    peaks[:, :margin] = False
    peaks[:, -margin:] = False
    yy, xx = np.where(peaks & (working > cutoff))
    if xx.size == 0:
        return []
    order = np.argsort(-working[yy, xx], kind="stable")
    picked = []
    for index in order:
        x, y = int(xx[index]), int(yy[index])
        if any((x - px) ** 2 + (y - py) ** 2 < PEAK_SEPARATION ** 2 for px, py in picked):
            continue
        picked.append((x, y))
        if len(picked) >= cap:
            break
    return picked


def _pair_peaks(peaks_a, peaks_b):
    pairs = []
    used = set()
    for xa, ya in peaks_a:
        best = None
        best_distance = POSITION_TOLERANCE
        for index, (xb, yb) in enumerate(peaks_b):
            if index in used:
                continue
            distance = math.hypot(xa - xb, ya - yb)
            if distance <= best_distance:
                best = index
                best_distance = distance
        if best is not None:
            used.add(best)
            pairs.append(((xa, ya), peaks_b[best]))
    return pairs


def _star_stamp(plane, x, y):
    radius = STAMP_RADIUS
    window = np.s_[y - radius:y + radius + 1, x - radius:x + radius + 1]
    values = plane.values[window]
    if values.shape != (2 * radius + 1, 2 * radius + 1):
        return None
    if not np.all(plane.valid[window]) or np.any(plane.saturated[window]) or not np.all(np.isfinite(values)):
        return None
    yy, xx = np.indices(values.shape, dtype=np.float64)
    rho = np.hypot(yy - radius, xx - radius)
    sky = robust_summary(values[rho >= radius - 1.5])
    if sky["median"] is None:
        return None
    net = values - sky["median"]
    floor = max(2.5 * (sky["sigma"] or 0.0), 1e-9)
    weights = np.where((rho <= radius - 2) & (net > floor), net, 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        return None
    cx = float(np.sum(xx * weights) / total)
    cy = float(np.sum(yy * weights) / total)
    if math.hypot(cx - radius, cy - radius) > radius * 0.45:
        return None
    gx, gy = xx - cx, yy - cy
    moment = np.array([
        [np.sum(gx ** 2 * weights), np.sum(gx * gy * weights)],
        [np.sum(gx * gy * weights), np.sum(gy ** 2 * weights)],
    ]) / total
    eigen = np.linalg.eigvalsh(moment)
    minor, major = math.sqrt(max(0.0, eigen[0])), math.sqrt(max(0.0, eigen[1]))
    if minor < 0.3 or major > radius * 0.85:
        return None
    flux = float(np.sum(net[rho <= radius - 2]))
    if flux <= 0:
        return None
    coords = (yy + cy - radius, xx + cx - radius)
    centered = map_coordinates(net, coords, order=1, mode="constant", cval=0.0, prefilter=False)
    return {"stamp": centered, "flux": flux, "cx": x - radius + cx, "cy": y - radius + cy}


def _paired_stars(a, b):
    pairs = _pair_peaks(_detect_peaks(a), _detect_peaks(b))
    stars_a, stars_b, ratios = [], [], []
    for (xa, ya), (xb, yb) in pairs:
        star_a = _star_stamp(a, xa, ya)
        star_b = _star_stamp(b, xb, yb)
        if star_a is None or star_b is None:
            continue
        if math.hypot(star_a["cx"] - star_b["cx"], star_a["cy"] - star_b["cy"]) > CENTROID_TOLERANCE:
            continue
        stars_a.append(star_a)
        stars_b.append(star_b)
        ratios.append(star_a["flux"] / star_b["flux"])
        if len(stars_a) >= MAX_PAIRS:
            break
    if len(stars_a) < MIN_PAIRS:
        raise DomainError(f"两图共同可用的未饱和星点不足（{len(stars_a)} 颗，至少需 {MIN_PAIRS} 颗）")
    return stars_a, stars_b, ratios


def _stack_psf(stars):
    normalized = []
    for star in stars:
        total = float(np.sum(star["stamp"]))
        if total > 0:
            normalized.append(star["stamp"] / total)
    if len(normalized) < MIN_PAIRS:
        raise DomainError("共同星点的星象估计失败，请检查星点是否过暗或拖尾过长")
    return np.median(np.stack(normalized), axis=0)


def _finalize_kernel(psf):
    radius = (psf.shape[0] - 1) // 2
    yy, xx = np.indices(psf.shape, dtype=np.float64)
    clipped = np.clip(psf, 0.0, None)
    total = float(np.sum(clipped))
    if total <= 0:
        raise DomainError("星象估计失败：共同星点堆叠没有正流量")
    cy = float(np.sum(yy * clipped) / total)
    cx = float(np.sum(xx * clipped) / total)
    coords = (yy + cy - radius, xx + cx - radius)
    centered = map_coordinates(clipped, coords, order=1, mode="constant", cval=0.0, prefilter=False)
    centered = np.clip(centered, 0.0, None)
    peak = float(np.max(centered))
    if peak <= 0:
        raise DomainError("星象估计失败：共同星点堆叠没有正流量")
    ys, xs = np.where(centered > peak * TRIM_FRACTION)
    reach = min(int(max(np.max(np.abs(ys - radius)), np.max(np.abs(xs - radius)))), radius)
    kernel = centered[radius - reach:radius + reach + 1, radius - reach:radius + reach + 1]
    return kernel / float(np.sum(kernel))


def _convolve_masked(plane, kernel):
    good = plane.valid & ~plane.saturated & np.isfinite(plane.values)
    sums = fftconvolve(np.where(good, plane.values, 0.0), kernel, mode="same")
    support = fftconvolve(good.astype(np.float64), kernel, mode="same")
    full = support >= 1.0 - 1e-6
    values = np.where(full, sums / np.maximum(support, 1e-12), np.nan)
    return values, full


def _kernel_profile(kernel):
    total = float(np.sum(kernel))
    yy, xx = np.indices(kernel.shape, dtype=np.float64)
    cx = float(np.sum(xx * kernel) / total)
    cy = float(np.sum(yy * kernel) / total)
    var_x = float(np.sum((xx - cx) ** 2 * kernel) / total)
    var_y = float(np.sum((yy - cy) ** 2 * kernel) / total)
    return {"fwhm_x": FWHM_FACTOR * math.sqrt(max(var_x, 0.0)), "fwhm_y": FWHM_FACTOR * math.sqrt(max(var_y, 0.0))}


def _data_uri(data):
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _kernel_image(kernel):
    peak = float(np.max(kernel))
    scaled = kernel / peak if peak > 0 else kernel
    stretched = np.arcsinh(scaled * 25) / np.arcsinh(25)
    gray = np.rint(np.clip(stretched, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(gray, "L")
    factor = max(2, 120 // max(1, image.width))
    image = image.resize((image.width * factor, image.height * factor), Image.Resampling.NEAREST)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return _data_uri(buffer.getvalue())


def matched_preview(a, b, normalizer):
    if a.values.shape != b.values.shape:
        raise DomainError("两帧画幅不一致，无法做星像匹配差分")
    stars_a, stars_b, ratios = _paired_stars(a, b)
    psf_a = _finalize_kernel(_stack_psf(stars_a))
    psf_b = _finalize_kernel(_stack_psf(stars_b))
    scale = float(np.median(np.asarray(ratios)))
    if not math.isfinite(scale) or scale <= 0:
        raise DomainError("共同星点的流量比例估计失败")
    matched_a, full_a = _convolve_masked(a, psf_b)
    matched_b, full_b = _convolve_masked(b, psf_a)
    both = full_a & full_b
    if np.count_nonzero(both) < max(20, a.values.size * 0.1):
        raise DomainError("匹配后两帧的共同有效像素不足，请检查坏区与覆盖范围")
    difference = matched_a - matched_b * scale
    shift = float(np.median(difference[both])) if normalizer == "median" else 0.0
    plane = Plane(np.where(both, difference - shift, np.nan), both, np.zeros(a.values.shape, dtype=bool))
    rendered = render_difference(plane)
    common = fftconvolve(psf_a, psf_b, mode="full")
    return {
        "image": _data_uri(rendered["image"]),
        "offset": shift,
        "scale": scale,
        "limit": rendered["limit"],
        "valid": rendered["valid"],
        "stars_used": len(stars_a),
        "common": _kernel_profile(common),
        "sides": {
            "a": {
                "psf": _kernel_profile(psf_a),
                "kernel": _kernel_profile(psf_b),
                "kernel_size": [int(psf_b.shape[1]), int(psf_b.shape[0])],
                "kernel_image": _kernel_image(psf_b),
            },
            "b": {
                "psf": _kernel_profile(psf_b),
                "kernel": _kernel_profile(psf_a),
                "kernel_size": [int(psf_a.shape[1]), int(psf_a.shape[0])],
                "kernel_image": _kernel_image(psf_a),
            },
        },
    }
