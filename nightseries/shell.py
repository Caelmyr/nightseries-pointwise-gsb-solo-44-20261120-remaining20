"""Measure one shell on two sides of a center with adjacent radial profiles.

Every ray is measured on its own: its background, its peak, and its inner and
outer rise/return crossings.  A side is summarised by robust medians of the
per-ray positions and per-ray widths.  The profiles are never averaged into a
single trace, so a shell whose radius shifts from ray to ray keeps that shift
as a spread instead of being flattened into one thicker shell.

A ray that is cut short by the frame border, a masked bad region, saturation or
a gap keeps whichever bright edge it actually crossed and is reported as
width-incomplete rather than having the missing side guessed.  A ray carrying a
star or bad patch is rejected by its disagreement with its neighbouring rays;
because the comparison is made ray against ray, a smooth radius or thickness
change across the fan passes while a single lifted or displaced flank does not.
"""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d, map_coordinates

from .calibration import robust_summary
from .domain import DomainError, dimensions_of, enum, exposure_row, identifier, integer, number
from .preview import png, stretch_pixels

SHELL_STREAMS = {"raw", "calibrated"}
MAX_PROFILES = 15

# Where a ray stopped before returning to background.
CUT_NONE = None
CUT_FRAME = "frame"
CUT_CENTER = "center"
CUT_GAP = "gap"
CUT_SATURATED = "saturated"

# Per-ray states.
RAY_COMPLETE = "measured"
RAY_INCOMPLETE = "width_incomplete"
RAY_INNER_MISSING = "partial_inner"
RAY_OUTER_MISSING = "partial_outer"
RAY_NARROW = "narrow"
RAY_REJECTED = "rejected"
RAY_EDGES = {RAY_COMPLETE, RAY_INCOMPLETE, RAY_INNER_MISSING, RAY_OUTER_MISSING, RAY_NARROW}

# Rejection reasons.
STAR_SPIKE = "point_source"
WIDTH_OUTLIER = "width_outlier"

# Side states.
SIDE_MEASURED = "measured"
SIDE_FEW = "few_profiles"
SIDE_INCOMPLETE = "width_incomplete"
SIDE_NONE = "no_shell"


@dataclass(frozen=True)
class ShellRequest:
    frame: dict
    stream: str
    center: tuple
    angles: tuple
    profiles: int
    spread: float
    step: float
    edge_fraction: float
    min_snr: float
    background_gap: float
    min_width: float
    search_radius: float

    @classmethod
    def requested(cls, document, options):
        dims = dimensions_of(document)
        chosen = options.get("frame") or document["reference"]
        if not chosen:
            raise DomainError("请先指定一张科学曝光")
        frame = exposure_row(document, identifier(chosen, "环壳曝光"))
        if frame["kind"] != "science":
            raise DomainError("环壳剖面只测量科学曝光")
        center = dims.xy(
            number(options.get("center_x"), "中心横坐标", 0, dims.width - 1),
            number(options.get("center_y"), "中心纵坐标", 0, dims.height - 1),
        )
        angles = tuple(
            number(options.get(name, default), label, -720, 720) % 360
            for name, label, default in (("angle_a", "方向 A 的角度", 0), ("angle_b", "方向 B 的角度", 180))
        )
        profiles = integer(options.get("profiles", 5), "每侧相邻剖面数", 1, MAX_PROFILES)
        if profiles % 2 == 0:
            raise DomainError("每侧相邻剖面数需为奇数，好让标称方向本身落在一条剖面上")
        spread = number(options.get("spread", 4), "相邻剖面张角", 0, 45)
        if profiles > 1 and spread < 0.05:
            raise DomainError("有多条相邻剖面时，张角需大于 0")
        return cls(
            frame,
            enum(options.get("stream", "calibrated"), "环壳影像层", SHELL_STREAMS),
            center,
            angles,
            profiles,
            spread,
            number(options.get("step", 0.5), "径向步长", 0.1, 5),
            number(options.get("edge_fraction", 0.5), "边缘阈值", 0.05, 0.95),
            number(options.get("min_snr", 5), "最低峰值信噪比", 1, 50),
            number(options.get("background_gap", 10), "峰外背景起点", 0, 200),
            number(options.get("min_width", 1.5), "最短亮段", 0, 50),
            number(options.get("search_radius", 0), "首轮峰值搜索半径", 0, 2000),
        )

    def ray_angles(self, nominal):
        if self.profiles == 1:
            return [(nominal, 0.0)]
        offsets = np.linspace(-self.spread, self.spread, self.profiles)
        return [((nominal + offset) % 360, float(offset)) for offset in offsets]


def plane_for(images, request):
    if request.stream == "calibrated":
        return images.calibrated(request.frame["id"])
    return images.raw(request.frame["id"], True)


# ------------------------------------------------------------------ sampling


def _sample_ray(plane, center, angle, step):
    """Sample one ray to the frame border; return radii, values, usability and cut reasons."""
    height, width = plane.values.shape
    cx, cy = center
    theta = math.radians(angle)
    direction = (math.cos(theta), math.sin(theta))
    limits = []
    for offset, start, upper in ((direction[0], cx, width - 1), (direction[1], cy, height - 1)):
        if offset > 1e-12:
            limits.append((upper - start) / offset)
        elif offset < -1e-12:
            limits.append((0 - start) / offset)
    reach = min([value for value in limits if value >= 0], default=0.0)
    if not math.isfinite(reach) or reach <= 0:
        return None
    samples = int(math.floor(reach / step)) + 1
    if samples < 6:
        return None
    radii = np.arange(samples, dtype=float) * step
    coordinates = (cy + direction[1] * radii, cx + direction[0] * radii)
    support = map_coordinates(
        plane.valid.astype(float), coordinates, order=1, mode="constant", cval=0, prefilter=False
    )
    saturated = map_coordinates(
        plane.saturated.astype(float), coordinates, order=1, mode="constant", cval=0, prefilter=False
    )
    values = map_coordinates(
        np.where(plane.valid, plane.values, 0.0), coordinates,
        order=1, mode="constant", cval=0, prefilter=False,
    )
    usable = (support > 1 - 1e-7) & (saturated <= 1e-7) & np.isfinite(values)
    # The nearest pixel tells a masked/blank pixel apart from a saturated one.
    nearest_saturated = map_coordinates(
        plane.saturated.astype(float), coordinates, order=0, mode="nearest", prefilter=False
    ) > 0.5
    cut = np.where(usable, "", np.where(nearest_saturated, CUT_SATURATED, CUT_GAP))
    return radii, values, usable, cut, float(reach)


def _smooth(values, usable, step, sigma=None):
    """Normalised Gaussian trace; positions with little support stay NaN.

    The default kernel is narrow enough that a shell only a few pixels thick is
    not visibly broadened, yet wide enough on a 0.1 px grid to suppress sampling
    noise.  It is capped in samples so a coarse step does not over-smooth.
    """
    if sigma is None:
        sigma = max(0.6, min(4.0, 0.4 / step))
    numerator = gaussian_filter1d(np.where(usable, values, 0.0), sigma, mode="constant", cval=0)
    support = gaussian_filter1d(usable.astype(float), sigma, mode="constant", cval=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        smooth = numerator / support
    smooth[support < 0.6] = np.nan
    # The kernel has a fixed width in pixels, so a sample-count sigma is scaled
    # back to pixels before working out how much white noise it suppresses.
    width_px = sigma * step
    return smooth, support, math.sqrt(1.0 / (2.0 * math.sqrt(math.pi) * width_px))


def _background(values, usable, radii, peak_radius, gap):
    """Sky level away from the peak, preferring the outer side and falling back inward."""
    outer = usable & (radii >= peak_radius + gap)
    inner = usable & (radii <= max(0.0, peak_radius - gap)) & (radii <= 0.6 * peak_radius)
    if np.count_nonzero(outer) >= 8:
        sample, source = values[outer], "outer"
    elif np.count_nonzero(inner) >= 8:
        sample, source = values[inner], "inner"
    else:
        pool = values[usable]
        if pool.size < 8:
            return None
        lower = pool[pool <= np.median(pool)]
        sample = lower if lower.size >= 8 else pool
        source = "lower_half"
    level = float(np.median(sample))
    sigma = 1.482602218505602 * float(np.median(np.abs(sample - level)))
    return {"level": level, "sigma": sigma, "source": source, "count": int(sample.size)}


def _crossing(radii, smooth, usable, cut, start, threshold, direction, evidence=None):
    """Walk away from the peak until the threshold is crossed or the ray is cut.

    ``evidence`` is the low-support kernel sum; a crossing without enough
    support behind it is not a measured edge, it is the boundary of the data.
    """
    last = start
    index = start + direction
    while 0 <= index < radii.size:
        if not usable[index] or not np.isfinite(smooth[index]):
            return None, cut[index] or CUT_GAP
        if smooth[index] < threshold:
            if evidence is not None and evidence[index] < 0.6:
                return None, cut[index] or CUT_GAP
            high, low = float(smooth[last]), float(smooth[index])
            fraction = 0.5 if high <= low else (high - threshold) / (high - low)
            fraction = min(1.0, max(0.0, fraction))
            return float(radii[last] + (radii[index] - radii[last]) * fraction), CUT_NONE
        last, index = index, index + direction
    return None, CUT_FRAME if direction > 0 else CUT_CENTER


def _peak_offset(before, middle, after):
    """Parabolic sub-step peak position, in steps."""
    denominator = before - 2 * middle + after
    if not math.isfinite(denominator) or denominator >= -1e-12:
        return 0.0
    return float(np.clip(0.5 * (before - after) / denominator, -0.75, 0.75))


def _measure_ray(plane, center, angle, offset, settings, window=None):
    """Measure one ray.  ``window`` = (radius, half-width) limits the peak search
    so a star elsewhere on the ray cannot win it."""
    record = {
        "angle": round(angle, 3),
        "offset_deg": round(offset, 3),
        "state": "no_signal",
        "reject_reason": None,
        "ray_length": None,
        "peak_radius": None,
        "peak_adu": None,
        "background_adu": None,
        "background_sigma": None,
        "background_source": None,
        "amplitude": None,
        "snr": None,
        "inner_edge": None,
        "outer_edge": None,
        "inner_cut": None,
        "outer_cut": None,
        "width": None,
        "valid_fraction": None,
        "trace": None,
    }
    sampled = _sample_ray(plane, center, angle, settings.step)
    if sampled is None:
        return {**record, "state": "too_short"}
    radii, values, usable, cut, reach = sampled
    record["ray_length"] = reach
    record["valid_fraction"] = float(np.mean(usable))
    if np.count_nonzero(usable) < 8:
        return {**record, "state": "no_support"}
    smooth, support, kernel = _smooth(values, usable, settings.step)
    stride = max(1, int(math.ceil(radii.size / 160)))
    record["trace"] = [
        {"r": round(float(radii[index]), 2), "v": None if not np.isfinite(smooth[index]) else round(float(smooth[index]), 2)}
        for index in range(0, radii.size, stride)
    ]
    margin = max(1, int(round(1.0 / settings.step)))
    low, high = margin, radii.size - margin
    if window is not None:
        seed, half = window
        low = max(low, int(np.searchsorted(radii, seed - half)))
        high = min(high, int(np.searchsorted(radii, seed + half, side="right")))
    if high - low < 2 or not np.any(np.isfinite(smooth[low:high])):
        return {**record, "state": "too_short"}
    window_values = smooth[low:high]
    peak = low + int(np.nanargmax(np.where(np.isfinite(window_values), window_values, -np.inf)))
    if window is not None and (peak <= low or peak >= high - 1):
        # The feature sits outside the supplied window: fall back to the whole ray
        # rather than reporting a clipped peak.  A star found this way is rejected later.
        if seed - half <= radii[0]:
            return {**record, "state": "too_short"}
        return _measure_ray(plane, center, angle, offset, settings)
    record["peak_radius"] = float(radii[peak])
    if 0 < peak < radii.size - 1:
        neighbours = (smooth[peak - 1], smooth[peak], smooth[peak + 1])
        if all(np.isfinite(neighbours)):
            record["peak_radius"] = float(radii[peak] + _peak_offset(*neighbours) * settings.step)
    record["peak_adu"] = float(smooth[peak])
    background = _background(values, usable, radii, radii[peak], settings.background_gap)
    if background is None:
        return {**record, "state": "no_background"}
    amplitude = float(smooth[peak]) - background["level"]
    record.update(
        background_adu=background["level"],
        background_sigma=background["sigma"],
        background_source=background["source"],
        amplitude=amplitude,
        snr=amplitude / max(1e-9, background["sigma"] * kernel),
    )
    if amplitude <= 0 or record["snr"] < settings.min_snr:
        return {**record, "state": "no_signal"}
    threshold = background["level"] + settings.edge_fraction * amplitude
    inner, inner_cut = _crossing(radii, smooth, usable, cut, peak, threshold, -1, support)
    outer, outer_cut = _crossing(radii, smooth, usable, cut, peak, threshold, 1, support)
    record.update(inner_edge=inner, outer_edge=outer, inner_cut=inner_cut, outer_cut=outer_cut)
    if inner is None and outer is None:
        return {**record, "state": "no_edges"}
    if inner is None:
        return {**record, "state": RAY_INNER_MISSING}
    if outer is None:
        return {**record, "state": RAY_OUTER_MISSING}
    record["width"] = float(outer - inner)
    inside = np.s_[int(np.searchsorted(radii, inner)):int(np.searchsorted(radii, outer, side="right"))]
    if not np.all(usable[inside]):
        return {**record, "state": RAY_INCOMPLETE}
    if record["width"] < settings.min_width:
        return {**record, "state": RAY_NARROW}
    return {**record, "state": RAY_COMPLETE}


def _robust_median(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if values.size == 0:
        return None
    for _ in range(4):
        middle = float(np.median(values))
        spread = 1.482602218505602 * float(np.median(np.abs(values - middle)))
        keep = np.abs(values - middle) <= max(1.0, 3.0 * spread)
        if np.count_nonzero(keep) < 2 or np.count_nonzero(keep) == values.size:
            break
        values = values[keep]
    return float(np.median(values))


def _reject_rays(rays, snr_floor):
    """Drop rays whose bright segment disagrees with the rest of the fan.

    A star, hot pixel or bad patch either outshines the shell, squeezes the
    segment it brightens, or lands on a wholly different radius.  The reference
    is a trimmed median over the fan, so it follows a shell that drifts or
    thickens smoothly across the fan; median-like statistics also keep one
    noisy ray from voting itself in.  Only noise-level peaks are exempt from the
    radius comparison, because their peak position is not meaningful; they are
    held back by the minimum peak signal-to-noise (``snr_floor``) instead.
    """
    candidates = [row for row in rays if row["state"] in RAY_EDGES]
    alive = list(candidates)
    for _ in range(3):
        amplitudes = _robust_median(row["amplitude"] for row in alive)
        widths = _robust_median(row["width"] for row in alive)
        radii = _robust_median(row["peak_radius"] for row in alive if (row["snr"] or 0) >= snr_floor)
        rejected = 0
        for row in alive:
            if amplitudes and row["amplitude"] is not None and row["amplitude"] > 2.0 * amplitudes:
                row["state"] = RAY_REJECTED
                row["reject_reason"] = STAR_SPIKE
            elif radii is not None and row["peak_radius"] is not None and (row["snr"] or 0) >= snr_floor \
                    and abs(row["peak_radius"] - radii) > max(3.0, 0.1 * radii):
                # A different radius from the rest of the fan: a star or patch won the peak.
                row["state"] = RAY_REJECTED
                row["reject_reason"] = STAR_SPIKE
            elif widths and row["width"] is not None and abs(row["width"] - widths) > max(0.25 * widths, 1.0):
                row["state"] = RAY_REJECTED
                row["reject_reason"] = WIDTH_OUTLIER
            else:
                continue
            rejected += 1
        if not rejected:
            break
        alive = [row for row in alive if row["state"] in RAY_EDGES]


def _seed_from_peaks(rays):
    """Robust shared peak radius from the rays, ignoring star-shifted candidates."""
    peaks = [row["peak_radius"] for row in rays if row["peak_radius"] is not None]
    if len(peaks) < 2:
        return None, None
    values = np.asarray(peaks, dtype=float)
    middle = float(np.median(values))
    for _ in range(4):
        spread = 1.482602218505602 * float(np.median(np.abs(values - middle)))
        keep = np.abs(values - middle) <= max(2.0, 3.0 * spread)
        if np.count_nonzero(keep) < max(2, len(values) - 2) or np.count_nonzero(keep) == values.size:
            break
        values = values[keep]
        middle = float(np.median(values))
    if values.size < 2:
        return None, None
    spread = 1.482602218505602 * float(np.median(np.abs(values - middle)))
    return float(middle), float(spread)


def _measure_side(plane, nominal, settings):
    fans = settings.ray_angles(nominal)
    rays = [
        _measure_ray(plane, settings.center, angle, offset, settings)
        for angle, offset in fans
    ]
    if settings.search_radius > 0:
        # A supplied radius pins every ray to the same feature.
        window = (settings.search_radius, max(8.0, 0.6 * settings.search_radius))
    else:
        seed, spread = _seed_from_peaks(rays)
        window = None if seed is None else (seed, min(0.5 * seed, max(6.0, 4.0 * spread + 4.0)))
    if window is not None:
        # Re-measure inside the shared annulus so a star elsewhere on a ray cannot
        # win the peak; the shell under it is then measured on its own terms.
        rays = [
            _measure_ray(plane, settings.center, angle, offset, settings, window)
            for angle, offset in fans
        ]
    _reject_rays(rays, settings.min_snr)
    return rays


# ------------------------------------------------------------------ summary


def _cut_notes(rays):
    notes = []
    labels = {CUT_FRAME: "画幅切边", CUT_GAP: "坏区截断", CUT_SATURATED: "饱和截断", CUT_CENTER: "内侧未回背景"}
    for row in rays:
        for cut in (row["inner_cut"], row["outer_cut"]):
            if cut in labels and labels[cut] not in notes:
                notes.append(labels[cut])
    return notes


def _side_summary(key, nominal, rays, settings):
    used = [row for row in rays if row["state"] in RAY_EDGES]
    clean = [row["width"] for row in used if row["state"] == RAY_COMPLETE]
    peaks = [row["peak_radius"] for row in used if row["peak_radius"] is not None]
    inner_edges = [row["inner_edge"] for row in used if row["inner_edge"] is not None]
    outer_edges = [row["outer_edge"] for row in used if row["outer_edge"] is not None]
    peak_figure = robust_summary(peaks)
    inner_figure = robust_summary(inner_edges)
    outer_figure = robust_summary(outer_edges)
    width_figure = robust_summary(clean)
    measured_edges = [
        name for name, figure in (("inner", inner_figure), ("outer", outer_figure)) if figure["count"]
    ]
    # A width comes only from rays whose bright segment returned to background on
    # both sides and is at least the shortest believable shell.  One stray noise
    # sliver in a wide fan is reported as such, not as a measured width.
    needed = 2 if settings.profiles > 1 else 1
    if width_figure["count"] >= needed and width_figure["median"] >= settings.min_width:
        state = SIDE_MEASURED
    elif width_figure["count"] and width_figure["median"] >= settings.min_width:
        state = SIDE_FEW
    elif measured_edges:
        state = SIDE_INCOMPLETE
    else:
        state = SIDE_NONE
    notes = _cut_notes(used)
    if state == SIDE_INCOMPLETE:
        edges = "与".join("内缘" if edge == "inner" else "外缘" for edge in measured_edges)
        notes.append(f"仅{edges}可量，宽度未量全")
    if state == SIDE_FEW:
        notes.append("仅 1 条完整剖面给出宽度")
    if settings.profiles == 1:
        notes.append("每侧仅 1 条剖面，无法与相邻剖面互校")
    rejected = [row for row in rays if row["state"] == RAY_REJECTED]
    if rejected:
        reasons = {row["reject_reason"] for row in rejected}
        detail = "星点或坏点" if STAR_SPIKE in reasons else "与相邻剖面宽度不一致"
        notes.append(f"{len(rejected)} 条剖面已剔除（{detail}）")
    return {
        "key": key,
        "angle": round(nominal, 3),
        "state": state,
        "radius": peak_figure["median"],
        "radius_spread": peak_figure["sigma"],
        "inner_edge": inner_figure["median"],
        "outer_edge": outer_figure["median"],
        "width": width_figure["median"],
        "width_spread": width_figure["sigma"],
        "measured_edges": measured_edges,
        "profiles": settings.profiles,
        "used_profiles": len(used),
        "complete_profiles": width_figure["count"],
        "partial_profiles": sum(
            1 for row in used if row["state"] in {RAY_INNER_MISSING, RAY_OUTER_MISSING, RAY_INCOMPLETE, RAY_NARROW}
        ),
        "rejected_profiles": len(rejected),
        "note": " · ".join(notes),
        "rays": [{**row, "trace": None} for row in rays],
        "trace": _side_trace(rays),
    }


def _side_trace(rays):
    """Trace nearest the nominal direction, preferring a ray that still counts."""
    ordered = sorted(rays, key=lambda row: (row["state"] not in RAY_EDGES, abs(row["offset_deg"])))
    middle = ordered[0]
    return {
        "angle": middle["angle"],
        "state": middle["state"],
        "reject_reason": middle["reject_reason"],
        "inner_edge": middle["inner_edge"],
        "outer_edge": middle["outer_edge"],
        "background_adu": middle["background_adu"],
        "points": middle["trace"] or [],
    }


def _comparison(sides):
    first, second = sides
    a, b = first["width"], second["width"]
    if first["state"] != SIDE_MEASURED or second["state"] != SIDE_MEASURED:
        return {
            "width_a": a,
            "width_b": b,
            "difference": None,
            "ratio": None,
            "state": "incomplete",
            "note": "至少一侧宽度未量全或只有单条剖面，两侧差值暂不可用",
        }
    return {
        "width_a": a,
        "width_b": b,
        "difference": b - a,
        "ratio": b / a if a else None,
        "state": "measured",
        "note": "",
    }


def shell_profile(images, options):
    settings = ShellRequest.requested(images.document, options)
    plane = plane_for(images, settings)
    sides = [
        _side_summary(key, nominal, _measure_side(plane, nominal, settings), settings)
        for key, nominal in zip(("A", "B"), settings.angles)
    ]
    return {
        "revision": images.document["revision"],
        "frame": settings.frame["id"],
        "filename": settings.frame["filename"],
        "stream": settings.stream,
        "center": {"x": settings.center[0], "y": settings.center[1]},
        "profiles_per_side": settings.profiles,
        "spread_deg": settings.spread,
        "step_px": settings.step,
        "threshold_fraction": settings.edge_fraction,
        "min_snr": settings.min_snr,
        "min_width_px": settings.min_width,
        "background_gap_px": settings.background_gap,
        "search_radius_px": settings.search_radius,
        "directions": sides,
        "comparison": _comparison(sides),
    }, plane


# ------------------------------------------------------------------ rendering

SIDE_COLORS = {"A": (124, 226, 216), "B": (239, 186, 117)}
PARTIAL = (247, 195, 134)
NARROW = (197, 151, 219)
REJECTED = (150, 112, 164)
UNUSED = (96, 108, 128)
RAY_SHADES = {
    RAY_COMPLETE: None,
    RAY_INCOMPLETE: PARTIAL,
    RAY_INNER_MISSING: PARTIAL,
    RAY_OUTER_MISSING: PARTIAL,
    RAY_NARROW: NARROW,
    RAY_REJECTED: REJECTED,
}


def _perpendicular(point, angle, length):
    theta = math.radians(angle)
    dx, dy = -math.sin(theta) * length, math.cos(theta) * length
    x, y = point
    return x - dx, y - dy, x + dx, y + dy


def draw_shell(plane, result, stretch):
    gray, _ = stretch_pixels(plane, stretch)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[~plane.valid] = [57, 21, 60]
    rgb[plane.valid & plane.saturated] = [253, 167, 77]
    image = Image.fromarray(rgb, "RGB")
    image.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
    scale = image.width / plane.values.shape[1]
    factor = max(1, round(scale))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=12)

    def point(radius, angle):
        theta = math.radians(angle)
        return (
            (result["center"]["x"] + math.cos(theta) * radius) * scale,
            (result["center"]["y"] + math.sin(theta) * radius) * scale,
        )

    cx, cy = point(0, 0)
    draw.line((cx - 5 * factor, cy, cx + 5 * factor, cy), fill=(236, 241, 255), width=1)
    draw.line((cx, cy - 5 * factor, cx, cy + 5 * factor), fill=(236, 241, 255), width=1)

    for side in result["directions"]:
        side_color = SIDE_COLORS[side["key"]]
        # The fan is contiguous in offsets from the nominal direction, so ordering by
        # offset keeps an arc from wrapping the long way round when it crosses 0°.
        drawn = sorted(
            (row for row in side["rays"] if row["state"] in RAY_EDGES),
            key=lambda row: row["offset_deg"],
        )
        for radius in (side["inner_edge"], side["outer_edge"]):
            if radius is None or not drawn:
                continue
            path = [point(radius, row["angle"]) for row in drawn]
            if len(path) >= 2:
                draw.line(path, fill=side_color, width=max(1, factor))

        for ray in side["rays"]:
            state = ray["state"]
            shade = side_color if state in {RAY_COMPLETE, RAY_INNER_MISSING, RAY_OUTER_MISSING} else RAY_SHADES.get(state, UNUSED)
            weight = max(1, round(1.4 * factor)) if state in RAY_EDGES else 1
            draw.line((point(0, ray["angle"]), point(ray["ray_length"] or 0, ray["angle"])), fill=shade, width=weight)
            if state in RAY_EDGES:
                for radius in (ray["inner_edge"], ray["outer_edge"]):
                    if radius is not None:
                        draw.line(_perpendicular(point(radius, ray["angle"]), ray["angle"], 3.2 * factor),
                                  fill=shade, width=max(1, factor))
                if ray["peak_radius"] is not None:
                    px, py = point(ray["peak_radius"], ray["angle"])
                    draw.ellipse((px - 1.6 * factor, py - 1.6 * factor, px + 1.6 * factor, py + 1.6 * factor),
                                 fill=shade)

        theta = math.radians(side["angle"])
        anchor = (side["outer_edge"] or side["radius"] or 0) + 8
        tx = (result["center"]["x"] + math.cos(theta) * anchor) * scale
        ty = (result["center"]["y"] + math.sin(theta) * anchor) * scale
        tx = min(max(tx, 2), max(2, image.width - 74))
        ty = min(max(ty - 8, 2), max(2, image.height - 16))
        if side["width"] is not None:
            label = f"{side['key']} r={side['radius']:.1f} w={side['width']:.1f}"
        elif side["radius"] is not None:
            label = f"{side['key']} r={side['radius']:.1f} width incomplete"
        else:
            label = f"{side['key']} no shell"
        draw.text((tx + 4, ty), label, fill=side_color, font=font)
    return png(image)


# --------------------------------------------------------------------- CSV


def shell_csv(result):
    from .exports import csv_bytes

    columns = (
        "kind", "side", "angle_deg", "profile_offset_deg", "state", "reject_reason",
        "radius_px", "radius_spread_px", "inner_edge_px", "outer_edge_px", "width_px",
        "width_spread_px", "snr", "background_adu", "background_source", "inner_cut",
        "outer_cut", "profiles", "used_profiles", "complete_profiles", "partial_profiles",
        "rejected_profiles", "measured_edges", "note", "frame", "stream", "center_x", "center_y",
    )
    rows = []
    for side in result["directions"]:
        common = [result["filename"], result["stream"], result["center"]["x"], result["center"]["y"]]
        rows.append([
            "side", side["key"], side["angle"], None, side["state"], None,
            side["radius"], side["radius_spread"], side["inner_edge"], side["outer_edge"],
            side["width"], side["width_spread"], None, None, None, None, None,
            side["profiles"], side["used_profiles"], side["complete_profiles"],
            side["partial_profiles"], side["rejected_profiles"],
            " ".join(side["measured_edges"]), side["note"], *common,
        ])
        for ray in side["rays"]:
            rows.append([
                "profile", side["key"], ray["angle"], ray["offset_deg"], ray["state"],
                ray["reject_reason"], ray["peak_radius"], None, ray["inner_edge"],
                ray["outer_edge"], ray["width"], None, ray["snr"], ray["background_adu"],
                ray["background_source"], ray["inner_cut"], ray["outer_cut"],
                None, None, None, None, None, None, "", *common,
            ])
    return csv_bytes(columns, rows)
