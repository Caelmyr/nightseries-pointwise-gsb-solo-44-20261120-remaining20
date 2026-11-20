"""Compare repeated calibration exposures to find detector defects."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, find_objects, label, median_filter

from .calibration import ObservationImages, finite_median, robust_summary
from .domain import DomainError, enum, integer, number
from .preview import png

DETECTOR_KINDS = {"hot", "unstable", "low_response"}
MIN_CALIBRATION_FRAMES = 3
MONITOR_PIXELS = 14_000_000


@dataclass(frozen=True)
class MonitorSettings:
    kind: str
    minimum: int
    fraction: float
    threshold: float
    band: str
    padding: int

    @classmethod
    def requested(cls, options):
        kind = enum(options.get("kind", "hot"), "坏点检查类型", DETECTOR_KINDS)
        defaults = {"hot": 15, "unstable": 12, "low_response": 0.55}
        bounds = (0.05, 0.95) if kind == "low_response" else (0.01, 10000)
        band = options.get("band", "")
        if not isinstance(band, str) or len(band) > 24:
            raise DomainError("平场滤镜标签无效")
        return cls(
            kind=kind,
            minimum=integer(options.get("minimum", 3), "最低采样曝光数", 3, 48),
            fraction=number(options.get("fraction", 0.75), "持续比例", 0.5, 1),
            threshold=number(options.get("threshold", defaults[kind]), "候选判定阈值", *bounds),
            band=band,
            padding=integer(options.get("padding", 1), "区域外扩像素", 0, 4),
        )


def frame_pool(images, settings):
    kind = {"hot": "dark", "unstable": "bias", "low_response": "flat"}[settings.kind]
    items = [
        row for row in images.document["calibration_frames"]
        if row["enabled"] and row["kind"] == kind and (kind != "flat" or row["band"] == settings.band)
    ]
    if len(items) < settings.minimum:
        raise DomainError(f"当前至少需要 {settings.minimum} 张有效的同类型校准曝光")
    if len(items) * images.dims.pixels > MONITOR_PIXELS:
        raise DomainError("坏点监测处理 1400 万校准像素以内，请暂时挂起多出的参考")
    if settings.kind == "hot" and images.master_bias() is None:
        exposure = items[0]["exposure"]
        if any(not math.isclose(exposure, row["exposure"], rel_tol=1e-6, abs_tol=1e-6) for row in items):
            raise DomainError("不同曝光时长的暗场需要偏置参考才能比较每秒信号")
    return items


def _normalized_masked(plane):
    allowed = plane.valid & ~plane.saturated
    result = np.full(plane.values.shape, np.nan)
    result[allowed] = plane.values[allowed]
    return result


def _local_deviation(values, divisor=1.0):
    stats = robust_summary(values)
    if stats["median"] is None:
        return np.full(values.shape, np.nan)
    filled = np.where(np.isfinite(values), values, stats["median"])
    smooth = median_filter(filled, size=5, mode="reflect")
    return (values - smooth) / divisor


def _extract_plane(images, row, mode):
    plane = images.raw(row["id"], False)
    values = _normalized_masked(plane)
    if mode == "unstable":
        return _local_deviation(values)
    if mode == "hot":
        bias = images.master_bias()
        if bias is not None:
            values -= bias
        return _local_deviation(values, row["exposure"])
    values, _ = images.linear_values(plane, row["exposure"])
    values = np.where(plane.valid & ~plane.saturated, values, np.nan)
    center = robust_summary(values[values > 0])["median"]
    return values / center if center is not None and center > 0 else np.full(values.shape, np.nan)


def _stable_threshold(data, options):
    valid = np.isfinite(data)
    center = finite_median(data, axis=0)
    support = np.count_nonzero(valid, axis=0)
    required = np.ceil(options.fraction * support).astype(np.int32)
    if options.kind == "hot":
        detected = data >= options.threshold
        severity = center / options.threshold
    elif options.kind == "low_response":
        detected = valid & (data <= options.threshold)
        severity = options.threshold / np.maximum(1e-9, center)
    else:
        absolute = np.abs(data - center)
        deviation = finite_median(absolute, axis=0) * 1.4826
        detected = absolute >= options.threshold * 0.5
        severity = deviation / options.threshold
    count = np.count_nonzero(detected, axis=0)
    flags = (support >= options.minimum) & (count >= required)
    if options.kind == "unstable":
        flags &= severity >= 1
    return center, flags, count, support, severity


def _regions(flags, severity, votes, supports, padding):
    if padding:
        expanded = binary_dilation(flags, iterations=padding)
    else:
        expanded = flags
    labeled, n_components = label(expanded, structure=np.ones((3, 3), dtype=bool))
    neighborhoods = find_objects(labeled)
    kept = []
    for segment, window in enumerate(neighborhoods, start=1):
        component = labeled[window] == segment
        candidates = component & flags[window]
        if not np.any(candidates):
            continue
        level = np.where(candidates, severity[window], -np.inf)
        iy, ix = np.unravel_index(np.argmax(level), level.shape)
        sy, sx = window
        fingerprint = f"{sx.start}-{sy.start}-{sx.stop}-{sy.stop}"
        kept.append({
            "id": fingerprint,
            "x": int(sx.start + ix),
            "y": int(sy.start + iy),
            "score": float(level[iy, ix]),
            "left": int(sx.start),
            "top": int(sy.start),
            "right": int(sx.stop),
            "bottom": int(sy.stop),
            "pixels": int(np.count_nonzero(candidates)),
            "votes": int(votes[window][iy, ix]),
            "support": int(supports[window][iy, ix]),
        })
    kept.sort(key=lambda row: (-row["score"], row["y"], row["x"]))
    return kept[:48], n_components


def detector_results(images, options):
    settings = MonitorSettings.requested(options)
    diagnostic = ObservationImages({**images.document, "masks": []}, images.bodies)
    frames = frame_pool(diagnostic, settings)
    field = np.stack([_extract_plane(diagnostic, row, settings.kind) for row in frames])
    center, flags, votes, support, severity = _stable_threshold(field, settings)
    regions, total_regions = _regions(flags, severity, votes, support, settings.padding)
    report = {
        "kind": settings.kind,
        "revision": images.document["revision"],
        "threshold": settings.threshold,
        "frames_used": [row["filename"] for row in frames],
        "max_votes": int(len(frames)),
        "available_fraction": float(np.mean(support >= settings.minimum)),
        "flagged_pixels": int(np.count_nonzero(flags)),
        "total_regions": total_regions,
        "regions": regions,
        "unit": "ADU/s" if settings.kind == "hot" else "fraction" if settings.kind == "low_response" else "ADU",
    }
    return report, center, flags


def detector_map(center, flagged, report):
    height, width = center.shape
    integers = np.isfinite(center)
    values = center[integers]
    if values.size:
        low, high = np.percentile(values, [1, 99])
    else:
        low, high = 0, 1
    gray = np.clip((np.where(integers, center, low) - low) / max(high - low, 1e-7), 0, 1)
    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    buffer[:] = np.stack((22 + 91 * gray, 29 + 95 * gray, 39 + 106 * gray), axis=-1).astype(np.uint8)
    buffer[~integers] = [45, 28, 63]
    buffer[flagged] = [255, 190, 105]
    image = Image.fromarray(buffer, "RGB")
    magnify = min(3, max(1, 900 // max(height, width)))
    if magnify > 1:
        image = image.resize((width * magnify, height * magnify), Image.Resampling.NEAREST)
    marker = ImageDraw.Draw(image)
    for item in report["regions"]:
        box = tuple(magnify * item[k] for k in ("left", "top", "right", "bottom"))
        marker.rectangle(box, outline=(255, 198, 115), width=1)
    image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    return png(image)


def as_masks(report, selected, *, maximum):
    if not isinstance(selected, list) or not 1 <= len(selected) <= maximum or any(not isinstance(key, str) for key in selected):
        raise DomainError(f"请选择 1 至 {maximum} 个候选坏区")
    if len(set(selected)) != len(selected):
        raise DomainError("同一坏区不能重复选中")
    by_id = {row["id"]: row for row in report["regions"]}
    if not set(selected).issubset(by_id):
        raise DomainError("坏区候选已改变，请重跑检测")
    masks = []
    lookup = {"hot": "热点", "unstable": "不稳定", "low_response": "低响应"}
    for token in selected:
        source = by_id[token]
        masks.append({
            "label": f"{lookup[report['kind']]} {source['x']},{source['y']}",
            "left": source["left"],
            "top": source["top"],
            "right": source["right"],
            "bottom": source["bottom"],
            "frame": None,
        })
    return masks
