"""Map smooth detector background from confirmed empty-sky patches or an automatic grid."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, median_filter

from .domain import DomainError, enum, integer, number
from .preview import png
from .registration import offset_for

BACKGROUND_MODELS = {"plane", "curved"}
BACKGROUND_SAMPLES = {"auto", "anchors"}
ANCHOR_TERMS = {
    "plane": ((0, 0), (1, 0), (0, 1)),
    "curved": ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)),
}
ANCHOR_LABELS = ("constant", "x", "y", "x²", "xy", "y²")
MAX_ANCHOR_SAMPLES = 20000
MIN_ANCHOR_PIXELS = 64


@dataclass(frozen=True)
class SkyModel:
    tiles: int
    rejection: float
    kind: str
    min_support: float
    grow: int
    sample: str
    reach: float
    conflict: float
    error_limit: float

    @classmethod
    def requested(cls, options):
        sample = enum(options.get("sample", "auto"), "背景样本来源", BACKGROUND_SAMPLES)
        return cls(
            integer(options.get("tiles", 8), "网格每行块数", 4, 18),
            number(options.get("rejection", 3.2), "背景剪裁倍数", 1.5, 10),
            enum(options.get("kind", "plane"), "背景趋势类型", BACKGROUND_MODELS),
            number(options.get("min_support", 0.25 if sample == "anchors" else 0.38), "每块最低背景覆盖", 0.2, 0.9),
            integer(options.get("grow", 2), "星点排除宽度像素", 0, 5),
            sample,
            number(options.get("reach", 64), "局部照顾半径像素", 8, 512),
            number(options.get("conflict_limit", 3.2), "空天样本冲突阈值", 1.5, 8),
            number(options.get("error_limit", 1.0), "局部预测不确定度上限", 0.25, 4),
        )


def _local_sky(data, mask, rejection):
    current = data[mask]
    initial = current.size
    if initial < 12:
        return None, None, -1
    for _ in range(4):
        middle = np.median(current)
        scale = max(1e-7, np.median(np.abs(current - middle)) * 1.4826)
        accepted = current[np.abs(current - middle) <= rejection * scale]
        if accepted.size < 12:
            return None, None, -1
        if accepted.size == current.size:
            break
        current = accepted
    return float(np.median(current)), float(np.median(np.abs(current - np.median(current))) * 1.4826), initial - int(current.size)


def _detect_star_patch(data, valid, rejection, radius):
    if not np.any(valid):
        return valid
    good = data[valid]
    center = float(np.median(good))
    noise = float(np.median(np.abs(good - center)) * 1.4826)
    grid = np.where(valid, data, center)
    local = median_filter(grid, size=7, mode="reflect")
    stars = (grid - local) > max(rejection, 3) * max(noise, 1e-7)
    if radius:
        stars = binary_dilation(stars, iterations=radius)
    return valid & ~stars


def _tile_model(plane, cfg):
    height, width = plane.values.shape
    y_edges = np.rint(np.linspace(0, height, cfg.tiles + 1)).astype(int)
    x_edges = np.rint(np.linspace(0, width, cfg.tiles + 1)).astype(int)
    support = plane.valid & ~plane.saturated
    starless = _detect_star_patch(plane.values, support, cfg.rejection, cfg.grow)
    result = []
    for i, (lo, hi) in enumerate(zip(y_edges, y_edges[1:])):
        for j, (left, right) in enumerate(zip(x_edges, x_edges[1:])):
            cut = np.s_[lo:hi, left:right]
            if (hi - lo) * (right - left) < 16:
                continue
            center, sigma, removed = _local_sky(plane.values[cut], starless[cut], cfg.rejection)
            used = int(np.count_nonzero(starless[cut])) - max(0, removed)
            fraction = used / starless[cut].size
            good = center is not None and fraction >= cfg.min_support
            result.append({
                "column": j,
                "row": i,
                "bounds": [int(left), int(lo), int(right), int(hi)],
                "x": float((left + right - 1) * 0.5),
                "y": float((lo + hi - 1) * 0.5),
                "state": "measured" if good else "insufficient_sky",
                "level": center if good else None,
                "sigma": sigma if good else None,
                "support": fraction,
                "remaining": int(max(used, 0)),
                "clipped": int(max(removed, 0)),
            })
    return result


def _transforms(u, v, mode):
    labels = ["constant", "x", "y"]
    fields = [np.ones(np.broadcast_shapes(u.shape, v.shape)), u, v]
    if mode == "curved":
        labels += ["x²", "xy", "y²"]
        fields += [u**2, u * v, v**2]
    return labels, np.broadcast_arrays(*fields)


def _center_coord(values, length):
    return 2 * np.asarray(values, dtype=float) / max(1, length - 1) - 1.0


def _trend(tiles, dimensions, mode):
    used = [entry for entry in tiles if entry["state"] == "measured"]
    target = np.array([entry["level"] for entry in used])
    u = _center_coord([entry["x"] for entry in used], dimensions[1])
    v = _center_coord([entry["y"] for entry in used], dimensions[0])
    names, matrices = _transforms(u, v, mode)
    design = np.column_stack(matrices)
    count = len(names)
    if len(used) < count * 2:
        raise DomainError("可用背景块不足以支持当前背景阶次")
    floor = max(0.01, np.median([entry["sigma"] for entry in used]) * 0.4)
    weights = np.asarray([entry["support"] / max(entry["sigma"], floor)**2 for entry in used])
    hold = np.ones(target.shape, dtype=bool)
    for _ in range(5):
        inner = np.sqrt(weights[hold])
        coeff, _, rank, singular = np.linalg.lstsq(design[hold] * inner[:, None], target[hold] * inner, rcond=None)
        if rank != count or singular[-1] / singular[0] < 0.025:
            raise DomainError("可用背景块在画幅上分布过于集中")
        residuals = target - design @ coeff
        median = np.median(residuals[hold])
        noise = max(0.01, np.median(np.abs(residuals[hold] - median)) * 1.4826)
        updated = np.abs(residuals - median) <= noise * 3.6
        if np.count_nonzero(updated) < count * 2 or np.array_equal(updated, hold):
            break
        hold = updated
    else:
        inner = np.sqrt(weights[hold])
        coeff = np.linalg.lstsq(design[hold] * inner[:, None], target[hold] * inner, rcond=None)[0]
        residuals = target - design @ coeff
    for item, accepted, error in zip(used, hold, residuals):
        item["fit"] = bool(accepted)
        item["residual"] = float(error)
    return names, coeff, float(np.sqrt(np.mean(residuals[hold]**2)))


def _rectangle_distance(bounds, xs, ys):
    left, top, right, bottom = bounds
    dx = np.maximum(np.maximum(left - xs, 0.0), xs - (right - 1))
    dy = np.maximum(np.maximum(top - ys, 0.0), ys - (bottom - 1))
    return np.hypot(dx, dy)


def _evaluate_anchor(model, xs, ys):
    cx, cy = model["center"]
    u = (xs - cx) / model["scale"]
    v = (ys - cy) / model["scale"]
    basis = np.stack([u**a * v**b for a, b in model["terms"]])
    predicted = np.tensordot(model["coefficients"], basis, axes=1)
    quadratic = np.sum(np.tensordot(model["inverse"], basis, axes=1) * basis, axis=0)
    uncertainty = np.sqrt(np.maximum(model["variance"] * quadratic, 0.0))
    return predicted, uncertainty


def _anchor_field(model, cfg, xs, ys):
    predicted, uncertainty = _evaluate_anchor(model, xs, ys)
    distance = _rectangle_distance(model["bounds"], xs, ys)
    usable = (distance <= cfg.reach) & np.isfinite(predicted) & np.isfinite(uncertainty)
    usable &= uncertainty <= cfg.error_limit * model["noise"]
    return usable, predicted, uncertainty


def _anchor_fit(plane, starless, region, cfg):
    """Robust local polynomial from one confirmed sky patch, plus its coverage box."""
    left, top, right, bottom = region["bounds"]
    height, width = plane.values.shape
    area = max(1, (right - left) * (bottom - top))
    stride = max(1, int(math.ceil(math.sqrt(area / MAX_ANCHOR_SAMPLES))))
    grid = starless[top:bottom:stride, left:right:stride]
    ys, xs = np.nonzero(grid)
    xs = xs * stride + left
    ys = ys * stride + top
    values = plane.values[ys, xs]
    finite = np.isfinite(values)
    xs, ys, values = xs[finite], ys[finite], values[finite]
    report = {
        "label": region["label"], "bounds": [int(left), int(top), int(right), int(bottom)],
        "state": "insufficient_sample", "samples": int(values.size),
        "support": float(values.size / max(1, grid.size)), "reason": None,
        "level": None, "noise": None, "fit_rms": None, "coverage": 0,
    }
    keep = np.ones(values.size, dtype=bool)
    for _ in range(4):
        current = values[keep]
        if current.size < 12:
            break
        middle = float(np.median(current))
        scale = max(1e-7, float(np.median(np.abs(current - middle))) * 1.4826)
        updated = keep & (np.abs(values - middle) <= cfg.rejection * scale)
        if np.count_nonzero(updated) < 12 or np.array_equal(updated, keep):
            break
        keep = updated
    values, xs, ys = values[keep], xs[keep], ys[keep]
    report["samples"] = int(values.size)
    report["support"] = float(values.size / max(1, grid.size))
    terms = ANCHOR_TERMS[cfg.kind]
    if values.size < max(12, len(terms) * 4):
        report["reason"] = "样本内有效像素不足，可能是饱和或坏区面积过大"
        return None, report
    if report["support"] < cfg.min_support:
        report["reason"] = "样本内可当天空的像素比例太低"
        return None, report
    center_x = (left + right - 1) * 0.5
    center_y = (top + bottom - 1) * 0.5
    scale = max(cfg.reach / 2.5, 1.0)
    u = (xs - center_x) / scale
    v = (ys - center_y) / scale
    design = np.column_stack([u**a * v**b for a, b in terms])
    normal = design.T @ design
    try:
        inverse = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None, report
    coefficients = inverse @ design.T @ values
    rss = float(np.sum((values - design @ coefficients) ** 2))
    variance = max(rss / max(1, values.size - len(terms)), 1e-12)
    noise = max(1e-9, float(np.median(np.abs(values - np.median(values))) * 1.4826))
    pad = int(math.ceil(cfg.reach)) + 2
    report.update({
        "state": "measured", "level": float(np.median(values)), "noise": noise,
        "fit_rms": math.sqrt(variance),
        "coefficients": {name: float(value) for name, value in zip(ANCHOR_LABELS, coefficients)},
    })
    model = {
        "bounds": (left, top, right, bottom),
        "box": (max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad)),
        "center": (center_x, center_y), "scale": scale, "terms": terms,
        "coefficients": coefficients, "inverse": inverse,
        "variance": variance, "noise": noise,
    }
    return model, report


def _anchor_conflicts(models, noise, cfg, shape):
    """Pixels where two confirmed patches extrapolate to materially incompatible levels."""
    conflict = np.zeros(shape, dtype=bool)
    floor = max(1e-9, 1.5 * noise)
    for first in range(len(models)):
        for second in range(first + 1, len(models)):
            one, two = models[first], models[second]
            el = max(one["box"][0], two["box"][0])
            et = max(one["box"][1], two["box"][1])
            er = min(one["box"][2], two["box"][2])
            eb = min(one["box"][3], two["box"][3])
            if er <= el or eb <= et:
                continue
            yy, xx = np.indices((eb - et, er - el), dtype=float)
            usable = np.ones(xx.shape, dtype=bool)
            fields = []
            for model in (one, two):
                ok, predicted, uncertainty = _anchor_field(model, cfg, xx + el, yy + et)
                usable &= ok
                fields.append((predicted, uncertainty))
            if not np.any(usable):
                continue
            gap = np.abs(fields[0][0] - fields[1][0])
            combined = np.sqrt(fields[0][1] ** 2 + fields[1][1] ** 2)
            conflict[et:eb, el:er] |= usable & (gap > floor) & (gap > cfg.conflict * combined)
    return conflict


def _anchored_trend(plane, regions, cfg):
    usable = plane.valid & ~plane.saturated
    starless = _detect_star_patch(plane.values, usable, cfg.rejection, cfg.grow)
    height, width = plane.values.shape
    models = []
    reports = []
    for region in regions:
        model, report = _anchor_fit(plane, starless, region, cfg)
        reports.append(report)
        if model is not None:
            models.append(model)
    if not models:
        raise DomainError("所选空天样本的可用像素不足以拟合局部背景")
    weight = np.zeros((height, width))
    weighted = np.zeros((height, width))
    contributors = np.zeros((height, width), dtype=np.int16)
    measured = [row for row in reports if row["state"] == "measured"]
    for model, report in zip(models, measured):
        el, et, er, eb = model["box"]
        yy, xx = np.indices((eb - et, er - el), dtype=float)
        ok, predicted, uncertainty = _anchor_field(model, cfg, xx + el, yy + et)
        cut = np.s_[et:eb, el:er]
        w = np.where(ok, 1.0 / np.maximum(uncertainty, 1e-9) ** 2, 0.0)
        weight[cut] += w
        weighted[cut] += w * np.where(ok, predicted, 0.0)
        contributors[cut] += ok.astype(np.int16)
        report["coverage"] = int(np.count_nonzero(ok))
    active = weight > 0
    surface = np.where(active, weighted / np.maximum(weight, 1e-12), np.nan)
    noise = float(np.median([model["noise"] for model in models]))
    conflict = _anchor_conflicts(models, noise, cfg, (height, width))
    supported = active & ~conflict
    predicted = np.where(supported, surface, np.nan)
    residual = np.where(supported & usable, plane.values - surface, np.nan)
    low, high = (float(np.min(predicted[supported])), float(np.max(predicted[supported]))) if np.any(supported) else (0.0, 0.0)
    report = {
        "sample": "anchors", "mode": cfg.kind, "anchors": reports,
        "reach": cfg.reach, "conflict_limit": cfg.conflict, "error_limit": cfg.error_limit,
        "valid_tiles": len(measured),
        "fit_rms_adu": float(np.median([row["fit_rms"] for row in measured])),
        "minimum": low, "maximum": high, "range": high - low,
        "supported_pixels": int(np.count_nonzero(supported)),
        "conflict_pixels": int(np.count_nonzero(conflict)),
        "supported_fraction": float(np.mean(supported)),
        "conflict_fraction": float(np.mean(conflict)),
        "retained_fraction": float(np.mean(usable & ~supported)),
        "good_fraction": float(np.mean(supported)),
    }
    layers = {
        "supported": supported, "conflict": conflict, "contributors": contributors,
        "valid": usable,
        "anchors": [model["bounds"] for model in models],
    }
    return report, predicted, residual, layers


def anchor_regions(document, frame_id, layer, shape):
    if document is None or frame_id is None:
        raise DomainError("局部空天背景需要当前曝光与观测记录")
    height, width = shape
    shift_x = shift_y = 0.0
    offset = offset_for(document, frame_id)
    if layer == "aligned":
        if offset is None:
            raise DomainError("本曝光还没有对准偏移，无法在参考坐标使用空天样本")
        shift_x, shift_y = -offset.dx, -offset.dy
    elif offset is not None:
        # Samples are stored in reference coordinates; move them into this native frame.
        shift_x, shift_y = offset.dx, offset.dy
    entries = document.get("sky_samples", [])
    chosen = []
    for entry in entries:
        if entry.get("frame") and entry["frame"] != frame_id:
            continue
        left = max(0, int(math.floor(entry["left"] + shift_x)))
        top = max(0, int(math.floor(entry["top"] + shift_y)))
        right = min(width, int(math.ceil(entry["right"] + shift_x)))
        bottom = min(height, int(math.ceil(entry["bottom"] + shift_y)))
        if right - left < 2 or bottom - top < 2:
            continue
        if (right - left) * (bottom - top) < MIN_ANCHOR_PIXELS:
            continue
        chosen.append({
            "label": entry.get("label") or f"空天样本 {len(chosen) + 1}",
            "bounds": (left, top, right, bottom),
        })
    if not chosen:
        raise DomainError("当前曝光没有可用的空天样本，请先在影像阅览框选确认的空天")
    return chosen


def sky_surface(plane, plan, document=None, frame_id=None, layer="calibrated"):
    opts = SkyModel.requested(plan)
    if opts.sample == "anchors":
        regions = anchor_regions(document, frame_id, layer, plane.values.shape)
        return _anchored_trend(plane, regions, opts)
    tiles = _tile_model(plane, opts)
    names, coeff, rms = _trend(tiles, plane.values.shape, opts.kind)
    height, width = plane.values.shape
    y = _center_coord(np.arange(height), height)[:, None]
    x = _center_coord(np.arange(width), width)[None, :]
    _, basis = _transforms(x, y, opts.kind)
    predicted = np.zeros(plane.values.shape)
    for component, field in zip(coeff, basis):
        predicted += component * field
    usable = plane.valid & ~plane.saturated
    residual = np.where(usable, plane.values - predicted, np.nan)
    low, high = float(np.min(predicted)), float(np.max(predicted))
    report = {
        "sample": "auto", "mode": opts.kind, "tiles": tiles, "anchors": [],
        "coefficients": {name: float(value) for name, value in zip(names, coeff)},
        "fit_rms_adu": rms, "minimum": low, "maximum": high, "range": high - low,
        "valid_tiles": sum(tile.get("fit", False) for tile in tiles),
        "good_fraction": float(np.mean(np.isfinite(residual))),
        "supported_fraction": float(np.mean(usable)),
        "conflict_fraction": 0.0,
        "retained_fraction": 0.0,
    }
    layers = {
        "supported": usable, "conflict": np.zeros(plane.values.shape, dtype=bool),
        "contributors": np.ones(plane.values.shape, dtype=np.int16), "valid": usable, "anchors": [],
    }
    return report, predicted, residual, layers


def render_sky(values, residual):
    valid = np.isfinite(values)
    if not np.any(valid):
        raise DomainError("背景图没有可显示的像素")
    if residual:
        bound = max(0.001, np.percentile(np.abs(values[valid]), 98))
        scaled = np.clip(np.where(valid, values, 0) / bound, -1, 1)
        r = (21 + np.maximum(scaled, 0) * 205).astype(np.uint8)
        g = (27 + np.abs(scaled) * 104).astype(np.uint8)
        b = (30 + np.maximum(-scaled, 0) * 220).astype(np.uint8)
    else:
        low, high = np.percentile(values[valid], [1, 99])
        scaled = np.clip((np.where(valid, values, low) - low) / max(0.01, high - low), 0, 1)
        r = (19 + scaled**2 * 182).astype(np.uint8)
        g = (34 + scaled * 171).astype(np.uint8)
        b = (63 + scaled * 130).astype(np.uint8)
    im = Image.fromarray(np.stack([r, g, b], axis=2), "RGB")
    im.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    return png(im)


def render_support(layers):
    """Colour where the sky level is constrained, where patches conflict, and what stays untouched."""
    supported = layers["supported"]
    conflict = layers["conflict"]
    valid = layers["valid"]
    contributors = np.clip(np.asarray(layers["contributors"], dtype=float), 1, 4) / 4.0
    height, width = supported.shape
    rgb = np.zeros((height, width, 3), dtype=float)
    rgb[..., 0] = 20
    rgb[..., 1] = 15
    rgb[..., 2] = 30
    rgb[valid] = (46, 38, 58)
    rgb[..., 0] = np.where(supported, 26 + 40 * contributors, rgb[..., 0])
    rgb[..., 1] = np.where(supported, 96 + 108 * contributors, rgb[..., 1])
    rgb[..., 2] = np.where(supported, 96 + 90 * contributors, rgb[..., 2])
    rgb[conflict] = (218, 150, 62)
    image = Image.fromarray(np.rint(np.clip(rgb, 0, 255)).astype(np.uint8), "RGB")
    image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    scale_x = image.width / width
    scale_y = image.height / height
    for left, top, right, bottom in layers.get("anchors", []):
        draw.rectangle(
            (
                left * scale_x, top * scale_y,
                max(left * scale_x, (right - 1) * scale_x), max(top * scale_y, (bottom - 1) * scale_y),
            ),
            outline=(124, 236, 226), width=1,
        )
    return png(image)
