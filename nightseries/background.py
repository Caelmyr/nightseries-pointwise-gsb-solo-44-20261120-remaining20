"""Map smooth detector background while masking stars, saturation and unsupported tiles."""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, median_filter

from .calibration import Plane
from .domain import DomainError, enum, integer, number
from .preview import png

BACKGROUND_MODELS = {"plane", "curved"}


@dataclass(frozen=True)
class SkyModel:
    tiles: int
    rejection: float
    kind: str
    min_support: float
    grow: int

    @classmethod
    def requested(cls, options):
        return cls(
            integer(options.get("tiles", 8), "网格每行块数", 4, 18),
            number(options.get("rejection", 3.2), "背景剪裁倍数", 1.5, 10),
            enum(options.get("kind", "plane"), "背景趋势类型", BACKGROUND_MODELS),
            number(options.get("min_support", 0.38), "每格最低背景覆盖", 0.2, 0.9),
            integer(options.get("grow", 2), "星点排除宽度像素", 0, 5),
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


def sky_surface(plane, plan):
    opts = SkyModel.requested(plan)
    tiles = _tile_model(plane, opts)
    names, coeff, rms = _trend(tiles, plane.values.shape, opts.kind)
    height, width = plane.values.shape
    y = _center_coord(np.arange(height), height)[:, None]
    x = _center_coord(np.arange(width), width)[None, :]
    _, basis = _transforms(x, y, opts.kind)
    predicted = np.zeros(plane.values.shape)
    for component, field in zip(coeff, basis):
        predicted += component * field
    residual = np.where(plane.valid & ~plane.saturated, plane.values - predicted, np.nan)
    low, high = float(np.min(predicted)), float(np.max(predicted))
    return {
        "mode": opts.kind, "tiles": tiles,
        "coefficients": {name: float(value) for name, value in zip(names, coeff)},
        "fit_rms_adu": rms, "minimum": low, "maximum": high, "range": high - low,
        "valid_tiles": sum(tile.get("fit", False) for tile in tiles),
        "good_fraction": float(np.mean(np.isfinite(residual))),
    }, predicted, residual


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
