"""Quantitative intensity comparison of two registered science exposures."""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, uniform_filter

from .calibration import Plane, robust_summary
from .domain import DomainError, exposure_row, integer, number
from .preview import png
from .registration import preview_stream

MAX_FIT_SAMPLES = 72_000


@dataclass(frozen=True)
class PairOptions:
    threshold: float
    grid: int
    sigma: float
    star_only: bool

    @classmethod
    def requested(cls, plan):
        return cls(
            number(plan.get("low_quantile", 0.84), "亮度样本百分位", 0.05, 0.99),
            integer(plan.get("grid", 5), "残差分块数", 3, 12),
            number(plan.get("sigma", 3), "稳健残差截断倍数", 1.5, 8),
            plan.get("star_only", True) is True,
        )


def _common_values(images, left_id, right_id):
    if left_id == right_id:
        raise DomainError("请选择两张不同的曝光")
    d = images.document
    left = exposure_row(d, left_id)
    right = exposure_row(d, right_id)
    if left["band"] != right["band"]:
        raise DomainError("配对强度只对同滤镜曝光成立")
    first = preview_stream(images, left_id, "aligned")
    second = preview_stream(images, right_id, "aligned")
    valid = first.valid & second.valid & ~first.saturated & ~second.saturated
    count = np.count_nonzero(valid)
    if count < 100 or count < valid.size * 0.15:
        raise DomainError("公共有效画幅不足以比较强度")
    xa = np.where(valid, first.values / left["exposure"], np.nan)
    yb = np.where(valid, second.values / right["exposure"], np.nan)
    return first, second, xa, yb, valid


def _typical_background(plane, valid):
    body = np.where(valid, plane, 0)
    weight = uniform_filter(valid.astype(float), size=15, mode="constant", cval=0)
    base = uniform_filter(body, size=15, mode="constant", cval=0)
    base = np.divide(base, weight, where=weight >= .65, out=np.full(base.shape, np.nan))
    return base, weight


def _select_pairs(x, y, valid, options):
    stable = binary_erosion(valid, iterations=2)
    if options.star_only:
        sky_a, cover_a = _typical_background(x, valid)
        sky_b, cover_b = _typical_background(y, valid)
        sx = np.abs(x - sky_a)
        sy = np.abs(y - sky_b)
        stable &= (cover_a >= 0.65) & (cover_b >= 0.65)
        if np.count_nonzero(stable) < 100:
            raise DomainError("公共内部有效画幅太少")
        edge_x = np.quantile(sx[stable], options.threshold)
        edge_y = np.quantile(sy[stable], options.threshold)
        stable &= (x - sky_a >= edge_x) | (y - sky_b >= edge_y)
    if np.count_nonzero(stable) < 72:
        raise DomainError("可用于配对的像素样本不足")
    i, j = np.where(stable)
    if len(i) > MAX_FIT_SAMPLES:
        positions = np.linspace(0, len(i) - 1, MAX_FIT_SAMPLES).astype(int)
        i, j = i[positions], j[positions]
    return x[i, j], y[i, j], i, j


def _linear(x, y):
    center = float(np.median(x))
    scale = float(np.std(x))
    if scale <= 1e-8:
        raise DomainError("像素亮度分布太窄，无法区分斜率与零位")
    normalized = (x - center) / scale
    coef, _, rank, _ = np.linalg.lstsq(np.stack([normalized, np.ones(x.size)], axis=1), y, rcond=None)
    if rank < 2:
        raise DomainError("像素配对无法识别线性关系")
    slope = float(coef[0] / scale)
    intercept = float(coef[1] - center * slope)
    return slope, intercept


def _robust_fit(x, y, sigma):
    first = np.percentile(x, (25, 75))
    second = np.percentile(y, (25, 75))
    slope = (second[1] - second[0]) / max(first[1] - first[0], 1e-8)
    intercept = float(np.median(y - x * slope))
    keep = np.ones(x.shape, dtype=bool)
    for iteration in range(7):
        residual = y - (x * slope + intercept)
        summary = robust_summary(residual[keep])
        new_keep = np.abs(residual - summary["median"]) <= max(1e-8, sigma * summary["sigma"])
        if np.count_nonzero(new_keep) < max(36, x.size // 3):
            raise DomainError("两帧的像素强度关系缺少一致样本")
        slope, intercept = _linear(x[new_keep], y[new_keep])
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if not 0.02 < slope < 50:
        raise DomainError("拟合强度比例超出工作台的可信范围")
    residual = y - (x * slope + intercept)
    return slope, intercept, keep, residual


def _tile_summary(residual, valid, tile_count):
    height, width = residual.shape
    row_edges = np.rint(np.linspace(0, height, tile_count + 1)).astype(int)
    col_edges = np.rint(np.linspace(0, width, tile_count + 1)).astype(int)
    report = []
    for row, (y0, y1) in enumerate(zip(row_edges, row_edges[1:])):
        for col, (x0, x1) in enumerate(zip(col_edges, col_edges[1:])):
            view = np.s_[y0:y1, x0:x1]
            keep = valid[view]
            values = residual[view][keep]
            summary = robust_summary(values)
            report.append({
                "column": col,
                "row": row,
                "bounds": [int(x0), int(y0), int(x1), int(y1)],
                "support": float(np.mean(keep)),
                "median": summary["median"],
                "sigma": summary["sigma"],
            })
    return report


def _scatter(x, y, keep):
    low, high = np.percentile(x, (1, 99))
    if high <= low:
        high = low + 1
    match = np.clip(np.digitize(x, np.linspace(low, high, 20)), 0, 19)
    bins = []
    for bucket in range(20):
        sample = match == bucket
        retained = sample & keep
        if not np.any(sample):
            continue
        bins.append({
            "rate_a": float(np.median(x[sample])),
            "rate_b": float(np.median(y[sample])),
            "retained_b": float(np.median(y[retained])) if np.any(retained) else None,
            "count": int(np.count_nonzero(sample)),
        })
    return bins


def pair_report(images, left_id, right_id, options):
    settings = PairOptions.requested(options)
    first, second, a, b, mask = _common_values(images, left_id, right_id)
    x, y, row, col = _select_pairs(a, b, mask, settings)
    slope, offset, chosen, residual = _robust_fit(x, y, settings.sigma)
    after = np.where(mask, b - (a * slope + offset), np.nan)
    quality = robust_summary(residual[chosen])
    report = {
        "slope": slope,
        "offset": offset,
        "support": float(np.mean(mask)),
        "rejected": int(np.count_nonzero(~chosen)),
        "included": int(np.count_nonzero(chosen)),
        "residual": quality,
        "scatter": _scatter(x, y, chosen),
        "tile_grid": settings.grid,
        "tiles": _tile_summary(after, mask, settings.grid),
    }
    values = after[mask]
    barrier = max(0.002, float(np.quantile(np.abs(values), 0.995)))
    scaled = np.clip(np.where(mask, after, 0) / barrier, -1, 1)
    heat = np.stack(
        [22 + 199 * np.maximum(scaled, 0), 26 + 80 * np.abs(scaled), 32 + 210 * np.maximum(-scaled, 0)],
        axis=2,
    )
    heat[~mask] = (54, 36, 64)
    rendered = Image.fromarray(np.rint(heat).astype(np.uint8), "RGB")
    rendered.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    return report, png(rendered)
