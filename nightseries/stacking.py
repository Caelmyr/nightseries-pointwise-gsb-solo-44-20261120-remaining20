"""Integration in reference coordinates with explicit per-pixel support."""
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .calibration import Plane, finite_median, robust_summary
from .domain import DomainError, MAX_ACTIVE_PIXELS, boolean, enum, integer, usable_science_frames
from .photometry import filter_photometry_frames
from .preview import png
from .registration import offset_for, warp_to_reference

STACK_MODES = {"mean", "median", "exposure"}


@dataclass(frozen=True)
class StackProduct:
    plane: Plane
    coverage: np.ndarray
    saturated: np.ndarray
    report: dict


def _center_sky(plane, seconds, center):
    keep = plane.valid & ~plane.saturated
    rate = np.where(keep, plane.values / seconds, np.nan)
    level = robust_summary(rate)["median"] if center else 0.0
    if level is None:
        return rate, None
    return rate - level, level


def _mean_accumulate(sum_values, weights, rate, valid, weight):
    sum_values[valid] += rate[valid] * weight
    weights[valid] += weight


def _preview_shape(image):
    image.thumbnail((1000, 1000), Image.Resampling.NEAREST)
    return image


def coverage_preview(coverage, maximum):
    level = coverage.astype(float) / max(1, maximum)
    rgb = np.stack(
        (20 + 95 * level, 20 + 190 * level, 26 + 212 * level**0.65),
        axis=2,
    )
    rgb[coverage == 0] = [13, 14, 22]
    return png(_preview_shape(Image.fromarray(np.rint(rgb).astype(np.uint8), "RGB")))


def compute_stack(images, options):
    doc = images.document
    band, frames = filter_photometry_frames(doc, options.get("band"))
    mode = enum(options.get("mode", "mean"), "叠加方式", STACK_MODES)
    min_support = integer(options.get("min_support", 2), "最低覆盖帧数", 1, len(frames))
    subtract_background = boolean(options.get("subtract_background", False), "扣除帧背景")
    # "all" counts every frame where the pixel was on the detector, saturated or not,
    # so a saturated pixel is reported as covered but unusable rather than absent.
    coverage_mode = enum(options.get("coverage", "measured"), "覆盖计数方式", {"measured", "all"})
    count_all = coverage_mode == "all"
    if len(frames) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("当前叠加超过 2400 万像素，先挂起部分科学曝光")
    if mode == "median" and len(frames) * images.dims.pixels > 12_000_000:
        raise DomainError("中位叠加当前最多 1200 万采样，可用均值处理更大一组")
    coverage = np.zeros(images.dims.shape, dtype=np.uint16)
    saturated = np.zeros(images.dims.shape, dtype=np.uint16)
    totals = np.zeros(images.dims.shape, dtype=np.float64)
    weights = np.zeros(images.dims.shape, dtype=np.float64)
    planes = []
    used = []
    skipped = []
    for frame in frames:
        frame_id = frame["id"]
        offset = offset_for(doc, frame_id)
        if offset is None:
            skipped.append({"frame": frame_id, "filename": frame["filename"], "state": "unaligned"})
            continue
        plane = warp_to_reference(images.calibrated(frame_id), offset)
        rates, removed = _center_sky(plane, frame["exposure"], subtract_background)
        valid = np.isfinite(rates)
        if not np.any(plane.valid) or (not count_all and not np.any(valid)):
            skipped.append({"frame": frame_id, "filename": frame["filename"], "state": "no_support"})
            continue
        # Saturation is counted only for frames that are actually used, so a skipped
        # frame cannot make a pixel look covered-but-saturated.
        saturated += plane.saturated.astype(np.uint16)
        coverage += (plane.valid if count_all else valid)
        if mode == "median":
            planes.append(rates)
        else:
            weight = frame["exposure"] if mode == "exposure" else 1.0
            _mean_accumulate(totals, weights, rates, valid, weight)
        used.append({
            "frame": frame_id,
            "filename": frame["filename"],
            "exposure": frame["exposure"],
            "removed_background": removed,
            "valid_fraction": float(np.mean(valid)),
            "offset": [offset.dx, offset.dy],
        })
    if len(used) < min_support:
        raise DomainError("已对准且有有效像素的曝光数量不足以达到最低覆盖")
    valid = coverage >= min_support
    if mode == "median":
        values = finite_median(np.stack(planes), axis=0)
    else:
        values = np.divide(totals, weights, out=np.full(images.dims.shape, np.nan), where=valid)
    values[~valid] = np.nan
    result = Plane(values, valid, np.zeros(valid.shape, dtype=bool))
    report = {
        "revision": doc["revision"],
        "reference": doc["reference"],
        "band": band,
        "mode": mode,
        "min_support": min_support,
        "subtract_background": subtract_background,
        "used": used,
        "skipped": skipped,
        "width": images.dims.width,
        "height": images.dims.height,
        "coverage_histogram": np.bincount(coverage.ravel(), minlength=len(used) + 1).tolist(),
        "qualified_fraction": float(np.mean(valid)),
        "saturated_pixels": int(np.count_nonzero(saturated)),
        "unit": "ADU/s",
    }
    return StackProduct(result, coverage, saturated, report)
