"""Estimate detector conversion gain from matched flat and bias pairs."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from .calibration import robust_summary
from .domain import DomainError, MAX_ACTIVE_PIXELS, integer, number, text

MAX_PAIRS = 18


@dataclass(frozen=True)
class GainRequest:
    band: str
    tile_size: int
    maximum_level: float
    stability: float
    low: float
    high: float

    @classmethod
    def request(cls, options, saturation):
        maximum = number(options.get("maximum_level", saturation * 0.82), "最高平场 ADU", 10, saturation)
        low = number(options.get("low", 0.15), "下方裁边比例", 0.02, 0.42)
        high = number(options.get("high", 0.85), "上方裁边比例", 0.58, 0.98)
        return cls(
            text(options.get("band", ""), "平场滤镜", 24, required=False),
            integer(options.get("tile_size", 32), "取样块宽度", 12, 96),
            maximum,
            number(options.get("stability", 0.025), "配对平均亮度允许差", 0.001, 0.10),
            low,
            high,
        )


def calibration_pairs(document, kind, filter_name):
    found = defaultdict(list)
    for row in document["calibration_frames"]:
        if row["enabled"] and row["kind"] == kind and (kind == "bias" or row["band"] == filter_name):
            found[round(row["exposure"], 6)].append(row)
    ordered = []
    for key, group in sorted(found.items()):
        group.sort(key=lambda row: row["filename"])
        ordered.extend(combinations(group, 2))
    if not ordered:
        raise DomainError("需要同曝光时长的两张偏置参考" if kind == "bias" else "需要同滤镜、同曝光时长的两张平场")
    return ordered


def _pick_representative_pairs(pairs, limit=MAX_PAIRS):
    if len(pairs) <= limit:
        return pairs
    at = np.rint(np.linspace(0, len(pairs) - 1, limit)).astype(int)
    return [pairs[index] for index in at]


def sample_blocks(shape, size, low, high):
    height, width = shape
    y0, x0 = int(height * low), int(width * low)
    y1, x1 = int(height * high), int(width * high)
    if min(y1 - y0, x1 - x0) < size:
        raise DomainError("中心可用区域放不下该采样块")
    cuts = []
    for y in range(y0, y1 - size + 1, size):
        for x in range(x0, x1 - size + 1, size):
            cuts.append(np.s_[y:y + size, x:x + size])
    return cuts


def _trimmed_variance(values):
    stats = robust_summary(values)
    floor = max(1e-8, stats["sigma"])
    diff = values - stats["median"]
    mask = np.abs(diff) <= 5 * floor
    retained = int(np.count_nonzero(mask))
    if retained < 30 or retained < values.size * 0.85:
        return None, retained
    return float(np.var(values[mask], ddof=1)) * 0.5, retained


def _compare_bias(a, b, blocks):
    passed = []
    for block in blocks:
        mask = a.valid[block] & b.valid[block] & ~a.saturated[block] & ~b.saturated[block]
        if np.mean(mask) < 0.75:
            continue
        first, second = a.values[block][mask], b.values[block][mask]
        variance, count = _trimmed_variance(first - second)
        if variance is None:
            continue
        mean_level = float(np.median((first + second) * 0.5))
        passed.append({
            "variance_adu2": variance,
            "center_adu": mean_level,
            "samples": count,
            "x": block[1].start,
            "y": block[0].start,
        })
    return passed


def compare_pair(pair, images, blocks, result_fn):
    left, right = pair
    first = images.raw(left["id"], True)
    second = images.raw(right["id"], True)
    measured = result_fn(first, second, blocks)
    return {
        "first": left["filename"],
        "second": right["filename"],
        "exposure": left["exposure"],
        "samples": measured,
        "valid_patches": len(measured),
    }


def _flat_fn(bias, variance_floor, spec):
    def compare(first, second, blocks):
        kept = []
        for block in blocks:
            mask = first.valid[block] & second.valid[block] & ~first.saturated[block] & ~second.saturated[block]
            if bias is not None:
                bias_plane = bias[block]
                mask &= np.isfinite(bias_plane)
            else:
                bias_plane = None
            if float(np.mean(mask)) < 0.75:
                continue
            a = first.values[block][mask]
            b = second.values[block][mask]
            if bias_plane is not None:
                correction = bias_plane[mask]
                a = a - correction
                b = b - correction
            level_a, level_b = float(np.median(a)), float(np.median(b))
            level = 0.5 * (level_a + level_b)
            if level < 30 or level > spec.maximum_level:
                continue
            if abs(level_a - level_b) > spec.stability * level:
                continue
            variance, n = _trimmed_variance(a - b)
            if variance is None or variance <= variance_floor * 1.1:
                continue
            electron_ratio = level / (variance - variance_floor)
            if not 0.02 <= electron_ratio <= 500:
                continue
            kept.append({
                "center_adu": level,
                "variance_adu2": variance,
                "gain": electron_ratio,
                "pair_difference": (level_a - level_b) / level,
                "samples": n,
                "x": block[1].start,
                "y": block[0].start,
            })
        return kept
    return compare


def _robust_pair_level(entries, key):
    accepted = []
    rejected = []
    for pair in entries:
        if not pair["samples"]:
            rejected.append({
                **{name: pair[name] for name in ("first", "second", "exposure")},
                "state": "no_valid_tiles",
                "median": None,
                "sigma": None,
            })
            continue
        metrics = robust_summary([row[key] for row in pair["samples"]])
        accepted.append({
            **{name: pair[name] for name in ("first", "second", "exposure")},
            "state": "measured",
            "median": metrics["median"],
            "sigma": metrics["sigma"],
            "patches": len(pair["samples"]),
        })
    return accepted + rejected, robust_summary([row["median"] for row in accepted])


def gain_report(images, options):
    doc = images.document
    spec = GainRequest.request(options, images.saturation)
    bias_pairs = _pick_representative_pairs(calibration_pairs(doc, "bias", spec.band), 8)
    flat_pairs = _pick_representative_pairs(calibration_pairs(doc, "flat", spec.band), 10)
    if (len(bias_pairs) + len(flat_pairs)) * 2 * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("当前配对超过 2400 万校准像素，可停用部分重复平场")
    blocks = sample_blocks(images.dims.shape, spec.tile_size, spec.low, spec.high)
    bias_rows = [compare_pair(pair, images, blocks, _compare_bias) for pair in bias_pairs]
    bias_checks, base = _robust_pair_level(bias_rows, "variance_adu2")
    if base["count"] < 1:
        raise DomainError("偏置对缺少可比较的有效样本块")
    first_levels = [row["center_adu"] for result in bias_rows for row in result["samples"]]
    level = robust_summary(first_levels)
    # Gain inspection deliberately takes enabled bias exposures even when the science master is switched off.
    bias_planes = []
    for key in dict.fromkeys(row["id"] for pair in bias_pairs for row in pair):
        raw = images.raw(key, True)
        bias_planes.append(np.where(raw.valid & ~raw.saturated, raw.values, np.nan))
    from .calibration import master_median

    corrected = master_median(bias_planes)
    compare = _flat_fn(corrected, base["median"], spec)
    flat_rows = [compare_pair(pair, images, blocks, compare) for pair in flat_pairs]
    flat_checks, measured = _robust_pair_level(flat_rows, "gain")
    noise = math.sqrt(base["median"])
    return {
        "revision": doc["revision"],
        "band": spec.band,
        "gain_e_per_adu": measured["median"],
        "gain_pair_spread": measured["sigma"],
        "read_noise_adu": noise,
        "read_noise_e": noise * measured["median"] if measured["median"] is not None else None,
        "bias_level_adu": level["median"],
        "bias_pair_count": base["count"],
        "flat_pair_count": measured["count"],
        "flat_pairs": flat_checks,
        "bias_pairs": bias_checks,
        "tile_size": spec.tile_size,
        "tile_count": len(blocks),
        "estimates": [
            {"first": check["first"], "second": check["second"], **point}
            for check in flat_rows for point in check["samples"]
        ][:240],
    }
