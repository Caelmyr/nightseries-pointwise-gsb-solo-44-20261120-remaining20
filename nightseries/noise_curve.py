"""Noise versus brightness curve from repeated flat levels and bias pairs.

Flats are grouped by their *measured* median brightness rather than exposure
time, so lamp changes form their own levels.  Every level keeps the identities
of the flat frames that actually entered a measured pair: two pairs sharing one
original frame only count as two distinct frames, never as extra independent
repeats.  A brightness level without enough independent frames (or with
saturation) is reported as evidence-incomplete and can never lower the stable
working upper bound on its own.
"""
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from .calibration import master_median, robust_summary
from .domain import DomainError, MAX_ACTIVE_PIXELS, integer, number, text
from .gain import (
    _compare_bias, _pick_representative_pairs, _trimmed_variance,
    calibration_pairs, compare_pair, sample_blocks,
)

MAX_PAIRS_PER_LEVEL = 12
GAIN_LIMITS = (0.02, 500.0)

# Machine-readable state codes are emitted by ``classify_levels``; the browser
# owns the Chinese labels and keeps this list aligned.
TILE_REJECTIONS = {
    "low_coverage": "有效像素覆盖不足",
    "low_signal": "净亮度低于最低信号",
    "lamp_drift": "配对灯源亮度变动",
    "trim_failed": "剪裁后样本不足",
    "floor_not_exceeded": "未超过读出噪声底",
    "gain_out_of_range": "增益估计越界",
}


@dataclass(frozen=True)
class NoiseCurveRequest:
    band: str
    tile_size: int
    low: float
    high: float
    stability: float
    level_gap: float
    tolerance: float
    min_independent: int
    min_net_adu: float

    @classmethod
    def request(cls, options):
        return cls(
            text(options.get("band", ""), "平场滤镜", 24, required=False),
            integer(options.get("tile_size", 32), "取样块宽度", 12, 96),
            number(options.get("low", 0.15), "下方裁边比例", 0.02, 0.42),
            number(options.get("high", 0.85), "上方裁边比例", 0.58, 0.98),
            number(options.get("stability", 0.025), "配对平均亮度允许差", 0.001, 0.10),
            number(options.get("level_gap", 0.08), "亮度分档相对间隔", 0.02, 0.50),
            number(options.get("tolerance", 0.10), "稳定允许相对偏差", 0.03, 0.35),
            integer(options.get("min_independent", 3), "独立复测最少平场数", 2, 8),
            number(options.get("min_net_adu", 200), "稳定判定最低净 ADU", 10, 50000),
        )


def enabled_flats(document, band):
    flats = [
        row for row in document["calibration_frames"]
        if row["enabled"] and row["kind"] == "flat" and row["band"] == band
    ]
    flats.sort(key=lambda row: (row["exposure"], row["filename"]))
    if len(flats) < 2:
        raise DomainError("该滤镜至少需要两张启用的平场才能比较重复噪声")
    return flats


def bias_floor(images, blocks):
    """Read-noise variance floor and master bias plane, reusing gain tool rules."""
    doc = images.document
    bias_pairs_all = calibration_pairs(doc, "bias", "")
    bias_pairs = _pick_representative_pairs(bias_pairs_all, 8)
    rows = [compare_pair(pair, images, blocks, _compare_bias) for pair in bias_pairs]
    checks, base = _pair_level_summary(rows, "variance_adu2")
    if base["count"] < 1:
        raise DomainError("偏置对缺少可比较的有效样本块，无法建立读出噪声底")
    distinct = {}
    for pair in bias_pairs:
        for frame in pair:
            distinct[frame["id"]] = frame
    planes = []
    for frame in distinct.values():
        raw = images.raw(frame["id"], True)
        planes.append(np.where(raw.valid & ~raw.saturated, raw.values, np.nan))
    master = master_median(planes)
    levels = [point["center_adu"] for row in rows for point in row["samples"]]
    return {
        "floor_variance_adu2": base["median"],
        "read_noise_adu": math.sqrt(max(base["median"], 0.0)),
        "bias_level_adu": robust_summary(levels)["median"],
        "bias_pair_count": base["count"],
        "bias_pairs": checks,
        "master": master,
        "bias_frame_count": len(distinct),
    }


def _pair_level_summary(entries, key):
    accepted, rejected = [], []
    for pair in entries:
        if not pair["samples"]:
            rejected.append({
                "first": pair["first"],
                "second": pair["second"],
                "exposure": pair["exposure"],
                "state": "no_valid_tiles",
                "median": None,
                "sigma": None,
                "patches": 0,
            })
            continue
        metrics = robust_summary([row[key] for row in pair["samples"]])
        accepted.append({
            "first": pair["first"],
            "second": pair["second"],
            "exposure": pair["exposure"],
            "state": "measured",
            "median": metrics["median"],
            "sigma": metrics["sigma"],
            "patches": len(pair["samples"]),
        })
    return accepted + rejected, robust_summary([row["median"] for row in accepted])


def frame_profile(images, frame, blocks, bias_plane):
    """Whole-frame saturation and per-tile bias-corrected net brightness."""
    raw = images.raw(frame["id"], True)
    valid = raw.valid
    valid_count = int(np.count_nonzero(valid))
    saturated = int(np.count_nonzero(raw.saturated))
    tile_levels = []
    for block in blocks:
        mask = valid[block] & ~raw.saturated[block] & np.isfinite(bias_plane[block])
        if float(np.mean(mask)) < 0.75:
            continue
        net = raw.values[block][mask] - bias_plane[block][mask]
        tile_levels.append(float(np.median(net)))
    return {
        "id": frame["id"],
        "filename": frame["filename"],
        "exposure": frame["exposure"],
        "net_level_adu": float(np.median(tile_levels)) if tile_levels else None,
        "tile_levels": tile_levels,
        "saturated_fraction": saturated / valid_count if valid_count else 1.0,
        "invalid_fraction": 1.0 - valid_count / raw.values.size,
    }


def group_levels(profiles, gap):
    """Greedy clustering on measured net brightness with a relative gap."""
    measurable = [row for row in profiles if row["net_level_adu"] is not None]
    measurable.sort(key=lambda row: row["net_level_adu"])
    groups = []
    for row in measurable:
        if not groups:
            groups.append([row])
            continue
        anchor = float(np.median([item["net_level_adu"] for item in groups[-1]]))
        if anchor <= 0 or abs(row["net_level_adu"] - anchor) / anchor > gap:
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def measure_pair(images, left, right, blocks, bias_plane, floor_variance, spec):
    first = images.raw(left["id"], True)
    second = images.raw(right["id"], True)
    kept = []
    rejections = {}

    def reject(code):
        rejections[code] = rejections.get(code, 0) + 1

    for block in blocks:
        mask = first.valid[block] & second.valid[block] & ~first.saturated[block] & ~second.saturated[block]
        mask &= np.isfinite(bias_plane[block])
        if float(np.mean(mask)) < 0.75:
            reject("low_coverage")
            continue
        a = first.values[block][mask] - bias_plane[block][mask]
        b = second.values[block][mask] - bias_plane[block][mask]
        level_a, level_b = float(np.median(a)), float(np.median(b))
        level = 0.5 * (level_a + level_b)
        if level < spec.min_net_adu:
            reject("low_signal")
            continue
        if abs(level_a - level_b) > spec.stability * level:
            reject("lamp_drift")
            continue
        variance, _ = _trimmed_variance(a - b)
        if variance is None:
            reject("trim_failed")
            continue
        if variance <= floor_variance * 1.1:
            reject("floor_not_exceeded")
            continue
        gain = level / (variance - floor_variance)
        if not GAIN_LIMITS[0] <= gain <= GAIN_LIMITS[1]:
            reject("gain_out_of_range")
            continue
        kept.append({
            "net_level_adu": level,
            "variance_adu2": variance,
            "gain_e_per_adu": gain,
            "x": block[1].start,
            "y": block[0].start,
        })
    pair = {
        "first_id": left["id"],
        "second_id": right["id"],
        "first": left["filename"],
        "second": right["filename"],
        "exposure": left["exposure"],
        "samples": kept,
        "rejection_reasons": rejections,
        "tiles_total": len(blocks),
    }
    if kept:
        gains = robust_summary([row["gain_e_per_adu"] for row in kept])
        variances = robust_summary([row["variance_adu2"] for row in kept])
        levels = robust_summary([row["net_level_adu"] for row in kept])
        pair.update({
            "state": "measured",
            "net_level_adu": levels["median"],
            "gain_median": gains["median"],
            "gain_sigma": gains["sigma"],
            "variance_median": variances["median"],
            "noise_adu": math.sqrt(max(variances["median"], 0.0)),
            "tiles_kept": len(kept),
        })
    else:
        pair.update({
            "state": "no_valid_tiles",
            "net_level_adu": None,
            "gain_median": None,
            "gain_sigma": None,
            "variance_median": None,
            "noise_adu": None,
            "tiles_kept": 0,
        })
    return pair


def reference_transfer(level_rows, floor_variance):
    """Photon-transfer fit variance = floor + net_level / gain.

    Runs before level classification, so candidate levels are chosen on
    objective evidence alone (no saturation, enough independent frames) and a
    robust median gain guards against an outlier bright level dragging the fit.
    """
    candidates = [
        row for row in level_rows
        if not row["saturated_present"] and row["variance_median"] is not None
        and row["median_net_adu"] > 0 and row["independent_frames"] >= 2
    ]
    if not candidates:
        return None, "none"
    raw_gains = np.array([
        row["median_net_adu"] / max(row["variance_median"] - floor_variance, 1e-9)
        for row in candidates
    ], dtype=np.float64)
    raw_gains = raw_gains[np.isfinite(raw_gains) & (raw_gains > 0)]
    if not raw_gains.size:
        return None, "none"
    center = float(np.median(raw_gains))
    spread = 1.482602218505602 * float(np.median(np.abs(raw_gains - center))) or center * 0.05
    clean = [
        row for row in candidates
        if abs(row["median_net_adu"] / max(row["variance_median"] - floor_variance, 1e-9) - center) <= 2.5 * spread
    ]
    levels = np.array([row["median_net_adu"] for row in clean], dtype=np.float64)
    variances = np.array([row["variance_median"] for row in clean], dtype=np.float64)
    denominator = float(np.sum(levels**2))
    if len(clean) >= 2 and denominator > 0:
        slope = float(np.sum(levels * (variances - floor_variance)) / denominator)
        if slope > 0:
            return 1.0 / slope, "ptc_fit"
    return center, "median_fallback"


def classify_levels(level_rows, spec, gain_ref):
    for row in level_rows:
        # Frame-level evidence is checked before tile-level measurement
        # outcomes so a single suspicious shot can never masquerade as a
        # measured-and-rejected bright ceiling.
        if row["saturated_present"]:
            row["state"] = "saturated_present"
            row["state_reason"] = "档内平场出现饱和像素，不能据此压低工作上限"
            continue
        if row["independent_frames"] < spec.min_independent:
            row["state"] = "insufficient_independent"
            row["state_reason"] = (
                f"只有 {row['independent_frames']} 张独立平场进入测量，"
                f"少于要求的 {spec.min_independent} 张"
            )
            continue
        if row["pair_count_measured"] == 0:
            row["state"] = "no_valid_tiles"
            row["state_reason"] = "没有任何通过剪裁的取样块"
            continue
        if row["median_net_adu"] < spec.min_net_adu:
            row["state"] = "low_signal"
            row["state_reason"] = "净亮度低于稳定判定最低信号"
            continue
        if gain_ref is None:
            row["state"] = "no_model"
            row["state_reason"] = "缺少可拟合的光子转移参考"
            continue
        expected_variance = row["floor_variance_adu2"] + row["median_net_adu"] / gain_ref
        expected_noise = math.sqrt(max(expected_variance, 0.0))
        gain_dev = abs(row["gain_median"] - gain_ref) / gain_ref
        noise_dev = abs(row["noise_adu"] - expected_noise) / expected_noise if expected_noise > 0 else 1.0
        row["expected_noise_adu"] = expected_noise
        row["gain_deviation"] = gain_dev
        row["noise_deviation"] = noise_dev
        if gain_dev <= spec.tolerance and noise_dev <= spec.tolerance:
            row["state"] = "stable"
            row["state_reason"] = "增益与噪声均在参考曲线允许偏差内"
        else:
            row["state"] = "deviation_confirmed"
            causes = []
            if gain_dev > spec.tolerance:
                causes.append(f"增益偏离 {gain_dev:.1%}")
            if noise_dev > spec.tolerance:
                causes.append(f"噪声偏离 {noise_dev:.1%}")
            row["state_reason"] = "；".join(causes) + "，且有足够独立复测"


def stable_window(level_rows, spec):
    stable = [row for row in level_rows if row["state"] == "stable"]
    if not stable:
        return {
            "lower_adu": None,
            "upper_adu": None,
            "status": "no_stable_level",
            "note": "没有任何亮度档同时满足独立复测与偏差要求，无法给出稳定工作范围",
            "stable_level_count": 0,
            "evidence_incomplete_levels": [],
        }
    lower = min(row["median_net_adu"] for row in stable)
    upper = max(row["median_net_adu"] for row in stable)
    brighter = [row for row in level_rows if row["median_net_adu"] > upper]
    unverified_gaps = [
        row["index"] for row in level_rows
        if lower < row["median_net_adu"] < upper and row["state"] != "stable"
    ]
    result = {
        "lower_adu": lower,
        "upper_adu": upper,
        "stable_level_count": len(stable),
        "unverified_gap_levels": unverified_gaps,
        "evidence_incomplete_levels": [],
    }
    deviation = next((row for row in brighter if row["state"] == "deviation_confirmed"), None)
    if deviation is not None:
        result.update({
            "status": "confirmed_ceiling",
            "note": (
                f"更亮的第 {deviation['index'] + 1} 档（{deviation['median_net_adu']:.0f} ADU）"
                f"经 {deviation['independent_frames']} 张独立平场复测确认偏离，工作上限止于其上一档"
            ),
            "first_deviation_level": deviation["index"],
        })
        return result
    if brighter:
        blockers = []
        for row in brighter:
            blockers.append({"level": row["index"], "state": row["state"], "reason": row["state_reason"]})
        result.update({
            "status": "evidence_incomplete",
            "evidence_incomplete_levels": blockers,
            "note": (
                "更亮的档位缺少合格独立复测或存在饱和，证据不足；"
                "工作上限保持在已验证的最高稳定档，不据可疑图对压低上限"
            ),
        })
        return result
    result.update({
        "status": "verified_top",
        "note": "没有比最高稳定档更亮的平场，上限即为当前已验证的最高亮度档",
    })
    return result


def noise_curve_report(images, options):
    doc = images.document
    spec = NoiseCurveRequest.request(options)
    flats = enabled_flats(doc, spec.band)
    blocks = sample_blocks(images.dims.shape, spec.tile_size, spec.low, spec.high)
    bias = bias_floor(images, blocks)
    floor_variance = bias["floor_variance_adu2"]

    distinct_ids = {frame["id"] for frame in flats} | {
        frame["id"] for pair in calibration_pairs(doc, "bias", "") for frame in pair
    }
    if len(distinct_ids) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("当前平场与偏置超过 2400 万校准像素，可挂起部分重复平场")

    profiles = [frame_profile(images, frame, blocks, bias["master"]) for frame in flats]
    excluded = [
        {"filename": row["filename"], "reason": "中心取样区没有足够有效像素"}
        for row in profiles if row["net_level_adu"] is None
    ]
    groups = group_levels([row for row in profiles if row["net_level_adu"] is not None], spec.level_gap)

    level_rows = []
    for index, group in enumerate(groups):
        all_pairs = list(combinations(group, 2))
        selected = _pick_representative_pairs(all_pairs, MAX_PAIRS_PER_LEVEL)
        pairs = [
            measure_pair(images, left, right, blocks, bias["master"], floor_variance, spec)
            for left, right in selected
        ]
        measured_pairs = [pair for pair in pairs if pair["state"] == "measured"]
        used_ids = set()
        for pair in measured_pairs:
            used_ids.add(pair["first_id"])
            used_ids.add(pair["second_id"])
        frame_rows = []
        for profile in group:
            spread = robust_summary(profile["tile_levels"])
            frame_rows.append({
                "id": profile["id"],
                "filename": profile["filename"],
                "exposure": profile["exposure"],
                "net_level_adu": profile["net_level_adu"],
                "tile_level_spread_adu": spread["sigma"],
                "saturated_fraction": profile["saturated_fraction"],
                "invalid_fraction": profile["invalid_fraction"],
                "used_in_pairs": profile["id"] in used_ids,
            })
        level_median = float(np.median([row["net_level_adu"] for row in group]))
        level_spread = robust_summary([row["net_level_adu"] for row in group])
        if measured_pairs:
            gain = robust_summary([pair["gain_median"] for pair in measured_pairs])
            noise = robust_summary([pair["noise_adu"] for pair in measured_pairs])
            variance = robust_summary([pair["variance_median"] for pair in measured_pairs])
        else:
            gain = {"median": None, "sigma": None}
            noise = {"median": None, "sigma": None}
            variance = {"median": None}
        level_rows.append({
            "index": index,
            "median_net_adu": level_median,
            "level_spread_adu": level_spread["sigma"],
            "frame_count": len(group),
            "independent_frames": len(used_ids),
            "pair_count_total": len(all_pairs),
            "pair_count_measured": len(measured_pairs),
            "pair_count_capped": len(all_pairs) > len(selected),
            "gain_median": gain["median"],
            "gain_sigma": gain["sigma"],
            "noise_adu": noise["median"],
            "noise_sigma": noise["sigma"],
            "variance_median": variance["median"],
            "expected_noise_adu": None,
            "gain_deviation": None,
            "noise_deviation": None,
            "max_saturated_fraction": max(row["saturated_fraction"] for row in group),
            "saturated_present": any(row["saturated_fraction"] > 0 for row in group),
            "floor_variance_adu2": floor_variance,
            "state": "no_valid_tiles",
            "state_reason": "",
            "frames": frame_rows,
            "pairs": pairs,
        })

    gain_ref, reference_method = reference_transfer(level_rows, floor_variance)
    classify_levels(level_rows, spec, gain_ref)
    window = stable_window(level_rows, spec)

    if gain_ref is not None and level_rows:
        lows = [row["median_net_adu"] for row in level_rows]
        span = np.linspace(max(0.0, min(lows)), max(lows), 60)
        model_curve = [
            {
                "net_level_adu": float(level),
                "expected_variance_adu2": float(floor_variance + level / gain_ref),
                "expected_noise_adu": float(math.sqrt(max(floor_variance + level / gain_ref, 0.0))),
            }
            for level in span
        ]
    else:
        model_curve = []

    return {
        "revision": doc["revision"],
        "band": spec.band,
        "tile_size": spec.tile_size,
        "tile_count": len(blocks),
        "level_gap": spec.level_gap,
        "tolerance": spec.tolerance,
        "min_independent": spec.min_independent,
        "min_net_adu": spec.min_net_adu,
        "stability": spec.stability,
        "saturation_adu": images.saturation,
        "read_noise_adu": bias["read_noise_adu"],
        "bias_level_adu": bias["bias_level_adu"],
        "bias_pair_count": bias["bias_pair_count"],
        "bias_frame_count": bias["bias_frame_count"],
        "bias_pairs": bias["bias_pairs"],
        "reference_gain_e_per_adu": gain_ref,
        "reference_method": reference_method,
        "flat_frame_count": len(flats),
        "excluded_frames": excluded,
        "stable_range": window,
        "model_curve": model_curve,
        "levels": level_rows,
    }
