"""Horizontal readout stripe removal gated on bias-frame evidence.

The detector may imprint a periodic row-wise pattern whose affected rows and
strength differ from exposure to exposure.  A stripe model is only accepted when
the same period can be found in the raw bias frames; each science exposure is
then asked to confirm that period on its own pixels.  Star halos, saturated and
masked pixels never participate in the estimate, and a slowly varying sky level
is removed before the periodic fit.  Frames without enough evidence keep the
ordinary calibrated plane unchanged, and the report states why.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import binary_dilation, median_filter

from .calibration import Plane, robust_summary
from .domain import DomainError

# At least this many enabled bias frames must carry the same stripe period.
MIN_BIAS_FRAMES = 2
# A bias row / science row needs this many clean pixels to be usable.
MIN_ROW_SUPPORT = 8
# Period candidates live inside these bounds, in detector rows.
MIN_PERIOD = 3.0
# Spectral peak must stand out from the local continuum by this factor.
SPECTRUM_SIGNIFICANCE = 3.0
# Per-frame fit must reach this signal-to-noise ratio to be applied.
FIT_MIN_SNR = 3.0
# Fraction of rows surviving all masks for a science fit to be credible.
FIT_MIN_ROW_FRACTION = 0.55
# Each phase bin needs this many rows for a robust phase-median estimate.
MIN_PHASE_BIN_SUPPORT = 3
# Detrending window in confirmed-period units, short enough to follow a slow
# sky trend but several times longer than the readout stripe.
TREND_WINDOW_PERIODS = 4
# A second, high-pass window used only to estimate the stochastic row noise.
NOISE_WINDOW_PERIODS = 2.5

REASON_LABELS = {
    "no_bias": "没有可用偏置图，无法确认读出条纹周期",
    "few_bias": "偏置图数量不足，无法交叉确认周期",
    "no_period": "偏置图中未找到一致的周期性读出条纹",
    "few_rows": "本帧干净行不足，无法独立复现条纹",
    "period_mismatch": "本帧行信号与偏置确认的周期对不上",
    "weak_signal": "本帧条纹幅度低于读出噪声，证据不足",
    "no_support": "有效像素被坏区或饱和占满，无法估计条纹",
}


@dataclass(frozen=True)
class StripeEvidence:
    period: float | None
    reason: str | None
    bias_rows: tuple
    frame_count: int
    agreeing: int
    fit_rms_adu: float
    detail: dict

    @property
    def confirmed(self):
        return self.period is not None


def _odd_window(length, minimum):
    size = min(length if length % 2 else length - 1, int(math.ceil(minimum)) // 2 * 2 + 1)
    return max(3, size | 1)


def row_signal(values, valid, *, min_support=MIN_ROW_SUPPORT):
    """Per-row robust level, with rows lacking clean pixels left undefined."""
    height = values.shape[0]
    levels = np.full(height, np.nan)
    counts = np.zeros(height, dtype=np.int32)
    for y in range(height):
        samples = values[y][valid[y]]
        finite = samples[np.isfinite(samples)]
        if finite.size >= min_support:
            levels[y] = float(np.median(finite))
            counts[y] = finite.size
    return levels, counts


def detrend_rows(levels, max_period):
    """Remove slow variations while preserving periods up to max_period."""
    defined = np.isfinite(levels)
    if defined.sum() < 9:
        return levels, None
    fill = float(np.nanmedian(levels))
    window = _odd_window(levels.size, max_period * TREND_WINDOW_PERIODS)
    trend = median_filter(np.where(defined, levels, fill), size=window, mode="reflect")
    residual = levels - trend
    residual[~defined] = np.nan
    return residual, trend


def row_spectrum(levels, max_period):
    """Normalised amplitude spectrum of a detrended row-level series."""
    residual, _ = detrend_rows(levels, max_period)
    defined = np.isfinite(residual)
    count = int(np.count_nonzero(defined))
    if count < 16:
        return None
    filled = np.where(defined, residual, 0.0)
    spectrum = np.abs(np.fft.rfft(filled)) / max(1, count)
    height = levels.size
    freqs = np.fft.rfftfreq(height)
    upper = max_period
    lower = min(MIN_PERIOD, upper)
    candidates = np.where((height * freqs >= lower) & (height * freqs <= upper))[0]
    candidates = candidates[candidates > 0]
    if candidates.size == 0:
        return None
    return spectrum, freqs, candidates


def _peak_period(spectrum, freqs, candidates, height):
    band = spectrum[candidates]
    order = np.argsort(band)[::-1]
    peak_index = candidates[order[0]]
    # Robust local continuum: median of the other candidate bins.
    continuum = float(np.median(np.delete(band, np.where(candidates == peak_index)[0])))
    scatter = float(np.median(np.abs(band - continuum))) or 1e-12
    strength = float(spectrum[peak_index] / max(continuum, 1e-12))
    prominence = float((spectrum[peak_index] - continuum) / scatter)
    period = height / peak_index
    return {
        "period": float(period),
        "bin": int(peak_index),
        "strength": strength,
        "prominence": prominence,
        "continuum": continuum,
        "accepted": strength >= SPECTRUM_SIGNIFICANCE and prominence >= SPECTRUM_SIGNIFICANCE,
    }


def _period_match(residual, usable_rows, expected_period, height):
    """Check whether the bias-confirmed period is itself strong in this frame.

    Unlike a global argmax, this stays valid when saturated bands or detector
    edges put more power into unrelated low frequencies.
    """
    series = np.where(usable_rows, residual, 0.0)
    count = max(1, int(np.count_nonzero(usable_rows)))
    spectrum = np.abs(np.fft.rfft(series)) / count
    target = int(round(height / expected_period))
    if not 0 < target < spectrum.size:
        return None
    neighbors = np.r_[spectrum[max(1, target - 2):target], spectrum[target + 1:target + 3]]
    continuum = float(np.median(neighbors)) if neighbors.size else float(np.median(spectrum[1:]))
    scatter = float(np.median(np.abs(spectrum[1:] - np.median(spectrum[1:])))) or 1e-12
    prominence = float((spectrum[target] - continuum) / scatter)
    strength = float(spectrum[target] / max(continuum, 1e-12))
    if strength < SPECTRUM_SIGNIFICANCE or prominence < SPECTRUM_SIGNIFICANCE:
        return None
    return {
        "period": float(expected_period),
        "bin": target,
        "strength": strength,
        "prominence": prominence,
        "continuum": continuum,
        "accepted": True,
    }


def detect_bias_stripes(images):
    """Confirm a common readout period from the enabled raw bias frames."""
    bias_frames = [
        row for row in images.document["calibration_frames"]
        if row["enabled"] and row["kind"] == "bias"
    ]
    height = images.dims.height
    max_period = max(MIN_PERIOD + 1.0, height / 3.0)
    rows = []
    votes = []
    if len(bias_frames) < MIN_BIAS_FRAMES:
        reason = "few_bias" if bias_frames else "no_bias"
        return StripeEvidence(None, reason, tuple(rows), len(bias_frames), 0, 0.0, {})
    noise_levels = []
    for frame in bias_frames:
        plane = images.raw(frame["id"], use_detector_mask=True)
        usable = plane.valid & ~plane.saturated
        levels, counts = row_signal(plane.values, usable)
        residual, _ = detrend_rows(levels, max_period)
        defined = residual[np.isfinite(residual)]
        noise_levels.append(float(np.median(np.abs(defined - np.median(defined))) * 1.4826))
        spectrum = row_spectrum(levels, max_period)
        if spectrum is None:
            rows.append({"frame": frame["id"], "filename": frame["filename"], "accepted": False})
            continue
        peak = _peak_period(*spectrum, height)
        row = {"frame": frame["id"], "filename": frame["filename"], **peak}
        rows.append(row)
        if peak["accepted"]:
            votes.append(peak["period"])
    if len(votes) < MIN_BIAS_FRAMES:
        return StripeEvidence(
            None, "no_period", tuple(rows), len(bias_frames), len(votes),
            float(np.median(noise_levels)), {},
        )
    votes = np.asarray(votes)
    # Group votes whose periods agree within 10%; choose the largest cluster.
    voted = votes[np.argsort(votes)]
    best_cluster = []
    for anchor in voted:
        cluster = voted[np.abs(voted - anchor) <= 0.1 * anchor]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    agreeing = int(best_cluster.size)
    if agreeing < MIN_BIAS_FRAMES or agreeing < math.ceil(0.6 * len(bias_frames)):
        return StripeEvidence(
            None, "no_period", tuple(rows), len(bias_frames), agreeing,
            float(np.median(noise_levels)), {},
        )
    period = float(np.median(best_cluster))
    detail = {
        "spectrum_bins": _spectrum_trace(images, bias_frames, max_period, height),
        "period_spread": float(np.std(best_cluster)),
        "noise_adu": float(np.median(noise_levels)),
    }
    return StripeEvidence(
        period, None, tuple(rows), len(bias_frames), agreeing,
        float(np.median(noise_levels)), detail,
    )


def _spectrum_trace(images, bias_frames, max_period, height, limit=256):
    """Down-sampled mean bias spectrum for the diagnostic chart."""
    traces = []
    for frame in bias_frames:
        plane = images.raw(frame["id"], use_detector_mask=True)
        levels, _ = row_signal(plane.values, plane.valid & ~plane.saturated)
        spectrum = row_spectrum(levels, max_period)
        if spectrum is not None:
            traces.append(spectrum[0])
    if not traces:
        return {"periods": [], "power": []}
    mean = np.mean(np.stack(traces), axis=0)
    bins = np.arange(mean.size)
    keep = bins[1:limit + 1]
    periods = np.divide(height, keep, out=np.full_like(keep, np.nan, dtype=float), where=keep > 0)
    return {"periods": periods.tolist(), "power": mean[keep].tolist()}


def _clean_science_mask(plane):
    """Valid, unsaturated pixels with stars and their halos removed."""
    usable = plane.valid & ~plane.saturated
    values = plane.values
    stats = robust_summary(values[usable])
    if stats["median"] is None:
        return usable
    floor = max(3.0 * max(stats["sigma"], 1e-9), 1e-9)
    bright = usable & (values > stats["median"] + floor)
    # Saturated cores and bright pixels both carry star structure; dilate their
    # union so a 3-pixel halo (including bleed into neighbouring rows) is kept
    # out of the row medians.
    seeds = bright | (plane.valid & plane.saturated)
    return usable & ~binary_dilation(seeds, iterations=3)


def fit_frame_stripes(plane, evidence):
    """Estimate the per-row stripe matching the bias-confirmed period."""
    if not evidence.confirmed:
        return _skip(evidence.reason or "no_period", evidence, plane)
    height, width = plane.values.shape
    clean = _clean_science_mask(plane)
    if np.count_nonzero(clean) < height * MIN_ROW_SUPPORT:
        return _skip("no_support", evidence, plane)
    levels, counts = row_signal(plane.values, clean)
    usable_rows = np.isfinite(levels)
    if usable_rows.sum() < max(8, int(height * FIT_MIN_ROW_FRACTION)):
        return _skip("few_rows", evidence, plane)
    # Detrend on a window tied to the confirmed period: long enough to preserve
    # the stripe, short enough to follow a genuinely slow sky gradient.
    residual, trend = detrend_rows(levels, evidence.period * (TREND_WINDOW_PERIODS / 2))
    frame_peak = _period_match(residual, usable_rows, evidence.period, height)
    if frame_peak is None:
        return _skip("period_mismatch", evidence, plane,
                     rows_used=int(usable_rows.sum()), row_fraction=float(usable_rows.sum() / height))
    # Phase-bin robust medians at the bias period.  Rows sample the phase at
    # integer positions, so never create more bins than one period can fill.
    phase = (np.arange(height) / evidence.period) % 1.0
    distinct_phases = np.unique(np.round(phase, decimals=6)).size
    bin_count = max(4, min(16, int(round(evidence.period)), distinct_phases))
    bins = np.floor(phase * bin_count).astype(int)
    profile = np.full(bin_count, np.nan)
    support = np.zeros(bin_count, dtype=int)
    for index in range(bin_count):
        members = residual[(bins == index) & usable_rows]
        members = members[np.isfinite(members)]
        support[index] = members.size
        if members.size:
            profile[index] = float(np.median(members))
    if np.count_nonzero(support >= MIN_PHASE_BIN_SUPPORT) < max(3, bin_count * 0.7):
        return _skip("few_rows", evidence, plane, peak=frame_peak,
                     rows_used=int(usable_rows.sum()), row_fraction=float(usable_rows.sum() / height))
    # Close the periodic profile and interpolate onto every row.
    filled = profile.copy()
    missing = ~np.isfinite(filled)
    if missing.any():
        anchors = np.isfinite(filled)
        filled[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(anchors), filled[anchors])
    extended = np.concatenate([filled, filled[:1]])
    model = np.interp(phase * bin_count, np.arange(bin_count + 1), extended, period=bin_count)
    # Stochastic row-noise floor: high-pass the residual at a couple of stripe
    # periods so slow sky structure never inflates the noise, then take the
    # robust scatter.  The fitted phase bins average several rows, so the SNR
    # compares the coherent amplitude with the averaged uncertainty.
    highpassed, _ = detrend_rows(residual, evidence.period * (NOISE_WINDOW_PERIODS / 2))
    noise_series = highpassed[usable_rows] - model[usable_rows]
    noise = float(np.median(np.abs(noise_series - np.median(noise_series))) * 1.4826)
    amplitude = float(np.nanmax(model) - np.nanmin(model))
    rows_per_bin = max(1.0, usable_rows.sum() / bin_count)
    fit_uncertainty = noise / math.sqrt(rows_per_bin)
    snr = amplitude / max(2.0 * fit_uncertainty, 1e-12)
    if snr < FIT_MIN_SNR:
        return _skip("weak_signal", evidence, plane, peak=frame_peak,
                     amplitude=amplitude, noise=noise, snr=snr,
                     rows_used=int(usable_rows.sum()), row_fraction=float(usable_rows.sum() / height))
    correction = np.broadcast_to(model[:, None], plane.values.shape).copy()
    corrected = np.where(plane.valid, plane.values - correction, np.nan)
    report = {
        "state": "measured",
        "reason": None,
        "period": evidence.period,
        "amplitude_adu": amplitude,
        "rms_adu": float(np.sqrt(np.nanmean(model**2))),
        "noise_adu": noise,
        "fit_uncertainty_adu": fit_uncertainty,
        "snr": float(snr),
        "rows_used": int(usable_rows.sum()),
        "row_fraction": float(usable_rows.sum() / height),
        "frame_period": frame_peak["period"],
        "frame_prominence": frame_peak["prominence"],
        "phase_bins": bin_count,
        "row_profile": _row_trace(levels, trend, model, usable_rows),
    }
    result = Plane(corrected, plane.valid, plane.saturated,
                   plane.notes + ("读出横带去除（偏置确认周期）",))
    return result, report, correction


def _skip(reason, evidence, plane, *, peak=None, amplitude=None, noise=None, snr=None,
          rows_used=0, row_fraction=0.0):
    report = {
        "state": "skipped",
        "reason": reason,
        "reason_text": REASON_LABELS.get(reason, reason),
        "period": evidence.period,
        "amplitude_adu": amplitude,
        "noise_adu": noise,
        "snr": snr,
        "frame_period": peak["period"] if peak else None,
        "frame_prominence": peak["prominence"] if peak else None,
        "rows_used": int(rows_used),
        "row_fraction": float(row_fraction),
        "row_profile": {"rows": [], "observed": [], "model": []},
    }
    return plane, report, np.zeros(plane.values.shape)


def _row_trace(levels, trend, model, usable, limit=240):
    height = levels.size
    step = max(1, height // limit)
    rows = np.arange(0, height, step)
    observed = levels - trend
    return {
        "rows": rows.tolist(),
        "observed": [float(observed[y]) if usable[y] else None for y in rows],
        "model": [float(model[y]) for y in rows],
    }


def stripe_diagnosis(images, frame_id):
    """Full evidence pack for the preview side panel and charts."""
    evidence = images.bias_stripe_evidence()
    frame = images.rows[frame_id]
    if frame["kind"] != "science":
        raise DomainError("去横带只用于科学曝光")
    plane = images.calibrated(frame_id)
    _, report, correction = fit_frame_stripes(plane, evidence)
    return {
        "frame": frame_id,
        "filename": frame["filename"],
        "revision": images.document["revision"],
        "confirmed": evidence.confirmed,
        "period": evidence.period,
        "reason": evidence.reason,
        "reason_text": REASON_LABELS.get(evidence.reason, "") if evidence.reason else "",
        "bias": {
            "frames": evidence.frame_count,
            "agreeing": evidence.agreeing,
            "required": MIN_BIAS_FRAMES,
            "fit_rms_adu": evidence.fit_rms_adu,
            "per_frame": list(evidence.bias_rows),
            "spectrum": evidence.detail.get("spectrum_bins", {"periods": [], "power": []}),
        },
        "fit": report,
        "correction_summary": robust_summary(correction[np.isfinite(correction)]) if correction.any() else None,
    }
