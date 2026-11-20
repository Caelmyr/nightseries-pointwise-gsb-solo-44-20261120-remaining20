"""Integration in reference coordinates with explicit per-pixel support.

Median stacking removes most cosmic-ray hits, but the surviving high values are
not enough evidence on their own: a single-frame bright value can equally be a
real star whose sub-pixel alignment and exposure length differ.  The per-pixel
rejection below therefore requires both repeated coverage and a robust
deviation against the other co-registered samples.  Pixels with too few
repeated samples keep every available value and are reported as unresolved
instead of having their data discarded.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from PIL import Image

from .calibration import Plane, finite_median, robust_summary
from .domain import DomainError, MAX_ACTIVE_PIXELS, boolean, enum, integer
from .photometry import filter_photometry_frames
from .preview import png
from .registration import offset_for, warp_to_reference

STACK_MODES = {"mean", "median", "exposure"}

# Per-pixel support states, shared by the report, the coverage preview and the
# downloadable status FITS.
STATUS_UNCOVERED = 0
STATUS_CLEAN = 1
STATUS_REJECTED = 2
STATUS_UNRESOLVED = 3
STATUS_NAMES = {
    STATUS_UNCOVERED: "uncovered",
    STATUS_CLEAN: "clean",
    STATUS_REJECTED: "rejected",
    STATUS_UNRESOLVED: "unresolved",
}
# A robust high-side outlier call needs repeated, corroborating samples.  With
# two or three co-registered values a star peak and a cosmic ray are not
# separable, so those pixels stay unresolved and keep their existing samples.
MIN_DECISION_SUPPORT = 4
# Conservative sigma cut: a genuine star only varies smoothly between aligned
# frames, while a cosmic ray is a sharp, much larger positive excursion.
REJECT_SIGMA = 5.0
# A temporal MAD far above the read/sky noise floor means the samples split
# into separate levels (for example half the frames landing on a different
# registration position).  Such a pixel cannot be assigned a single reference
# level, so it is reported as unresolved rather than force-combined.
BIMODAL_SIGMA = 6.0

# Preview palette for the four support states.
STATUS_COLORS = {
    STATUS_UNCOVERED: (13, 14, 22),
    STATUS_CLEAN: (52, 132, 142),
    STATUS_REJECTED: (226, 118, 74),
    STATUS_UNRESOLVED: (198, 162, 74),
}


@dataclass(frozen=True)
class StackProduct:
    plane: Plane
    coverage: np.ndarray
    status: np.ndarray
    report: dict


def _center_sky(plane, seconds, center):
    keep = plane.valid & ~plane.saturated
    rate = np.where(keep, plane.values / seconds, np.nan)
    level = robust_summary(rate)["median"] if center else 0.0
    if level is None:
        return rate, None
    return rate - level, level


def _preview_shape(image):
    image.thumbnail((1000, 1000), Image.Resampling.NEAREST)
    return image


def coverage_preview(coverage, status, maximum):
    """Colour the four support states, with brightness tracking frame count."""
    level = coverage.astype(float) / max(1, maximum)
    rgb = np.zeros((*coverage.shape, 3), dtype=np.float64)
    for state, color in STATUS_COLORS.items():
        rgb[status == state] = color
    # Clean and unresolved regions keep a visible coverage gradient.
    graded = (STATUS_CLEAN, STATUS_UNRESOLVED)
    for state in graded:
        mask = status == state
        base = np.array(STATUS_COLORS[state], dtype=np.float64)
        rgb[mask] = base * (0.45 + 0.55 * level[mask])[:, None]
    rgb[status == STATUS_UNCOVERED] = STATUS_COLORS[STATUS_UNCOVERED]
    return png(_preview_shape(Image.fromarray(np.rint(rgb).astype(np.uint8), "RGB")))


def _nan_median(values, axis):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _temporal_noise_floor(stack, present):
    """Median per-pixel MAD across the co-registered frames, in ADU/s.

    Local MAD collapses to zero in quiet regions or when only a handful of
    frames cover a pixel, which would let read noise masquerade as an outlier.
    The global median temporal spread supplies a minimum threshold instead.
    """
    medians = _nan_median(stack, axis=0)
    spread = _nan_median(np.abs(stack - medians[None, :, :]), axis=0)
    sigma = 1.482602218505602 * spread
    covered = present.any(axis=0)
    candidates = sigma[covered & (sigma > 0)]
    return float(np.median(candidates)) if candidates.size else 0.0


def reject_cosmic_rays(stack, present):
    """Return a per-sample rejection mask and the per-pixel support status.

    Only high-side (bright) deviations are considered, and a sample is rejected
    only when the remaining repeated samples agree tightly enough to provide a
    reference level.  Everything else is either clean or explicitly
    unresolved; unresolved pixels retain all of their original samples.
    """
    _, height, width = stack.shape
    coverage = present.sum(axis=0).astype(np.uint16)
    status = np.full((height, width), STATUS_UNCOVERED, dtype=np.uint8)
    status[coverage > 0] = STATUS_CLEAN
    enough = coverage >= MIN_DECISION_SUPPORT
    # Thin edges and gaps: samples exist but repeated evidence does not.
    status[(coverage > 0) & ~enough] = STATUS_UNRESOLVED
    if not np.any(enough):
        return np.zeros_like(present, dtype=bool), status, coverage

    work = np.where(present, stack, np.nan)
    median = _nan_median(work, axis=0)
    mad = _nan_median(np.abs(work - median[None, :, :]), axis=0)
    local_sigma = 1.482602218505602 * mad
    noise_floor = _temporal_noise_floor(stack, present)
    # Samples splitting into distant levels have no defensible single
    # reference; keep them all and flag the pixel as unresolved.
    bimodal = enough & (local_sigma > BIMODAL_SIGMA * max(noise_floor, 1e-12))
    threshold = _deviation_threshold(local_sigma, median, noise_floor)
    deviation = work - median[None, :, :]
    candidate = (
        present
        & enough[None, :, :]
        & ~bimodal[None, :, :]
        & (deviation > threshold[None, :, :])
    )
    candidate_count = candidate.sum(axis=0)

    # The minority must be rejectable against a surviving reference of at
    # least two corroborating samples; rejecting half or more is not evidence
    # of cosmic rays, so split pixels stay unresolved.
    rejectable = (
        enough
        & (candidate_count >= 1)
        & (candidate_count * 2 < coverage)
        & (coverage - candidate_count >= 2)
    )

    # Re-run the deviation test against the surviving reference so an unusual
    # but self-consistent group cannot force a rejection.
    reference = np.where(candidate & rejectable[None, :, :], np.nan, work)
    reference_median = _nan_median(reference, axis=0)
    reference_mad = _nan_median(np.abs(reference - reference_median[None, :, :]), axis=0)
    reference_threshold = _deviation_threshold(
        1.482602218505602 * reference_mad, reference_median, noise_floor
    )
    confirmed = candidate & rejectable[None, :, :] & (
        deviation > reference_threshold[None, :, :]
    )
    # Every candidate at a pixel must survive the stricter re-test; otherwise
    # the evidence is ambiguous and the whole pixel remains unresolved.
    fully_confirmed = confirmed.sum(axis=0) == candidate_count
    final_reject_pixels = rejectable & fully_confirmed
    rejected = candidate & final_reject_pixels[None, :, :]

    ambiguous = enough & ((candidate_count > 0) | bimodal) & ~final_reject_pixels
    status[final_reject_pixels] = STATUS_REJECTED
    status[ambiguous] = STATUS_UNRESOLVED
    return rejected, status, coverage


def _deviation_threshold(sigma, center, noise_floor):
    threshold = REJECT_SIGMA * np.maximum(sigma, noise_floor)
    # A zero threshold (for example an entirely flat synthetic field) would
    # flag every deviation; fall back to a tiny relative floor on the level.
    return np.maximum(threshold, 1e-9 * (1.0 + np.abs(center)))


def _combine(stack, present, weights, mode):
    if mode == "median":
        values = finite_median(np.where(present, stack, np.nan), axis=0)
    else:
        weighted = stack * weights[:, None, None]
        totals = np.nansum(np.where(present, weighted, 0.0), axis=0)
        weight_total = np.nansum(np.where(present, weights[:, None, None], 0.0), axis=0)
        values = np.divide(
            totals, weight_total, out=np.full(stack.shape[1:], np.nan), where=weight_total > 0
        )
    return values


def compute_stack(images, options):
    doc = images.document
    band, frames = filter_photometry_frames(doc, options.get("band"))
    mode = enum(options.get("mode", "mean"), "叠加方式", STACK_MODES)
    min_support = integer(options.get("min_support", 2), "最低覆盖帧数", 1, len(frames))
    subtract_background = boolean(options.get("subtract_background", False), "扣除帧背景")
    if len(frames) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("当前叠加超过 2400 万像素，先挂起部分科学曝光")
    if len(frames) * images.dims.pixels > 12_000_000:
        raise DomainError("当前叠加最多 1200 万采样，可挂起部分科学曝光后再处理")
    shape = images.dims.shape
    stack = np.full((len(frames), *shape), np.nan, dtype=np.float64)
    present = np.zeros((len(frames), *shape), dtype=bool)
    weights = np.ones(len(frames), dtype=np.float64)
    used = []
    skipped = []
    slot = 0
    for frame in frames:
        frame_id = frame["id"]
        offset = offset_for(doc, frame_id)
        if offset is None:
            skipped.append({"frame": frame_id, "filename": frame["filename"], "state": "unaligned"})
            continue
        plane = warp_to_reference(images.calibrated(frame_id), offset)
        rates, removed = _center_sky(plane, frame["exposure"], subtract_background)
        valid = np.isfinite(rates)
        if not np.any(valid):
            skipped.append({"frame": frame_id, "filename": frame["filename"], "state": "no_support"})
            continue
        stack[slot] = np.where(valid, rates, np.nan)
        present[slot] = valid
        weights[slot] = frame["exposure"] if mode == "exposure" else 1.0
        used.append({
            "frame": frame_id,
            "filename": frame["filename"],
            "exposure": frame["exposure"],
            "removed_background": removed,
            "valid_fraction": float(np.mean(valid)),
            "offset": [offset.dx, offset.dy],
            "rejected_pixels": 0,
            "rejected_fraction": 0.0,
        })
        slot += 1
    if len(used) < min_support:
        raise DomainError("已对准且有有效像素的曝光数量不足以达到最低覆盖")

    stack = stack[:slot]
    present = present[:slot]
    weights = weights[:slot]
    rejected, status, raw_coverage = reject_cosmic_rays(stack, present)
    # The coverage count and every output grid derive from the same accepted
    # sample set: rejected samples are removed before combining and counting.
    accepted = present & ~rejected
    coverage = accepted.sum(axis=0).astype(np.uint16)
    for index, row in enumerate(used):
        count = int(np.count_nonzero(rejected[index]))
        row["rejected_pixels"] = count
        row["rejected_fraction"] = float(count / max(1, int(np.count_nonzero(present[index]))))

    qualified = coverage >= min_support
    values = _combine(stack, accepted, weights, mode)
    values[~qualified] = np.nan
    result = Plane(values, qualified, np.zeros(shape, dtype=bool))

    raw_total = int(raw_coverage.sum())
    rejected_total = int(rejected.sum())
    status_counts = {
        name: int(np.count_nonzero(status == code)) for code, name in STATUS_NAMES.items()
    }
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
        "raw_coverage_histogram": np.bincount(
            raw_coverage.ravel(), minlength=len(used) + 1
        ).tolist(),
        "status_counts": status_counts,
        "qualified_fraction": float(np.mean(qualified)),
        "rejected_samples": rejected_total,
        "rejected_sample_fraction": rejected_total / raw_total if raw_total else 0.0,
        "unresolved_fraction": status_counts["unresolved"] / shape[0] / shape[1],
        "unit": "ADU/s",
    }
    return StackProduct(result, coverage, status, report)
