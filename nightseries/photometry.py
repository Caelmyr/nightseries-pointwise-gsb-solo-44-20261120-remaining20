"""Native-pixel aperture photometry of fixed reference-frame sources."""
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

import numpy as np

from .calibration import robust_summary
from .domain import (
    DomainError, MAX_ACTIVE_PIXELS, MAX_APERTURES, identifier, list_value, usable_science_frames,
)
from .registration import offset_for

SUBPIXEL_GRID = (-0.375, -0.125, 0.125, 0.375)
MAGNITUDE_FACTOR = 2.5 / math.log(10)
# A comparison star joins the shared reference only when it has at least two
# measurable exposures in common with an already-linked star. One shared frame
# is too little to trust the zero-point bridge between two groups of stars.
MIN_LINK_EXPOSURES = 2

STATE_TEXT = {
    "measured": "已测量",
    "saturated": "饱和",
    "insufficient_aperture": "孔径覆盖不足",
    "insufficient_sky": "背景环不足",
    "nonpositive": "净光度不为正",
    "unaligned": "待对准",
}


@dataclass(frozen=True)
class SampleRegion:
    x: float
    y: float
    radius: float
    sky_inner: float
    sky_outer: float

    def cut(self, detector_shape):
        height, width = detector_shape
        x0 = max(0, math.floor(self.x - self.sky_outer - 1))
        y0 = max(0, math.floor(self.y - self.sky_outer - 1))
        x1 = min(width, math.ceil(self.x + self.sky_outer + 1))
        y1 = min(height, math.ceil(self.y + self.sky_outer + 1))
        return slice(y0, max(y0, y1)), slice(x0, max(x0, x1))

    def weights(self, cut):
        yy = np.arange(cut[0].start, cut[0].stop)[:, None] - self.y
        xx = np.arange(cut[1].start, cut[1].stop)[None, :] - self.x
        stamp = np.broadcast_shapes(xx.shape, yy.shape)
        aperture = np.zeros(stamp, dtype=np.float64)
        annulus = np.zeros(stamp, dtype=np.float64)
        for dx in SUBPIXEL_GRID:
            for dy in SUBPIXEL_GRID:
                rho2 = (xx + dx)**2 + (yy + dy)**2
                aperture += rho2 <= self.radius**2
                annulus += (rho2 <= self.sky_outer**2) & (rho2 >= self.sky_inner**2)
        divisor = len(SUBPIXEL_GRID)**2
        return aperture / divisor, annulus / divisor


def weighted_median(values, weights):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    partial = np.cumsum(weights)
    index = int(np.searchsorted(partial, partial[-1] * 0.5))
    return float(values[index])


def sky_background(values, valid, weights):
    mask = valid & (weights >= 0.25)
    effective_area = float(np.sum(weights[mask]))
    if effective_area < 12:
        return None
    values = values[mask]
    weights = weights[mask]
    center = weighted_median(values, weights)
    deviations = np.abs(values - center)
    noise = weighted_median(deviations, weights) * 1.482602218505602
    use = np.ones(values.size, dtype=bool) if noise < 1e-6 else deviations <= 3 * noise
    kept_area = float(np.sum(weights[use]))
    if kept_area < 10:
        return None
    return {
        "level": weighted_median(values[use], weights[use]),
        "noise": noise,
        "area": kept_area,
        "count": int(np.count_nonzero(use)),
        "rejected": int(np.count_nonzero(~use)),
    }


def aperture_measurement(plane, source, offset, settings, seconds):
    x, y = offset.native_xy(source["x"], source["y"])
    region = SampleRegion(x, y, settings["radius"], settings["sky_inner"], settings["sky_outer"])
    cut = region.cut(plane.values.shape)
    values = plane.values[cut]
    aperture, annulus = region.weights(cut)
    valid = plane.valid[cut]
    saturated = plane.saturated[cut]
    use = valid & ~saturated & (aperture > 0)
    actual_area = float(np.sum(aperture[use]))
    expected_area = math.pi * region.radius**2
    coverage = min(1.0, actual_area / expected_area)
    result = {
        "source": source["id"],
        "label": source["label"],
        "role": source["role"],
        "native_x": x,
        "native_y": y,
        "coverage": coverage,
        "flux": None,
        "flux_per_second": None,
        "uncertainty_per_second": None,
        "background": None,
        "peak": None,
    }
    if np.any(saturated & (aperture >= 0.25)):
        return {**result, "state": "saturated"}
    if values.size == 0 or coverage + 0.01 < settings["min_fraction"]:
        return {**result, "state": "insufficient_aperture"}
    sky = sky_background(values, valid & ~saturated, annulus)
    if sky is None:
        return {**result, "state": "insufficient_sky"}
    pixel_weights = aperture[use]
    counts = values[use]
    flux = float(np.sum(pixel_weights * (counts - sky["level"])))
    variance_sky = sky["noise"]**2 * (float(np.sum(pixel_weights**2)) + actual_area**2 / sky["area"])
    variance_source = max(0.0, flux) / settings["gain"]
    sigma = math.sqrt(variance_sky + variance_source)
    return {
        **result,
        "state": "measured" if flux > 0 else "nonpositive",
        "flux": flux,
        "flux_per_second": flux / seconds,
        "uncertainty_per_second": sigma / seconds,
        "background": sky,
        "peak": float(np.max(counts)),
    }


def measure_missing(source):
    return {
        "source": source["id"],
        "label": source["label"],
        "role": source["role"],
        "state": "unaligned",
        "native_x": None,
        "native_y": None,
        "coverage": None,
        "flux": None,
        "flux_per_second": None,
        "uncertainty_per_second": None,
        "peak": None,
        "background": None,
    }


def time_midpoint(started, exposure):
    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        middle = start + timedelta(seconds=exposure * 0.5)
    except (ValueError, OverflowError) as exc:
        raise DomainError("曝光中点时间超出范围") from exc
    return middle.isoformat(timespec="microseconds").replace("+00:00", "Z")


def per_frame_measurements(images, frames, sources):
    for frame in frames:
        offset = offset_for(images.document, frame["id"])
        if offset is None:
            measures = [measure_missing(source) for source in sources]
            calibration = []
        else:
            plane = images.calibrated(frame["id"])
            settings = images.document["photometry"]
            measures = [
                aperture_measurement(plane, source, offset, settings, frame["exposure"])
                for source in sources
            ]
            calibration = list(plane.notes)
        yield {
            "frame": frame["id"],
            "filename": frame["filename"],
            "started": frame["started"],
            "midpoint": time_midpoint(frame["started"], frame["exposure"]),
            "time_seconds": datetime.fromisoformat(frame["started"].replace("Z", "+00:00")).timestamp() + frame["exposure"] * 0.5,
            "exposure": frame["exposure"],
            "band": frame["band"],
            "alignment": offset.source if offset else None,
            "calibration": calibration,
            "measurements": measures,
        }


def instrument_magnitude(measurement):
    """Magnitude and uncertainty of a usable measurement, else None."""
    if measurement["state"] != "measured":
        return None
    flux = measurement["flux_per_second"]
    if flux is None or flux <= 0 or not math.isfinite(flux):
        return None
    sigma = measurement["uncertainty_per_second"]
    if sigma is None or not math.isfinite(sigma) or sigma < 0:
        return None
    return -2.5 * math.log10(flux), MAGNITUDE_FACTOR * sigma / flux


def comparison_samples(rows, comparison_sources):
    """Measurable comparison magnitudes per star and per frame."""
    samples = {source["id"]: {} for source in comparison_sources}
    for row_index, row in enumerate(rows):
        for measure in row["measurements"]:
            if measure["role"] != "comparison" or measure["source"] not in samples:
                continue
            value = instrument_magnitude(measure)
            if value is not None:
                samples[measure["source"]][row_index] = value
    return samples


def linked_components(samples):
    """Union-find groups joined by at least MIN_LINK_EXPOSURES common frames."""
    ids = list(samples)
    joint = {}
    for first in range(len(ids)):
        for second in range(first + 1, len(ids)):
            a, b = ids[first], ids[second]
            shared = len(samples[a].keys() & samples[b].keys())
            joint[(a, b)] = shared
    parent = {value: value for value in ids}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for (a, b), shared in joint.items():
        if shared >= MIN_LINK_EXPOSURES:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a
    groups = {}
    for value in ids:
        groups.setdefault(find(value), []).append(value)
    return list(groups.values()), joint


def first_measurement_frame(samples, ids):
    indexes = [index for value in ids for index in samples[value]]
    return min(indexes, default=math.inf)


def link_reference(rows, comparison_sources):
    """Build the shared comparison-star reference and its provenance."""
    samples = comparison_samples(rows, comparison_sources)
    components, joint = linked_components(samples)
    if not components:
        return {"members": [], "anchor": None, "excluded": [], "links": [], "components": [],
                "offsets": {}, "references": {}}
    components.sort(key=lambda group: (
        -len(set().union(*(samples[value].keys() for value in group))),
        -len(group),
        first_measurement_frame(samples, group),
    ))
    primary = components[0]
    by_id = {source["id"]: source for source in comparison_sources}
    positions = {source["id"]: index for index, source in enumerate(comparison_sources)}
    anchor = min(primary, key=lambda value: (-len(samples[value]), positions[value]))

    offsets, offset_errors = solve_offsets(samples, primary, anchor)
    linked = [value for value in primary if value in offsets]
    references = frame_references(rows, samples, linked, offsets, offset_errors)

    excluded = []
    for group in components[1:]:
        bridge = max(
            (joint.get(tuple(sorted((member, other), key=positions.get)), 0)
             for member in group for other in primary),
            default=0,
        )
        for value in group:
            excluded.append({
                "id": value,
                "label": by_id[value]["label"],
                "joint_frames": bridge,
                "reason": "no_shared_exposure" if bridge == 0 else "single_shared_exposure",
            })
    for value in primary:
        if value not in linked:
            excluded.append({
                "id": value,
                "label": by_id[value]["label"],
                "joint_frames": 0,
                "reason": "no_shared_exposure",
            })
    links = [
        {"a": a, "b": b, "joint_frames": count}
        for (a, b), count in sorted(joint.items(), key=lambda item: (-item[1], positions[item[0][0]], positions[item[0][1]]))
    ]
    return {
        "members": [by_id[value] for value in linked],
        "anchor": anchor,
        "excluded": excluded,
        "links": links,
        "components": [
            {
                "anchor": min(group, key=lambda value: (-len(samples[value]), positions[value])),
                "members": [by_id[value]["label"] for value in group],
                "frames": len(set().union(*(samples[value].keys() for value in group))),
            }
            for group in components
        ],
        "offsets": {value: (offsets[value], offset_errors[value]) for value in linked},
        "references": references,
    }


def solve_offsets(samples, members, anchor):
    """Tie every member star to the anchor through shared exposures.

    For two stars a, b the difference of instrumental magnitudes over a shared
    frame is M_a - M_b, independent of the frame. The median difference over the
    shared frames therefore gives the relative brightness offset; the bridge is
    grown from the best-connected pair outward.
    """
    absolute = {anchor: (0.0, 0.0)}
    remaining = [value for value in members if value != anchor]
    while remaining:
        best = None
        for value in remaining:
            for known, (known_offset, known_error) in absolute.items():
                shared = samples[value].keys() & samples[known].keys()
                if len(shared) < MIN_LINK_EXPOSURES:
                    continue
                deltas = np.array([samples[value][index][0] - samples[known][index][0] for index in shared])
                scatter = 1.482602218505602 * float(np.median(np.abs(deltas - np.median(deltas))))
                scatter_error = scatter / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
                error = math.hypot(known_error, scatter_error)
                candidate = (len(shared), -error, value, known_offset, error, float(np.median(deltas)))
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        if best is None:
            break
        _, _, value, known_offset, error, delta = best
        absolute[value] = (known_offset + delta, error)
        remaining.remove(value)
    return ({value: absolute[value][0] for value in absolute},
            {value: absolute[value][1] for value in absolute})


def frame_references(rows, samples, members, offsets, offset_errors):
    references = {}
    for row_index in range(len(rows)):
        estimates = []
        for value in members:
            sample = samples[value].get(row_index)
            if sample is None:
                continue
            magnitude, error = sample
            estimates.append((magnitude - offsets[value], math.hypot(error, offset_errors[value])))
        if not estimates:
            continue
        weights = np.array([1.0 / max(error, 1e-9)**2 for _, error in estimates])
        values = np.array([value for value, _ in estimates])
        zero_point = float(np.sum(values * weights) / np.sum(weights))
        references[row_index] = {
            "magnitude": zero_point,
            "error": float(math.sqrt(1.0 / np.sum(weights))),
            "used": [
                value for value in members
                if samples[value].get(row_index) is not None
            ],
        }
    return references


def reference_for_row(row, reference, link, comparison_sources):
    """Describe which comparison stars define this frame's brightness reference."""
    member_ids = {source["id"] for source in link["members"]}
    used_ids = set(reference["used"]) if reference else set()
    comparisons = []
    unavailable = []
    for measure in row["measurements"]:
        if measure["role"] != "comparison":
            continue
        if measure["source"] in used_ids:
            sample = instrument_magnitude(measure)
            comparisons.append({
                "id": measure["source"],
                "label": measure["label"],
                "magnitude": sample[0] if sample else None,
                "error": sample[1] if sample else None,
                "offset": link["offsets"][measure["source"]][0],
                "state": "used",
            })
        elif measure["source"] in member_ids:
            unavailable.append({
                "id": measure["source"],
                "label": measure["label"],
                "state": measure["state"],
                "reason": "measurement_unavailable",
            })
        else:
            entry = next((item for item in link["excluded"] if item["id"] == measure["source"]), None)
            if entry is None:
                continue
            unavailable.append({
                "id": measure["source"],
                "label": measure["label"],
                "state": measure["state"],
                "reason": entry["reason"],
                "joint_frames": entry["joint_frames"],
            })
    if row["alignment"] is None:
        state, note = "unaligned", "该曝光尚未对准，无法在原生像素上量测"
    elif reference is None or not comparisons:
        state, note = "unlinked", unlinked_note(unavailable, link)
    else:
        state, note = "measured", None
    return {
        "state": state,
        "magnitude": reference["magnitude"] if state == "measured" else None,
        "error": reference["error"] if state == "measured" else None,
        "comparisons": comparisons if state == "measured" else [],
        "unavailable": unavailable,
        "note": note,
    }


def unlinked_note(unavailable, link):
    if not link["members"]:
        return "没有可用于建立共同参照的比较星"
    reasons = []
    for entry in unavailable:
        if entry["reason"] == "measurement_unavailable":
            reasons.append(f"{entry['label']} {STATE_TEXT.get(entry['state'], entry['state'])}")
        elif entry["reason"] == "no_shared_exposure":
            reasons.append(f"{entry['label']} 与主参照没有共同可量测的曝光")
        elif entry["reason"] == "single_shared_exposure":
            reasons.append(
                f"{entry['label']} 与主参照仅在 {entry['joint_frames']} 张曝光上共同可量测"
                f"（至少需 {MIN_LINK_EXPOSURES} 张）"
            )
    if not reasons:
        return "本帧没有主参照比较星的可用读数"
    return "参照接不上：" + "；".join(dict.fromkeys(reasons))


def differential(target, reference):
    if target["state"] != "measured":
        return {"state": f"target_{target['state']}", "ratio": None, "magnitude": None, "error": None}
    if reference["state"] != "measured":
        return {"state": reference["state"], "ratio": None, "magnitude": None, "error": None,
                "note": reference.get("note")}
    magnitude = instrument_magnitude(target)
    if magnitude is None:
        return {"state": "target_nonpositive", "ratio": None, "magnitude": None, "error": None,
                "note": None}
    value = magnitude[0] - reference["magnitude"]
    error = math.hypot(magnitude[1], reference["error"] or 0.0)
    return {"state": "measured", "ratio": 10 ** (-0.4 * value), "magnitude": value, "error": error,
            "note": None}


def choose_sources(doc, options):
    by_id = {source["id"]: source for source in doc["apertures"]}
    requested = {}
    for field in ("target", "check"):
        value = options.get(field)
        if not value:
            continue
        source = by_id.get(identifier(value, "孔径"))
        if not source or source["role"] != field:
            raise DomainError(f"所选 {field} 孔径不存在或角色不符")
        requested[field] = source
    comparisons = options.get("comparisons")
    if comparisons is None and options.get("comparison"):
        comparisons = [options["comparison"]]
    if comparisons is not None:
        comparisons = list_value(comparisons, "比较星", MAX_APERTURES)
        chosen = []
        seen = set()
        for value in comparisons:
            source = by_id.get(identifier(value, "孔径"))
            if not source or source["role"] != "comparison":
                raise DomainError("所选比较星孔径不存在或角色不符")
            if source["id"] in seen:
                continue
            seen.add(source["id"])
            chosen.append(source)
        requested["comparisons"] = chosen
    if "target" not in requested:
        raise DomainError("请指定一颗目标星")
    if not requested.get("comparisons"):
        raise DomainError("请至少选择一颗比较星")
    return requested


def ordered_sources(requested):
    sources = [requested["target"], *requested["comparisons"]]
    if "check" in requested:
        sources.append(requested["check"])
    return sources


def filter_photometry_frames(doc, band):
    frames = usable_science_frames(doc)
    filters = sorted({frame["band"] for frame in frames})
    if band is None:
        if len(filters) > 1:
            raise DomainError("存在多种滤镜，请选定一种滤镜再计算曲线")
        band = next(iter(filters), "")
    chosen = [frame for frame in frames if frame["band"] == band]
    if not chosen:
        raise DomainError("所选滤镜没有可用曝光")
    return band, chosen


def light_curve(images, options):
    doc = images.document
    requested = choose_sources(doc, options)
    band, frames = filter_photometry_frames(doc, options.get("band"))
    if len(frames) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("请暂时挂起部分曝光，使单次曲线图像总量降至 2400 万像素内")
    rows = list(per_frame_measurements(images, frames, ordered_sources(requested)))
    link = link_reference(rows, requested["comparisons"])
    excluded_by_id = {entry["id"]: entry for entry in link["excluded"]}
    for row_index, row in enumerate(rows):
        reference = reference_for_row(row, link["references"].get(row_index), link, requested["comparisons"])
        row["reference"] = reference
        by_id = {measure["source"]: measure for measure in row["measurements"]}
        row["target"] = differential(by_id[requested["target"]["id"]], reference)
        row["check"] = (
            differential(by_id[requested["check"]["id"]], reference) if "check" in requested else None
        )
        used_ids = {entry["id"] for entry in reference["comparisons"]}
        row["comparisons"] = []
        for measure in row["measurements"]:
            if measure["role"] != "comparison":
                continue
            if measure["source"] in used_ids:
                state = "used"
            elif measure["source"] in excluded_by_id:
                state = excluded_by_id[measure["source"]]["reason"]
            else:
                state = "not_used"
            row["comparisons"].append({
                "id": measure["source"],
                "label": measure["label"],
                "state": measure["state"],
                "reference_state": state,
                "flux_per_second": measure["flux_per_second"],
            })
    accepted = [row for row in rows if row["target"]["state"] == "measured"]
    unlinked = [row for row in rows if row["target"]["state"] == "unlinked"]
    return {
        "kind": "photometry",
        "revision": doc["revision"],
        "band": band,
        "unit": "mag",
        "reference": doc["reference"],
        "sources": {
            "target": requested["target"],
            "comparisons": requested["comparisons"],
            "comparison": requested["comparisons"][0] if len(requested["comparisons"]) == 1 else None,
            "check": requested.get("check"),
        },
        "ensemble": {
            "anchor": link["anchor"],
            "members": [
                {
                    "id": source["id"],
                    "label": source["label"],
                    "offset": link["offsets"][source["id"]][0],
                    "offset_error": link["offsets"][source["id"]][1],
                }
                for source in link["members"]
            ],
            "excluded": link["excluded"],
            "links": link["links"],
            "components": link["components"],
            "min_link_exposures": MIN_LINK_EXPOSURES,
        },
        "settings": dict(doc["photometry"]),
        "rows": rows,
        "summary": {
            "frames": len(rows),
            "measured": len(accepted),
            "unavailable": len(rows) - len(accepted),
            "unlinked": len(unlinked),
            "target_baseline": robust_summary([row["target"]["magnitude"] for row in accepted]),
        },
    }
