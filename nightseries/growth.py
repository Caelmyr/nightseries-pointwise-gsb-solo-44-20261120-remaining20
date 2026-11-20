"""Check aperture radii against the same native exposure and sky annulus."""
from dataclasses import dataclass
import math

import numpy as np

from .calibration import robust_summary
from .domain import DomainError, MAX_ACTIVE_PIXELS, exposure_row, identifier, integer, number
from .photometry import aperture_measurement, time_midpoint
from .registration import offset_for


@dataclass(frozen=True)
class SweepSpec:
    minimum: float
    maximum: float
    count: int
    plateau: float
    minimum_frames: int

    @classmethod
    def request(cls, doc, options, available):
        ceiling = min(28, doc["photometry"]["sky_inner"] - 1)
        if ceiling < 2:
            raise DomainError("背景内径太小，无法检查多个孔径")
        low = number(options.get("minimum", 1.5), "最小孔径半径", 1.5, ceiling)
        high = number(options.get("maximum", min(7, ceiling)), "最大孔径半径", low, ceiling)
        if high - low < 0.5:
            raise DomainError("最大孔径需比最小孔径至少大 0.5 px")
        return cls(
            low,
            high,
            integer(options.get("count", 10), "试测档数", 3, 24),
            number(options.get("plateau", 0.98), "目标累计光度比例", 0.75, 0.999),
            integer(options.get("minimum_frames", min(3, available)), "最少有效曝光", 1, available),
        )

    @property
    def radii(self):
        return np.linspace(self.minimum, self.maximum, self.count).tolist()


def choose_frames(document, options):
    band = options.get("band")
    frame_id = options.get("frame")
    if frame_id is not None:
        return [exposure_row(document, identifier(frame_id, "增长曲线科学曝光"))]
    frames = [frame for frame in document["frames"] if frame["enabled"] and (band is None or frame["band"] == band)]
    if len({frame["band"] for frame in frames}) > 1:
        raise DomainError("增长曲线合并前需选择一种滤镜")
    if not frames:
        raise DomainError("选定范围内无科学曝光")
    return frames


def choose_aperture(document, source_id):
    token = identifier(source_id, "增长曲线孔径")
    by_id = {row["id"]: row for row in document["apertures"]}
    if token not in by_id:
        raise DomainError("孔径不存在")
    return by_id[token]


def _single_sweep(image, source, offset, default, radii, exposure):
    samples = []
    for radius in radii:
        settings = {**default, "radius": radius}
        measurement = aperture_measurement(image, source, offset, settings, exposure)
        flux = measurement["flux_per_second"]
        error = measurement["uncertainty_per_second"]
        valid = measurement["state"] == "measured" and error is not None
        samples.append({
            "radius": radius,
            "state": measurement["state"],
            "coverage": measurement["coverage"],
            "flux": flux,
            "uncertainty": error,
            "snr": flux / error if valid and error > 0 else None,
        })
    return samples


def normalize_sweep(samples):
    largest = samples[-1]
    if largest["state"] != "measured" or not largest["flux"]:
        return [{**row, "fraction": None} for row in samples]
    return [
        {**row, "fraction": row["flux"] / largest["flux"] if row["state"] == "measured" else None}
        for row in samples
    ]


def _context_summary(sweeps, radii, minimum_frames):
    traces = [row["samples"] for row in sweeps if row["samples"]]
    levels = []
    for index, radius in enumerate(radii):
        rows = [trace[index] for trace in traces]
        fractions = [row["fraction"] for row in rows if row["fraction"] is not None]
        snr = [row["snr"] for row in rows if row["snr"] is not None and row["fraction"] is not None]
        area = [row["coverage"] for row in rows if row["fraction"] is not None]
        enough = len(fractions) >= minimum_frames
        measure = robust_summary(fractions)
        levels.append({
            "radius": radius,
            "count": len(fractions),
            "fraction": measure["median"] if enough else None,
            "fraction_spread": measure["sigma"] if enough else None,
            "median_snr": robust_summary(snr)["median"] if enough else None,
            "median_overlap": robust_summary(area)["median"] if enough else None,
        })
    return levels


def find_plateau(sweep, threshold):
    for index, level in enumerate(sweep[:-1]):
        if level["fraction"] is None or level["fraction"] < threshold:
            continue
        next_level = sweep[index + 1]
        if next_level["fraction"] is not None and next_level["fraction"] >= threshold:
            return level["radius"]
    return None


def flux_growth(images, options):
    doc = images.document
    frames = choose_frames(doc, options)
    source = choose_aperture(doc, options.get("source"))
    settings = SweepSpec.request(doc, options, len(frames))
    if len(frames) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("增长曲线超过 2400 万像素，请收窄参与曝光")
    radii = settings.radii
    results = []
    for frame in frames:
        off = offset_for(doc, frame["id"])
        row = {
            "frame": frame["id"],
            "filename": frame["filename"],
            "midpoint": time_midpoint(frame["started"], frame["exposure"]),
            "state": "measured" if off else "unaligned",
            "samples": [],
        }
        if off:
            image = images.calibrated(frame["id"])
            sweep = _single_sweep(image, source, off, doc["photometry"], radii, frame["exposure"])
            row["samples"] = normalize_sweep(sweep)
            if sweep[-1]["state"] != "measured":
                row["state"] = "largest_" + sweep[-1]["state"]
        results.append(row)
    trace = _context_summary(results, radii, settings.minimum_frames)
    available = [
        level for level in trace if level["median_snr"] is not None and math.isfinite(level["median_snr"])
    ]
    best = max(available, key=lambda row: row["median_snr"])["radius"] if available else None
    return {
        "reference": doc["reference"],
        "revision": doc["revision"],
        "source": source,
        "bands": sorted({frame["band"] for frame in frames}),
        "radii": trace,
        "rows": results,
        "recommended_plateau_radius": find_plateau(trace, settings.plateau),
        "max_snr_radius": best,
        "min_frames": settings.minimum_frames,
        "background": {
            "inner": doc["photometry"]["sky_inner"],
            "outer": doc["photometry"]["sky_outer"],
        },
    }
