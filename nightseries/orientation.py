"""Shared star orientation across one exposure, local zone deviations, and explicit gaps.

One frame's median elongation cannot say whether the whole frame trails along one
direction or a few corners trail along their own. This module groups the measured
star stamps into zones, estimates each zone's axis, and combines the reliable zones
with equal weight so a star-dense corner cannot outvote the rest of the frame.
Zones without enough evidence are reported as such, never as consistent.
"""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from .calibration import robust_summary
from .domain import integer, number
from .preview import png, stretch_pixels, stretch_value

AXIAL = 180.0
ZONE_COLORS = {
    "consistent": (110, 214, 201),
    "deviating": (247, 169, 95),
    "insufficient": (156, 139, 196),
}


@dataclass(frozen=True)
class ZoneSpec:
    divisions: int = 3
    min_stars: int = 3
    deviation: float = 15.0
    elongation_change: float = 0.20
    elongation_floor: float = 1.10
    concentration: float = 0.35
    masked_floor: float = 0.50

    @classmethod
    def requested(cls, options):
        return cls(
            integer(options.get("zones", 3), "分区每边数", 2, 6),
            integer(options.get("zone_min_stars", 3), "每区最少星点", 3, 30),
            number(options.get("deviation", 15), "方向偏离阈值（度）", 3, 60),
            number(options.get("elongation_change", 0.20), "拖尾变化阈值", 0.05, 1),
        )

    def describe(self):
        return {
            "zones": self.divisions,
            "zone_min_stars": self.min_stars,
            "deviation": self.deviation,
            "elongation_change": self.elongation_change,
            "elongation_floor": self.elongation_floor,
            "concentration": self.concentration,
            "masked_floor": self.masked_floor,
        }


def _axial_mean(angles):
    """Mean axis of undirected angles (degrees, 0-180) and its resultant length."""
    radians = np.radians(np.asarray(angles, dtype=float)) * 2.0
    cosine = float(np.mean(np.cos(radians)))
    sine = float(np.mean(np.sin(radians)))
    return math.degrees(0.5 * math.atan2(sine, cosine)) % AXIAL, math.hypot(cosine, sine)


def _axial_std(resultant):
    if resultant is None or resultant <= 1e-9:
        return None
    if resultant >= 1.0:
        return 0.0
    return math.degrees(math.sqrt(-2.0 * math.log(resultant)) * 0.5)


def _axial_offset(angle, reference):
    return (angle - reference + 0.5 * AXIAL) % AXIAL - 0.5 * AXIAL


def _grid(length, divisions):
    return np.rint(np.linspace(0, length, divisions + 1)).astype(int)


def _cell(value, edges):
    index = int(np.searchsorted(edges, value, side="right") - 1)
    return min(len(edges) - 2, max(0, index))


def _zone_measure(plane, stars, spec):
    height, width = plane.values.shape
    y_edges = _grid(height, spec.divisions)
    x_edges = _grid(width, spec.divisions)
    zones = []
    for line in range(spec.divisions):
        y0, y1 = int(y_edges[line]), int(y_edges[line + 1])
        for column in range(spec.divisions):
            x0, x1 = int(x_edges[column]), int(x_edges[column + 1])
            area = plane.valid[y0:y1, x0:x1]
            valid_fraction = float(np.mean(area)) if area.size else 0.0
            members = [star for star in stars if x0 <= star["x"] < x1 and y0 <= star["y"] < y1]
            elongations = [star["elongation"] for star in members if star.get("elongation")]
            summary = robust_summary(elongations)
            angle = resultant = spread = error = None
            if members:
                angle, resultant = _axial_mean([star["angle"] for star in members])
                spread = _axial_std(resultant)
                error = None if spread is None else spread / math.sqrt(len(members))
            zones.append({
                "line": line,
                "column": column,
                "x": x0,
                "y": y0,
                "width": x1 - x0,
                "height": y1 - y0,
                "stars": len(members),
                "valid_fraction": valid_fraction,
                "elongation": summary["median"],
                "elongation_spread": summary["sigma"],
                "angle": angle,
                "angle_spread": spread,
                "angle_error": error,
                "concentration": resultant,
                "angle_offset": None,
                "elongation_ratio": None,
                "direction": bool(
                    len(members) >= spec.min_stars
                    and summary["median"] is not None
                    and summary["median"] >= spec.elongation_floor
                    and resultant is not None
                    and resultant >= spec.concentration
                ),
            })
    return zones


def _blocking_reason(zones, spec):
    """Why no zone resolves a direction: separate 'stars are round' from 'no evidence'."""
    crowded = [zone for zone in zones if zone["stars"] >= spec.min_stars]
    if not crowded:
        return "no_common_direction"
    if all(zone["valid_fraction"] < spec.masked_floor for zone in crowded):
        return "masked_frames"
    if all(
        zone["elongation"] is None or zone["elongation"] < spec.elongation_floor
        for zone in crowded
    ):
        return "round_frames"
    return "scatter"


def _common_direction(zones, spec):
    """Equal-weight robust shared axis over zones with a resolved direction."""
    resolved = [zone for zone in zones if zone["direction"]]
    common = {
        "state": "insufficient",
        "angle": None,
        "elongation": None,
        "angle_spread": None,
        "elongation_spread": None,
        "angle_error": None,
        "zones_used": 0,
        "zones_resolved": len(resolved),
        "reasons": [],
    }
    if not resolved:
        common["reasons"] = [_blocking_reason(zones, spec)]
        return common
    angles = np.array([zone["angle"] for zone in resolved], dtype=float)
    keep = np.ones(angles.size, dtype=bool)
    floor = 0.5 * spec.deviation
    for _ in range(6):
        centre, _ = _axial_mean(angles[keep])
        offsets = np.array([_axial_offset(value, centre) for value in angles[keep]])
        middle = float(np.median(offsets))
        sigma = 1.4826 * float(np.median(np.abs(offsets - middle)))
        updated = keep.copy()
        updated[np.where(keep)[0]] = np.abs(offsets - middle) <= max(3.0 * sigma, floor)
        if np.count_nonzero(updated) < 2 or np.array_equal(updated, keep):
            break
        keep = updated
    kept = [zone for zone, taken in zip(resolved, keep) if taken]
    angle, resultant = _axial_mean(angles[keep])
    spread = _axial_std(resultant)
    elongation = robust_summary([zone["elongation"] for zone in kept])
    common.update(
        angle=angle,
        elongation=elongation["median"],
        angle_spread=spread,
        elongation_spread=elongation["sigma"],
        angle_error=None if spread is None else spread / math.sqrt(len(kept)),
        zones_used=len(kept),
    )
    if len(kept) < 2:
        common["reasons"] = ["one_zone_only"]
    elif spread is None or spread > spec.deviation:
        common["reasons"] = ["zones_disagree"]
    else:
        common["state"] = "defined"
    return common


def _classify(zones, common, spec):
    for zone in zones:
        reasons = []
        state = "insufficient"
        if zone["valid_fraction"] < spec.masked_floor:
            reasons = ["masked_zone"]
        elif zone["stars"] == 0:
            reasons = ["no_stars"]
        elif zone["stars"] < spec.min_stars:
            reasons = ["few_stars"]
        elif zone["elongation"] is None or zone["elongation"] < spec.elongation_floor:
            reasons = ["round_stars"]
        elif zone["concentration"] is None or zone["concentration"] < spec.concentration:
            reasons = ["scatter"]
        elif common["state"] != "defined":
            reasons = ["no_common_direction"]
        else:
            offset = _axial_offset(zone["angle"], common["angle"])
            limit = max(
                spec.deviation,
                2.0 * math.hypot(zone["angle_error"] or 0.0, common["angle_error"] or 0.0),
            )
            zone["angle_offset"] = offset
            if abs(offset) > limit:
                reasons.append("angle_shift")
            if common["elongation"]:
                ratio = zone["elongation"] / common["elongation"]
                zone["elongation_ratio"] = ratio
                scatter = math.hypot(
                    (zone["elongation_spread"] or 0.0) / math.sqrt(max(1, zone["stars"])),
                    (common["elongation_spread"] or 0.0) / math.sqrt(max(1, common["zones_used"])),
                ) / common["elongation"]
                if abs(ratio - 1.0) > max(spec.elongation_change, 2.0 * scatter):
                    reasons.append("elongation_change")
            state = "deviating" if reasons else "consistent"
        zone["state"] = state
        zone["reasons"] = reasons


def orientation_summary(stars, plane, options, *, detail=False):
    """Summarise shared and local star axes from the measured stamps of one plane."""
    spec = ZoneSpec.requested(options)
    measured = [
        star for star in stars
        if star.get("state") == "measured"
        and star.get("angle") is not None
        and star.get("elongation")
    ]
    zones = _zone_measure(plane, measured, spec)
    common = _common_direction(zones, spec)
    _classify(zones, common, spec)
    for zone in zones:
        zone.pop("direction", None)
    report = {
        "settings": spec.describe(),
        "common": common,
        "zones": zones,
        "summary": {
            "zones": len(zones),
            "zones_deviating": sum(zone["state"] == "deviating" for zone in zones),
            "zones_insufficient": sum(zone["state"] == "insufficient" for zone in zones),
        },
    }
    if detail:
        y_edges = _grid(plane.values.shape[0], spec.divisions)
        x_edges = _grid(plane.values.shape[1], spec.divisions)
        lookup = {(zone["line"], zone["column"]): zone for zone in zones}
        report["stars"] = []
        for star in measured:
            line = _cell(star["y"], y_edges)
            column = _cell(star["x"], x_edges)
            zone = lookup[(line, column)]
            report["stars"].append({
                "x": float(star["x"]),
                "y": float(star["y"]),
                "angle": float(star["angle"]),
                "elongation": float(star["elongation"]),
                "fwhm": float(star["fwhm"]),
                "line": line,
                "column": column,
                "state": zone["state"],
                "reasons": list(zone["reasons"]),
            })
    return report


def draw_orientation_map(plane, report, settings=None, *, max_side=1100):
    """Overlay measured star positions and axes, zone verdicts, and the shared axis."""
    gray, _ = stretch_pixels(plane, stretch_value(settings or {}))
    base = np.stack((15 + gray * 0.32, 20 + gray * 0.38, 32 + gray * 0.48), axis=2)
    image = Image.fromarray(np.clip(np.rint(base), 0, 255).astype(np.uint8), "RGB")
    height, width = gray.shape
    factor = min(1.0, max_side / max(height, width))
    if factor < 1.0:
        image = image.resize((max(1, round(width * factor)), max(1, round(height * factor))),
                             Image.Resampling.LANCZOS)
    scale_x = image.width / max(1, width)
    scale_y = image.height / max(1, height)
    draw = ImageDraw.Draw(image, "RGBA")
    for zone in report["zones"]:
        color = ZONE_COLORS.get(zone["state"], (200, 210, 230))
        left = zone["x"] * scale_x
        top = zone["y"] * scale_y
        right = (zone["x"] + zone["width"]) * scale_x - 1
        bottom = (zone["y"] + zone["height"]) * scale_y - 1
        draw.rectangle((left, top, right, bottom), outline=(*color, 175), width=1)
        if right - left > 54 and zone["stars"]:
            label = f"n{zone['stars']} a{zone['angle']:.0f}" if zone["angle"] is not None else f"n{zone['stars']}"
            draw.text((left + 3, top + 2), label, fill=(*color, 220))
    reach_scale = 0.5 * (scale_x + scale_y)
    for star in report.get("stars", []):
        color = ZONE_COLORS.get(star["state"], (200, 210, 230))
        x, y = star["x"] * scale_x, star["y"] * scale_y
        radians = math.radians(star["angle"])
        reach = (3.5 + 5.0 * max(0.0, min(star["elongation"] - 1.0, 1.2))) * reach_scale
        dx, dy = math.cos(radians) * reach, math.sin(radians) * reach
        draw.line((x - dx, y - dy, x + dx, y + dy), fill=(*color, 230), width=1)
        draw.ellipse((x - 1.3, y - 1.3, x + 1.3, y + 1.3), fill=(241, 247, 255, 240))
    legend_height = 34
    canvas = Image.new("RGB", (image.width, image.height + legend_height), (11, 15, 26))
    canvas.paste(image, (0, 0))
    legend = ImageDraw.Draw(canvas)
    common = report["common"]
    if common["state"] == "defined":
        text = f"common {common['angle']:.1f}deg  elongation {common['elongation']:.2f}  {common['zones_used']}/{len(report['zones'])} zones"
    else:
        text = "common unresolved: " + " + ".join(common["reasons"])
    legend.text((8, image.height + 3), text, fill=(196, 210, 232))
    cursor = 8
    for label, state in (("shared", "consistent"), ("local", "deviating"), ("unclear", "insufficient")):
        color = ZONE_COLORS[state]
        legend.rectangle((cursor, image.height + 21, cursor + 7, image.height + 28), fill=color)
        legend.text((cursor + 11, image.height + 18), label, fill=color)
        cursor += 18 + int(legend.textlength(label))
    legend.line((8, image.height + legend_height - 4, image.width - 8, image.height + legend_height - 4),
                fill=(36, 46, 64))
    return png(canvas)
