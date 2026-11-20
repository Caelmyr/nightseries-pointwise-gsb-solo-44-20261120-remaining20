"""Audit WCS consistency after mounting each exposure onto the reference field."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from .astrometry import Point, angular_distance, from_header, reference_projection
from .calibration import robust_summary
from .domain import DomainError, dimensions_of, integer, number, usable_science_frames
from .preview import png
from .registration import offset_for


@dataclass(frozen=True)
class PlateCheck:
    subdivisions: int
    tolerance: float
    interior: float
    filter_name: str | None

    @classmethod
    def from_options(cls, options):
        return cls(
            integer(options.get("subdivisions", 5), "检点每行数量", 3, 12),
            number(options.get("tolerance", 3.5), "角距离容差", 0.05, 120),
            number(options.get("interior", 0.1), "画幅内缩比例", 0, 0.40),
            options.get("band"),
        )


def _grid(dimensions, count, edge):
    inset_x = edge * (dimensions.width - 1)
    inset_y = edge * (dimensions.height - 1)
    x = np.linspace(inset_x, dimensions.width - 1 - inset_x, count)
    y = np.linspace(inset_y, dimensions.height - 1 - inset_y, count)
    return [Point(float(xx), float(yy)) for yy in y for xx in x]


def _test_frame(row, reference, samples, dimensions, shift, tolerance):
    result = {
        "frame": row["id"], "filename": row["filename"], "band": row["band"],
        "state": "no_header", "points": [], "median_error": None, "maximum_error": None,
        "rms_error": None, "used": 0, "out_of_frame": 0,
    }
    if shift is None:
        return {**result, "state": "unaligned"}
    try:
        second = from_header(row["header"])
    except DomainError as error:
        return {**result, "state": "unsupported_header", "reason": str(error)}
    if second is None:
        return result
    if reference.system != second.system:
        return {**result, "state": "coordinate_system_conflict", "reason": f"{reference.system} / {second.system}"}
    offsets = []
    for point in samples:
        x, y = shift.native_xy(point.x, point.y)
        if not 0 <= x <= dimensions.width - 1 or not 0 <= y <= dimensions.height - 1:
            result["out_of_frame"] += 1
            continue
        expected = reference.pixel_to_sky(point)
        current = second.pixel_to_sky(Point(x, y))
        magnitude = angular_distance(expected, current) * 3600
        delta_ra = ((current.x - expected.x + 180) % 360 - 180) * 3600 * math.cos(math.radians(expected.y))
        delta_dec = (current.y - expected.y) * 3600
        result["points"].append({
            "x": point.x, "y": point.y, "native_x": x, "native_y": y,
            "east_arcsec": delta_ra, "north_arcsec": delta_dec, "error_arcsec": magnitude,
            "in_tolerance": magnitude <= tolerance,
        })
        offsets.append(magnitude)
    if len(offsets) < len(samples) // 2:
        return {**result, "state": "insufficient_shared_field", "used": len(offsets)}
    center = robust_summary(offsets)
    max_error = max(offsets)
    rms = math.sqrt(math.fsum(value**2 for value in offsets) / len(offsets))
    return {
        **result,
        "state": "consistent" if max_error <= tolerance else "mismatch",
        "median_error": center["median"], "maximum_error": max_error, "rms_error": rms,
        "used": len(offsets), "deviation": center["sigma"],
    }


def audit_plates(document, options):
    plan = PlateCheck.from_options(options)
    dims = dimensions_of(document)
    reference = reference_projection(document)
    if reference is None:
        raise DomainError("参考帧缺少可比较的天球解")
    all_frames = usable_science_frames(document, plan.filter_name)
    frames = [row for row in all_frames if row["id"] != document["reference"]]
    if not frames:
        raise DomainError("没有可与参考帧比较的科学曝光")
    samples = _grid(dims, plan.subdivisions, plan.interior)
    rows = [
        _test_frame(row, reference, samples, dims, offset_for(document, row["id"]), plan.tolerance)
        for row in frames
    ]
    successful = [row for row in rows if row["state"] in {"consistent", "mismatch"}]
    worst = max(successful, key=lambda row: row["maximum_error"]) if successful else None
    return {
        "revision": document["revision"], "reference": document["reference"],
        "system": reference.system, "tolerance": plan.tolerance,
        "width": dims.width, "height": dims.height,
        "tested": len(successful), "mismatches": sum(row["state"] == "mismatch" for row in rows),
        "worst": worst["frame"] if worst else None,
        "rows": rows,
    }


def draw_plate_differences(report, frame_id):
    selected = next((row for row in report["rows"] if row["frame"] == frame_id), None)
    if selected is None or not selected["points"]:
        raise DomainError("所选曝光没有可画的天球控制点")
    tolerance = report["tolerance"]
    width = 720
    height = min(900, max(180, round(width * report["height"] / report["width"])))
    image = Image.new("RGB", (width, height), (15, 22, 35))
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, width - 17, height - 17), outline=(76, 91, 112))
    def coordinate(point):
        return (
            16 + point["x"] / max(1, report["width"] - 1) * (width - 33),
            16 + point["y"] / max(1, report["height"] - 1) * (height - 33),
        )
    for point in selected["points"]:
        x, y = coordinate(point)
        east, north = point["east_arcsec"], point["north_arcsec"]
        factor = min(16 / tolerance, 35 / max(math.hypot(east, north), 1e-6))
        x_final, y_final = x + east * factor, y - north * factor
        color = (122, 226, 209) if point["in_tolerance"] else (246, 170, 108)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        draw.line((x, y, x_final, y_final), fill=color, width=2)
        draw.ellipse((x_final - 1, y_final - 1, x_final + 1, y_final + 1), fill=(227, 239, 254))
    draw.text((24, height - 32), f"Reference controls / East right, north up / limit {tolerance:g} arcsec", fill=(163, 183, 206))
    return png(image)
