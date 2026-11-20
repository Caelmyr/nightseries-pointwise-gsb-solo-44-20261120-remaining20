"""Measure extended target and background regions in the common field."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from .calibration import robust_summary
from .domain import DomainError, MAX_ACTIVE_PIXELS, number, object_value, usable_science_frames
from .photometry import time_midpoint
from .preview import png
from .registration import offset_for, warp_to_reference


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_data(cls, region, dimensions, description):
        region = object_value(region, description)
        left = math.floor(number(region.get("left"), description + "左边界", 0, dimensions.width - 3))
        top = math.floor(number(region.get("top"), description + "上边界", 0, dimensions.height - 3))
        right = math.ceil(number(region.get("right"), description + "右边界", left + 3, dimensions.width))
        bottom = math.ceil(number(region.get("bottom"), description + "下边界", top + 3, dimensions.height))
        return cls(left, top, right, bottom)

    @property
    def cut(self):
        return np.s_[self.top:self.bottom, self.left:self.right]

    @property
    def area(self):
        return (self.right - self.left) * (self.bottom - self.top)

    def overlaps(self, other):
        return self.left < other.right and self.right > other.left and self.top < other.bottom and self.bottom > other.top

    def serialize(self):
        return {
            "left": self.left, "top": self.top, "right": self.right, "bottom": self.bottom,
            "pixel_area": self.area,
        }


def _region_samples(plane, box, exposure):
    swath = box.cut
    allowed = plane.valid[swath] & ~plane.saturated[swath]
    values = plane.values[swath][allowed] / exposure
    return values, allowed.size, int(np.count_nonzero(plane.saturated[swath] & plane.valid[swath]))


def _background(observed, rejection):
    running = observed
    for _ in range(4):
        if running.size < 9:
            return None
        model = robust_summary(running)
        retained = np.abs(running - model["median"]) <= max(1e-6, rejection * model["sigma"])
        if np.count_nonzero(retained) == running.size:
            break
        running = running[retained]
    else:
        if running.size < 9:
            return None
        model = robust_summary(running)
    return {**model, "removed": observed.size - running.size}


def measure_region(plane, exposure, target, sky, required_fraction, rejection):
    target_values, size, saturation = _region_samples(plane, target, exposure)
    sky_values, sky_size, sky_saturation = _region_samples(plane, sky, exposure)
    result = {
        "target_fraction": target_values.size / size,
        "sky_fraction": sky_values.size / sky_size,
        "target_saturated": saturation,
        "sky_saturated": sky_saturation,
        "target_pixels": int(target_values.size),
        "sky_pixels": int(sky_values.size),
        "background": None,
        "net_rate": None,
        "surface_rate": None,
        "uncertainty": None,
    }
    if min(result["target_fraction"], result["sky_fraction"]) < required_fraction:
        return {**result, "state": "clipped"}
    density = _background(sky_values, rejection)
    if density is None:
        return {**result, "state": "insufficient_sky"}
    region = target_values - density["median"]
    surface_mean = float(np.mean(region))
    aperture_rate = float(np.sum(region))
    sky_count = density["count"]
    area = target_values.size
    independent_sample_variance = density["sigma"]**2 * (1 / area + 1.57 / sky_count)
    return {
        **result, "state": "measured", "background": density,
        "net_rate": aperture_rate, "surface_rate": surface_mean,
        "uncertainty": math.sqrt(independent_sample_variance),
    }


def _common_options(images, options):
    dimensions = images.dims
    target = Box.from_data(options.get("target"), dimensions, "目标区")
    background = Box.from_data(options.get("sky"), dimensions, "天空区")
    if target.overlaps(background):
        raise DomainError("目标和天空采样区不能重叠")
    fraction = number(options.get("fraction", 0.85), "最低区域可用比例", 0.5, 1)
    rejection = number(options.get("rejection", 3.5), "天空强点剪裁倍数", 1.5, 8)
    return target, background, fraction, rejection


def surface_brightness(images, options):
    target, sky, fraction, rejection = _common_options(images, options)
    rows = usable_science_frames(images.document, options.get("band"))
    if not rows:
        raise DomainError("没有可用于区域光度的科学曝光")
    if len({frame["band"] for frame in rows}) > 1:
        raise DomainError("扩展目标的区域光度请分滤镜计算")
    if len(rows) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("区域光度超过 2400 万样本，请挂起一部分曝光")
    result = []
    for row in rows:
        offset = offset_for(images.document, row["id"])
        entry = {
            "frame": row["id"],
            "filename": row["filename"],
            "band": row["band"],
            "timestamp": time_midpoint(row["started"], row["exposure"]),
            "state": "unaligned",
        }
        if offset is not None:
            plane = warp_to_reference(images.calibrated(row["id"]), offset)
            entry.update(measure_region(plane, row["exposure"], target, sky, fraction, rejection))
        result.append(entry)
    finished = [row for row in result if row["state"] == "measured"]
    return {
        "revision": images.document["revision"], "reference": images.document["reference"],
        "target": target.serialize(), "sky": sky.serialize(), "minimum_fraction": fraction,
        "unit": "ADU/s/pixel", "rows": result,
        "surface_summary": robust_summary([row["surface_rate"] for row in finished]),
        "net_summary": robust_summary([row["net_rate"] for row in finished]),
    }


def draw_regions_on_reference(images, options):
    from .preview import stretch_pixels

    frame_id = images.document["reference"]
    if not frame_id:
        raise DomainError("没有参考曝光可供显示")
    target, sky, _, _ = _common_options(images, options)
    plane = images.calibrated(frame_id)
    gray, _ = stretch_pixels(plane, images.document["stretch"])
    image = Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), "RGB")
    draw = ImageDraw.Draw(image)
    for item, name, color in [(target, "TARGET", (136, 233, 213)), (sky, "SKY", (247, 195, 134))]:
        draw.rectangle((item.left, item.top, item.right - 1, item.bottom - 1), outline=color, width=2)
        draw.text((item.left + 3, max(0, item.top - 14)), name, fill=color)
    image.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
    return png(image)
