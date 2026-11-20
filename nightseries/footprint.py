"""Find usable interior areas from aligned detector support rather than rectangles of image bounds."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .domain import DomainError, MAX_ACTIVE_PIXELS, enum, integer, usable_science_frames
from .preview import png
from .registration import offset_for, warp_to_reference

MODES = {"detector", "valid", "unsaturated"}


@dataclass(frozen=True)
class FootprintSpec:
    layer: str
    minimum: int
    margin: int
    band: str | None

    @classmethod
    def from_options(cls, options, count):
        return cls(
            enum(options.get("layer", "valid"), "覆盖图依据", MODES),
            integer(options.get("minimum", min(2, count)), "最低覆盖曝光", 1, count),
            integer(options.get("margin", 0), "边缘留量像素", 0, 32),
            options.get("band"),
        )


def _geometric_support(dimensions, offset, margin):
    x = np.arange(dimensions.width, dtype=float) + offset.dx
    y = np.arange(dimensions.height, dtype=float) + offset.dy
    cats = (x >= margin) & (x <= dimensions.width - 1 - margin)
    arcs = (y >= margin) & (y <= dimensions.height - 1 - margin)
    return arcs[:, None] & cats[None, :]


def calculate_occupancy(images, frames, spec):
    counter = np.zeros(images.dims.shape, dtype=np.uint16)
    details = []
    for row in frames:
        frame_id = row["id"]
        offset = offset_for(images.document, frame_id)
        if offset is None:
            details.append({"frame": frame_id, "filename": row["filename"], "state": "unaligned", "fraction": None})
            continue
        support = _geometric_support(images.dims, offset, spec.margin)
        if spec.layer != "detector":
            remapped = warp_to_reference(images.calibrated(frame_id), offset)
            support &= remapped.valid
            if spec.layer == "unsaturated":
                support &= ~remapped.saturated
        counter += support
        details.append({
            "frame": frame_id,
            "filename": row["filename"],
            "state": "used" if np.any(support) else "no_support",
            "fraction": float(np.mean(support)),
        })
    return counter, details


def largest_inside(mask):
    height, width = mask.shape
    hist = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = None
    for row in range(height):
        hist = np.where(mask[row], hist + 1, 0)
        increasing = []
        for column in range(width + 1):
            prefix = column
            value = int(hist[column]) if column < width else 0
            while increasing and increasing[-1][1] > value:
                start, high = increasing.pop()
                area = (column - start) * high
                candidate = (row - high + 1, start, row + 1, column)
                if area > best_area or area == best_area and best is not None and candidate < best:
                    best_area, best = area, candidate
                prefix = start
            if value and (not increasing or value > increasing[-1][1]):
                increasing.append((prefix, value))
    if best is None:
        return None
    top, left, bottom, right = best
    return {"left": left, "top": top, "right": right, "bottom": bottom, "pixels": best_area}


def describe_selection(counter, roi):
    left = integer(roi.get("left"), "覆盖左边界", 0, counter.shape[1] - 1)
    top = integer(roi.get("top"), "覆盖上边界", 0, counter.shape[0] - 1)
    right = integer(roi.get("right"), "覆盖右边界", left + 1, counter.shape[1])
    bottom = integer(roi.get("bottom"), "覆盖下边界", top + 1, counter.shape[0])
    area = counter[top:bottom, left:right]
    return {
        "left": left, "top": top, "right": right, "bottom": bottom,
        "minimum": int(np.min(area)), "maximum": int(np.max(area)), "mean": float(np.mean(area)),
        "data": area,
    }


def draw_coverage(counter, used, best, settings):
    scale = counter.astype(float) / max(1, used)
    ready = counter >= settings.minimum
    colored = np.stack(
        [21 + scale * 92 + ready * 30, 20 + scale * 132 + ready * 50, 43 + scale * 136 + ready * 28],
        axis=2,
    )
    colored[counter == 0] = (10, 13, 25)
    image = Image.fromarray(np.rint(colored).astype(np.uint8), "RGB")
    factor = 2 if max(image.size) < 500 else 1
    if factor > 1:
        image = image.resize((image.width * factor, image.height * factor), Image.Resampling.NEAREST)
    if best:
        draw = ImageDraw.Draw(image)
        bounds = [best["left"], best["top"], best["right"] - 1, best["bottom"] - 1]
        draw.rectangle([factor * n for n in bounds], outline=(252, 211, 139), width=max(1, factor))
    image.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
    return png(image)


def footprint_plan(images, options):
    enabled = usable_science_frames(images.document, options.get("band"))
    if not enabled:
        raise DomainError("没有参与覆盖检查的科学曝光")
    chosen = FootprintSpec.from_options(options, len(enabled))
    if chosen.margin * 2 >= min(images.dims.shape):
        raise DomainError("边缘留量过大")
    if len(enabled) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("覆盖检查超过 2400 万像素，请挂起部分曝光")
    count, rows = calculate_occupancy(images, enabled, chosen)
    used = sum(row["state"] == "used" for row in rows)
    sufficient = count >= chosen.minimum
    best = largest_inside(sufficient)
    selection = None
    if isinstance(options.get("roi"), dict):
        selection = describe_selection(count, options["roi"])
        data = selection.pop("data")
        selection["adequate"] = float(np.mean(data >= chosen.minimum))
    return {
        "revision": images.document["revision"],
        "layer": chosen.layer,
        "minimum": chosen.minimum,
        "margin": chosen.margin,
        "adequate": float(np.mean(sufficient)),
        "used": used,
        "best": best,
        "selection": selection,
        "count_histogram": np.bincount(count.ravel(), minlength=used + 1).tolist(),
        "rows": rows,
    }, count, chosen
