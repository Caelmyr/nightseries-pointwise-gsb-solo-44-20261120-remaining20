"""Produce uniform-scale inspection contact sheets from selected exposure images."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .domain import DomainError, MAX_ACTIVE_PIXELS, STREAMS, boolean, enum, integer, list_value
from .preview import percentiles, png
from .registration import offset_for, preview_stream

PANEL_SIZES = {200, 268, 336}
BACKGROUNDS = {"dark": (17, 22, 34), "print": (236, 240, 246)}
MAX_SHEET_FRAMES = 24


@dataclass(frozen=True)
class SheetSettings:
    columns: int
    panel_size: int
    stream: str
    theme: str
    linked_scale: bool
    flag_invalid: bool

    @classmethod
    def from_options(cls, options):
        size = integer(options.get("panel_size", 268), "缩略图宽度", 200, 336)
        if size not in PANEL_SIZES:
            raise DomainError("缩略图宽度可选 200、268 或 336")
        return cls(
            integer(options.get("columns", 4), "排版列数", 2, 6),
            size,
            enum(options.get("stream", "calibrated"), "图层", STREAMS),
            enum(options.get("theme", "dark"), "背景", set(BACKGROUNDS)),
            boolean(options.get("linked_scale", True), "统一亮度范围"),
            boolean(options.get("flag_invalid", False), "标示无效区"),
        )


def _choose_frames(document, selection):
    by_id = {row["id"]: row for row in document["frames"]}
    if selection is None:
        return [row for row in document["frames"] if row["enabled"]][:MAX_SHEET_FRAMES]
    ids = list_value(selection, "预览册曝光", MAX_SHEET_FRAMES)
    if any(not isinstance(token, str) or token not in by_id for token in ids) or len(set(ids)) != len(ids):
        raise DomainError("预览册曝光编号不存在或重复")
    return [by_id[token] for token in ids]


def _planes(images, frames, stream):
    for frame in frames:
        frame_id = frame["id"]
        if stream == "aligned" and offset_for(images.document, frame_id) is None:
            yield frame, None
            continue
        yield frame, preview_stream(images, frame_id, stream)


def _scale_bound(plane, display):
    return percentiles(plane, display) if plane is not None else None


def _tone(scale, display):
    if display["mode"] == "asinh":
        result = np.arcsinh(scale * 10) / np.arcsinh(10)
    elif display["mode"] == "log":
        result = np.log1p(scale * 30) / np.log1p(30)
    else:
        result = scale
    return 1 - result if display["invert"] else result


def preview_chip(plane, options, width, height, bounds, display):
    fill = (14, 16, 23) if options.theme == "dark" else (216, 221, 228)
    chip = Image.new("RGB", (width, height), fill)
    if plane is None:
        draw = ImageDraw.Draw(chip)
        color = (163, 177, 199) if options.theme == "dark" else (56, 63, 82)
        draw.line((0, 0, width, height), fill=color, width=2)
        draw.line((width, 0, 0, height), fill=color, width=2)
        return chip
    low, high = bounds
    clipped = np.clip((np.where(plane.valid, plane.values, low) - low) / max(high - low, 1e-7), 0, 1)
    gray = np.rint(255 * _tone(clipped, display)).astype(np.uint8)
    tile = np.repeat(gray[:, :, None], 3, axis=2)
    tile[~plane.valid] = (91, 58, 108) if options.flag_invalid else (12, 12, 12)
    if options.flag_invalid:
        tile[plane.saturated & plane.valid] = (238, 175, 95)
    thumb = Image.fromarray(tile, "RGB")
    thumb.thumbnail((width, height), Image.Resampling.LANCZOS)
    chip.paste(thumb, ((width - thumb.width) // 2, (height - thumb.height) // 2))
    return chip


def _report_count(planes, bounds):
    return sum(plane is not None and bound is not None for (_, plane), bound in zip(planes, bounds))


def _label_draw(draw, location, frame, bounds, options, font):
    x, y = location
    foreground = (212, 223, 242) if options.theme == "dark" else (19, 30, 49)
    quiet = (149, 167, 196) if options.theme == "dark" else (56, 73, 94)
    filename = frame["filename"].encode("ascii", "replace").decode()
    limit = options.panel_size - 12
    while len(filename) > 10 and font.getlength(filename) > limit:
        filename = filename[:-4] + "..."
    draw.text((x + 1, y + 2), filename, fill=foreground, font=font)
    started = frame["started"][11:19] + " UTC"
    text = f"{started}   {frame['exposure']:g}s"
    if bounds is None:
        text += "   NO ALIGN"
    elif not frame["enabled"]:
        text += "   HELD"
    draw.text((x + 1, y + 20), text, fill=quiet, font=font)


def contact_sheet(images, options):
    settings = SheetSettings.from_options(options)
    selected = _choose_frames(images.document, options.get("frames"))
    if not selected:
        raise DomainError("请选取编入预览册的科学曝光")
    if len(selected) * images.dims.pixels > MAX_ACTIVE_PIXELS:
        raise DomainError("预览册单次最多处理 2400 万像素")
    panels = list(_planes(images, selected, settings.stream))
    display = images.document["stretch"]
    bands = {row["band"] for row in selected}
    if settings.linked_scale and len(bands) > 1:
        raise DomainError("统一亮度范围时请选择同滤镜曝光，或改用单帧自动范围")
    if settings.linked_scale:
        seed = next((plane for row, plane in panels if row["id"] == images.document["reference"] and plane is not None), None)
        if seed is None:
            seed = next((plane for _, plane in panels if plane is not None), None)
        if seed is None:
            raise DomainError("选定曝光没有能绘制的影像")
        shared = _scale_bound(seed, display)
        bounds = [shared if plane is not None else None for _, plane in panels]
    else:
        bounds = [_scale_bound(plane, display) for _, plane in panels]
    panel_width = settings.panel_size
    panel_height = max(110, int(panel_width * min(1, images.dims.height / images.dims.width)))
    horizontal, vertical = 18, 12
    columns = min(settings.columns, len(selected))
    row_count = math.ceil(len(selected) / columns)
    width = horizontal + columns * (panel_width + horizontal)
    height = 76 + row_count * (panel_height + 43 + vertical)
    if width * height > 12_000_000:
        raise DomainError("预览册画幅过大，请减少图块或使用较小缩略图")
    color = BACKGROUNDS[settings.theme]
    out = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=12)
    header = ImageFont.load_default(size=21)
    head_color = (159, 226, 221) if settings.theme == "dark" else (18, 81, 83)
    draw.text((horizontal, 17), "NightSeries  /  Exposure Contact Sheet", fill=head_color, font=header)
    descriptor = f"{len(selected)} exposures   |   {settings.stream.upper()}   |   {'Common range' if settings.linked_scale else 'Independent ranges'}"
    draw.text((horizontal + 1, 46), descriptor, fill=head_color, font=font)
    for index, ((frame, plane), bound) in enumerate(zip(panels, bounds)):
        row, col = divmod(index, columns)
        x = horizontal + col * (panel_width + horizontal)
        y = 70 + row * (panel_height + 43 + vertical)
        tile = preview_chip(plane, settings, panel_width, panel_height, bound, display)
        out.paste(tile, (x, y))
        _label_draw(draw, (x, y + panel_height + 4), frame, bound, settings, font)
    result = {
        "count": len(selected),
        "displayed": _report_count(panels, bounds),
        "width": width,
        "height": height,
        "tiles": [
            {"frame": frame["id"], "filename": frame["filename"], "bounds": list(bound) if bound else None,
             "state": "ok" if bound else "unaligned"}
            for (frame, _), bound in zip(panels, bounds)
        ],
    }
    return png(out), result
