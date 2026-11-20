"""Three-filter colour composite that separates luminance evidence from chroma.

Each filter is integrated in reference coordinates on its own exposure-time scale,
with its own smooth sky model removed.  The per-pixel colour is frozen as a ratio
against that pixel's mean band rate, and the display stretch applies only to
luminance, so changing the display brightness cannot change a measured colour
relationship.

A pixel is coloured only where all three filters carry signal above their local noise.
Elsewhere no missing band is treated as zero: the pixel falls back to measured
grayscale (the mean of whatever bands did measure), and the per-pixel reason — frame
edge, saturation, noise — is kept in a categorical label and summarised per class.
"""
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter

from .calibration import robust_summary
from .domain import DomainError, boolean, number
from .preview import png
from .stacking import compute_stack

ROLE_CODES = {"red": "r", "green": "g", "blue": "b"}
ROLE_ORDER = (("red", "红"), ("green", "绿"), ("blue", "蓝"))
CHANNEL_ORDER = ("r", "g", "b")
SIGNAL_TO_NOISE = 2.0
TILE_NOISE_FLOOR = 0.7
TILE_FLOOR_FRACTION = 0.6
MAX_RENDER_SIDE = 1200
DEFAULT_STRENGTH = 0.8

# code, Chinese reason (page), ASCII reason (rendered PNG), legend tone.
# Generated images stay ASCII because the minimal container has no CJK font.
REASONS = (
    ("covered", "三波段均有信号，按测得的颜色比着色", "colour: all 3 bands", "ok"),
    ("saturated", "有波段饱和，改用灰度", "grey: band saturated", "saturated"),
    ("noise", "有波段只到噪声，改用灰度", "grey: band is noise", "noise"),
    ("two_channels", "缺一个波段（画幅或未对准），改用灰度", "grey: 1 band missing", "partial"),
    ("one_channel", "只剩一个波段，改用灰度", "grey: 2 bands missing", "partial"),
    ("outside", "该处没有任何波段覆盖", "grey: no coverage", "outside"),
)
REASON_INDEX = {code: index for index, (code, _, _, _) in enumerate(REASONS)}
REASON_TEXT = {code: text for code, text, _, _ in REASONS}
PALETTE = {
    "ok": (112, 190, 122),
    "partial": (208, 158, 72),
    "saturated": (208, 94, 62),
    "noise": (66, 158, 166),
    "outside": (104, 88, 158),
}


@dataclass(frozen=True)
class BandLayer:
    band: str
    role: str
    plane: object
    coverage: np.ndarray
    saturated: np.ndarray
    report: dict
    noise: float
    tile_noise: np.ndarray


def _band_name(entry, options, fallback, available):
    value = entry.get("filter") if entry.get("filter") not in (None, "") else options.get("band") or fallback
    if not isinstance(value, str) or not value.strip():
        raise DomainError("请为三个颜色通道分别指定滤镜")
    chosen = value.strip()
    if chosen not in available:
        raise DomainError(f"观测中没有「{chosen}」滤镜参与的科学曝光")
    return chosen


def channel_assignment(options, doc):
    entries = options.get("channels")
    if entries is None:
        entries = [{"role": role, "filter": options.get("band") or role} for role, _ in ROLE_ORDER]
    if not isinstance(entries, list) or len(entries) != 3:
        raise DomainError("彩色合成需要红、绿、蓝三个通道")
    available = {frame["band"] for frame in doc["frames"] if frame["enabled"]}
    chosen = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise DomainError("每个颜色通道必须为对象")
        role = entry.get("role", ROLE_ORDER[index][0])
        if role not in ROLE_CODES:
            raise DomainError("颜色通道只支持红、绿、蓝")
        chosen.append((role, _band_name(entry, options, ROLE_ORDER[index][0], available)))
    if len({role for role, _ in chosen}) != 3:
        raise DomainError("红、绿、蓝三个通道须各一")
    if len({band for _, band in chosen}) != 3:
        raise DomainError("三个颜色通道需要三种不同的滤镜")
    return {ROLE_CODES[role]: band for role, band in chosen}


def stack_mode(options):
    mode = options.get("mode", "median")
    if mode not in {"mean", "median"}:
        raise DomainError("叠加方式只支持均值或中位")
    return mode


def stack_support(options):
    return int(number(options.get("min_support", 1), "最低覆盖帧数", 1, 48))


def noise_floor(plane, limit=750_000):
    """Stellar-robust sky scatter of an integrated plane, in ADU/s."""
    samples = np.where(plane.valid & np.isfinite(plane.values), plane.values, np.nan)
    if samples.size > limit:
        step = int(np.ceil(np.sqrt(samples.size / limit)))
        samples = samples[::step, ::step]
    values = samples[np.isfinite(samples)]
    if values.size < 64:
        raise DomainError("有效像素太少，无法判断该波段噪声")
    return _sigma(values)


def _sigma(values):
    summary = robust_summary(values)
    sigma = summary["sigma"]
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        sigma = max(abs(summary["median"] or 0.0) * 1e-4, 1e-6)
    return float(sigma)


def noise_grid(plane, tiles=24):
    """Per-tile sky scatter, so a residual sky gradient is not read as signal.

    A feature is only coloured when it clears the local scatter of its own patch.
    Single-pixel noise that happens to correlate across filters is handled separately
    by the 3×3 coherence test in classify(); this grid handles broad gradients.
    """
    values = np.where(plane.valid & np.isfinite(plane.values), plane.values, np.nan)
    height, width = values.shape
    rows = np.linspace(0, height, min(tiles, height) + 1).astype(int)
    columns = np.linspace(0, width, min(tiles, width) + 1).astype(int)
    grid = np.zeros(values.shape, dtype=np.float64)
    fallback = noise_floor(plane)
    known = []
    for row in range(len(rows) - 1):
        for column in range(len(columns) - 1):
            patch = values[rows[row]:rows[row + 1],
                           columns[column]:columns[column + 1]]
            kept = patch[np.isfinite(patch)]
            sigma = _sigma(kept) if kept.size >= 24 else None
            grid[rows[row]:rows[row + 1], columns[column]:columns[column + 1]] = sigma or fallback
            if sigma:
                known.append(sigma)
    # A locally flat tile can underestimate the scatter, so hold every tile to a
    # fraction of the frame-wide value: tiles still follow a gradient, but cannot
    # claim implausibly low noise and turn ordinary background into "signal".
    if known:
        grid = np.maximum(grid, sorted(known)[len(known) // 2] * TILE_FLOOR_FRACTION)
    return grid


def box_mean(values, valid, size=3):
    """Mean over the valid pixels of a size×size box, and that box's valid count."""
    filled = np.where(valid, np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    weight = uniform_filter(valid.astype(np.float64), size=size, mode="nearest")
    total = uniform_filter(filled, size=size, mode="nearest")
    mean = np.divide(total, weight, out=np.zeros_like(total), where=weight > 1e-9)
    return mean, weight * size * size


def build_layers(images, options):
    doc = images.document
    if not doc["reference"]:
        raise DomainError("请先选择一张参考曝光")
    assignment = channel_assignment(options, doc)
    mode = stack_mode(options)
    support = stack_support(options)
    center = boolean(options.get("subtract_background", True), "扣除各波段天空背景")
    layers = {}
    for role in CHANNEL_ORDER:
        band = assignment[role]
        product = compute_stack(images, {
            "band": band, "mode": mode, "min_support": support,
            "subtract_background": center, "coverage": "all",
        })
        plane = product.plane
        if center:
            # A per-band smooth sky model removes what a constant median leaves behind:
            # a gradient difference between filters would otherwise read as colour.
            plane = _sky_subtracted(plane)
        layers[role] = BandLayer(
            band, role, plane, product.coverage, product.saturated,
            product.report, noise_floor(plane), noise_grid(plane),
        )
    return assignment, [layers[role] for role in CHANNEL_ORDER]


def _sky_subtracted(plane):
    from .background import sky_surface

    if np.count_nonzero(plane.valid) < 400:
        return plane
    try:
        _, predicted, _ = sky_surface(plane, {"tiles": 10, "kind": "curved"})
    except DomainError:
        return plane
    values = np.where(plane.valid, plane.values - predicted, np.nan)
    return type(plane)(values, plane.valid, plane.saturated, plane.notes)


def classify(layers):
    """Per-pixel hue ratio where every band has clean signal, plus an evidence label.

    A band counts as evidence only when its rate clears the local noise both pixel by
    pixel and on a 3×3 average.  The 3×3 test rejects single-pixel noise correlations
    across filters while keeping compact real features (a few pixels across).
    """
    shape = layers[0].plane.values.shape
    support = np.stack([layer.coverage > 0 for layer in layers], axis=-1)
    saturated = np.stack([layer.saturated > 0 for layer in layers], axis=-1)
    values = np.stack([layer.plane.values for layer in layers], axis=-1)
    rates = np.where(support & np.isfinite(values), values, np.nan)
    local_noise = np.stack([layer.tile_noise for layer in layers], axis=-1)
    local_noise = np.maximum(local_noise, TILE_NOISE_FLOOR * np.array([layer.noise for layer in layers]))
    signal = support & np.isfinite(rates) & (rates >= SIGNAL_TO_NOISE * local_noise)
    for index in range(3):
        smooth, effective = box_mean(rates[..., index], np.isfinite(rates[..., index]))
        coherent = smooth >= SIGNAL_TO_NOISE * local_noise[..., index] / np.sqrt(np.maximum(effective, 1.0))
        signal[..., index] &= coherent
    clean = signal & ~saturated

    with np.errstate(all="ignore"):
        level = np.nanmean(rates, axis=-1)
    covered = np.all(clean, axis=-1) & np.isfinite(level) & (level > 0)
    hue = np.ones(shape + (3,), dtype=np.float64)
    if np.any(covered):
        with np.errstate(all="ignore"):
            ratio = rates / level[..., None]
        hue[covered] = np.where(np.isfinite(ratio[covered]), ratio[covered], 1.0)

    label = np.full(shape, REASON_INDEX["outside"], dtype=np.uint8)
    label[covered] = REASON_INDEX["covered"]
    available = np.count_nonzero(support, axis=-1)
    unfilled = label == REASON_INDEX["outside"]
    label[(available == 2) & unfilled] = REASON_INDEX["two_channels"]
    label[(available == 1) & unfilled] = REASON_INDEX["one_channel"]
    any_support = support.any(axis=-1)
    label[any_support & ~covered & np.any(saturated, axis=-1)] = REASON_INDEX["saturated"]
    label[any_support & ~covered & (label == REASON_INDEX["outside"])] = REASON_INDEX["noise"]
    return hue, label, rates


def _stretch_bounds(rates, options):
    low = number(options.get("low", 1.0), "黑点百分位", 0.0, 99.0)
    high = number(options.get("high", 99.5), "白点百分位", 0.1, 100.0)
    if low >= high:
        raise DomainError("白点百分位须高于黑点")
    with np.errstate(all="ignore"):
        level = np.nanmean(rates, axis=-1)
    sample = level[np.isfinite(level)]
    if sample.size < 16:
        raise DomainError("可测像素太少，无法确定显示范围")
    first, last = (float(value) for value in np.percentile(sample, [low, high]))
    if not np.isfinite(last) or last <= first:
        last = first + max(abs(first) * 1e-3, 1.0)
    return first, last


def display_rgb(hue, label, rates, options):
    """Luminance-only display map with an exactly fixed hue.

    The colour of a pixel is the measured rate vector, frozen at classification.  The
    display stretch produces one scalar per pixel and multiplies that whole vector, so
    no display setting can change the ratio between channels — only brightness.  Where
    colour evidence is incomplete the same scalar is shown in neutral gray.
    """
    amount = number(options.get("strength", DEFAULT_STRENGTH), "着色强度", 0.0, 1.0)
    gain = number(options.get("gain", 1.0), "亮度增益", 0.05, 50.0)
    low, high = _stretch_bounds(rates, options)
    with np.errstate(all="ignore"):
        level = np.nanmean(rates, axis=-1)
    any_measure = np.isfinite(level)
    level = np.where(any_measure, level, 0.0)
    scaled = np.clip((level - low) / max(high - low, 1e-9), 0.0, None) * gain
    luminance = np.arcsinh(scaled * 10.0) / np.arcsinh(10.0)
    luminance = np.where(any_measure, luminance, 0.0)

    mixed = np.repeat(luminance[..., None], 3, axis=-1)
    colored = label == REASON_INDEX["covered"]
    if np.any(colored):
        chroma = np.clip(hue / np.maximum(hue.mean(axis=-1, keepdims=True), 1e-12), 0.0, None)
        chroma = np.clip(1.0 + amount * (chroma - 1.0), 0.25, 3.0)
        # Unit mean again: colour must contribute no brightness of its own.
        chroma = chroma / np.maximum(chroma.mean(axis=-1, keepdims=True), 1e-12)
        mixed[colored] = luminance[colored, None] * chroma[colored]
    # When the brightest channel tops out, compress all three by the same peak factor:
    # the pixel reaches white without per-channel clipping, so its colour ratio survives.
    peak = mixed.max(axis=-1, keepdims=True)
    mixed = np.where(peak > 1.0, mixed / np.maximum(peak, 1e-12), mixed)
    return np.clip(mixed, 0.0, 1.0), (low, high)


def compute_composite(images, options):
    if not isinstance(options, dict):
        raise DomainError("合成选项必须为对象")
    assignment, layers = build_layers(images, options)
    hue, label, rates = classify(layers)
    if not np.any(np.isfinite(rates)):
        raise DomainError("三个滤镜在参考画幅内没有共同覆盖的像素，无法合成彩色图")
    if not np.any(label == REASON_INDEX["covered"]):
        raise DomainError("没有像素同时取得三个波段的可靠信号，无法确定颜色关系")
    rgb, scale = display_rgb(hue, label, rates, options)
    pixels = np.rint(rgb * 255).astype(np.uint8)
    if max(pixels.shape[:2]) > MAX_RENDER_SIDE:
        step = int(np.ceil(max(pixels.shape[:2]) / MAX_RENDER_SIDE))
        pixels = np.asarray(
            Image.fromarray(pixels, "RGB").resize(
                (max(1, pixels.shape[1] // step), max(1, pixels.shape[0] // step)),
                Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
        label = label[::step, ::step]
    report = {
        "revision": images.document["revision"],
        "reference": images.document["reference"],
        "width": images.dims.width,
        "height": images.dims.height,
        "channels": assignment,
        "mode": stack_mode(options),
        "min_support": stack_support(options),
        "subtract_background": bool(options.get("subtract_background", True)),
        "strength": number(options.get("strength", DEFAULT_STRENGTH), "着色强度", 0.0, 1.0),
        "gain": number(options.get("gain", 1.0), "亮度增益", 0.05, 50.0),
        "snr_threshold": SIGNAL_TO_NOISE,
        "scale": [float(scale[0]), float(scale[1])],
        "bands": [_band_report(layer) for layer in layers],
        "counts": _counts(label),
    }
    return report, pixels, label


def _band_report(layer):
    used = layer.report["used"]
    return {
        "band": layer.band,
        "role": layer.role,
        "frames": len(used),
        "exposure_seconds": float(sum(row["exposure"] for row in used)),
        "noise": layer.noise,
        "filenames": [row["filename"] for row in used],
        "skipped": layer.report["skipped"],
        "qualified_fraction": layer.report["qualified_fraction"],
    }


def _counts(label):
    total = int(label.size)
    rows = []
    for code, text, english, tone in REASONS:
        amount = int(np.count_nonzero(label == REASON_INDEX[code]))
        rows.append({
            "code": code, "label": text, "english": english, "tone": tone,
            "pixels": amount, "fraction": amount / total if total else 0.0,
        })
    return rows


LEGEND_GAP = 26
LEGEND_ROW = 30


def _wrap(draw, text, font, width):
    lines, line = [], ""
    for word in text.split(" "):
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines or [""]


def legend(report, channels, width=460):
    entries = [row for row in report["counts"] if row["pixels"]]
    red, green, blue = (channels[role] for role in CHANNEL_ORDER)
    note = (f"NightSeries colour composite   R={red}  G={green}  B={blue}"
            "   per-filter linear rate, luminance-only stretch")
    subtitle = "Grey keeps the measured shape wherever colour evidence is incomplete"
    font = ImageFont.load_default(size=12)
    small = ImageFont.load_default(size=11)
    width = int(max(320, width))
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    head = _wrap(draw, note, font, width - 32) + _wrap(draw, subtitle, small, width - 32)
    # Lay the evidence classes out in as many rows as the image width needs.
    rows = [[]]
    x = 16.0
    for row in entries:
        item = 19 + max(small.getlength(row["english"]), small.getlength(f"{row['fraction'] * 100:.1f}%")) + LEGEND_GAP
        if rows[-1] and x + item > width - 16:
            rows.append([])
            x = 16.0
        rows[-1].append((row, x))
        x += item
    top = 9 + len(head) * 18 + 8
    height = top + len(rows) * LEGEND_ROW + 6
    canvas = Image.new("RGB", (width, height), (12, 15, 23))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(head):
        draw.text((16, 9 + index * 18), line,
                  fill=(184, 199, 220) if index == 0 else (139, 153, 175),
                  font=font if index == 0 else small)
    for index, line in enumerate(rows):
        row_top = top + index * LEGEND_ROW
        for item, left in line:
            draw.rounded_rectangle((left, row_top + 3, left + 13, row_top + 17), radius=3,
                                   fill=PALETTE[item["tone"]])
            draw.text((left + 19, row_top - 2), item["english"], fill=(172, 186, 206), font=small)
            draw.text((left + 19, row_top + 12), f"{item['fraction'] * 100:.1f}%",
                      fill=(122, 135, 156), font=small)
    return png(canvas)


def preview(report, pixels):
    return png(Image.fromarray(pixels, "RGB"))


def download(report, pixels):
    body = Image.fromarray(pixels, "RGB")
    width = max(body.width, 520)
    key = Image.open(BytesIO(legend(report, report["channels"], width))).convert("RGB")
    width = max(body.width, key.width)
    out = Image.new("RGB", (width, body.height + key.height + 6), (12, 15, 23))
    out.paste(body, ((width - body.width) // 2, 0))
    out.paste(key, ((width - key.width) // 2, body.height + 6))
    return png(out)
