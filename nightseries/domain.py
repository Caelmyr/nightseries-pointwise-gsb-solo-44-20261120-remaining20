"""Validated values shared by the observing log and image pipeline."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import re

MAX_FRAMES = 48
MAX_PIXELS = 1_048_576
MAX_ACTIVE_PIXELS = 24_000_000
MAX_ASSET_BYTES = 10 * 1024 * 1024
MAX_SERIES_BYTES = 96 * 1024 * 1024
MAX_APERTURES = 20
MAX_MASKS = 32
MAX_SKY_SAMPLES = 8
MAX_SERIES = 50
MAX_SNAPSHOTS = 30

ASSET_KINDS = {"science", "bias", "dark", "flat"}
SOURCE_ROLES = {"target", "comparison", "check"}
STREAMS = {"raw", "calibrated", "aligned"}

ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")


class DomainError(ValueError):
    """A user input or observation state cannot satisfy the requested action."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def object_value(value, description="输入"):
    if not isinstance(value, dict):
        raise DomainError(f"{description}必须为对象")
    return value


def list_value(value, description, limit):
    if not isinstance(value, list) or len(value) > limit:
        raise DomainError(f"{description}必须是至多 {limit} 项的列表")
    return value


def text(value, description, limit=100, required=True):
    if not isinstance(value, str):
        raise DomainError(f"{description}必须为文字")
    clean = value.strip()
    if required and not clean:
        raise DomainError(f"请填写{description}")
    if len(clean) > limit or any(ord(ch) < 32 for ch in clean):
        raise DomainError(f"{description}含非法字符或超过 {limit} 字符")
    return clean


def enum(value, description, options):
    if not isinstance(value, str) or value not in options:
        raise DomainError(f"{description}无效")
    return value


def number(value, description, lower, upper):
    if isinstance(value, bool):
        raise DomainError(f"{description}必须是数字")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DomainError(f"{description}必须是数字") from exc
    if not math.isfinite(result) or not lower <= result <= upper:
        raise DomainError(f"{description}必须在 {lower:g} 至 {upper:g} 之间")
    return result


def integer(value, description, lower, upper):
    result = number(value, description, lower, upper)
    if result != int(result):
        raise DomainError(f"{description}必须为整数")
    return int(result)


def boolean(value, description):
    if not isinstance(value, bool):
        raise DomainError(f"{description}必须为开关")
    return value


def identifier(value, description="编号"):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise DomainError(f"{description}无效")
    return value


def iso_time(value, description="曝光开始时间"):
    original = text(value, description, 38)
    if "T" not in original and " " not in original:
        raise DomainError(f"{description}须精确到时间")
    try:
        result = datetime.fromisoformat(original.replace("Z", "+00:00"))
    except (ValueError, OverflowError) as exc:
        raise DomainError(f"{description}须为 ISO 8601 日期时间") from exc
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    try:
        utc = result.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise DomainError(f"{description}换算成 UTC 后超出范围") from exc
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Dimensions:
    width: int
    height: int

    @classmethod
    def checked(cls, width, height):
        width = integer(width, "宽度", 16, MAX_PIXELS // 16)
        height = integer(height, "高度", 16, MAX_PIXELS // 16)
        if width * height > MAX_PIXELS:
            raise DomainError(f"一张图最多 {MAX_PIXELS:,} 像素")
        return cls(width, height)

    @property
    def pixels(self):
        return self.width * self.height

    @property
    def shape(self):
        return self.height, self.width

    def xy(self, x, y):
        return (
            number(x, "横坐标", 0, self.width - 1),
            number(y, "纵坐标", 0, self.height - 1),
        )


def science_metadata(source, image_metadata=None):
    source = object_value(source, "影像元数据")
    header = image_metadata or {}
    started = source.get("started") or header.get("started")
    exposure = source.get("exposure")
    if exposure in ("", None):
        exposure = header.get("exposure")
    band = source.get("band") or header.get("band") or ""
    start = iso_time(started)
    seconds = number(exposure, "曝光秒数", 0.001, 86400)
    try:
        finish = datetime.fromisoformat(start.replace("Z", "+00:00")) + timedelta(seconds=seconds)
    except (ValueError, OverflowError) as exc:
        raise DomainError("曝光完成时间超出支持的日历范围") from exc
    if finish.year < 1900:
        raise DomainError("观测时间需为 1900 年或之后")
    return {
        "started": start,
        "exposure": seconds,
        "band": text(band, "滤镜", 24, required=False),
        "enabled": boolean(source.get("enabled", True), "参与状态"),
        "notes": text(source.get("notes", ""), "备注", 300, required=False),
    }


def calibration_metadata(kind, source, image_metadata=None):
    source = object_value(source, "校准元数据")
    header = image_metadata or {}
    exposure = source.get("exposure")
    if exposure in ("", None):
        exposure = header.get("exposure")
    return {
        "exposure": 0.0 if kind == "bias" else number(exposure, "曝光秒数", 0.001, 86400),
        "band": text(source.get("band") or header.get("band") or "", "滤镜", 24, required=False),
        "enabled": boolean(source.get("enabled", True), "参与状态"),
        "notes": text(source.get("notes", ""), "备注", 300, required=False),
    }


def photometry_settings(value):
    data = object_value(value, "孔径设置")
    radius = number(data.get("radius", 4.5), "孔径半径", 1.5, 28)
    inner = number(data.get("sky_inner", 8), "背景内径", 3, 46)
    outer = number(data.get("sky_outer", 13), "背景外径", 4, 60)
    if radius + 1 > inner or inner + 2 > outer:
        raise DomainError("背景环须与测光孔径分开且至少宽 2 像素")
    return {
        "radius": radius,
        "sky_inner": inner,
        "sky_outer": outer,
        "min_fraction": number(data.get("min_fraction", 0.85), "最低孔径覆盖", 0.5, 1),
        "gain": number(data.get("gain", 1.0), "增益 e⁻/ADU", 0.02, 1000),
        "saturation": number(data.get("saturation", 65000), "饱和值", 1, 1e12),
    }


def aperture_value(value, dimensions):
    data = object_value(value, "孔径")
    x, y = dimensions.xy(data.get("x"), data.get("y"))
    return {
        "label": text(data.get("label"), "天体标签", 42),
        "role": enum(data.get("role", "target"), "孔径角色", SOURCE_ROLES),
        "x": x,
        "y": y,
    }


def mask_value(value, dimensions):
    data = object_value(value, "掩膜")
    left, top = dimensions.xy(data.get("left"), data.get("top"))
    right = number(data.get("right"), "掩膜右边界", 1, dimensions.width)
    bottom = number(data.get("bottom"), "掩膜下边界", 1, dimensions.height)
    if right - left < 1 or bottom - top < 1:
        raise DomainError("掩膜需覆盖至少一个像素")
    return {
        "label": text(data.get("label"), "掩膜名称", 50),
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "frame": identifier(data["frame"], "掩膜曝光") if data.get("frame") else None,
    }


def sky_sample_value(value, dimensions):
    data = object_value(value, "空天样本")
    left, top = dimensions.xy(data.get("left"), data.get("top"))
    right = number(data.get("right"), "空天样本右边界", 1, dimensions.width)
    bottom = number(data.get("bottom"), "空天样本下边界", 1, dimensions.height)
    if right - left < 2 or bottom - top < 2:
        raise DomainError("空天样本需覆盖至少 2 × 2 像素")
    if (right - left) * (bottom - top) < 64:
        raise DomainError("空天样本面积过小，至少要 8 × 8 像素")
    return {
        "label": text(data.get("label"), "空天样本名称", 50),
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "frame": identifier(data["frame"], "空天样本曝光") if data.get("frame") else None,
    }


def alignment_value(value):
    data = object_value(value, "对准偏移")
    return {
        "dx": number(data.get("dx", 0), "横向偏移", -512, 512),
        "dy": number(data.get("dy", 0), "纵向偏移", -512, 512),
        "source": "manual",
    }


def stretch_value(value):
    data = object_value(value, "显示比例")
    mode = enum(data.get("mode", "asinh"), "拉伸方式", {"linear", "asinh", "log"})
    low = number(data.get("low", 1), "黑点百分位", 0, 99.9)
    high = number(data.get("high", 99.7), "白点百分位", 0.1, 100)
    if low >= high:
        raise DomainError("白点百分位须高于黑点")
    return {"mode": mode, "low": low, "high": high, "invert": boolean(data.get("invert", False), "反相")}


def calibration_settings(value):
    data = object_value(value, "校准设置")
    return {
        "bias": boolean(data.get("bias", True), "偏置"),
        "dark": boolean(data.get("dark", True), "暗场"),
        "flat": boolean(data.get("flat", True), "平场"),
        "flat_floor": number(data.get("flat_floor", 0.2), "平场最低响应", 0.05, 0.9),
    }


def new_document(name, synthetic=False):
    return {
        "name": text(name, "观测组名称", 80),
        "synthetic": synthetic,
        "notes": "",
        "frames": [],
        "calibration_frames": [],
        "dimensions": None,
        "reference": None,
        "alignments": {},
        "apertures": [],
        "masks": [],
        "sky_samples": [],
        "photometry": photometry_settings({}),
        "calibration": calibration_settings({}),
        "stretch": stretch_value({}),
        "plate": None,
        "catalog": [],
    }


def dimensions_of(document):
    value = document.get("dimensions")
    if value is None:
        raise DomainError("请先导入一张科学曝光")
    return Dimensions.checked(value["width"], value["height"])


def usable_science_frames(document, band=None):
    frames = [
        frame for frame in document["frames"]
        if frame["enabled"] and (band is None or frame["band"] == band)
    ]
    return sorted(frames, key=lambda frame: (frame["started"], frame["id"]))


def exposure_row(document, frame_id, include_calibration=False):
    frames = document["frames"]
    if include_calibration:
        frames = frames + document["calibration_frames"]
    for frame in frames:
        if frame["id"] == frame_id:
            return frame
    raise DomainError("找不到指定曝光", 404)
