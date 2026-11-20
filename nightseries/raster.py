"""Decode original detector images and encode downloadable quantitative grids."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import math
import re

import numpy as np
from PIL import Image, UnidentifiedImageError

from .astrometry import SKY_KEYS
from .domain import Dimensions, DomainError, MAX_ASSET_BYTES

FITS_RECORD = 2880
FITS_CARD = 80
FITS_TYPE = {
    8: "u1",
    16: ">i2",
    32: ">i4",
    -32: ">f4",
    -64: ">f8",
}
FITS_SUFFIX = {".fits", ".fit", ".fts"}
RASTER_SUFFIX = {".png", ".tif", ".tiff"}
KEYWORD = re.compile(r"[A-Z0-9_-]{1,8}$")
INTERESTING_KEYS = (
    "DATE-OBS", "TIMESYS", "EXPTIME", "FILTER", "OBJECT", "TELESCOP",
    "INSTRUME", "BUNIT", "SATURATE", "BSCALE", "BZERO", "BLANK", "CROTA1", "CROTA2",
) + SKY_KEYS


@dataclass(frozen=True)
class ExposureImage:
    samples: np.ndarray
    metadata: dict
    kind: str

    @property
    def dimensions(self):
        height, width = self.samples.shape
        return Dimensions(width, height)


def fits_value(field):
    field = field.lstrip()
    if not field:
        return None
    if field.startswith("'"):
        index = 1
        chars = []
        while index < len(field):
            char = field[index]
            if char == "'":
                if field[index + 1:index + 2] != "'":
                    return "".join(chars).rstrip()
                index += 1
            chars.append(char)
            index += 1
        raise DomainError("FITS 有未闭合的字符串卡")
    token = field.split("/", 1)[0].strip()
    if token in {"T", "F"}:
        return token == "T"
    if not token:
        return None
    try:
        if "." in token or "E" in token or "D" in token:
            return float(token.replace("D", "E"))
        return int(token)
    except ValueError:
        return token


def read_fits_header(blob):
    data = {}
    header_limit = min(len(blob), 128 * FITS_RECORD)
    for start in range(0, header_limit - FITS_RECORD + 1, FITS_RECORD):
        record = blob[start:start + FITS_RECORD]
        for offset in range(0, FITS_RECORD, FITS_CARD):
            try:
                card = record[offset:offset + FITS_CARD].decode("ascii")
            except UnicodeDecodeError as exc:
                raise DomainError("FITS 主头含非 ASCII 字节") from exc
            keyword = card[:8].strip()
            if keyword == "END":
                return data, start + FITS_RECORD
            if not KEYWORD.fullmatch(keyword) or card[8:10] != "= ":
                continue
            value = fits_value(card[10:])
            if keyword in data and (keyword.startswith("NAXIS") or keyword in {"SIMPLE", "BITPIX"}):
                raise DomainError(f"FITS 主头重复声明 {keyword}")
            data[keyword] = value
    raise DomainError("FITS 主头未在允许范围内结束")


def header_number(header, key, default):
    value = header.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DomainError(f"FITS {key} 必须为有限数值")
    return value


def fit_metadata(header):
    timesys = str(header.get("TIMESYS", "UTC")).upper()
    observation_time = header.get("DATE-OBS") if timesys in {"UTC", "UT"} else None
    exposure = header.get("EXPTIME", header.get("EXPOSURE"))
    metadata = {
        "started": observation_time if isinstance(observation_time, str) else None,
        "exposure": exposure if type(exposure) in (int, float) and math.isfinite(exposure) else None,
        "band": str(header.get("FILTER", ""))[:24],
        "header": {
            key: header[key] for key in INTERESTING_KEYS
            if key in header and (not isinstance(header[key], float) or math.isfinite(header[key]))
        },
    }
    if any(key.startswith(("PV1_", "PV2_", "A_ORDER", "B_ORDER")) for key in header):
        metadata["header"]["DISTORTED"] = True
    if timesys not in {"UTC", "UT"}:
        metadata["timing_note"] = f"源时间系统 {timesys}，需填写换算后的 UTC"
    return metadata


def read_fits(blob):
    header, offset = read_fits_header(blob)
    if header.get("SIMPLE") is not True:
        raise DomainError("只读取标准 FITS 主 HDU")
    if header.get("NAXIS") != 2 or header.get("GROUPS") is True:
        raise DomainError("只读取二维主图像 FITS，图像扩展、数据立方体和表格不在本工作台输入范围")
    bitpix = header.get("BITPIX")
    if type(bitpix) is not int or bitpix not in FITS_TYPE:
        raise DomainError("FITS 只读取 8/16/32 位整数和 32/64 位浮点二维图")
    dims = Dimensions.checked(header.get("NAXIS1"), header.get("NAXIS2"))
    dtype = np.dtype(FITS_TYPE[bitpix])
    needed = dims.pixels * dtype.itemsize
    data_end = offset + needed
    padded_end = offset + int(math.ceil(needed / FITS_RECORD)) * FITS_RECORD
    if len(blob) < padded_end:
        raise DomainError("FITS 图像数据或记录填充不完整")
    encoded = np.frombuffer(blob[offset:data_end], dtype=dtype).reshape(dims.shape)
    invalid = ~np.isfinite(encoded)
    if bitpix > 0 and "BLANK" in header:
        blank = header_number(header, "BLANK", None)
        if type(blank) is not int:
            raise DomainError("FITS BLANK 必须为整数")
        invalid |= encoded == blank
    scale = header_number(header, "BSCALE", 1.0)
    zero = header_number(header, "BZERO", 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        pixels = encoded.astype(np.float64) * scale + zero
    pixels[invalid | ~np.isfinite(pixels) | (np.abs(pixels) > 1e15)] = np.nan
    if not np.any(np.isfinite(pixels)):
        raise DomainError("图像没有任何可用像素")
    return ExposureImage(pixels, fit_metadata(header), "FITS")


def read_raster(blob, suffix):
    try:
        with Image.open(BytesIO(blob), formats=["PNG", "TIFF"]) as image:
            if image.format != ("PNG" if suffix == ".png" else "TIFF"):
                raise DomainError("文件后缀与实际灰度图格式不一致")
            if getattr(image, "n_frames", 1) != 1:
                raise DomainError("请将多页图像分成单曝光再导入")
            Dimensions.checked(*image.size)
            if image.mode not in {"L", "I", "F", "I;16", "I;16B", "I;16L"}:
                raise DomainError("需要灰度探测器图；请先从仪器导出灰度图")
            pixels = np.asarray(image, dtype=np.float64).copy()
            pixels[~np.isfinite(pixels) | (np.abs(pixels) > 1e15)] = np.nan
            if not np.any(np.isfinite(pixels)):
                raise DomainError("图像没有任何可用像素")
            return ExposureImage(pixels, {"header": {}}, image.format)
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError, SyntaxError) as exc:
        raise DomainError("无法读取灰度图，文件可能不完整") from exc


def read_image(name, blob):
    if not 0 < len(blob) <= MAX_ASSET_BYTES:
        raise DomainError("文件为空或超过单张 10 MiB 上限")
    suffix = Path(name).suffix.lower()
    if suffix in FITS_SUFFIX:
        return read_fits(blob)
    if suffix in RASTER_SUFFIX:
        return read_raster(blob, suffix)
    raise DomainError("只支持 FITS、灰度 PNG 和单页 TIFF")


def make_card(key, value):
    if isinstance(value, bool):
        string = f"{'T' if value else 'F':>20}"
    elif isinstance(value, (int, float)):
        string = f"{value:>20}"
    else:
        string = "'" + str(value).encode("ascii", "replace").decode("ascii").replace("'", "''") + "'"
    return f"{key:<8}= {string}"[:FITS_CARD].ljust(FITS_CARD).encode("ascii")


def make_fits(pixels, headers=None, *, unsigned=False):
    height, width = pixels.shape
    Dimensions.checked(width, height)
    header = [
        ("SIMPLE", True),
        ("BITPIX", 16 if unsigned else -32),
        ("NAXIS", 2),
        ("NAXIS1", width),
        ("NAXIS2", height),
    ]
    if unsigned:
        header.extend([("BSCALE", 1), ("BZERO", 32768)])
    reserved = {key for key, _ in header} | {"END"}
    header.extend((key, val) for key, val in (headers or {}).items() if key not in reserved)
    encoded = b"".join(make_card(key, value) for key, value in header) + b"END".ljust(FITS_CARD)
    encoded = encoded.ljust(-(-len(encoded) // FITS_RECORD) * FITS_RECORD, b" ")
    if unsigned:
        samples = (np.clip(np.rint(pixels), 0, 65535) - 32768).astype(">i2")
    else:
        samples = np.asarray(pixels, dtype=">f4")
    return encoded + samples.tobytes().ljust(-(-samples.nbytes // FITS_RECORD) * FITS_RECORD, b"\0")


def image_summary(pixels):
    good = pixels[np.isfinite(pixels)]
    percent = np.percentile(good, [0, 1, 25, 50, 75, 99, 100])
    return {
        "valid_pixels": int(good.size),
        "invalid_pixels": int(pixels.size - good.size),
        "minimum": float(percent[0]),
        "p01": float(percent[1]),
        "p25": float(percent[2]),
        "median": float(percent[3]),
        "p75": float(percent[4]),
        "p99": float(percent[5]),
        "maximum": float(percent[6]),
    }
