"""Deterministic synthetic star fields for rehearsing the observing workflow."""
from datetime import datetime, timedelta, timezone
import math

import numpy as np

from .domain import new_document
from .raster import make_fits
from .store import Store, new_id

WIDTH = 368
HEIGHT = 296
EPOCH = datetime(2026, 9, 12, 20, 38, 0, tzinfo=timezone.utc)

SCENARIOS = {
    "variable": {
        "name": "N-021 · 疏散星场",
        "notes": "合成演示。14 张 V 波段科学曝光，暗化的是目标星，背景亮度随时间变动。",
        "seed": 4825,
        "count": 14,
        "band": "V",
        "sources": (
            ("V-17 目标星", "target", 161.3, 138.8, 965.0),
            ("C-04 比较星", "comparison", 236.8, 90.1, 780.0),
            ("C-09 比较星", "comparison", 77.6, 179.4, 565.0),
            ("K-02 检查星", "check", 271.7, 205.2, 540.0),
        ),
    },
    "tracking": {
        "name": "N-034 · 导星漂移",
        "notes": "合成演示。16 张 R 波段科学曝光，连续横向漂移，最后几帧增加拖尾和背景梯度。",
        "seed": 9402,
        "count": 16,
        "band": "R",
        "sources": (
            ("A-05 目标星", "target", 207.4, 121.4, 900.0),
            ("C-11 比较星", "comparison", 144.8, 159.3, 850.0),
            ("K-03 检查星", "check", 310.6, 222.2, 685.0),
        ),
    },
    "trails": {
        "name": "N-047 · 交差拖尾",
        "notes": "合成演示。6 张 V 波段科学曝光，前三张横向拖尾、后三张纵向拖尾，后段出现一颗新星。",
        "seed": 6108,
        "count": 6,
        "band": "V",
        "sources": (
            ("T-07 目标星", "target", 172.4, 131.6, 900.0),
            ("C-15 比较星", "comparison", 250.2, 96.8, 790.0),
            ("K-06 检查星", "check", 118.3, 216.9, 600.0),
        ),
    },
}


def timestamps(frame_index):
    return (EPOCH + timedelta(seconds=93 * frame_index)).isoformat(timespec="seconds").replace("+00:00", "Z")


def optical_fields():
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    xn = (xx - WIDTH / 2) / (WIDTH / 2)
    yn = (yy - HEIGHT / 2) / (HEIGHT / 2)
    flat = 1.11 - 0.19 * (xn**2 + yn**2) + 0.04 * np.cos(xx / 61)
    dust = np.exp(-((xx - 301)**2 / 13**2 + (yy - 63)**2 / 16**2) / 2)
    flat *= 1 - 0.27 * dust
    bias = 730 + 1.9 * np.sin(yy / 6.0) + 5 * xn
    dark = np.full_like(xx, 0.42) + 0.06 * (xn + 1)
    dark[33, 130] = 240
    dark[211, 18] = 315
    return yy, xx, flat, bias, dark


def insert_star(image, x, y, flux, sigma_x, sigma_y, angle):
    radius = int(math.ceil(5.5 * max(sigma_x, sigma_y)))
    height, width = image.shape
    x0, x1 = max(0, int(x) - radius), min(width, int(x) + radius + 2)
    y0, y1 = max(0, int(y) - radius), min(height, int(y) + radius + 2)
    if x1 <= x0 or y1 <= y0:
        return
    gx = np.arange(x0, x1)[None, :] - x
    gy = np.arange(y0, y1)[:, None] - y
    cosine, sine = math.cos(angle), math.sin(angle)
    u = (gx * cosine - gy * sine) / sigma_x
    v = (gx * sine + gy * cosine) / sigma_y
    normalized = np.exp(-0.5 * (u**2 + v**2)) / (2 * math.pi * sigma_x * sigma_y)
    image[y0:y1, x0:x1] += flux * normalized


def star_catalog(scenario, rng):
    named = list(scenario["sources"])
    stars = []
    while len(stars) < 222:
        x = float(rng.uniform(5, WIDTH - 5))
        y = float(rng.uniform(5, HEIGHT - 5))
        if any(math.hypot(x - source[2], y - source[3]) < 17 for source in named):
            continue
        strength = float(np.clip(rng.lognormal(3.8, 1.18), 7.0, 900.0))
        stars.append((x, y, strength))
    return stars


def synthetic_exposure(scenario, scenario_id, index, fields, catalog, rng):
    yy, xx, flat, bias, dark = fields
    if scenario_id == "tracking":
        dx = index * 0.83 + 0.6 * math.sin(index * 1.15)
        dy = index * -0.47 + math.sin(index / 2.2) * 0.55
    elif scenario_id == "trails":
        dx = 0.9 * math.sin(index * 1.31) + 0.3 * index
        dy = 0.8 * math.cos(index * 1.07) - 0.2 * index
    else:
        dx = 2.4 * math.sin(index * 0.67) + index * 0.14
        dy = 2.1 * math.sin(index * 0.41) - index * 0.09
    exposure = 48.0 if scenario_id == "trails" else (42.0 if index % 3 == 0 else 48.0)
    transparency = 0.91 + 0.1 * math.cos(index * 0.43) - 0.18 * math.exp(-((index - 9.5) / 2.3)**2)
    if scenario_id == "trails":
        transparency = 0.96 + 0.02 * math.sin(index * 0.9)
    seeing = 1.25 + 0.13 * math.sin(index * 0.9)
    drifting = scenario_id == "tracking" and index > 10
    if scenario_id == "trails":
        minor = 1.1 + 0.05 * math.sin(index * 0.8)
        major = minor * 2.8
        angle = 0.0 if index < 3 else math.pi / 2
    else:
        minor, major = seeing, seeing * (1.85 if drifting else 1.04)
        angle = 0.45 if drifting else 0.1
    sky = 11 + index * 0.14 + 2 * (xx / WIDTH)
    sky += (2.5 if drifting else 0.7) * np.cos((yy + 0.34 * xx) / 63 + index * 0.08)
    signal = sky * exposure
    for x, y, strength in catalog:
        insert_star(signal, x + dx, y + dy, strength * transparency * exposure, major, minor, angle)
    for label, role, x, y, strength in scenario["sources"]:
        eclipse = role == "target" and scenario_id == "variable"
        modulation = 1 - 0.24 * math.exp(-((index - 7.1) / 2.15)**4) if eclipse else 1
        insert_star(
            signal, x + dx, y + dy,
            strength * transparency * exposure * modulation,
            major, minor, angle,
        )
    if scenario_id == "trails" and index >= 3:
        insert_star(signal, 214.6 + dx, 158.4 + dy, 320 * transparency * exposure, major, minor, angle)
    detector = bias + dark * exposure + flat * signal
    detector += rng.normal(0, np.sqrt(np.maximum(1, flat * signal)) * 0.41 + 2.5)
    for _ in range(4):
        y, x = rng.integers(3, HEIGHT - 3), rng.integers(3, WIDTH - 3)
        detector[y, x] += rng.uniform(450, 7400)
    return detector, exposure, dx, dy


def science_header(scenario, index, exposure, dx, dy):
    return {
        "OBJECT": scenario["name"].split(" ", 1)[0],
        "DATE-OBS": timestamps(index),
        "TIMESYS": "UTC",
        "EXPTIME": exposure,
        "FILTER": scenario["band"],
        "INSTRUME": "Synthetic CCD",
        "BUNIT": "ADU",
        "SATURATE": 65000,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRPIX1": WIDTH / 2 + 1 + dx,
        "CRPIX2": HEIGHT / 2 + 1 + dy,
        "CRVAL1": 0.021,
        "CRVAL2": 23.4,
        "CD1_1": -0.00042,
        "CD1_2": 0.00003,
        "CD2_1": 0.00003,
        "CD2_2": 0.00042,
        "RADESYS": "ICRS",
    }


def calibration_assets(scenario, fields, rng):
    _, _, flat, bias, dark = fields
    prepared = []
    shots = (
        ("bias", 0, "", bias, 2.1, 3),
        ("dark", 60, "", bias + 60 * dark, 3.5, 3),
        ("flat", 3, scenario["band"], bias + 3 * dark + 18500 * flat, 22, 3),
    )
    for kind, exposure, band, values, noise, count in shots:
        for index in range(count):
            data = values + rng.normal(0, noise, size=values.shape)
            headers = {"EXPTIME": exposure, "FILTER": band, "BUNIT": "ADU", "IMAGETYP": kind}
            filename = f"{kind}-{index + 1:02d}.fits"
            prepared.append(Store.prepare_asset(filename, make_fits(data, headers, unsigned=True), kind, {}))
    return prepared


def make_demo(scenario_id, store):
    scenario = SCENARIOS[scenario_id]
    rng = np.random.default_rng(scenario["seed"])
    fields = optical_fields()
    catalog = star_catalog(scenario, rng)
    prepared = []
    alignments = {}
    for index in range(scenario["count"]):
        pixels, exposure, dx, dy = synthetic_exposure(scenario, scenario_id, index, fields, catalog, rng)
        filename = f"field-{scenario['band']}-{index + 1:02d}.fits"
        body = make_fits(pixels, science_header(scenario, index, exposure, dx, dy), unsigned=True)
        model = Store.prepare_asset(filename, body, "science", {})
        prepared.append(model)
        if scenario_id in ("variable", "trails") and index:
            alignments[model[0]["id"]] = {"dx": dx, "dy": dy, "source": "manual"}
    prepared.extend(calibration_assets(scenario, fields, rng))
    doc = new_document(scenario["name"], synthetic=True)
    initial = {
        "notes": scenario["notes"],
        "apertures": [
            {"id": new_id(), "label": label, "role": role, "x": x, "y": y}
            for label, role, x, y, _ in scenario["sources"]
        ],
        "masks": [
            {"id": new_id(), "label": "热像素 1", "left": 129, "top": 32,
             "right": 132, "bottom": 35, "frame": None},
            {"id": new_id(), "label": "热像素 2", "left": 17, "top": 210,
             "right": 20, "bottom": 213, "frame": None},
        ],
        "alignments": alignments,
    }
    return store.create_populated(doc, prepared, initial)
