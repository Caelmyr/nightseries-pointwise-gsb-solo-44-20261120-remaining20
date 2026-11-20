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
    "nebula": {
        "name": "N-047 · 三色星云",
        "notes": "合成演示。B、V、R 三组科学曝光各 4 张、曝光时长不同；B 组画幅向右下偏移，"
                 "右缘与底边缺少 B 覆盖；亮星在 R 组饱和；星云偏红、反射区偏蓝，另有一块接近噪声的淡斑。"
                 "用于彩色合成演练。",
        "seed": 6713,
        "count": 12,
        "band": "",
        "bands": (
            {"band": "V", "role": "green", "count": 4, "exposure": 50.0, "shift": (0, 0)},
            {"band": "B", "role": "blue", "count": 4, "exposure": 75.0, "shift": (14, 11)},
            {"band": "R", "role": "red", "count": 4, "exposure": 32.0, "shift": (-4, 3)},
        ),
        "sources": (
            ("V-21 亮星", "target", 249.6, 96.4, 620.0),
            ("C-07 比较星", "comparison", 118.3, 208.7, 710.0),
            ("K-12 检查星", "check", 302.5, 158.2, 610.0),
        ),
        "bright": {
            "x": 68.0, "y": 268.0, "sigma": 1.35,
            "flux": {"B": 5200.0, "V": 9400.0, "R": 34000.0},
        },
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
    else:
        dx = 2.4 * math.sin(index * 0.67) + index * 0.14
        dy = 2.1 * math.sin(index * 0.41) - index * 0.09
    exposure = 42.0 if index % 3 == 0 else 48.0
    transparency = 0.91 + 0.1 * math.cos(index * 0.43) - 0.18 * math.exp(-((index - 9.5) / 2.3)**2)
    seeing = 1.25 + 0.13 * math.sin(index * 0.9)
    drifting = scenario_id == "tracking" and index > 10
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
    flat_bands = [entry["band"] for entry in scenario.get("bands", ())] or [scenario["band"]]
    shots = (
        [("bias", 0, "", bias, 2.1, 3)]
        + [("dark", 60, "", bias + 60 * dark, 3.5, 3)]
        + [("flat", 3, band, bias + 3 * dark + 18500 * flat, 22, 1) for band in flat_bands]
    )
    for kind, exposure, band, values, noise, count in shots:
        for index in range(count):
            data = values + rng.normal(0, noise, size=values.shape)
            headers = {"EXPTIME": exposure, "FILTER": band, "BUNIT": "ADU", "IMAGETYP": kind}
            filename = f"{kind}-{band + '-' if band else ''}{index + 1:02d}.fits"
            prepared.append(Store.prepare_asset(filename, make_fits(data, headers, unsigned=True), kind, {}))
    return prepared


def nebula_rate(dx=0.0, dy=0.0):
    """Extended emission pinned to reference coordinates, sampled at native (x+dx, y+dy).

    Deliberately gives the field a clear red/blue structure: an H-alpha-like core, a
    blue reflection lobe, a green rim, a dust lane, and one faint patch that stays
    under the noise in two filters so the composite must fall back to grey there.
    """
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    xx = xx - dx
    yy = yy - dy
    core = np.exp(-(((xx - 150) ** 2) / (2 * 22.0**2) + ((yy - 238) ** 2) / (2 * 20.0**2)))
    halo = np.exp(-(((xx - 140) ** 2) / (2 * 62.0**2) + ((yy - 244) ** 2) / (2 * 54.0**2)))
    reflect = np.exp(-(((xx - 196) ** 2) / (2 * 27.0**2) + ((yy - 196) ** 2) / (2 * 19.0**2)))
    rim = np.exp(-(((xx - 266) ** 2) / (2 * 17.0**2) + ((yy - 104) ** 2) / (2 * 15.0**2)))
    faint = np.exp(-(((xx - 66) ** 2) / (2 * 21.0**2) + ((yy - 70) ** 2) / (2 * 18.0**2)))
    knot = np.exp(-(((xx - 244) ** 2) / (2 * 2.3**2) + ((yy - 272) ** 2) / (2 * 2.3**2)))
    lane = np.exp(-(((xx - 172) ** 2) / (2 * 9.0**2) + ((yy - 224) ** 2) / (2 * 34.0**2)))
    clear = 1 - 0.78 * lane
    blue = 1.1 * core * clear + 1.6 * halo + 11.0 * reflect + 1.3 * rim + 0.05 * faint + 16.0 * knot
    green = 3.4 * core * clear + 1.4 * halo + 2.2 * reflect + 4.4 * rim + 0.09 * faint + 7.5 * knot
    red = 11.5 * core * clear + 2.8 * halo + 0.7 * reflect + 1.1 * rim + 0.16 * faint + 0.9 * knot
    return {"B": np.maximum(blue, 0), "V": np.maximum(green, 0), "R": np.maximum(red, 0)}


def star_tints(count, rng):
    palette = ((1.45, 1.0, 0.62), (1.0, 1.0, 1.0), (0.78, 0.95, 1.16), (0.52, 0.86, 1.42))
    weights = (0.22, 0.46, 0.22, 0.10)
    picks = rng.choice(len(palette), size=count, p=weights)
    return [palette[index] for index in picks]


def spectral_exposure(scenario, spec, index, fields, catalog, tints, rng, drift):
    yy, xx, flat, bias, dark = fields
    band = spec["band"]
    band_index = {"B": 0, "V": 1, "R": 2}[band]
    exposure = spec["exposure"]
    dx = drift["dx"] + spec["shift"][0]
    dy = drift["dy"] + spec["shift"][1]
    # Each filter sees its own sky brightness and gradient, as a real run does.
    sky_level = {"B": 13.6, "V": 11.2, "R": 9.4}[band]
    sky_slope = {"B": 2.3, "V": 1.5, "R": 1.1}[band]
    sky_phase = {"B": 0.35, "V": 0.62, "R": 0.9}[band]
    sky = sky_level + 0.09 * index + sky_slope * (xx / WIDTH)
    sky += 0.6 * np.cos((yy + sky_phase * xx) / 71 + index * 0.1)
    sky += 0.4 * np.sin((yy - xx) / 97 + sky_phase)
    signal = (sky + nebula_rate(dx, dy)[band]) * exposure
    for (x, y, strength), tint in zip(catalog, tints):
        insert_star(signal, x + dx, y + dy, strength * tint[band_index] * exposure, 1.22, 1.30, 0.1)
    for label, role, x, y, strength in scenario["sources"]:
        tint = (0.95, 0.98, 1.02)[band_index]
        insert_star(signal, x + dx, y + dy, strength * tint * exposure, 1.2, 1.28, 0.12)
    bright = scenario["bright"]
    insert_star(
        signal, bright["x"] + dx, bright["y"] + dy,
        bright["flux"][band] * exposure, bright["sigma"], bright["sigma"] * 1.04, 0.08,
    )
    detector = bias + dark * exposure + flat * signal
    detector += rng.normal(0, np.sqrt(np.maximum(1, flat * signal)) * 0.41 + 2.5)
    for _ in range(4):
        row, column = rng.integers(3, HEIGHT - 3), rng.integers(3, WIDTH - 3)
        detector[row, column] += rng.uniform(450, 7400)
    return detector, exposure, dx, dy


def spectral_header(scenario, spec, index, exposure, dx, dy):
    return science_header({**scenario, "band": spec["band"]}, index, exposure, dx, dy)


def make_demo(scenario_id, store):
    scenario = SCENARIOS[scenario_id]
    rng = np.random.default_rng(scenario["seed"])
    fields = optical_fields()
    catalog = star_catalog(scenario, rng)
    prepared = []
    alignments = {}
    if "bands" in scenario:
        tints = star_tints(len(catalog), rng)
        drift = {"dx": 0.0, "dy": 0.0}
        for spec in scenario["bands"]:
            for index in range(spec["count"]):
                pixels, exposure, dx, dy = spectral_exposure(
                    scenario, spec, index, fields, catalog, tints, rng, drift,
                )
                filename = f"field-{spec['band']}-{index + 1:02d}.fits"
                header = spectral_header(scenario, spec, len(prepared), exposure, dx, dy)
                model = Store.prepare_asset(filename, make_fits(pixels, header, unsigned=True), "science", {})
                prepared.append(model)
                alignments[model[0]["id"]] = {"dx": dx, "dy": dy, "source": "manual"}
            drift = {"dx": drift["dx"] + 0.55, "dy": drift["dy"] - 0.35}
    else:
        for index in range(scenario["count"]):
            pixels, exposure, dx, dy = synthetic_exposure(scenario, scenario_id, index, fields, catalog, rng)
            filename = f"field-{scenario['band']}-{index + 1:02d}.fits"
            body = make_fits(pixels, science_header(scenario, index, exposure, dx, dy), unsigned=True)
            model = Store.prepare_asset(filename, body, "science", {})
            prepared.append(model)
            if scenario_id == "variable" and index:
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
