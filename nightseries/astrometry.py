"""Bidirectional TAN sky coordinates and a small-field plate fit."""
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from .domain import DomainError, Dimensions, integer, list_value, number, object_value, text

SKY_KEYS = (
    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2", "PC1_1", "PC1_2", "PC2_1", "PC2_2",
    "CDELT1", "CDELT2", "RADESYS", "EQUINOX", "LONPOLE", "LATPOLE",
)
MAX_ANCHORS = 24
MAX_CATALOG = 1000


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @classmethod
    def pixel(cls, data, dimensions):
        x, y = dimensions.xy(data.get("x"), data.get("y"))
        return cls(x, y)

    @classmethod
    def celestial(cls, data):
        return cls(
            number(data.get("ra"), "赤经角度", -360, 720) % 360,
            number(data.get("dec"), "赤纬角度", -90, 90),
        )


@dataclass(frozen=True)
class Projection:
    pixel_origin: Point
    center: Point
    matrix: np.ndarray
    origin: str
    system: str = "ICRS"

    @staticmethod
    def check_matrix(matrix):
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise DomainError("图幅坐标变换缺少有限系数")
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[-1] < 1e-8 or singular[0] > 5 or singular[0] / singular[-1] > 30:
            raise DomainError("图幅坐标变换的尺度或条件数无效")
        return matrix

    def pixel_to_sky(self, point):
        dx = np.array([point.x - self.pixel_origin.x, point.y - self.pixel_origin.y])
        projected = self.matrix @ dx
        return tangent_to_sky(self.center, Point(*projected))

    def sky_to_pixel(self, sky):
        projected = sky_to_tangent(self.center, sky)
        offsets = np.linalg.solve(self.matrix, np.array([projected.x, projected.y]))
        return Point(self.pixel_origin.x + float(offsets[0]), self.pixel_origin.y + float(offsets[1]))

    def summary(self):
        singular = np.linalg.svd(self.matrix, compute_uv=False)
        return {
            "system": self.system,
            "origin": self.origin,
            "origin_x": self.pixel_origin.x,
            "origin_y": self.pixel_origin.y,
            "center_ra": self.center.x,
            "center_dec": self.center.y,
            "matrix": self.matrix.tolist(),
            "major_scale_arcsec": float(singular[0] * 3600),
            "minor_scale_arcsec": float(singular[-1] * 3600),
            "x_bearing": math.degrees(math.atan2(self.matrix[0, 0], self.matrix[1, 0])) % 360,
            "mirror": bool(np.linalg.det(self.matrix) < 0),
        }


def sky_to_tangent(center, sky):
    offset = math.radians((sky.x - center.x + 180) % 360 - 180)
    d, d0 = math.radians(sky.y), math.radians(center.y)
    cosd, sind = math.cos(d), math.sin(d)
    cos0, sin0 = math.cos(d0), math.sin(d0)
    dot = sind * sin0 + cosd * cos0 * math.cos(offset)
    if dot <= 1e-6:
        raise DomainError("此天点位于参考切平面的背面")
    xi = cosd * math.sin(offset) / dot
    eta = (cos0 * sind - sin0 * cosd * math.cos(offset)) / dot
    return Point(math.degrees(xi), math.degrees(eta))


def tangent_to_sky(center, tangent):
    x, y = math.radians(tangent.x), math.radians(tangent.y)
    dec0, ra0 = math.radians(center.y), math.radians(center.x)
    top = math.sin(dec0) + y * math.cos(dec0)
    aside = math.cos(dec0) - y * math.sin(dec0)
    dec = math.atan2(top, math.hypot(aside, x))
    ra = ra0 + math.atan2(x, aside)
    return Point(math.degrees(ra) % 360, math.degrees(dec))


def angular_distance(a, b):
    x, y = math.radians((a.x - b.x + 180) % 360 - 180), math.radians(a.y - b.y)
    term = math.sin(y * 0.5)**2 + math.cos(math.radians(a.y)) * math.cos(math.radians(b.y)) * math.sin(x * 0.5)**2
    return math.degrees(2 * math.asin(math.sqrt(min(1.0, max(0.0, term)))))


def _matrix_from_header(header):
    entries = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    if any(key in header for key in entries):
        values = [number(header.get(key), key, -5, 5) for key in entries]
        return np.asarray(values, dtype=float).reshape((2, 2))
    pc = np.array([
        [number(header.get("PC1_1", 1), "PC1_1", -30, 30), number(header.get("PC1_2", 0), "PC1_2", -30, 30)],
        [number(header.get("PC2_1", 0), "PC2_1", -30, 30), number(header.get("PC2_2", 1), "PC2_2", -30, 30)],
    ])
    scale = np.asarray([
        number(header.get("CDELT1"), "CDELT1", -5, 5),
        number(header.get("CDELT2"), "CDELT2", -5, 5),
    ])
    return pc * scale[:, None]


def from_header(header):
    if not header or ("CTYPE1" not in header and "CTYPE2" not in header):
        return None
    ctype = (header.get("CTYPE1"), header.get("CTYPE2"))
    if ctype != ("RA---TAN", "DEC--TAN"):
        raise DomainError("当前仅识别赤经／赤纬轴顺序的无畸变 TAN FITS 星图")
    if any(str(header.get(f"CUNIT{axis}", "deg")).lower() != "deg" for axis in [1, 2]):
        raise DomainError("FITS TAN 坐标须以度为单位")
    if any(key in header for key in ("CROTA1", "CROTA2", "DISTORTED")) or header.get("LONPOLE", 180) != 180:
        raise DomainError("此 FITS 变换含工作台未支持的旋转约定")
    system = str(header.get("RADESYS", "ICRS")).upper()
    if system not in {"ICRS", "FK5", "FK4"}:
        raise DomainError("FITS 头使用未知赤道参考系")
    pixel = Point(
        number(header.get("CRPIX1"), "CRPIX1", -1e6, 1e6) - 1,
        number(header.get("CRPIX2"), "CRPIX2", -1e6, 1e6) - 1,
    )
    center = Point.celestial({"ra": header.get("CRVAL1"), "dec": header.get("CRVAL2")})
    if abs(center.y) >= 89.9999:
        raise DomainError("参考切点靠近天极，当前工作台不使用此 FITS 坐标")
    return Projection(pixel, center, Projection.check_matrix(_matrix_from_header(header)), "FITS", system)


def reference_header(document):
    reference = document["reference"]
    selected = next((row for row in document["frames"] if row["id"] == reference), None)
    return selected.get("header", {}) if selected is not None else {}


def reference_projection(document):
    saved = document.get("plate")
    if saved and saved.get("reference") == document["reference"]:
        solution = saved["solution"]
        return Projection(
            Point(solution["origin_x"], solution["origin_y"]),
            Point(solution["center_ra"], solution["center_dec"]),
            Projection.check_matrix(np.asarray(solution["matrix"], dtype=float)),
            "anchors",
        )
    return from_header(reference_header(document))


def anchored_projection(center, coeff):
    matrix = Projection.check_matrix(coeff[:2].T)
    intercept = coeff[2]
    shift = np.linalg.solve(matrix, -intercept)
    return Projection(Point(float(shift[0]), float(shift[1])), center, matrix, "anchors")


def spherical_center(sky):
    angles = np.radians(np.asarray([(item.x, item.y) for item in sky]))
    x = np.cos(angles[:, 1]) * np.cos(angles[:, 0])
    y = np.cos(angles[:, 1]) * np.sin(angles[:, 0])
    z = np.sin(angles[:, 1])
    mx, my, mz = float(np.mean(x)), float(np.mean(y)), float(np.mean(z))
    if math.sqrt(mx**2 + my**2 + mz**2) < 0.8:
        raise DomainError("参考星未落在足够小的同一片天区")
    ra = math.degrees(math.atan2(my, mx)) % 360
    dec = math.degrees(math.atan2(mz, math.hypot(mx, my)))
    return Point(ra, dec)


def anchor_rows(values, dimensions):
    rows = list_value(values, "定标星", MAX_ANCHORS)
    if len(rows) < 3:
        raise DomainError("至少需要三个非共线的定标星")
    result = []
    for entry in rows:
        row = object_value(entry, "定标星")
        xy = Point.pixel(row, dimensions)
        sky = Point.celestial(row)
        precision = number(row.get("weight", 1), "定标星权重", 0.01, 100)
        result.append({
            "label": text(row.get("label"), "定标星标签", 48),
            "x": xy.x, "y": xy.y, "ra": sky.x, "dec": sky.y,
            "weight": precision,
        })
    if len({row["label"] for row in result}) != len(result):
        raise DomainError("同一拟合中定标星标签不能重复")
    return result


def design_and_sky(rows, center):
    positions = np.asarray([(row["x"], row["y"]) for row in rows])
    tangents = [sky_to_tangent(center, Point(row["ra"], row["dec"])) for row in rows]
    cloud = np.asarray([(item.x, item.y) for item in tangents])
    if np.max(np.linalg.norm(cloud, axis=1)) > 8:
        raise DomainError("人工定标只用于中心 8 度内的窄视场")
    return positions, cloud


def affine_coefficients(positions, tangents, weights):
    mean = np.average(positions, weights=weights, axis=0)
    shifted = positions - mean
    radius = max(1, float(np.sqrt(np.mean(shifted**2))))
    design = np.column_stack((shifted / radius, np.ones(len(weights))))
    weighted = np.sqrt(weights)[:, None]
    result, _, rank, singular = np.linalg.lstsq(design * weighted, tangents * weighted, rcond=None)
    if rank < 3 or singular[-1] / singular[0] < 0.027:
        raise DomainError("参考星的图像坐标过近或接近共线")
    result[:2] /= radius
    result[2] -= mean @ result[:2]
    return result


def nonlinear_residuals(center, coeff, rows):
    projection = anchored_projection(center, coeff)
    return np.asarray([
        angular_distance(
            projection.pixel_to_sky(Point(row["x"], row["y"])),
            Point(row["ra"], row["dec"]),
        ) * 3600 for row in rows
    ])


def best_fit_mask(center, positions, tangents, rows, weights, tolerance):
    size = len(rows)
    last_score = None
    best = None
    max_seeds = 600
    trials = np.array(list(combinations(range(size), 3)), dtype=int)
    selection = np.unique(np.rint(np.linspace(0, len(trials) - 1, min(len(trials), max_seeds))).astype(int))
    for index in selection:
        chosen = trials[index]
        try:
            coeff = affine_coefficients(positions[chosen], tangents[chosen], weights[chosen])
            error = nonlinear_residuals(center, coeff, rows)
        except (DomainError, np.linalg.LinAlgError):
            continue
        receive = error <= tolerance
        if np.count_nonzero(receive) < max(4, (size + 1) // 2):
            continue
        score = (float(np.sum(weights[receive])), -float(np.average(error[receive], weights=weights[receive])))
        if last_score is None or score > last_score:
            best = receive
            last_score = score
    if best is None:
        raise DomainError("没有足够定标星支持同一个容差内的图幅解")
    return best


def fit_sky_solution(rows, options, dimensions):
    valid = anchor_rows(rows, dimensions)
    sky = [Point(row["ra"], row["dec"]) for row in valid]
    center = spherical_center(sky)
    mode = options.get("mode", "all")
    if mode not in {"all", "robust"}:
        raise DomainError("未知定标方法")
    tolerance = number(options.get("tolerance", 6), "角距离容差（角秒）", 0.1, 120)
    positions, tangents = design_and_sky(valid, center)
    weights = np.asarray([row["weight"] for row in valid])
    if mode == "robust":
        if len(valid) < 5:
            raise DomainError("启用异常点筛选至少需 5 星")
        kept = best_fit_mask(center, positions, tangents, valid, weights, tolerance)
    else:
        kept = np.ones(len(valid), dtype=bool)
    coeff = affine_coefficients(positions[kept], tangents[kept], weights[kept])
    scale = anchored_projection(center, coeff)
    residual = nonlinear_residuals(center, coeff, valid)
    used = residual[kept]
    if mode == "robust" and float(np.max(used)) > tolerance:
        raise DomainError("采用保留点重算后图幅仍超出定标容差")
    return {
        "solution": scale.summary(),
        "mode": mode,
        "tolerance": tolerance,
        "anchors": [
            {**row, "included": bool(keep), "residual_arcsec": float(error)}
            for row, keep, error in zip(valid, kept, residual)
        ],
        "rms_arcsec": float(np.sqrt(np.mean(used**2))),
        "worst_arcsec": float(np.max(used)),
    }


def sky_from_pixel(projection, x, y, dimensions):
    xy = Point.pixel({"x": x, "y": y}, dimensions)
    sky = projection.pixel_to_sky(xy)
    return {"x": xy.x, "y": xy.y, "ra": sky.x, "dec": sky.y, "system": projection.system}


def pixel_from_sky(projection, row, dimensions):
    sky = Point.celestial(row)
    try:
        point = projection.sky_to_pixel(sky)
    except DomainError:
        return {**row, "x": None, "y": None, "state": "backside"}
    inside = (
        -0.5 <= point.x < dimensions.width - 0.5
        and -0.5 <= point.y < dimensions.height - 0.5
    )
    return {**row, "x": point.x, "y": point.y, "state": "inside" if inside else "outside"}


def describe_field(projection, dims):
    coords = [
        Point(0, 0), Point(dims.width - 1, 0),
        Point(dims.width - 1, dims.height - 1), Point(0, dims.height - 1),
    ]
    corners = [
        {"x": point.x, "y": point.y, "ra": sky.x, "dec": sky.y}
        for point in coords for sky in [projection.pixel_to_sky(point)]
    ]
    center = projection.pixel_to_sky(Point((dims.width - 1) / 2, (dims.height - 1) / 2))
    return {
        "solution": projection.summary(),
        "corners": corners,
        "center": {"ra": center.x, "dec": center.y},
        "width_arcmin": angular_distance(Point(corners[0]["ra"], corners[0]["dec"]), Point(corners[1]["ra"], corners[1]["dec"])) * 60,
        "height_arcmin": angular_distance(Point(corners[0]["ra"], corners[0]["dec"]), Point(corners[3]["ra"], corners[3]["dec"])) * 60,
    }


def catalog_rows(values):
    offered = list_value(values, "本地星表", MAX_CATALOG)
    safe = []
    names = set()
    for entry in offered:
        row = object_value(entry, "星表条目")
        label = text(row.get("label"), "星表名称", 55)
        if label in names:
            raise DomainError(f"星表名称重复：{label}")
        names.add(label)
        point = Point.celestial(row)
        candidate = {"label": label, "ra": point.x, "dec": point.y}
        value = row.get("magnitude")
        candidate["magnitude"] = number(value, "星等", -40, 40) if value not in ("", None) else None
        safe.append(candidate)
    return safe
