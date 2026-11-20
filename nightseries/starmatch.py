"""Automatic pattern matching between reference-frame stars and a local sky catalog.

The reference exposure may carry no celestial coordinates.  The operator supplies a
rough guess (tangent point, plate scale, orientation, possible mirroring) and this
module pairs detected image stars with catalog entries using mirror-invariant
triangle invariants.  Every distinct geometric solution is kept so the caller can
flag corner-only support or two equally plausible solutions instead of silently
adopting one of them.
"""
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from .astrometry import (
    Point, Projection, affine_coefficients, anchored_projection, angular_distance,
    sky_to_tangent, spherical_center,
)
from .domain import DomainError, dimensions_of, integer, number
from .preview import png, render_plane
from .quality import analyze_stars

MAX_MATCH_STARS = 120
MATCH_STAR_LIMIT = 30
MIN_PAIRS = 4
# A second solution is treated as a distinct rival when it both keeps a large
# fraction of the best support and disagrees on most of its pair identities.
RIVAL_SUPPORT_RATIO = 0.75
RIVAL_DISAGREEMENT = 0.35
CORNER_SPREAD_RATIO = 0.55


@dataclass(frozen=True)
class MatchPrior:
    center: Point
    scale_arcsec: float
    angle_deg: float
    mirror: bool
    uncertainty_arcsec: float
    match_tolerance_arcsec: float

    @classmethod
    def from_options(cls, options):
        ra = number(options.get("center_ra"), "中心赤经（度）", -360, 720) % 360
        dec = number(options.get("center_dec"), "中心赤纬（度）", -89.999, 89.999)
        return cls(
            Point(ra, dec),
            number(options.get("scale_arcsec"), "每像素角度（角秒）", 0.01, 3600),
            number(options.get("angle_deg", 0), "相机朝向（度）", -360, 360),
            bool(options.get("mirror", False)),
            number(options.get("uncertainty_arcsec", 120), "先验位置不确定度（角秒）", 1, 200000),
            number(options.get("tolerance_arcsec", 4), "配对残差容差（角秒）", 0.1, 120),
        )

    def matrix(self):
        angle = math.radians(self.angle_deg)
        scale = self.scale_arcsec / 3600.0
        rotation = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        matrix = rotation * scale
        if self.mirror:
            matrix[0, :] *= -1
        return Projection.check_matrix(matrix)

    def projection(self, dims):
        origin = Point((dims.width - 1) / 2, (dims.height - 1) / 2)
        return Projection(origin, self.center, self.matrix(), "guess")


def detect_reference_stars(plane, options):
    cap = integer(options.get("star_cap", 80), "检星上限", 8, MAX_MATCH_STARS)
    settings = {
        "threshold": options.get("threshold", 6),
        "stamp_radius": options.get("stamp_radius", 5),
        "cap": cap,
        "separation": options.get("separation", 12),
    }
    report = analyze_stars(plane, settings)
    stars = [star for star in report["stars"] if star["state"] == "measured"]
    if len(stars) < 5:
        raise DomainError("参考曝光可量测星点不足 5 颗，无法做几何配对")
    return stars, report


def _triangle_table(points, min_edge, max_edge):
    """Mirror-invariant triangle hashes keyed by sorted ratios of edge lengths.

    Callers pass a small bright-star subset so every triangle can be retained.
    """
    table = {}
    count = len(points)
    for vertices in combinations(range(count), 3):
        p = points[vertices, :]
        edges = np.array([
            np.linalg.norm(p[1] - p[2]),
            np.linalg.norm(p[0] - p[2]),
            np.linalg.norm(p[0] - p[1]),
        ])
        if edges.min() < min_edge or edges.max() > max_edge:
            continue
        order = np.argsort(edges)
        ordered = tuple(vertices[index] for index in order)
        shortest, middle, longest = edges[order]
        key = (round(middle / shortest, 2), round(longest / shortest, 2))
        table.setdefault(key, []).append((ordered, shortest, middle, longest))
    return table


def _triangle_hypotheses(image_xy, catalog_xy):
    image_table = _triangle_table(image_xy, min_edge=10, max_edge=1e9)
    catalog_table = _triangle_table(catalog_xy, min_edge=0.0, max_edge=1e9)
    hypotheses = []
    for key, image_rows in image_table.items():
        for d1 in (-1, 0, 1):
            for d2 in (-1, 0, 1):
                catalog_rows = catalog_table.get((key[0] + d1 * 0.01, key[1] + d2 * 0.01))
                if not catalog_rows:
                    continue
                for (i0, i1, i2), e0a, e1a, e2a in image_rows:
                    for (c0, c1, c2), e0b, e1b, e2b in catalog_rows:
                        scale = e0b / e0a
                        if not 0.5 < scale < 2.5:
                            continue
                        if abs(e1b / e1a - scale) > 0.10 * scale:
                            continue
                        if abs(e2b / e2a - scale) > 0.10 * scale:
                            continue
                        # Direct labeling and its mirror (swap the two wider-angle vertices).
                        hypotheses.append(((i0, c0), (i1, c1), (i2, c2)))
                        hypotheses.append(((i0, c0), (i1, c2), (i2, c1)))
    return hypotheses


def _fit_affine(image_xy, catalog_xy, pairs):
    image_indices = [pair[0] for pair in pairs]
    catalog_indices = [pair[1] for pair in pairs]
    source = image_xy[image_indices]
    target = catalog_xy[catalog_indices]
    design = np.column_stack((source, np.ones(len(pairs))))
    if len(pairs) == 3:
        # Exact fit for a triangle seed; cheaper and more stable than lstsq.
        coefficients = np.linalg.solve(design, target)
    else:
        coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def _predict(image_xy, coefficients):
    return np.column_stack((image_xy, np.ones(len(image_xy)))) @ coefficients


def _inlier_pairs(image_xy, catalog_xy, coefficients, tolerance_px):
    """Vectorized nearest-catalog assignment restricted to the match tolerance."""
    predicted = _predict(image_xy, coefficients)
    diff_x = predicted[:, None, 0] - catalog_xy[None, :, 0]
    diff_y = predicted[:, None, 1] - catalog_xy[None, :, 1]
    distances = np.hypot(diff_x, diff_y)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(image_xy)), nearest]
    candidates = np.where(nearest_distance <= tolerance_px)[0]
    # Enforce one-to-one assignment: keep the closest image star for each catalog star.
    unique_pairs = {}
    for image_index in candidates:
        catalog_index = int(nearest[image_index])
        distance = float(nearest_distance[image_index])
        incumbent = unique_pairs.get(catalog_index)
        if incumbent is None or distance < incumbent[1]:
            unique_pairs[catalog_index] = (int(image_index), distance)
    pairs = [(image_index, catalog_index)
             for catalog_index, (image_index, _) in unique_pairs.items()]
    return pairs, predicted


def _refit_loop(image_xy, catalog_xy, seed_pairs, tolerance_px, quick_xy=None, quick_pairs=None):
    """Grow a triangle seed into a maximal self-consistent pair set."""
    coefficients = _fit_affine(image_xy, catalog_xy, seed_pairs)
    previous = frozenset()
    for _ in range(6):
        pairs, _ = _inlier_pairs(image_xy, catalog_xy, coefficients, tolerance_px)
        current = frozenset(pairs)
        if current == previous or len(pairs) < 3:
            break
        previous = current
        coefficients = _fit_affine(image_xy, catalog_xy, pairs)
    return coefficients, pairs


def _support_spread(image_xy, pairs, dims):
    points = image_xy[[pair[0] for pair in pairs]]
    span = math.hypot(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    diagonal = math.hypot(dims.width, dims.height)
    return span / diagonal


def _pair_overlap(best_pairs, rival_pairs):
    best = set(best_pairs)
    rival = set(rival_pairs)
    if not best or not rival:
        return 0.0
    return len(best & rival) / len(best)


def match_stars(image_xy, catalog_xy, prior, dims, image_subset, catalog_subset):
    """Grow bright-star triangle seeds into distinct solutions on the full sets.

    Triangle invariants are built for small bright subsets only, which keeps the
    hash tables bounded.  Every distinct seed transform is ranked inside those
    subsets; a first wave expands the promising ones across all detected stars.
    When no decisive fit appears, a fallback scans the remaining seeds so a true
    match hidden by brightness/magnitude ranking differences is still recovered.
    """
    seed_image = image_xy[image_subset]
    seed_catalog = catalog_xy[catalog_subset]
    hypotheses = _triangle_hypotheses(seed_image, seed_catalog)
    if not hypotheses:
        raise DomainError("没有三角不变量能把图中星点与本地星表联系起来，请放宽先验或换用更亮的星")
    tolerance_px = prior.match_tolerance_arcsec / prior.scale_arcsec

    # Deduplicate and score each seed transform within the bright subsets.  Flux
    # and magnitude rankings need not agree, so this score only orders the work.
    seed_keys = set()
    ranked = []
    for hypothesis in hypotheses:
        seed_coefficients = _fit_affine(seed_image, seed_catalog, hypothesis)
        key = tuple(np.round(seed_coefficients, 4).ravel())
        if key in seed_keys:
            continue
        seed_keys.add(key)
        quick_pairs, _ = _inlier_pairs(seed_image, seed_catalog, seed_coefficients, tolerance_px)
        ranked.append((len(quick_pairs), hypothesis))
    if not ranked:
        raise DomainError(
            f"三角配对后没有任何解在 {prior.match_tolerance_arcsec:g} 角秒内容纳至少 {MIN_PAIRS} 对星"
        )
    ranked.sort(key=lambda item: -item[0])

    def map_seed_pairs(seed_pairs):
        return [(image_subset[i], catalog_subset[c]) for i, c in seed_pairs]

    def expand(hypothesis):
        mapped_seed = map_seed_pairs(hypothesis)
        return _refit_loop(image_xy, catalog_xy, mapped_seed, tolerance_px)

    signatures = set()
    solutions = []

    def absorb(coefficients, pairs):
        if len(pairs) < MIN_PAIRS:
            return False
        signature = frozenset(pairs)
        if signature in signatures:
            return False
        signatures.add(signature)
        predicted = _predict(image_xy, coefficients)
        residuals = [
            float(np.hypot(predicted[i, 0] - catalog_xy[c, 0],
                           predicted[i, 1] - catalog_xy[c, 1]))
            for i, c in pairs
        ]
        solutions.append({
            "coefficients": coefficients,
            "pairs": pairs,
            "predicted": predicted,
            "spread": _support_spread(image_xy, pairs, dims),
            "median_residual_px": float(np.median(residuals)),
        })
        return True

    # First wave: seeds explaining at least four bright subset stars.
    strong_seeds = [hypothesis for count, hypothesis in ranked if count >= 4]
    for hypothesis in strong_seeds:
        coefficients, pairs = expand(hypothesis)
        absorb(coefficients, pairs)
    leader = max(solutions, key=lambda item: len(item["pairs"]), default=None)
    # A decisive fit needs many field-spanning pairs AND tight residuals. Random
    # triangle collisions yield few pairs whose residuals approach the tolerance,
    # so this gate rejects them and the fallback scan still finds a true seed that
    # the brightness-ranking mismatch kept out of the first wave.
    well_supported = (
        leader is not None
        and len(leader["pairs"]) >= max(10, len(image_xy) // 6)
        and leader["spread"] >= CORNER_SPREAD_RATIO
        and leader["median_residual_px"] <= 0.35 * tolerance_px
    )
    # Fallback: if no decisive solution appeared, scan every remaining seed so a
    # correct match hidden by brightness-ranking mismatch is still recovered.
    if not well_supported:
        for _, hypothesis in ranked:
            if hypothesis in strong_seeds:
                continue
            coefficients, pairs = expand(hypothesis)
            absorb(coefficients, pairs)
    if not solutions:
        raise DomainError(
            f"三角配对后没有任何解在 {prior.match_tolerance_arcsec:g} 角秒内容纳至少 {MIN_PAIRS} 对星"
        )
    # Merge solutions whose pair sets largely agree, keeping the larger support.
    solutions.sort(key=lambda item: (-len(item["pairs"]), -item["spread"]))
    merged = []
    for candidate in solutions:
        for kept in merged:
            if _pair_overlap(kept["pairs"], candidate["pairs"]) >= 0.8:
                break
        else:
            merged.append(candidate)
    return merged


def _solution_sky_fit(solution, image_stars, catalog_rows):
    pairs = solution["pairs"]
    anchor_rows = []
    for image_index, catalog_index in pairs:
        star = image_stars[image_index]
        catalog = catalog_rows[catalog_index]
        anchor_rows.append({
            "label": catalog["label"],
            "x": star["x"],
            "y": star["y"],
            "ra": catalog["ra"],
            "dec": catalog["dec"],
            "weight": 1,
        })
    sky = [Point(row["ra"], row["dec"]) for row in anchor_rows]
    center = spherical_center(sky)
    positions = np.asarray([(row["x"], row["y"]) for row in anchor_rows])
    tangents = np.asarray([
        (lambda point: (point.x, point.y))(sky_to_tangent(center, Point(row["ra"], row["dec"])))
        for row in anchor_rows
    ])
    weights = np.ones(len(anchor_rows))
    coefficients = affine_coefficients(positions, tangents, weights)
    projection = anchored_projection(center, coefficients)
    residuals = np.asarray([
        angular_distance(projection.pixel_to_sky(Point(row["x"], row["y"])),
                         Point(row["ra"], row["dec"])) * 3600
        for row in anchor_rows
    ])
    return center, projection, anchor_rows, residuals


def classify_solutions(solutions, image_xy, dims):
    """Separate the best solution from corner-only support and genuine rivals."""
    best = solutions[0]
    best_count = len(best["pairs"])
    rivals = []
    for candidate in solutions[1:]:
        support_ratio = len(candidate["pairs"]) / best_count
        overlap = _pair_overlap(best["pairs"], candidate["pairs"])
        if support_ratio >= RIVAL_SUPPORT_RATIO and (1 - overlap) >= RIVAL_DISAGREEMENT:
            rivals.append(candidate)
    corner_only = best["spread"] < CORNER_SPREAD_RATIO
    if rivals:
        status = "ambiguous"
    elif corner_only:
        status = "corner_only"
    else:
        status = "unique"
    return status, rivals


def build_match_report(images, options):
    document = images.document
    if not document.get("reference"):
        raise DomainError("请先指定一张参考曝光")
    dims = dimensions_of(document)
    catalog_rows = document.get("catalog", [])
    if len(catalog_rows) < 5:
        raise DomainError("请先导入至少含 5 颗星的本地星表")
    prior = MatchPrior.from_options(options)
    plane = images.calibrated(document["reference"])
    image_stars, detection = detect_reference_stars(plane, options)
    image_xy = np.asarray([(star["x"], star["y"]) for star in image_stars])

    prior_projection = prior.projection(dims)
    catalog_pixels = []
    catalog_in_frame = []
    # Only catalog points plausibly overlapping detected stars can participate.
    # The margin is the smaller of the operator uncertainty and half the short side.
    uncertainty_px = prior.uncertainty_arcsec / prior.scale_arcsec
    margin_px = max(4.0, min(uncertainty_px, min(dims.width, dims.height) * 0.5))
    for index, row in enumerate(catalog_rows):
        point = prior_projection.sky_to_pixel(Point(row["ra"], row["dec"]))
        catalog_pixels.append((point.x, point.y))
        if (-margin_px <= point.x < dims.width + margin_px
                and -margin_px <= point.y < dims.height + margin_px):
            catalog_in_frame.append(index)
    if len(catalog_in_frame) < 4:
        raise DomainError("按给定先验，画幅附近的星表星不足 4 颗，请检查中心、尺度或不确定度")
    catalog_xy_full = np.asarray(catalog_pixels)
    catalog_xy = catalog_xy_full[catalog_in_frame]

    # Triangle hashing uses only the brightest stars on each side; refitting then
    # expands each seed across every detected star.
    image_brightness = np.asarray([
        star.get("net_counts") if star.get("net_counts") is not None else star.get("peak", 0)
        for star in image_stars
    ])
    image_subset = np.argsort(-image_brightness)[:MATCH_STAR_LIMIT]
    in_frame_rows = [catalog_rows[index] for index in catalog_in_frame]
    if any(row.get("magnitude") is not None for row in in_frame_rows):
        catalog_magnitudes = np.asarray([
            row["magnitude"] if row.get("magnitude") is not None else np.inf
            for row in in_frame_rows
        ])
        catalog_subset = np.argsort(catalog_magnitudes)[:MATCH_STAR_LIMIT]
    else:
        catalog_subset = np.arange(min(MATCH_STAR_LIMIT, len(catalog_in_frame)))
    if len(image_subset) < 4 or len(catalog_subset) < 4:
        raise DomainError("可用于三角配对的亮星不足 4 颗")

    solutions = match_stars(image_xy, catalog_xy, prior, dims, image_subset, catalog_subset)
    status, rivals = classify_solutions(solutions, image_xy, dims)

    def describe(solution):
        local_pairs = solution["pairs"]
        global_pairs = [(image_index, catalog_in_frame[catalog_index])
                        for image_index, catalog_index in local_pairs]
        center, projection, anchors, residuals = _solution_sky_fit(
            solution, image_stars, [catalog_rows[index] for index in catalog_in_frame],
        )
        pair_rows = []
        for (image_index, catalog_index), residual in zip(global_pairs, residuals):
            star = image_stars[image_index]
            catalog = catalog_rows[catalog_index]
            fitted_pixel = projection.sky_to_pixel(Point(catalog["ra"], catalog["dec"]))
            pair_rows.append({
                "label": catalog["label"],
                "x": float(star["x"]),
                "y": float(star["y"]),
                "ra": catalog["ra"],
                "dec": catalog["dec"],
                "magnitude": catalog.get("magnitude"),
                "residual_arcsec": float(residual),
                "fit_x": float(fitted_pixel.x),
                "fit_y": float(fitted_pixel.y),
            })
        fitted_matrix = projection.matrix
        singular = np.linalg.svd(fitted_matrix, compute_uv=False)
        recovered = {
            "mirror": bool(np.linalg.det(fitted_matrix) < 0),
            "angle_deg": float(math.degrees(math.atan2(fitted_matrix[0, 0],
                                                        fitted_matrix[1, 0])) % 360),
            "major_scale_arcsec": float(singular[0] * 3600),
            "minor_scale_arcsec": float(singular[-1] * 3600),
        }
        return {
            "pair_count": len(pair_rows),
            "spread_ratio": float(solution["spread"]),
            "rms_arcsec": float(np.sqrt(np.mean(residuals**2))),
            "worst_arcsec": float(np.max(residuals)),
            "center_ra": center.x,
            "center_dec": center.y,
            "matrix": projection.matrix.tolist(),
            "origin_x": projection.pixel_origin.x,
            "origin_y": projection.pixel_origin.y,
            "recovered": recovered,
            "pairs": pair_rows,
            "anchors": anchors,
        }

    report = {
        "kind": "star_match",
        "revision": document["revision"],
        "status": status,
        "corner_only": status == "corner_only",
        "ambiguous": status == "ambiguous",
        "tolerance_arcsec": prior.match_tolerance_arcsec,
        "prior": {
            "center_ra": prior.center.x,
            "center_dec": prior.center.y,
            "scale_arcsec": prior.scale_arcsec,
            "angle_deg": prior.angle_deg,
            "mirror": prior.mirror,
            "uncertainty_arcsec": prior.uncertainty_arcsec,
        },
        "detected_stars": len(image_stars),
        "catalog_stars_in_frame": len(catalog_in_frame),
        "triangle_image_stars": int(len(image_subset)),
        "triangle_catalog_stars": int(len(catalog_subset)),
        "solutions": [describe(solution) for solution in (solutions[:1] + rivals)],
        "all_solution_count": len(solutions),
        "image_stars": [
            {"x": float(star["x"]), "y": float(star["y"]), "peak": star.get("peak"),
             "fwhm": star.get("fwhm"), "net_counts": star.get("net_counts")}
            for star in image_stars
        ],
        "catalog_pixels": [
            {"label": catalog_rows[index]["label"], "x": float(catalog_xy_full[index, 0]),
             "y": float(catalog_xy_full[index, 1]), "in_frame": index in set(catalog_in_frame)}
            for index in range(len(catalog_rows))
        ],
        "detection": {
            "candidate_count": detection["candidate_count"],
            "stars_used": detection["stars_used"],
            "median_fwhm": detection["median_fwhm"],
        },
    }
    return report


def draw_match_preview(images, report, options):
    """Reference image with proposed pair links, residual vectors and status banner."""
    from io import BytesIO
    from PIL import Image, ImageDraw

    document = images.document
    dims = dimensions_of(document)
    plane = images.calibrated(document["reference"])
    rendered, _ = render_plane(plane, document["stretch"], max_side=1100)
    image = Image.open(BytesIO(rendered)).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale_x = image.width / dims.width
    scale_y = image.height / dims.height

    def px(point):
        return ((point[0] + 0.5) * scale_x, (point[1] + 0.5) * scale_y)

    status_colors = {
        "unique": (124, 226, 216),
        "corner_only": (245, 198, 130),
        "ambiguous": (246, 139, 122),
    }
    color = status_colors[report["status"]]
    rival_color = (170, 163, 255)
    prior_by_label = {row["label"]: row for row in report["catalog_pixels"]}

    # Detected image stars: small squares.
    for star in report["image_stars"]:
        x, y = px((star["x"], star["y"]))
        draw.rectangle((x - 2, y - 2, x + 2, y + 2), outline=(120, 140, 180))

    # Catalog prior positions: crosses, dimmed when outside the useful frame.
    for catalog in report["catalog_pixels"]:
        x, y = px((catalog["x"], catalog["y"]))
        if not (-20 < x < image.width + 20 and -20 < y < image.height + 20):
            continue
        tone = (150, 158, 178) if catalog["in_frame"] else (88, 96, 116)
        draw.line((x - 3, y, x + 3, y), fill=tone)
        draw.line((x, y - 3, x, y + 3), fill=tone)

    solution_index = integer(options.get("solution", 0), "候选解序号", 0,
                             max(0, len(report["solutions"]) - 1))
    solution = report["solutions"][solution_index]
    for row in solution["pairs"]:
        image_x, image_y = px((row["x"], row["y"]))
        # Link to the catalog star's rough prior position (the proposed identity).
        prior = prior_by_label.get(row["label"])
        if prior is not None:
            catalog_x, catalog_y = px((prior["x"], prior["y"]))
            draw.line((image_x, image_y, catalog_x, catalog_y), fill=(96, 110, 140), width=1)
            draw.ellipse((catalog_x - 2, catalog_y - 2, catalog_x + 2, catalog_y + 2),
                         outline=(170, 182, 206))
        # Residual vector: measured star -> fitted catalog position.
        fit_x, fit_y = px((row["fit_x"], row["fit_y"]))
        draw.line((image_x, image_y, fit_x, fit_y), fill=color, width=2)
        draw.ellipse((image_x - 3, image_y - 3, image_x + 3, image_y + 3), outline=color)
        draw.ellipse((fit_x - 2, fit_y - 2, fit_x + 2, fit_y + 2), fill=color)
        draw.text((image_x + 5, image_y - 6), row["label"][:10], fill=(226, 236, 250))

    if report["ambiguous"] and solution_index == 0 and len(report["solutions"]) > 1:
        rival = report["solutions"][1]
        for row in rival["pairs"][:24]:
            x, y = px((row["x"], row["y"]))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=rival_color)

    labels = {
        "unique": "UNIQUE solution",
        "corner_only": "CORNER-ONLY support - not applied",
        "ambiguous": "AMBIGUOUS - rival solutions exist, not applied",
    }
    draw.rectangle((0, 0, image.width, 22), fill=(16, 22, 34))
    draw.text((8, 5), f"{labels[report['status']]} | pairs={solution['pair_count']} "
                      f"rms={solution['rms_arcsec']:.2f} arcsec | solution {solution_index + 1}"
                      f"/{len(report['solutions'])}", fill=color)
    return png(image)
