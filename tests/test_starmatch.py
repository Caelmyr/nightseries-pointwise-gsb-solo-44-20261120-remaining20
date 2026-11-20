"""Unit tests for the automatic star-catalog pattern matcher."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nightseries.domain import Dimensions, DomainError
from nightseries.starmatch import (
    MatchPrior, RIVAL_DISAGREEMENT, RIVAL_SUPPORT_RATIO,
    _inlier_pairs, _refit_loop, _triangle_table, classify_solutions, match_stars,
)


class PriorTests(unittest.TestCase):

    def test_matrix_uses_degrees_and_mirror_flag(self):
        prior = MatchPrior.from_options({
            "center_ra": 10.0, "center_dec": 20.0, "scale_arcsec": 3.6,
            "angle_deg": 0.0, "mirror": False,
            "uncertainty_arcsec": 60.0, "tolerance_arcsec": 4.0,
        })
        matrix = prior.matrix()
        self.assertTrue(np.allclose(matrix, np.eye(2) * 0.001))
        mirrored = MatchPrior.from_options({
            "center_ra": 10.0, "center_dec": 20.0, "scale_arcsec": 3.6,
            "angle_deg": 0.0, "mirror": True,
            "uncertainty_arcsec": 60.0, "tolerance_arcsec": 4.0,
        }).matrix()
        self.assertAlmostEqual(np.linalg.det(mirrored), -np.linalg.det(matrix))

    def test_rejects_invalid_center(self):
        with self.assertRaises(DomainError):
            MatchPrior.from_options({"center_ra": 1, "center_dec": 95, "scale_arcsec": 1})


class TriangleTableTests(unittest.TestCase):

    def test_right_triangle_invariants(self):
        points = np.array([(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)])
        table = _triangle_table(points, min_edge=0.0, max_edge=1e9)
        # Sorted edges 3, 4, 5 -> ratios 4/3 and 5/3.
        self.assertIn((round(4 / 3, 2), round(5 / 3, 2)), table)

    def test_mirror_invariant(self):
        points = np.array([(0.0, 0.0), (4.0, 0.0), (1.0, 3.0)])
        mirrored = points * np.array([-1.0, 1.0])
        self.assertEqual(
            set(_triangle_table(points, 0.0, 1e9)),
            set(_triangle_table(mirrored, 0.0, 1e9)),
        )


class InlierTests(unittest.TestCase):

    def setUp(self):
        # Four image stars and a catalog reached by a known rotation+scale+offset.
        self.image = np.array([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)])
        angle = math.radians(30.0)
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]]) * 0.8
        self.catalog = (rotation @ self.image.T).T + np.array([5.0, -7.0])
        self.coefficients = np.vstack((rotation.T, np.array([5.0, -7.0])))

    def test_recovers_all_pairs(self):
        pairs, _ = _inlier_pairs(self.image, self.catalog, self.coefficients, 1.0)
        self.assertEqual(len(pairs), 4)
        mapping = dict(pairs)
        self.assertEqual([mapping[i] for i in range(4)], [0, 1, 2, 3])

    def test_one_to_one_when_two_images_share_a_catalog_star(self):
        image = np.array([(0.0, 0.0), (0.2, 0.0)])
        catalog = np.array([(0.0, 0.0), (50.0, 50.0)])
        coefficients = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        pairs, _ = _inlier_pairs(image, catalog, coefficients, 1.0)
        self.assertEqual(len(pairs), 1)

    def test_refit_loop_grows_seed(self):
        coefficients, pairs = _refit_loop(
            self.image, self.catalog, [(0, 0), (1, 1), (2, 2)], 1.0,
        )
        self.assertEqual(len(pairs), 4)


class ClassificationTests(unittest.TestCase):

    def setUp(self):
        self.dims = Dimensions(320, 260)
        self.image_xy = np.array([(x * 20.0 + 10, y * 20.0 + 10)
                                  for y in range(8) for x in range(8)])

    def solution(self, pairs, spread):
        return {"pairs": pairs, "spread": spread, "coefficients": np.eye(3)}

    def test_unique_when_one_dominant_solution(self):
        best = self.solution([(i, i) for i in range(20)], 0.9)
        weak = self.solution([(i, i + 100) for i in range(6)], 0.5)
        status, rivals = classify_solutions([best, weak], self.image_xy, self.dims)
        self.assertEqual(status, "unique")
        self.assertEqual(rivals, [])

    def test_corner_only_flag(self):
        corner = self.solution([(i, i) for i in range(8)], 0.2)
        status, _ = classify_solutions([corner], self.image_xy, self.dims)
        self.assertEqual(status, "corner_only")

    def test_ambiguous_with_strong_rival(self):
        best = self.solution([(i, i) for i in range(12)], 0.8)
        # Same support size, completely different pair identities.
        rival = self.solution([(i, 200 + i) for i in range(12)], 0.7)
        status, rivals = classify_solutions([best, rival], self.image_xy, self.dims)
        self.assertEqual(status, "ambiguous")
        self.assertEqual(len(rivals), 1)

    def test_similar_solutions_merge_into_unique(self):
        best = self.solution([(i, i) for i in range(12)], 0.8)
        almost = self.solution([(i, i) for i in range(11)] + [(50, 50)], 0.8)
        status, _ = classify_solutions([best, almost], self.image_xy, self.dims)
        self.assertEqual(status, "unique")


class MatchStarsTests(unittest.TestCase):

    def test_end_to_end_synthetic_field(self):
        # Full geometric pipeline on a modest synthetic field (no store/Flask).
        rng = np.random.default_rng(99)
        width, height = 240, 200
        dims = Dimensions(width, height)
        image_xy = np.column_stack((
            rng.uniform(12, width - 12, 28),
            rng.uniform(12, height - 12, 28),
        ))
        angle = math.radians(-40.0)
        # Both sets live in pixel space; the transform is near unit scale.
        matrix = np.array([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]]) * 1.05
        origin = np.array([(width - 1) / 2, (height - 1) / 2])
        catalog_xy = ((matrix @ (image_xy - origin).T).T + origin
                      + rng.normal(0, 0.15, image_xy.shape))
        prior = MatchPrior.from_options({
            "center_ra": 30.0, "center_dec": 10.0, "scale_arcsec": 2.8,
            "angle_deg": -40.0, "mirror": False,
            "uncertainty_arcsec": 120.0, "tolerance_arcsec": 4.0,
        })
        subset = np.arange(len(image_xy))
        solutions = match_stars(image_xy, catalog_xy, prior, dims, subset, subset)
        status, _ = classify_solutions(solutions, image_xy, dims)
        self.assertEqual(status, "unique")
        self.assertGreaterEqual(len(solutions[0]["pairs"]), 20)
        recovered = solutions[0]["coefficients"][:2].T
        self.assertTrue(np.allclose(recovered, matrix, atol=2e-3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
