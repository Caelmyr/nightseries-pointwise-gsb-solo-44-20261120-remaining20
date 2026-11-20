"""HTTP-level tests for the star matching endpoints via the Flask app."""
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nightseries.demo import SCENARIOS, make_demo, star_catalog
from nightseries.astrometry import Point, tangent_to_sky
from nightseries.store import Store
from nightseries.web import create_app


def catalog_rows():
    rng = np.random.default_rng(4825)
    catalog_xy = star_catalog(SCENARIOS["variable"], rng)
    center = Point(0.021, 23.4)
    true_matrix = np.array([[-0.00042, 0.00003], [0.00003, 0.00042]])
    crpix = np.array([368 / 2 + 1, 296 / 2 + 1]) - 1
    rows = []
    for index, (x, y, _) in enumerate(catalog_xy[:90]):
        tangent = true_matrix @ (np.array([x, y]) - crpix)
        sky = tangent_to_sky(center, Point(tangent[0], tangent[1]))
        rows.append({"label": f"CAT-{index:03d}", "ra": sky.x, "dec": sky.y, "magnitude": 12.0})
    return rows


class MatchApiTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        store = Store(f"{self.tmp}/observations.sqlite3")
        document = make_demo("variable", store)
        self.sid = document["id"]
        self.reference = document["reference"]
        store.set_catalog(self.sid, document["revision"], catalog_rows())
        with store.session(write=True) as db:
            doc = store._row(db, self.sid)
            frame = next(row for row in doc["frames"] if row["id"] == self.reference)
            frame["header"] = {key: value for key, value in frame["header"].items()
                               if not key.startswith(("CTYPE", "CUNIT", "CRPIX", "CRVAL", "CD", "RADESYS"))}
            store._edit(db, doc)
        self.app = create_app(self.tmp)
        self.client = self.app.test_client()
        doc = Store(f"{self.tmp}/observations.sqlite3").get(self.sid)
        self.version = doc["revision"]
        self.options = {
            "center_ra": 0.021 + 0.04, "center_dec": 23.4 - 0.03,
            "scale_arcsec": 1.512 * 1.10, "angle_deg": 4.09 + 15,
            "mirror": False, "uncertainty_arcsec": 600,
            "tolerance_arcsec": 4.0, "star_cap": 80, "version": self.version,
        }

    def test_match_preview_and_apply(self):
        response = self.client.post(f"/api/series/{self.sid}/sky/match", json=self.options)
        self.assertEqual(response.status_code, 200, response.data)
        report = response.get_json()
        self.assertEqual(report["status"], "unique")
        self.assertGreaterEqual(report["solutions"][0]["pair_count"], 20)

        image = self.client.post(
            f"/api/series/{self.sid}/sky/match",
            json={**self.options, "format": "image"},
        )
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.mimetype, "image/png")
        self.assertGreater(len(image.data), 1000)

        applied = self.client.post(
            f"/api/series/{self.sid}/sky/match/apply", json=self.options,
        )
        self.assertEqual(applied.status_code, 200, applied.data)
        plate = applied.get_json()["result"]
        self.assertEqual(plate["mode"], "starmatch")
        self.assertGreaterEqual(len(plate["anchors"]), 20)

        sky = self.client.get(f"/api/series/{self.sid}/sky").get_json()
        self.assertIsNotNone(sky["field"])
        self.assertAlmostEqual(sky["field"]["solution"]["major_scale_arcsec"], 1.512, delta=0.05)

    def test_apply_refuses_stale_revision(self):
        response = self.client.post(
            f"/api/series/{self.sid}/sky/match/apply",
            json={**self.options, "version": self.version + 99},
        )
        self.assertEqual(response.status_code, 409)

    def test_match_requires_catalog(self):
        empty_dir = self.tmp + "/empty"
        Path(empty_dir).mkdir(parents=True, exist_ok=True)
        empty = Store(f"{empty_dir}/observations.sqlite3")
        doc = make_demo("variable", empty)
        app = create_app(empty_dir)
        client = app.test_client()
        response = client.post(f"/api/series/{doc['id']}/sky/match", json={
            "center_ra": 0.0, "center_dec": 0.0, "scale_arcsec": 1.5,
            "angle_deg": 0.0, "mirror": False, "uncertainty_arcsec": 120,
            "tolerance_arcsec": 4.0, "version": doc["revision"],
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("星表", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
