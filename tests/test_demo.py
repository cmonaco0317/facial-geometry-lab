"""Public-demo regression tests use fictional fixtures only."""

import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from demo import demo_payload, render_demo
from examples.synthetic import frontal_points, illustrative_axes, profile_points
from local_app.landmark_review import calculate, save_review
from local_app.product_report import composite_score
from local_app.profile_v2 import measure, safe_profile


class DemoTests(unittest.TestCase):
    def test_actual_geometry_and_separate_score_fixture(self):
        rows = {r["id"]: r for r in calculate(frontal_points(), 400, 600)}
        self.assertAlmostEqual(rows["reviewed_facial_index"]["value"], 480 / 272, places=4)
        self.assertTrue(all(r["percentile"] is None for r in rows.values()))
        score = composite_score(illustrative_axes())
        self.assertAlmostEqual(score["value"], 75.3)
        self.assertEqual(score["coverage"], 1.0)

    def test_partial_profile_retains_independent_measurements(self):
        raw = profile_points()
        raw["upper_lip"] = None
        result = measure(raw, 400, 600, facing="right")
        self.assertEqual(result["unavailable"]["upper_eline"], "missing_anchors")
        self.assertIn("convexity", {row["id"] for row in result["measurements"]})
        self.assertIsNotNone(safe_profile({**result, "revision": 1}))

    def test_missing_components_reweight_without_fabricating_zero(self):
        axes = illustrative_axes()
        only = composite_score({"symmetry": axes["symmetry"]})
        self.assertEqual(only["value"], 78.0)
        self.assertEqual(only["status"], "partial")
        self.assertEqual(only["coverage"], 0.3)
        self.assertIsNone(composite_score({})["value"])

    def test_report_offline_no_network_resources(self):
        class Resources(HTMLParser):
            def __init__(self):
                super().__init__()
                self.external = []
                self.scripts = 0

            def handle_starttag(self, tag, attrs):
                self.scripts += tag == "script"
                for key, value in attrs:
                    if key in ("src", "href") and value and not value.startswith("#"):
                        self.external.append(value)

        with patch("socket.socket", side_effect=AssertionError("network forbidden")):
            page = render_demo()
        parser = Resources()
        parser.feed(page)
        self.assertEqual(parser.external, [])
        self.assertEqual(parser.scripts, 0)
        self.assertIn("Entirely synthetic demonstration", page)
        self.assertIn("Reviewed frontal measurements", page)
        self.assertIn("leave-one-input-out", page)
        self.assertEqual(page, render_demo())

    def test_export_is_strict_json_without_photo_data(self):
        payload = demo_payload()
        encoded = json.dumps(payload, allow_nan=False)
        for forbidden in ("source_path", "photo_id", "base64", "/Users/", "captures/"):
            self.assertNotIn(forbidden, encoded)
        self.assertIn("SYNTHETIC", payload["provenance"])
        self.assertEqual(len(payload["profile"]["measurements"]), 8)

    def test_original_review_is_immutable(self):
        current = dict(binding="synthetic", revision=1, points=frontal_points())
        before = copy.deepcopy(current)
        updated = save_review(current, frontal_points(), expected_revision=1,
                              binding="synthetic")
        self.assertEqual(current, before)
        self.assertEqual(updated["revision"], 2)

    def test_private_calibration_modules_are_absent(self):
        root = Path(__file__).resolve().parents[1]
        excluded = ("reference_values.py", "movement_reference.py",
                    "dimorphism_reference.py", "symmetry_reference.py",
                    "averageness_band.py", "stranger_movement_probe.py",
                    "local_app/analysis.py", "local_app/server.py")
        for name in excluded:
            self.assertFalse((root / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
