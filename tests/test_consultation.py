"""Report chapter, preference and manual profile contract checks."""

import math
import unittest

from local_app.consultation import build_chapters, preferences, styling_suggestions
from local_app.profile_review import measure_profile, validate_points


class ConsultationTests(unittest.TestCase):
    def test_unknown_preferences_are_rejected_and_defaults_are_neutral(self):
        self.assertEqual(preferences({})["goal"], "understand")
        for payload in ({"goal": "fix me"}, {"script": "<script>"}, {"hair": True}):
            with self.assertRaises(ValueError):
                preferences(payload)

    def test_chapters_use_actual_measurements_and_do_not_invent_missing_rows(self):
        rows = [
            dict(
                id="jaw_ratio",
                label="Jaw / cheek",
                value=0.8,
                percentile=90,
                unit="ratio",
                note="Measured on photo",
            )
        ]
        chapters = build_chapters(rows)
        jaw = next(c for c in chapters if c["id"] == "jaw")
        eyes = next(c for c in chapters if c["id"] == "eyes")
        self.assertEqual(jaw["measurements"][0]["value"], 0.8)
        self.assertIn("upper quarter", jaw["observations"][0])
        self.assertEqual(eyes["measurements"], [])
        self.assertIn("No usable", eyes["observations"][0])
        self.assertNotIn("ideal", str(chapters))

    def test_styling_is_opt_in_and_varies_with_goal_and_available_evidence(self):
        rows = [dict(id="facial_index", value=1.4, percentile=80)]
        self.assertEqual(styling_suggestions(rows, preferences({})), [])
        settings = preferences({"goal": "balance", "hair": "yes"})
        suggestions = styling_suggestions(rows, settings)
        self.assertIn("width", suggestions[0]["action"])
        self.assertEqual(styling_suggestions([], settings), [])
        self.assertEqual(
            styling_suggestions(rows, preferences({"goal": "balance"})), []
        )
        low = styling_suggestions(
            [dict(id="facial_index", value=1, percentile=10)], settings
        )
        self.assertNotEqual(suggestions[0]["action"], low[0]["action"])


class ProfileReviewTests(unittest.TestCase):
    def setUp(self):
        self.points = {
            "nasion": [0.4, 0.2],
            "tip": [0.7, 0.4],
            "subnasale": [0.5, 0.5],
            "upper_lip": [0.6, 0.6],
            "lower_lip": [0.6, 0.7],
            "pogonion": [0.6, 0.8],
            "menton": [0.5, 0.9],
        }

    def test_manual_profile_is_scale_and_mirror_invariant(self):
        a = measure_profile(self.points, 400, 600)
        b = measure_profile(self.points, 800, 1200)
        mirror = {k: [1 - p[0], p[1]] for k, p in self.points.items()}
        c = measure_profile(mirror, 400, 600)
        for x, y, z in zip(a["measurements"], b["measurements"], c["measurements"]):
            self.assertAlmostEqual(x["value"], y["value"])
            self.assertAlmostEqual(x["value"], z["value"])
        self.assertEqual(a["method"], "manual-profile-v1")
        self.assertEqual(len(a["measurements"]), 4)
        self.assertNotIn("percentile", str(a))

    def test_rejects_missing_nonfinite_outside_and_degenerate_points(self):
        for patch in (
            {"tip": [float("nan"), 0.4]},
            {"tip": [2, 0.4]},
            {"tip": [True, 0.4]},
            {"tip": [0.4]},
            {"extra": [0.4, 0.4]},
        ):
            with self.assertRaises(ValueError):
                validate_points({**self.points, **patch})
        with self.assertRaises(ValueError):
            validate_points({k: p for k, p in self.points.items() if k != "tip"})
        with self.assertRaises(ValueError):
            measure_profile({**self.points, "menton": self.points["nasion"]}, 400, 600)
        with self.assertRaises(ValueError):
            measure_profile(self.points, 0, 600)

    def test_profile_uses_image_aspect_ratio(self):
        a = measure_profile(self.points, 400, 600)
        b = measure_profile(self.points, 600, 400)
        self.assertNotEqual(
            a["measurements"][1]["value"], b["measurements"][1]["value"]
        )
        self.assertTrue(all(math.isfinite(r["value"]) for r in a["measurements"]))


class IllustrationTests(unittest.TestCase):
    def test_observed_diagram_keeps_aspect_and_exports_only_numeric_points(self):
        from types import SimpleNamespace
        import numpy as np
        from local_app.illustrations import (
            observed_diagram,
            public_diagram,
            render_diagrams,
        )

        points = np.array([[i % 26, i // 26] for i in range(468)], dtype=float)
        context = SimpleNamespace(
            kept=[dict(photo="private-photo-id", landmarks=points, yaw=0)]
        )
        diagram = observed_diagram(context, ("private-photo-id",))
        self.assertIsNotNone(diagram)
        self.assertNotIn("private-photo-id", str(diagram))
        self.assertEqual(len(render_diagrams(diagram)), 6)
        self.assertEqual(
            set(public_diagram({**diagram, "filename": "/private/file"})), {"points"}
        )
        points = diagram["points"].copy()
        points["10"] = [float("inf"), 1]
        self.assertIsNone(public_diagram({"points": points}))
        self.assertIsNone(public_diagram({"points": {"10": [1, 2]}}))
        self.assertIsNone(observed_diagram(context, ()))

    def test_export_and_chapters_escape_labels(self):
        from local_app.consultation import consultation_payload, render_consultation

        payload = consultation_payload(
            [
                dict(
                    id="eye_spacing",
                    label="<img src=x onerror=alert(1)>",
                    value=0.3,
                    unit="ratio",
                    percentile=45,
                )
            ]
        )
        page = render_consultation(payload)
        self.assertNotIn("<img", page)
        self.assertIn("&lt;img", page)


class ExportBoundaryTests(unittest.TestCase):
    def test_profile_export_rebuilds_from_finite_readings_only(self):
        from local_app.profile_review import public_profile

        fixture = ProfileReviewTests()
        fixture.setUp()
        raw = measure_profile(fixture.points, 400, 600)
        raw["photo_id"] = "PRIVATE"
        raw["note"] = "<local-path>"
        raw["measurements"][0]["label"] = "PRIVATE"
        raw["measurements"][0]["points"] = {"private": 1}
        clean = public_profile(raw)
        self.assertNotIn("PRIVATE", str(clean))
        self.assertNotIn("/Users", str(clean))
        self.assertNotIn("points", str(clean))
        raw["measurements"][0]["value"] = float("nan")
        self.assertIsNone(public_profile(raw))

    def test_canthal_angle_survives_without_reference_but_collapsed_eye_does_not(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from local_app.estimates import regional_measurements

        points = np.zeros((468, 2))
        points[33] = [0, 0]
        points[133] = [10, 2]
        points[263] = [30, 0]
        points[362] = [20, 2]
        context = SimpleNamespace(
            kept=[dict(photo="p", landmarks=points)], values=[{}], reference_id="test"
        )
        with patch("local_app.estimates._pool", return_value=None):
            rows = regional_measurements(context, ("p",))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "canthal_tilt")
        self.assertEqual(rows[0]["unit"], "deg")
        self.assertAlmostEqual(rows[0]["value"], 11.3099, places=3)
        self.assertIsNone(rows[0]["percentile"])
        points[33] = points[133]
        self.assertEqual(regional_measurements(context, ("p",)), [])
