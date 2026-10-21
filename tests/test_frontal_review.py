"""Reviewed photo geometry is distinct, partial and revision-bound."""

import copy
import math
import unittest

from local_app.landmark_review import ANCHORS, calculate, validate_points, save_review
from local_app.measurement_registry import METRICS, measurement_contract


def example_points():
    points = {k: None for k in ANCHORS}
    points.update(
        mesh_top=[0.5, 0.1],
        chin=[0.5, 0.9],
        cheek_a=[0.15, 0.5],
        cheek_b=[0.85, 0.5],
        brow_head_a=[0.4, 0.3],
        brow_head_b=[0.6, 0.3],
        nose_base=[0.5, 0.6],
        upper_lip=[0.5, 0.65],
        lower_lip=[0.5, 0.7],
        mouth_a=[0.35, 0.68],
        mouth_b=[0.65, 0.68],
        eye_outer_a=[0.25, 0.35],
        eye_inner_a=[0.4, 0.37],
        eye_inner_b=[0.6, 0.37],
        eye_outer_b=[0.75, 0.35],
        eye_top_a=[0.32, 0.34],
        eye_bottom_a=[0.32, 0.38],
        eye_top_b=[0.68, 0.34],
        eye_bottom_b=[0.68, 0.38],
    )
    return points


class FrontalReviewTests(unittest.TestCase):
    def test_known_geometry_aspect_and_new_hairline_ids(self):
        p = example_points()
        rows = {r["id"]: r for r in calculate(p, 400, 600)}
        self.assertAlmostEqual(
            rows["reviewed_facial_index"]["value"], 480 / 280, places=4
        )
        self.assertTrue(all(r["percentile"] is None for r in rows.values()))
        self.assertNotIn("hairline_upper_share", rows)
        p["hairline"] = [0.5, 0.05]
        rows = {r["id"]: r for r in calculate(p, 400, 600)}
        self.assertIn("hairline_upper_share", rows)
        self.assertEqual(rows["reviewed_facial_index"]["value"], round(480 / 280, 4))
        self.assertAlmostEqual(
            rows["hairline_upper_share"]["value"], 0.25 / 0.85, places=4
        )
        self.assertNotEqual(calculate(p, 400, 600), calculate(p, 600, 400))

    def test_partial_unavailable_does_not_erase_unrelated_geometry(self):
        p = example_points()
        p["chin"] = None
        rows = {r["id"] for r in calculate(p, 400, 600)}
        self.assertNotIn("reviewed_facial_index", rows)
        self.assertIn("reviewed_eye_spacing", rows)
        p["eye_inner_a"] = p["eye_outer_a"]
        self.assertNotIn(
            "reviewed_canthal_tilt", {r["id"] for r in calculate(p, 400, 600)}
        )

    def test_malformed_points_dimensions_and_unknown_fields(self):
        p = example_points()
        for changed in (
            {**p, "private_path": [0.2, 0.2]},
            {**p, "chin": [True, 0.9]},
            {**p, "chin": [math.nan, 0.5]},
            {**p, "chin": [-0.1, 0.2]},
            {**p, "chin": [0.2, 0.3, 5]},
            [],
        ):
            with self.assertRaises(ValueError):
                validate_points(changed)
        for dims in ((0, 600), (400, False), (400, float("inf"))):
            with self.assertRaises(ValueError):
                calculate(p, *dims)

    def test_atomic_revision_conflict_and_image_binding(self):
        base = {
            "binding": "a" * 64,
            "revision": 1,
            "automatic": example_points(),
            "points": example_points(),
            "reviewed": False,
        }
        original = copy.deepcopy(base)
        updated = save_review(
            base, example_points(), expected_revision=1, binding="a" * 64
        )
        self.assertEqual(base, original)
        self.assertEqual(updated["revision"], 2)
        self.assertTrue(updated["reviewed"])
        for revision, binding in ((1, "b" * 64), (0, "a" * 64), (True, "a" * 64)):
            with self.assertRaises(ValueError):
                save_review(
                    base, example_points(), expected_revision=revision, binding=binding
                )

    def test_registry_exact_definitions_and_contract(self):
        self.assertEqual(len(METRICS), 22)
        self.assertEqual(len({m.id for m in METRICS}), 22)
        self.assertTrue(all(m.unit in {"ratio", "deg"} for m in METRICS))
        contract = measurement_contract()
        self.assertEqual(contract, measurement_contract())
        self.assertEqual(len(contract), 64)


class ReviewGuardTests(unittest.TestCase):
    def test_swapped_corners_require_acknowledgment(self):
        from local_app.landmark_review import geometry_warnings

        p = example_points()
        p["eye_inner_a"], p["eye_outer_a"] = p["eye_outer_a"], p["eye_inner_a"]
        self.assertTrue(geometry_warnings(p))
        record = {
            "revision": 1,
            "binding": "x",
            "points": example_points(),
            "automatic": example_points(),
        }
        with self.assertRaises(ValueError):
            save_review(record, p, expected_revision=1, binding="x")
        accepted = save_review(
            record, p, expected_revision=1, binding="x", acknowledged=True
        )
        self.assertTrue(accepted["warnings"])

    def test_canonical_export_preserves_ordinal_and_rejects_bad_geometry(self):
        from local_app.landmark_review import (
            public_review,
            safe_review,
            render_reviewed,
        )

        raw = public_review(
            {"reviewed": True, "revision": 2, "points": example_points()}, 400, 600
        )
        safe = safe_review(
            {**raw, "photo_number": 3, "private_path": "/private/secret"}
        )
        self.assertEqual(safe["photo_number"], 3)
        self.assertIn("Photo 3", render_reviewed(safe))
        self.assertNotIn("private_path", str(safe))
        self.assertIsNone(safe_review({**raw, "photo_number": True}))
        self.assertIsNone(
            safe_review({**raw, "diagram": {**raw["diagram"], "points": {}}})
        )
