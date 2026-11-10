"""Hair coverage cannot turn an inferred/hidden forehead into observed anatomy."""

import unittest

from tests.test_frontal_review import example_points


class ForeheadVisibilityTests(unittest.TestCase):
    def test_hairline_hidden_retains_unrelated_readings_and_omits_hairline(self):
        from local_app.landmark_review import calculate, forehead_points

        points = {**example_points(), "hairline": [.5, .05]}
        result = forehead_points(points, "hairline_hidden")
        self.assertIsNone(result["hairline"])
        self.assertEqual(points["hairline"], [.5, .05])
        rows = {r["id"] for r in calculate(result, 400, 600)}
        self.assertIn("reviewed_eye_spacing", rows)
        self.assertFalse(any(k.startswith("hairline_") for k in rows))

    def test_forehead_covered_removes_mesh_top_and_outline_without_erasing_eyes(self):
        from local_app.landmark_review import calculate, forehead_points

        result = forehead_points(example_points(), "forehead_hidden")
        self.assertTrue(all(result[k] is None for k in ("mesh_top", "hairline", "forehead_a", "forehead_b")))
        rows = {r["id"] for r in calculate(result, 400, 600)}
        self.assertNotIn("reviewed_facial_index", rows)
        self.assertNotIn("reviewed_upper_third", rows)
        self.assertIn("reviewed_eye_spacing", rows)
        self.assertIn("reviewed_middle_lower_ratio", rows)

    def test_visible_hairline_requires_explicit_visibility_and_survives_exports(self):
        from local_app.landmark_review import public_review, safe_review, save_review

        points = {**example_points(), "hairline": [.5, .05]}
        record = dict(revision=1, binding="x", points=example_points(), automatic=example_points())
        with self.assertRaises(ValueError):
            save_review(record, points, expected_revision=1, binding="x")
        saved = save_review(record, points, expected_revision=1, binding="x", forehead_visibility="visible")
        payload = safe_review(public_review(saved, 400, 600))
        self.assertEqual(payload["forehead_visibility"], "visible")
        self.assertIn("hairline_upper_share", {r["id"] for r in payload["measurements"]})

    def test_covered_status_overrides_old_points_and_is_canonical_in_report(self):
        from local_app.landmark_review import public_review, safe_review, render_reviewed

        payload = public_review(dict(reviewed=True, revision=2, points={**example_points(), "hairline": [.5, .05]}, forehead_visibility="forehead_hidden"), 400, 600)
        safe = safe_review(payload)
        self.assertIsNone(safe["diagram"]["points"]["hairline"])
        self.assertIn("hair", render_reviewed(safe).lower())
        self.assertIn("clear", render_reviewed(safe).lower())
        self.assertFalse(any(r["id"].startswith("hairline_") for r in safe["measurements"]))
        self.assertIsNone(safe_review({**payload, "forehead_visibility": "<script>bad</script>"}))

    def test_automatic_regions_respect_coverage_only_for_their_actual_evidence(self):
        from dataclasses import replace
        from local_app.product_report import public_regions
        from tests.test_consumer_evaluation import report

        sample = report()
        rows = [dict(id="facial_index", value=1.2, evidence_photo_ids=["p1"]),
                dict(id="forehead_ratio", value=.8, evidence_photo_ids=["p2"]),
                dict(id="eye_spacing", value=.3, evidence_photo_ids=["p1"])]
        axes = {**sample.axes, "harmony": replace(sample.axes["harmony"], details={**sample.axes["harmony"].details, "regional_measurements": rows})}
        visible = public_regions(axes, frontal=[dict(evidence_photo_id="p1", forehead_visibility="forehead_hidden")])
        self.assertEqual({r["id"] for r in visible}, {"forehead_ratio", "eye_spacing"})
        self.assertNotIn("p1", str(visible))

    def test_restored_negative_coverage_can_be_carried_without_confirming_landmarks(self):
        from local_app.landmark_review import public_review, safe_review

        record = dict(reviewed=False, visibility_pending=True, revision=2,
                      points=example_points(), forehead_visibility="forehead_hidden")
        payload = safe_review(public_review(record, 400, 600))
        self.assertIsNotNone(payload)
        self.assertTrue(payload["visibility_only"])
        self.assertEqual(payload["measurements"], [])
        self.assertTrue(all(v is None for v in payload["diagram"]["points"].values()))


if __name__ == "__main__":
    unittest.main()
