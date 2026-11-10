"""Direct iris observations stay separate from the calibrated mesh instrument."""

import unittest
from unittest.mock import patch

import numpy as np

from tests.test_regional_evidence import context


def iris_points():
    points = np.full((478, 2), (50.0, 12.0))
    for index, pair in {
        33: (10, 20), 133: (30, 20), 362: (70, 20), 263: (90, 20),
        159: (20, 15), 145: (20, 25), 386: (80, 15), 374: (80, 25),
        468: (22, 21), 473: (78, 21), 70: (20, 10), 300: (80, 10),
    }.items():
        points[index] = pair
    return points


class IrisObservationTests(unittest.TestCase):
    def test_cache_fingerprint_reads_direct_detector_and_model_dependencies(self):
        from pathlib import Path
        from local_app.analysis import SetMeasurementAdapter

        original = Path.open
        opened = []

        def tracked(path, *args, **kwargs):
            opened.append(path.name)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", tracked):
            SetMeasurementAdapter().fingerprint()
        self.assertIn("fa_common.py", opened)
        self.assertIn("face_landmarker.task", opened)

    def test_regional_photo_ids_include_only_finite_contributing_observations(self):
        from local_app.estimates import regional_measurements

        sample = context([{"neutral": True}, {"neutral": True}])
        sample.values = [{"facial_index": 1.25}, {"facial_index": np.nan}]
        with patch("local_app.estimates._pool", return_value=None):
            row = next(r for r in regional_measurements(sample, ("p1", "p2")) if r["id"] == "facial_index")
        self.assertEqual(row["evidence_photo_count"], 1)
        self.assertEqual(row["evidence_photo_ids"], ["p1"])

    def test_real_iris_centres_not_eyelid_averages_and_scale_invariance(self):
        from local_app.anatomical_observations import iris_metrics

        points = iris_points()
        values = iris_metrics(points)
        self.assertAlmostEqual(values["iris_spacing_ratio"], 56 / 80)
        self.assertAlmostEqual(values["iris_brow_elevation"], 11 / 80)
        for key, value in values.items():
            self.assertAlmostEqual(iris_metrics(points * 3 + 17)[key], value)

    def test_missing_nonfinite_collapsed_or_outside_eye_does_not_fill_a_proxy(self):
        from local_app.anatomical_observations import iris_metrics

        points = iris_points()
        self.assertEqual(iris_metrics(points[:468]), {})
        for index, pair in ((468, (np.nan, 21)), (468, (70, 21)), (263, (10, 20))):
            changed = points.copy()
            changed[index] = pair
            self.assertEqual(iris_metrics(changed), {})

    def test_rows_use_eligible_photo_evidence_and_have_no_borrowed_ranks(self):
        from local_app.anatomical_observations import observed_iris_rows

        sample = context([{}, {"yaw": 30.0}])
        with patch("fa_common.detect_landmarks", return_value=(iris_points(), None, (100, 100))) as detect:
            rows = observed_iris_rows(sample, ("p1",), {"p1": "a.jpg", "p2": "b.jpg"})
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["percentile"] is None and r["evidence_photo_count"] == 1 for r in rows))
        self.assertTrue(all(r["method"] == "automatic-iris-v1" for r in rows))
        self.assertIn("iris", rows[0]["label"].lower())

    def test_no_primary_or_detection_failure_leaves_observations_unavailable(self):
        from local_app.anatomical_observations import observed_iris_rows

        with patch("fa_common.detect_landmarks", side_effect=RuntimeError("private path")):
            self.assertEqual(observed_iris_rows(context(), ("p1",), {"p1": "a.jpg"}), [])
        self.assertEqual(observed_iris_rows(context(), (), {"p1": "a.jpg"}), [])

    def test_public_labels_name_the_actual_proxy(self):
        from dataclasses import replace
        from local_app.product_report import public_regions
        from tests.test_consumer_evaluation import report

        sample = report()
        rows = [dict(id="brow_position", label="Pupil height", value=.1, unit="ratio"),
                dict(id="upper_third", label="Upper third", value=.2, unit="ratio"),
                dict(id="jaw_ratio", label="Bigonial width", value=.8, unit="ratio")]
        axes = {**sample.axes, "harmony": replace(sample.axes["harmony"], details={**sample.axes["harmony"].details, "regional_measurements": rows})}
        public = {r["id"]: r for r in public_regions(axes)}
        self.assertIn("eyelid", public["brow_position"]["label"].lower())
        self.assertIn("mesh", public["upper_third"]["label"].lower())
        self.assertNotIn("bigonial", public["jaw_ratio"]["label"].lower())


if __name__ == "__main__":
    unittest.main()
