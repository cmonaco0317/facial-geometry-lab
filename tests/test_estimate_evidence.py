"""Regression for discarding usable repeats when only image detail is low."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import assess_set
from local_app.analysis import PhotoInput, SetMeasurementAdapter
from local_app.contracts import CohortUse
from local_app.photos import PhotoEntry


class EstimateEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = CohortUse("m_test", "m_test", "exact", 100)

    def inputs(self, settings):
        photos, records = [], []
        for index, setting in enumerate(settings, 1):
            name = f"{index}.jpg"
            photos.append(PhotoInput(PhotoEntry(
                f"p{index}", name, name, str(index) * 64, "jpeg", 600, 800, 100, 100
            ), Path("/unused") / name))
            fact = {**dict(yaw=0.0, pitch=0.0, roll=0.0, ipd_px=80.0,
                           neutral=True, detected=True), **setting}
            records.append(dict(photo=name, landmarks=np.full((468, 2), index),
                                yaw=fact["yaw"], pitch=fact["pitch"],
                                roll=fact["roll"], ipd_px=fact["ipd_px"], fact=fact))
        return tuple(photos), records

    def route(self, settings):
        photos, records = self.inputs(settings)
        adapter = SetMeasurementAdapter()
        with (
            patch.object(assess_set, "gate_photos", return_value=(records, [])),
            patch.object(assess_set.BD, "metrics_for", side_effect=lambda p: {
                "asymmetry": float(p[0, 0]), "harm__horizontal_balance": float(p[0, 0])
            }),
        ):
            routes = adapter.route(photos, self.cohort)
        return adapter, routes, photos

    def test_pool_detail_only_repeats_and_keep_estimated_status(self) -> None:
        adapter, routes, photos = self.route([{}, {}, {}, {}])
        self.assertEqual(routes["symmetry"], ("p1", "p2", "p3", "p4"))
        with patch("local_app.estimates._symmetry_pool", return_value=np.arange(100.0)):
            reading = adapter.measure_axis("symmetry", photos, self.cohort)
        self.assertEqual(reading.status, "estimated")
        self.assertEqual(reading.photo_count, 4)
        self.assertEqual(reading.value, 2.5)
        self.assertEqual(reading.unit, "% outer-eye span asymmetry")
        self.assertIsNone(reading.interval)
        self.assertEqual(reading.details["capture_issues"], {"too_small": 4})

    def test_pooling_respects_axis_pose_and_expression_gates(self) -> None:
        _, routes, _ = self.route([{}, {"yaw": 8.0}, {"neutral": False}, {"pitch": 25.0}])
        self.assertEqual(routes["harmony"], ("p1", "p2"))
        self.assertEqual(routes["symmetry"], ("p1",))

    def test_strict_evidence_never_mixed_with_low_detail_observations(self) -> None:
        _, routes, _ = self.route([{}, {"ipd_px": 150.0}, {}])
        self.assertTrue(all(ids == ("p2",) for ids in routes.values()))

    def test_no_detail_only_candidate_preserves_single_best_observation(self) -> None:
        _, routes, _ = self.route([{"yaw": 20.0}, {"yaw": 30.0}])
        self.assertTrue(all(ids == ("p1",) for ids in routes.values()))


if __name__ == "__main__":
    unittest.main()
