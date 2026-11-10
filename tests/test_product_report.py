"""Product behavior: estimates survive imperfect evidence and scoring stays explicit."""

from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

import assess_set
from local_app.contracts import AxisResult, CohortUse, Demographics
from local_app.guidance import PhotoFact
from tests import test_render


class ProductReportTests(unittest.TestCase):
    def test_print_export_expands_score_input_audit_without_opening_live_panel(self):
        from html.parser import HTMLParser
        from local_app.report import build_export_payload, render_html_export
        from local_app.render import render_results
        from tests.test_consumer_evaluation import report

        class AuditState(HTMLParser):
            def __init__(self):
                super().__init__()
                self.open_states = []

            def handle_starttag(self, tag, attrs):
                attributes = dict(attrs)
                if tag == "details" and attributes.get("class") == "score-audit":
                    self.open_states.append("open" in attributes)

        fixture = report()
        payload = build_export_payload(fixture, Demographics(25, "male", ("white",)))
        for page, expected in (
            (render_html_export(payload), [True]),
            (render_results(fixture, csrf_token="test-only"), [False]),
        ):
            parser = AuditState()
            parser.feed(page)
            self.assertEqual(parser.open_states, expected)

    def setUp(self):
        self.cohort = CohortUse("m_test", "m_test", "exact", 100)
        self.fact = PhotoFact("p1", 20.0, 0.0, 0.0, 100.0, False)
        self.photo = {
            "photo": "p1",
            "landmarks": np.ones((468, 2)),
            "yaw": 20.0,
            "pitch": 0.0,
            "roll": 0.0,
            "ipd_px": 100.0,
        }
        self.values = {"harm__horizontal_balance": 0.1, "asymmetry": 2.0}
        self.context = assess_set.AssessmentContext(
            self.cohort, (self.photo,), (self.values,), (self.fact,)
        )

    def estimate(self, axis):
        from local_app.estimates import estimate_axis

        return estimate_axis(axis, self.context, ("p1",))

    def test_one_imperfect_photo_returns_symmetry_estimate_without_fake_interval(self):
        with mock.patch(
            "local_app.estimates._symmetry_pool", return_value=np.arange(100.0) / 10
        ):
            result = self.estimate("symmetry")
        self.assertEqual(result.status, "estimated")
        self.assertEqual(result.value, 2.0)
        self.assertEqual(result.unit, "% outer-eye span asymmetry")
        self.assertIsNone(result.interval)
        self.assertGreater(result.percentile, 70)
        self.assertIn("yaw", result.details["capture_issues"])
        self.assertEqual(AxisResult.from_dict(result.to_dict()), result)

        from local_app.report import build_export_payload, render_html_export, render_json_export
        from local_app.render import render_results
        from tests.test_consumer_evaluation import report
        source = report()
        current = replace(source, axes={**source.axes, "symmetry": result})
        payload = build_export_payload(current, Demographics(25, "male", ("white",)))
        self.assertEqual(payload["axes"]["symmetry"]["unit"], result.unit)
        for content in (render_results(current, csrf_token="test-only"), render_html_export(payload),
                        render_json_export(payload).decode()):
            self.assertIn("% outer-eye span asymmetry", content)
            self.assertNotIn("% IPD asymmetry", content)

    def test_raw_measurement_survives_missing_reference(self):
        with mock.patch("local_app.estimates._symmetry_pool", return_value=None):
            result = self.estimate("symmetry")
        self.assertEqual(result.value, 2.0)
        self.assertIsNone(result.percentile)
        self.assertEqual(result.status, "estimated")

    def test_harmony_returns_observed_components_when_reference_is_missing(self):
        with mock.patch.object(assess_set, "cohort_pool", return_value=None):
            result = self.estimate("harmony")
        self.assertEqual(result.status, "estimated")
        self.assertEqual(result.details["profile"]["horizontal_balance"]["value"], 0.1)
        self.assertIsNone(result.details["profile"]["horizontal_balance"]["percentile"])

    def test_empty_evidence_does_not_invent_a_measurement(self):
        from local_app.estimates import estimate_axis

        result = estimate_axis("symmetry", self.context, ())
        self.assertEqual(result.status, "needs_photo")
        self.assertIsNone(result.value)

    def test_estimated_contract_rejects_nonfinite_and_fake_uncertainty(self):
        with self.assertRaises(ValueError):
            AxisResult(
                axis="symmetry",
                status="estimated",
                summary="estimate",
                value=float("nan"),
                unit="ratio",
                cohort=self.cohort,
                photo_count=1,
            )
        with self.assertRaises(ValueError):
            AxisResult(
                axis="symmetry",
                status="estimated",
                summary="estimate",
                value=2.0,
                unit="ratio",
                cohort=self.cohort,
                photo_count=1,
                interval=(20.0, 80.0),
            )

    def test_quality_does_not_change_the_composite_or_dimorphism_determine_it(self):
        from local_app.product_report import composite_score

        fixture = test_render.ResultRenderTests()
        fixture.setUp()
        axes = fixture.report.axes.copy()
        axes["symmetry"] = AxisResult.measured(
            axis="symmetry",
            summary="symmetry",
            value=2.0,
            unit="ratio",
            percentile=80.0,
            interval=(70.0, 90.0),
            cohort=fixture.cohort,
            photo_count=4,
        )
        score = composite_score(axes)
        self.assertIsNotNone(score["value"])
        self.assertNotIn("dimorphism", [c["id"] for c in score["components"]])
        changed = {
            k: replace(v, details={**v.details, "capture_issues": ["yaw"]})
            for k, v in axes.items()
        }
        changed["dimorphism"] = replace(
            axes["dimorphism"], percentile=99.0, interval=(95.0, 100.0)
        )
        self.assertEqual(composite_score(changed)["value"], score["value"])
        self.assertEqual(score["coverage"], 0.7)
        self.assertEqual(score["status"], "partial")

    def test_harmony_centrality_is_not_raw_percentile_and_stale_scalar_is_ignored(self):
        from local_app.product_report import composite_score

        fixture = test_render.ResultRenderTests()
        fixture.setUp()
        score = composite_score(fixture.report.axes)
        self.assertEqual(score["components"][0]["value"], 92.0)
        self.assertEqual(score["value"], 92.0)
        self.assertEqual(score["coverage"], 0.4)

    def test_no_measurements_means_no_composite(self):
        from local_app.product_report import composite_score

        self.assertIsNone(composite_score({})["value"])

    def test_estimate_and_composite_reach_live_and_portable_reports(self):
        from local_app.report import build_export_payload, render_html_export
        from local_app.render import render_results

        fixture = test_render.ResultRenderTests()
        fixture.setUp()
        with mock.patch(
            "local_app.estimates._symmetry_pool", return_value=np.arange(100.0) / 10
        ):
            estimate = self.estimate("symmetry")
        axes = {
            **fixture.report.axes,
            "symmetry": replace(estimate, cohort=fixture.cohort),
        }
        report = replace(fixture.report, axes=axes)
        payload = build_export_payload(
            report, Demographics(30, "male", ("white",)), source_inventory={}
        )
        self.assertEqual(payload["axes"]["symmetry"]["status"], "estimated")
        self.assertEqual(payload["axes"]["symmetry"]["value"], 2.0)
        self.assertIsNone(payload["axes"]["symmetry"]["uncertainty_percentile"])
        self.assertIsNotNone(payload["composite"]["value"])
        for page in (
            render_results(report, csrf_token="secret"),
            render_html_export(payload),
        ):
            self.assertIn("Overall geometry score", page)
            self.assertIn("Estimate", page)
            self.assertIn("Scoring breakdown", page)
            self.assertNotIn(
                "No total beauty or attractiveness score is calculated", page
            )
        self.assertNotIn("p1", str(payload["axes"]["symmetry"]))

    def test_product_route_keeps_detected_off_angle_candidate(self):
        from local_app.analysis import SetMeasurementAdapter, PhotoInput
        from local_app.photos import PhotoEntry
        from pathlib import Path

        photo = PhotoInput(
            PhotoEntry(
                "p1", "one.jpg", "one.jpg", "a" * 64, "jpeg", 600, 800, 100, 100
            ),
            Path("/unused/one.jpg"),
        )
        rejected = {
            **self.photo,
            "photo": "one.jpg",
            "code": "yaw_out_of_range",
            "fact": {
                "yaw": 20.0,
                "pitch": 0.0,
                "roll": 0.0,
                "ipd_px": 100.0,
                "neutral": False,
                "detected": True,
            },
        }
        adapter = SetMeasurementAdapter()
        with (
            mock.patch.object(assess_set, "gate_photos", return_value=([], [rejected])),
            mock.patch.object(assess_set.BD, "metrics_for", return_value=self.values),
            mock.patch.object(
                assess_set, "cohort_pool", return_value=np.arange(100.0) / 10
            ),
        ):
            route = adapter.route((photo,), self.cohort)
            self.assertEqual(route["symmetry"], ("p1",))
            result = adapter.measure_axis("symmetry", (photo,), self.cohort)
        self.assertEqual(result.status, "estimated")

    def test_retain_candidates_does_not_change_strict_detector_default(self):
        landmark = np.arange(936.0).reshape(468, 2)
        detector = mock.Mock()
        with (
            mock.patch.object(assess_set.A2, "make_landmarker", return_value=detector),
            mock.patch.object(
                assess_set.AB, "posture_ok", return_value=(True, "neutral")
            ),
            mock.patch.object(
                assess_set.A2,
                "landmarks_and_expression",
                return_value=(landmark, (0, 25, 0), {}),
            ),
        ):
            strict, rejected = assess_set.gate_photos(["photo.jpg"])
            candidates, estimates = assess_set.gate_photos(
                ["photo.jpg"], retain_estimates=True
            )
        self.assertEqual(strict, [])
        self.assertNotIn("landmarks", rejected[0])
        self.assertIn("landmarks", estimates[0])
        self.assertTrue(estimates[0]["fact"]["neutral"])
        self.assertIsNone(rejected[0]["fact"]["neutral"])
        self.assertEqual(candidates, [])

    def test_missing_features_are_a_photo_issue_not_a_reference_error(self):
        with mock.patch.object(
            assess_set,
            "dimorphism",
            return_value={
                "refused": "missing",
                "cause": "missing_measurement",
                "features_used": ["jaw"],
            },
        ):
            result = self.estimate("dimorphism")
        self.assertEqual(result.status, "needs_photo")
        self.assertEqual(result.details["missing_features"], ["jaw"])

    def test_dimorphism_and_typicality_estimates_drop_inapplicable_intervals(self):
        for axis, raw in [
            ("dimorphism", {"percentile": 65.0, "alignment_probability": 0.7}),
            ("averageness", {"percentile": 55.0, "interval": [45.0, 65.0]}),
        ]:
            with (
                self.subTest(axis=axis),
                mock.patch.object(assess_set, axis, return_value=raw),
            ):
                result = self.estimate(axis)
            self.assertEqual(result.status, "estimated")
            self.assertIsNone(result.interval)
            self.assertEqual(result.percentile, raw["percentile"])
        with mock.patch.object(assess_set, "averageness", side_effect=OSError()):
            with self.assertRaises(assess_set.ReferenceDataError):
                self.estimate("averageness")

    def test_regional_export_redacts_paths_and_discards_invalid_percentiles(self):
        from local_app.product_report import public_regions
        from local_app.report import render_json_export

        fixture = test_render.ResultRenderTests()
        fixture.setUp()
        raw = [
            {
                "id": "ratio",
                "label": "/private/person/report",
                "value": 1.2,
                "unit": "ratio",
                "percentile": float("nan"),
                "note": r"C:\private\photo.jpg",
            }
        ]
        axes = dict(fixture.report.axes)
        axes["harmony"] = replace(
            axes["harmony"],
            details={**axes["harmony"].details, "regional_measurements": raw},
        )
        regions = public_regions(axes)
        self.assertIsNone(regions[0]["percentile"])
        self.assertEqual(regions[0]["label"], "[local path redacted]")
        self.assertEqual(regions[0]["note"], "[local path redacted]")
        self.assertNotIn(b"private", render_json_export({"regions": regions}))

    def test_four_good_photos_retain_raw_harmony_if_calibration_is_missing(self):
        from local_app.analysis import SetMeasurementAdapter

        context = assess_set.AssessmentContext(
            self.cohort,
            tuple({**self.photo, "photo": f"p{i}"} for i in range(4)),
            (self.values,) * 4,
            tuple(PhotoFact(f"p{i}", 0.0, 0.0, 0.0, 200.0, True) for i in range(4)),
        )
        adapter = SetMeasurementAdapter()
        photos = tuple(mock.Mock(entry=mock.Mock(photo_id=f"p{i}")) for i in range(4))
        ids = tuple(f"p{i}" for i in range(4))
        adapter._contexts[("harmony", self.cohort.used_id, ids)] = context
        with mock.patch.object(assess_set, "cohort_pool", return_value=None):
            result = adapter.measure_axis("harmony", photos, self.cohort)
        self.assertEqual(result.status, "estimated")
        self.assertEqual(result.details["profile"]["horizontal_balance"]["value"], 0.1)

    def test_same_symmetry_value_uses_same_rank_across_photo_count_transition(self):
        from local_app.estimates import _symmetry_pool, estimate_axis

        # The public ECDF was already gated during export; photo count cannot
        # substitute an ungated reference population.
        with mock.patch("public_reference.pool", return_value=np.array([1., 3.])) as reference:
            pool = _symmetry_pool(assess_set, "m_test")
            self.assertEqual(pool.tolist(), [1., 3.])
            for count in (1, 4):
                ids = tuple(f"p{i}" for i in range(count))
                context = assess_set.AssessmentContext(
                    self.cohort,
                    tuple({**self.photo, "photo": i} for i in ids),
                    tuple(dict(self.values) for _ in ids),
                    tuple(replace(self.fact, photo_id=i) for i in ids),
                )
                estimate = estimate_axis("symmetry", context, ids)
                self.assertEqual(estimate.percentile, 50.)
                self.assertEqual(estimate.percentile,
                                 100.-100.*float(np.mean(pool < estimate.value)))
            reference.assert_any_call("m_test", "_symmetry")


if __name__ == "__main__":
    unittest.main()
