"""Region estimates may use observed eye geometry without admitting mouth posture."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from local_app.estimates import regional_measurements
from local_app.guidance import PhotoFact


def context(settings=None):
    settings = settings or [{}, {}]
    photos, values, facts = [], [], []
    for index, changes in enumerate(settings, 1):
        expression = dict(mouthPucker=0.6, mouthSmileLeft=0.0,
                          mouthSmileRight=0.0, jawOpen=0.0,
                          eyeBlinkLeft=0.0, eyeBlinkRight=0.0,
                          eyeSquintLeft=0.0, eyeSquintRight=0.0)
        expression.update(changes.get("expression", {}))
        photo = dict(photo=f"p{index}", landmarks=np.zeros((468, 2)),
                     expression=expression)
        photos.append(photo)
        values.append(dict(eye_spacing=0.2 * index, naso_oral=0.4 * index))
        facts.append(PhotoFact(f"p{index}", changes.get("yaw", 0.0), 0.0, 0.0,
                               changes.get("ipd", 160.0), changes.get("neutral", False)))
    return SimpleNamespace(kept=photos, values=values, photo_facts=facts,
                           reference_id="test")


class RegionalEvidenceTests(unittest.TestCase):
    def rows(self, sample):
        with patch("local_app.estimates._pool", return_value=None):
            return {r["id"]: r for r in regional_measurements(sample, ("p1",))}

    def test_eye_estimate_pools_pucker_only_frames_but_mouth_stays_on_primary(self):
        rows = self.rows(context())
        self.assertAlmostEqual(rows["eye_spacing"]["value"], 0.3)
        self.assertEqual(rows["eye_spacing"]["evidence_photo_count"], 2)
        self.assertEqual(rows["eye_spacing"]["method"], "automatic-regional-v2")
        self.assertEqual(rows["naso_oral"]["value"], 0.4)
        self.assertEqual(rows["naso_oral"]["evidence_photo_count"], 1)
        self.assertIn("estimated", rows["eye_spacing"]["note"].lower())

    def test_strict_primary_never_mixed_with_expression_flagged_frames(self):
        rows = self.rows(context([{"neutral": True}, {}]))
        self.assertEqual(rows["eye_spacing"]["value"], 0.2)
        self.assertEqual(rows["eye_spacing"]["evidence_photo_count"], 1)

    def test_pose_smile_jaw_blink_and_unknown_scores_do_not_expand_pool(self):
        bad_settings = [
            {"yaw": 25.0}, {"expression": {"mouthSmileLeft": 0.5}},
            {"expression": {"jawOpen": 0.5}},
            {"expression": {"eyeBlinkLeft": 0.8}},
            {"expression": {"eyeSquintRight": 0.8}},
            {"expression": {"mouthPucker": float("nan")}},
            {"expression": {"mouthPucker": -1.0}},
            {"expression": {"jawOpen": None}},
        ]
        for bad in bad_settings:
            with self.subTest(bad=bad):
                rows = self.rows(context([{}, bad]))
                self.assertEqual(rows["eye_spacing"]["value"], 0.2)
                self.assertEqual(rows["eye_spacing"]["evidence_photo_count"], 1)

    def test_unknown_expression_keys_fail_closed(self):
        sample = context()
        sample.kept[1]["expression"] = {}
        self.assertEqual(self.rows(sample)["eye_spacing"]["value"], 0.2)

    def test_empty_primary_does_not_create_unrouted_measurements(self):
        with patch("local_app.estimates._pool", return_value=None):
            self.assertEqual(regional_measurements(context(), ()), [])


class RegionalRoutingTests(unittest.TestCase):
    def test_extra_eye_evidence_enters_cache_route_without_changing_axis_inputs(self):
        import assess_set
        from tests.test_estimate_evidence import EstimateEvidenceTests
        from local_app.analysis import SetMeasurementAdapter

        fixture = EstimateEvidenceTests()
        fixture.setUp()
        photos, records = fixture.inputs([
            {"ipd_px": 160.0, "neutral": False},
            {"ipd_px": 160.0, "neutral": False},
        ])
        for record in records:
            record["expression"] = context().kept[0]["expression"]
        adapter = SetMeasurementAdapter()
        with (
            patch.object(assess_set, "gate_photos", return_value=(records, [])),
            patch.object(assess_set.BD, "metrics_for", side_effect=lambda p: {
                "eye_spacing": float(p[0, 0]) * 0.2,
                "harm__horizontal_balance": float(p[0, 0]),
            }),
            patch("local_app.estimates._pool", return_value=None),
        ):
            routes = adapter.route(photos, fixture.cohort)
            result = adapter.measure_axis("harmony", photos, fixture.cohort)
        self.assertEqual(routes["harmony"], ("p1", "p2"))
        self.assertEqual(routes["symmetry"], ("p1",))
        self.assertEqual(result.status, "estimated")
        self.assertEqual(result.photo_count, 1)
        self.assertEqual(result.details["profile"]["horizontal_balance"]["value"], 1.0)
        eye = next(r for r in result.details["regional_measurements"] if r["id"] == "eye_spacing")
        self.assertEqual(eye["value"], 0.3)
        self.assertEqual(eye["evidence_photo_count"], 2)
        self.assertEqual(result.details["capture_diagnostics"]["selected_axis_photos"], 1)

    def test_aggregate_evidence_survives_exports_without_private_identifiers(self):
        from dataclasses import replace
        from local_app.product_report import public_regions
        from local_app.report import build_export_payload, render_html_export, public_capture_summary
        from local_app.contracts import Demographics
        from tests.test_consumer_evaluation import report

        sample = report()
        harmony = sample.axes["harmony"]
        summary = dict(submitted_photos=15, detected_photos=12,
                       eligible_photos=0, selected_axis_photos=1,
                       failure_counts={"yaw": 10, "expression": 3, "/private/photo": 1})
        row = dict(id="eye_spacing", label="Eye spacing", value=0.3, unit="ratio",
                   percentile=None, note="Estimated regional geometry", method="automatic-regional-v2",
                   evidence_photo_count=2, photo_id="PRIVATE-ID")
        sample = replace(sample, axes={**sample.axes, "harmony": replace(harmony, details={
            **harmony.details, "capture_diagnostics": summary, "regional_measurements": [row]
        })})
        payload = build_export_payload(sample, Demographics(25, "male", ("white",)), source_inventory={})
        self.assertEqual(payload["regional_measurements"][0]["evidence_photo_count"], 2)
        self.assertEqual(public_regions(sample.axes)[0]["method"], "automatic-regional-v2")
        page = render_html_export(payload)
        self.assertIn("Head turn: 10", page)
        self.assertIn("2 photo(s)", page)
        self.assertNotIn("PRIVATE-ID", str(payload))
        self.assertNotIn("/private/photo", str(payload))
        self.assertIsNone(public_capture_summary({**summary, "selected_axis_photos": True}))
        self.assertIsNone(public_capture_summary({**summary, "detected_photos": 16}))


class CaptureCacheTests(unittest.TestCase):
    def test_unrouted_add_refreshes_diagnostics_without_remeasuring(self):
        import tempfile
        from pathlib import Path
        from local_app.analysis import ProgressiveAnalysisService, SetMeasurementAdapter
        from local_app.contracts import Demographics
        from local_app.photos import PhotoCatalog
        from local_app.sessions import SessionStore
        from tests.test_progressive_analysis import FakeMeasurementAdapter, FixedResolver, jpeg_bytes
        from tests.test_estimate_evidence import EstimateEvidenceTests

        class Adapter(FakeMeasurementAdapter):
            refresh_capture_diagnostics = SetMeasurementAdapter.refresh_capture_diagnostics

            def route(self, photos, cohort):
                self.total = len(photos)
                self.bad = sum(p.entry.display_name.startswith("unusable") for p in photos)
                return super().route(photos, cohort)

            def routing_diagnostics(self):
                return {axis: dict(submitted_photos=self.total,
                    detected_photos=self.total-self.bad, eligible_photos=self.total-self.bad,
                    failure_counts={"face_not_detected": self.bad} if self.bad else {})
                    for axis in self.percentiles}

            def measure_axis(self, axis, photos, cohort):
                return self.refresh_capture_diagnostics(super().measure_axis(axis, photos, cohort))

        fixture = EstimateEvidenceTests()
        fixture.setUp()
        with tempfile.TemporaryDirectory() as temporary:
            store = SessionStore(Path(temporary))
            session = store.create(Demographics(25, "male", ("white",)))
            catalog = PhotoCatalog(store, session.session_id)
            catalog.add_bytes("good.jpg", jpeg_bytes(10))
            adapter = Adapter()
            service = ProgressiveAnalysisService(store, adapter=adapter, resolver=FixedResolver(fixture.cohort))
            first = service.analyze(session.session_id)
            catalog.add_bytes("unusable.jpg", jpeg_bytes(20))
            second = service.analyze(session.session_id)
            self.assertEqual(second.axes["harmony"].value, first.axes["harmony"].value)
            self.assertEqual(adapter.measure_calls["harmony"], 1)
            summary = second.axes["harmony"].details["capture_diagnostics"]
            self.assertEqual(summary["submitted_photos"], 2)
            self.assertEqual(summary["failure_counts"], {"face_not_detected": 1})
            store.cleanup_all()

    def test_malformed_method_is_incompatible_and_print_counts_stay_open(self):
        from dataclasses import replace
        from local_app.product_report import public_regions
        from local_app.report import render_capture_summary
        from tests.test_consumer_evaluation import report

        sample = report()
        for method in ([], {}, 3, None):
            row = dict(id="eye_spacing", label="Eye spacing", value=0.3, unit="ratio", method=method)
            axes = {**sample.axes, "harmony": replace(sample.axes["harmony"], details={**sample.axes["harmony"].details, "regional_measurements": [row]})}
            self.assertEqual(public_regions(axes)[0]["method"], "incompatible")
        summary = dict(submitted_photos=2, detected_photos=1, eligible_photos=1,
                       selected_axis_photos=1, failure_counts={"face_not_detected": 1})
        self.assertIn('class="capture-summary" open', render_capture_summary(summary, expanded=True))
        self.assertNotIn('class="capture-summary" open', render_capture_summary(summary))


if __name__ == "__main__":
    unittest.main()
