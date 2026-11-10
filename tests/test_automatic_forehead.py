"""Automatic hair overlap is local, photo-bound and never a hidden hairline guess."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np


def face_points():
    points = np.full((468, 2), (100., 130.))
    for index, point in {10: (100, 40), 54: (45, 65), 284: (155, 65),
                         70: (60, 110), 300: (140, 110), 55: (80, 110),
                         285: (120, 110), 33: (55, 125), 263: (145, 125),
                         152: (100, 200)}.items():
        points[index] = point
    return points


def masks(category):
    result = np.zeros((6, 240, 200), dtype=np.float32)
    result[category] = 1
    return result


class AutomaticForeheadTests(unittest.TestCase):
    def test_actual_confidence_mask_channel_shape_and_cropped_forehead_are_handled(self):
        from local_app.automatic_forehead import classify_forehead

        self.assertEqual(classify_forehead(masks(3)[..., None], face_points())["status"], "forehead_clear")
        shifted = face_points() - (100., 0.)
        self.assertEqual(classify_forehead(masks(3), shifted)["reason"], "geometry_unavailable")

    def test_model_failure_keeps_no_face_distinct_and_does_not_claim_clear_skin(self):
        from types import SimpleNamespace
        from local_app.automatic_forehead import observe_photos

        photos = [SimpleNamespace(entry=SimpleNamespace(photo_id=key)) for key in ("detected", "missing")]
        with patch("local_app.automatic_forehead.ForeheadSegmenter.__enter__", side_effect=ValueError("bad model")):
            findings = observe_photos(photos, [{"photo": "detected"}])
        self.assertEqual([r["status"] for r in findings], ["unavailable", "no_face"])
        self.assertTrue(all("hair_fraction" not in r for r in findings))

    def test_missing_local_model_never_downloads_during_inference(self):
        from pathlib import Path
        from local_app.automatic_forehead import ForeheadSegmenter

        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(ValueError, "missing"):
                with ForeheadSegmenter():
                    self.fail("A missing model cannot start segmentation")

    def test_hair_and_bare_skin_are_classified_without_manual_review(self):
        from local_app.automatic_forehead import classify_forehead

        self.assertEqual(classify_forehead(masks(1), face_points())["status"], "hair_overlap")
        clear = classify_forehead(masks(3), face_points())
        self.assertEqual(clear["status"], "forehead_clear")
        self.assertNotIn("hairline_point", clear)
        self.assertEqual(clear["skin_fraction"], 1.)
        self.assertEqual(clear["hair_fraction"], 0.)

    def test_hat_uncertain_nonfinite_or_malformed_masks_are_not_declared_clear(self):
        from local_app.automatic_forehead import classify_forehead

        self.assertEqual(classify_forehead(masks(5), face_points())["status"], "uncertain")
        self.assertEqual(classify_forehead(masks(3) / 6, face_points())["status"], "uncertain")
        invalid = masks(3)
        invalid[1, 70, 80] = np.nan
        self.assertEqual(classify_forehead(invalid, face_points())["status"], "uncertain")
        self.assertEqual(classify_forehead(masks(3)[:2], face_points())["status"], "uncertain")

    def test_forehead_mask_uses_head_geometry_not_hair_at_sides_or_background(self):
        from local_app.automatic_forehead import classify_forehead

        values = masks(3)
        values[:, :, :15] = 0
        values[1, :, :15] = 1
        self.assertEqual(classify_forehead(values, face_points())["status"], "forehead_clear")
        values[:, 40:110, 60:140] = 0
        values[1, 40:110, 60:140] = 1
        self.assertEqual(classify_forehead(values, face_points())["status"], "hair_overlap")

    def test_too_small_or_invalid_geometry_cannot_certify_visibility(self):
        from local_app.automatic_forehead import classify_forehead

        self.assertEqual(classify_forehead(masks(3), face_points() / 10)["status"], "uncertain")
        points = face_points()
        points[10] = np.nan
        self.assertEqual(classify_forehead(masks(3), points)["status"], "uncertain")

    def test_public_output_is_canonical_and_does_not_expose_photo_ids_or_paths(self):
        from local_app.automatic_forehead import public_forehead
        from tests.test_consumer_evaluation import report

        sample = report()
        raw = dict(photo_id="PRIVATE-ID", photo_number=1, status="hair_overlap",
                   reason="/private/name", hair_fraction=.6, skin_fraction=.3,
                   method="automatic-forehead-v1", filename="PRIVATE-NAME.jpg")
        axes = {**sample.axes, "harmony": replace(sample.axes["harmony"], details={
            **sample.axes["harmony"].details, "automatic_forehead": [raw]})}
        output = public_forehead(axes)
        self.assertEqual(output[0]["status"], "hair_overlap")
        self.assertIn("hair", output[0]["explanation"].lower())
        for private in ("PRIVATE-ID", "PRIVATE-NAME", "/private/name"):
            self.assertNotIn(private, str(output))

    def test_default_filter_uses_automatic_coverage_and_optional_manual_correction(self):
        from local_app.product_report import public_regions
        from tests.test_consumer_evaluation import report

        sample = report()
        raw = dict(photo_id="p1", photo_number=1, status="hair_overlap")
        rows = [dict(id="facial_index", value=1.2, evidence_photo_ids=["p1"]),
                dict(id="eye_spacing", value=.3, evidence_photo_ids=["p1"])]
        axes = {**sample.axes, "harmony": replace(sample.axes["harmony"], details={
            **sample.axes["harmony"].details, "regional_measurements": rows,
            "automatic_forehead": [raw]})}
        self.assertEqual({r["id"] for r in public_regions(axes)}, {"eye_spacing"})
        manual = dict(evidence_photo_id="p1", forehead_visibility="visible")
        self.assertEqual(len(public_regions(axes, frontal=[manual])), 2)
        self.assertEqual(len(public_regions(axes, frontal=[{**manual, "visibility_only": True}])), 1)

    def test_regional_measurement_uses_clear_photos_instead_of_discarding_the_whole_set(self):
        from local_app.estimates import regional_measurements
        from tests.test_regional_evidence import context

        sample = context([{"neutral": True}, {"neutral": True}])
        sample.values = [{"facial_index": 1.2}, {"facial_index": 1.8}]
        with patch("local_app.estimates._pool", return_value=None):
            rows = regional_measurements(sample, ("p1", "p2"), forehead_clear_ids={"p2"})
        row = next(r for r in rows if r["id"] == "facial_index")
        self.assertEqual(row["value"], 1.8)
        self.assertEqual(row["evidence_photo_count"], 1)
        self.assertEqual(row["evidence_photo_ids"], ["p2"])

    def test_optional_editor_uses_automatic_exclusion_without_a_manual_choice(self):
        from local_app.landmark_review import save_review, public_review
        from tests.test_frontal_review import example_points

        record = dict(points=example_points(), automatic=example_points(), revision=1,
                      binding="test", automatic_forehead_status="hair_overlap")
        saved = save_review(record, record["points"], expected_revision=1, binding="test")
        payload = public_review(saved, 400, 600)
        self.assertEqual(payload["forehead_visibility"], "unassessed")
        self.assertEqual(payload["automatic_forehead_status"], "hair_overlap")
        self.assertIsNone(payload["diagram"]["points"]["mesh_top"])
        self.assertNotIn("reviewed_facial_index", {r["id"] for r in payload["measurements"]})
        self.assertIn("reviewed_eye_spacing", {r["id"] for r in payload["measurements"]})

    def test_detailed_chapters_explain_missing_readings_for_each_region(self):
        from local_app.consultation import consultation_payload, render_consultation

        payload = consultation_payload([])
        self.assertEqual(len(payload["chapters"]), 6)
        self.assertTrue(all(c["unavailable_measurements"] for c in payload["chapters"]))
        self.assertIn("Not available", render_consultation(payload))

    def test_adapter_rebuilds_actual_medians_after_confirmed_visibility_corrections(self):
        import assess_set
        from local_app.analysis import SetMeasurementAdapter
        from local_app.product_report import public_regions
        from tests.test_estimate_evidence import EstimateEvidenceTests

        fixture = EstimateEvidenceTests()
        fixture.setUp()
        photos, records = fixture.inputs([{"ipd_px": 150.}, {"ipd_px": 150.}])
        findings = [dict(photo_id="p1", photo_number=1, status="hair_overlap"),
                    dict(photo_id="p2", photo_number=2, status="forehead_clear")]
        adapter = SetMeasurementAdapter()
        with (patch.object(assess_set, "gate_photos", return_value=(records, [])),
              patch.object(assess_set.BD, "metrics_for", side_effect=lambda p: {
                  "facial_index": float(p[0, 0]), "harm__horizontal_balance": .1}),
              patch("local_app.automatic_forehead.observe_photos", return_value=findings),
              patch("local_app.estimates._pool", return_value=None),
              patch("local_app.anatomical_observations.observed_iris_rows", return_value=[])):
            adapter.set_forehead_overrides({"p1": "visible"})
            adapter.route(photos, fixture.cohort)
            result = adapter.measure_axis("harmony", photos, fixture.cohort)
            row = next(r for r in public_regions({"harmony": result}) if r["id"] == "facial_index")
            self.assertEqual(row["value"], 1.5)
            self.assertEqual(row["evidence_photo_count"], 2)
            adapter.set_forehead_overrides({"p1": "visible", "p2": "forehead_hidden"})
            adapter.route(photos, fixture.cohort)
            result = adapter.measure_axis("harmony", photos, fixture.cohort)
            row = next(r for r in public_regions({"harmony": result}) if r["id"] == "facial_index")
            self.assertEqual(row["value"], 1.)
            self.assertEqual(row["evidence_photo_count"], 1)

    def test_cached_service_remeasures_when_a_photo_bound_visibility_override_changes(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import assess_set
        from local_app.analysis import ProgressiveAnalysisService, SetMeasurementAdapter
        from local_app.contracts import Demographics
        from local_app.photos import PhotoCatalog
        from local_app.sessions import SessionStore
        from local_app.frontal_routes import _record
        from local_app.landmark_review import save_review
        from local_app.product_report import public_regions
        from tests.test_server import FakeAdapter, FakeResolver, jpeg_bytes
        from tests.test_frontal_review import example_points

        class Adapter(SetMeasurementAdapter):
            def measure_axis(self, axis, photos, cohort):
                if axis == "harmony":
                    return super().measure_axis(axis, photos, cohort)
                return FakeAdapter.measure_axis(self, axis, photos, cohort)

        def gate(paths, **_kwargs):
            records = [dict(photo=Path(path).name, landmarks=np.full((468, 2), i),
                            yaw=0., pitch=0., roll=0., ipd_px=150.,
                            fact=dict(yaw=0., pitch=0., roll=0., ipd_px=150.,
                                      neutral=True, detected=True))
                       for i, path in enumerate(paths, 1)]
            return records, []

        def findings(photos, _records):
            return [dict(photo_id=p.entry.photo_id, photo_number=i,
                         status="hair_overlap" if i == 1 else "forehead_clear")
                    for i, p in enumerate(photos, 1)]

        with (TemporaryDirectory() as td,
              patch.object(assess_set, "gate_photos", side_effect=gate) as detection,
              patch.object(assess_set.BD, "metrics_for", side_effect=lambda p: {
                  "facial_index": float(p[0, 0]), "harm__horizontal_balance": .1}),
              patch("local_app.automatic_forehead.observe_photos", side_effect=findings),
              patch("local_app.frontal_routes.detected_points", return_value=example_points()),
              patch("local_app.estimates._pool", return_value=None),
              patch("local_app.anatomical_observations.observed_iris_rows", return_value=[])):
            store = SessionStore(Path(td))
            session = store.create(Demographics(25, "male", ("white",)))
            catalog = PhotoCatalog(store, session.session_id)
            first = catalog.add_bytes("one.jpg", jpeg_bytes(90))
            catalog.add_bytes("two.jpg", jpeg_bytes(100))
            service = ProgressiveAnalysisService(store, adapter=Adapter(), resolver=FakeResolver())
            before = service.analyze(session.session_id)
            self.assertEqual(next(r for r in public_regions(before.axes) if r["id"] == "facial_index")["value"], 2.)
            current = store.load(session.session_id)
            record = _record(current, store, first)
            current.state["frontal_reviews"][first.photo_id] = save_review(
                record, record["automatic"], expected_revision=record["revision"],
                binding=record["binding"], forehead_visibility="visible")
            store.save(current)
            after = service.analyze(session.session_id)
            self.assertEqual(next(r for r in public_regions(after.axes) if r["id"] == "facial_index")["value"], 1.5)
            calls = detection.call_count
            self.assertEqual(service.analyze(session.session_id), after)
            self.assertEqual(detection.call_count, calls)


if __name__ == "__main__":
    unittest.main()
