import html
import http.client
import json
from pathlib import Path
import socket
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from local_app.analysis import AnalysisReport, ProgressiveAnalysisService
from local_app.contracts import AxisResult, CohortUse, Demographics
from local_app.photos import PhotoCatalog
from local_app.report import (
    build_export_payload,
    render_html_export,
    render_json_export,
    write_export,
)
from local_app.server import make_server
from local_app.sessions import SessionStore

AXES = ("harmony", "dimorphism", "symmetry", "averageness")


def jpeg_bytes(value: int) -> bytes:
    image = np.full((20, 24, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("test JPEG encoding failed")
    return encoded.tobytes()


class FixedResolver:
    def __init__(self, cohort: CohortUse) -> None:
        self.cohort = cohort

    def resolve(self, _demographics: Demographics) -> CohortUse:
        return self.cohort


class OfflineAdapter:
    def fingerprint(self) -> str:
        return "offline-test-instrument-v1"

    def route(self, photos, _cohort):
        identifiers = tuple(photo.entry.photo_id for photo in photos)
        return {axis: identifiers for axis in AXES}

    def measure_axis(self, axis, photos, cohort):
        return AxisResult.measured(
            axis=axis,
            summary=f"{axis} measured offline",
            value=0.75,
            unit="test units",
            percentile=63.0,
            interval=(55.0, 71.0),
            cohort=cohort,
            photo_count=len(photos),
        )


class PrivacyAndExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base = Path(self.temp_dir.name)
        self.demographics = Demographics(
            age=31,
            sex="female",
            ethnicities=("east_asian", "white"),
        )
        self.cohort = CohortUse(
            requested_id="f_east_asian+white_31",
            used_id="f_all_ethnicities_30_39",
            fallback_level="same_age_sex_broader_ethnicity",
            sample_size=1_204,
            requested_label="East Asian and White women aged 31",
            used_label="Women of all available ethnicities aged 30-39",
            member_ids=("f_white_30_39", "f_east_asian_30_39"),
        )
        private_root = self.base / "sessions" / "private-session"
        private_photo = private_root / "uploads" / "original-portrait.jpg"
        axes = {
            axis: AxisResult.measured(
                axis=axis,
                summary=f"{axis.title()} was measured.",
                value=0.75,
                unit="test units",
                percentile=63.0,
                interval=(55.0, 71.0),
                cohort=self.cohort,
                photo_count=7,
                details={
                    "source_path": str(private_photo),
                    "display_name": private_photo.name,
                    "sha256": "a" * 64,
                    "evidence_photo_ids": ["private-photo-id"],
                    "profile": {
                        "horizontal_balance": {
                            "label": "Horizontal balance",
                            "value": 0.97,
                            "percentile": 57.0,
                            "interval": [49.0, 65.0],
                        }
                    },
                    **(
                        {
                            "overall_index": {
                                "value": 37.52,
                                "debug_path": str(private_photo),
                            }
                        }
                        if axis == "harmony"
                        else {}
                    ),
                },
            )
            for axis in AXES
        }
        axes["harmony"] = AxisResult.measured_profile(
            axis="harmony",
            summary="Geometric harmony index: 38/100.",
            cohort=self.cohort,
            photo_count=7,
            details={
                "source_path": str(private_photo),
                "display_name": private_photo.name,
                "evidence_photo_ids": ["private-photo-id"],
                "profile": {
                    "horizontal_balance": {
                        "label": "Horizontal balance",
                        "value": 0.97,
                        "percentile": 57.0,
                        "interval": [49.0, 65.0],
                    }
                },
                # A stale or injected report-only scalar must never reach exports.
                "overall_index": {
                    "value": 37.52,
                    "debug_path": str(private_photo),
                },
            },
        )
        self.report = AnalysisReport(
            session_id="private-session-id-abcdefghijklmnopqrstuvwxyz",
            cohort=self.cohort,
            axes=axes,
            accepted_photos=9,
            instrument_fingerprint="instrument-version-2026-09-18",
            photo_fingerprint="b" * 64,
        )
        self.inventory = {
            "f_white_30_39": {
                "source": "FairFace",
                "source_label_kind": "estimated then human-checked",
            },
            "f_east_asian_30_39": {
                "source": "FairFace",
                "source_label_kind": "estimated then human-checked",
            },
        }

    def test_export_is_complete_but_contains_no_biometric_or_local_identifiers(
        self,
    ) -> None:
        payload = build_export_payload(
            self.report,
            self.demographics,
            source_inventory=self.inventory,
        )

        self.assertEqual(payload["demographics"]["age"], 31)
        self.assertEqual(payload["demographics"]["sex"], "female")
        self.assertEqual(payload["accepted_photo_count"], 9)
        self.assertEqual(payload["instrument_version"], "instrument-version-2026-09-18")
        comparison = payload["comparison"]
        self.assertEqual(comparison["requested"]["id"], self.cohort.requested_id)
        self.assertEqual(comparison["used"]["id"], self.cohort.used_id)
        self.assertEqual(comparison["fallback_reason"], self.cohort.fallback_level)
        self.assertEqual(comparison["sample_size"], 1_204)
        self.assertEqual(
            comparison["reference_sources"][0]["label_meaning"],
            "estimated then human-checked",
        )
        for axis in AXES:
            exported = payload["axes"][axis]
            self.assertEqual(exported["status"], "measured")
            self.assertEqual(exported["evidence_photo_count"], 7)
            self.assertEqual(exported["comparison"]["sample_size"], 1_204)
            if axis == "harmony":
                self.assertFalse(exported["headline_available"])
                self.assertNotIn("uncertainty_percentile", exported)
            else:
                self.assertEqual(exported["uncertainty_percentile"], [55.0, 71.0])

        self.assertNotIn("overall_index", payload["axes"]["harmony"])
        self.assertNotIn("38/100", payload["axes"]["harmony"]["summary"])
        self.assertNotIn(
            "Geometric harmony index",
            payload["axes"]["harmony"]["summary"],
        )

        serialized = render_json_export(payload).decode("utf-8")
        for forbidden in (
            "private-session",
            "original-portrait.jpg",
            str(self.base),
            "source_path",
            "display_name",
            "sha256",
            "evidence_photo_ids",
            "private-photo-id",
            "csrf",
            "photo_fingerprint",
            "debug_path",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)

    def test_printable_html_is_self_contained_and_escapes_report_text(self) -> None:
        attack = "<script>alert(1)</script>"
        axes = dict(self.report.axes)
        axes["symmetry"] = AxisResult.measured(
            axis="symmetry",
            summary=attack,
            value=0.75,
            unit="test units",
            percentile=63.0,
            interval=(55.0, 71.0),
            cohort=self.cohort,
            photo_count=7,
        )
        hostile = AnalysisReport(
            session_id=self.report.session_id,
            cohort=self.cohort,
            axes=axes,
            accepted_photos=9,
            instrument_fingerprint=self.report.instrument_fingerprint,
            photo_fingerprint=self.report.photo_fingerprint,
        )
        payload = build_export_payload(
            hostile,
            self.demographics,
            source_inventory=self.inventory,
        )

        page = render_html_export(payload)

        self.assertIn("<!doctype html>", page.lower())
        self.assertIn("@media print", page)
        self.assertIn("63rd percentile", page)
        self.assertNotIn("63.0th percentile", page)
        for text in (
            "Four-pillar facial analysis",
            "Your results at a glance",
            "Harmony profile",
            "Proportion balance contributes to the overall score",
            "How to read this report",
            "Method and limits",
        ):
            with self.subTest(text=text):
                self.assertIn(text, page)
        self.assertEqual(page.count("1,204 reference identities"), 1)
        self.assertIn("@media screen and (max-width: 720px)", page)
        self.assertNotIn("38 / 100", page)
        self.assertNotIn("38/100", page)
        self.assertNotIn("Geometric harmony index", page)
        self.assertIn("available component proportions", page)
        self.assertNotIn("six measured proportions", page)
        self.assertIn("only nose–mouth–eye", page.lower())
        self.assertIn("60 Chinese volunteers", page)
        self.assertIn(html.escape(attack), page)
        self.assertNotIn(attack, page)
        self.assertNotIn("<script", page.lower())
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)

    def test_printable_method_matches_validated_harmony_scalar(self) -> None:
        axes = dict(self.report.axes)
        axes["harmony"] = AxisResult.measured(
            axis="harmony",
            summary="Validated geometric harmony is at the 72nd percentile.",
            value=72.0,
            unit="geometric harmony percentile",
            percentile=72.0,
            interval=(64.0, 80.0),
            cohort=self.cohort,
            photo_count=7,
            details=axes["harmony"].details,
        )
        report = AnalysisReport(
            session_id=self.report.session_id,
            cohort=self.report.cohort,
            axes=axes,
            accepted_photos=self.report.accepted_photos,
            instrument_fingerprint=self.report.instrument_fingerprint,
            photo_fingerprint=self.report.photo_fingerprint,
        )

        page = render_html_export(
            build_export_payload(
                report,
                self.demographics,
                source_inventory=self.inventory,
            )
        )

        self.assertIn("validated geometric measure", page)
        self.assertNotIn("Why harmony has no overall score", page)
        self.assertNotIn(
            "validation gate for a single harmony scalar has not passed", page
        )

    def test_printable_method_does_not_claim_profile_when_harmony_unavailable(
        self,
    ) -> None:
        axes = dict(self.report.axes)
        axes["harmony"] = AxisResult.processing_error(
            axis="harmony",
            error_code="measurement_failed",
            error_message="Harmony could not be measured.",
            retryable=True,
            cohort=self.cohort,
            photo_count=7,
        )
        report = AnalysisReport(
            session_id=self.report.session_id,
            cohort=self.report.cohort,
            axes=axes,
            accepted_photos=self.report.accepted_photos,
            instrument_fingerprint=self.report.instrument_fingerprint,
            photo_fingerprint=self.report.photo_fingerprint,
        )

        page = render_html_export(
            build_export_payload(
                report,
                self.demographics,
                source_inventory=self.inventory,
            )
        )

        self.assertIn("A harmony result is not available yet", page)
        self.assertNotIn("available component proportions are reported", page)

    def test_unresolved_averageness_export_leads_with_range(self) -> None:
        axes = dict(self.report.axes)
        axes["averageness"] = AxisResult.measured(
            axis="averageness",
            summary=(
                "Measured range: 15th–37th percentile. "
                "The range crosses a placement boundary, so placement is unresolved."
            ),
            value=25.7,
            unit="typicality percentile",
            percentile=25.7,
            interval=(14.8, 36.7),
            cohort=self.cohort,
            photo_count=5,
            details={
                "placement": "withheld — the interval straddles a placement cutoff"
            },
        )
        report = AnalysisReport(
            session_id=self.report.session_id,
            cohort=self.report.cohort,
            axes=axes,
            accepted_photos=self.report.accepted_photos,
            instrument_fingerprint=self.report.instrument_fingerprint,
            photo_fingerprint=self.report.photo_fingerprint,
        )
        payload = build_export_payload(
            report,
            self.demographics,
            source_inventory=self.inventory,
        )

        exported = payload["axes"]["averageness"]
        self.assertEqual(exported["placement"]["status"], "unresolved")
        page = render_html_export(payload)
        self.assertIn("Estimated range: 15th–37th percentile", page)
        self.assertIn("Placement unresolved", page)
        self.assertIn("Point estimate: 26th percentile", page)

    def test_explicit_file_export_must_be_outside_the_session_workspace(self) -> None:
        session_root = self.base / "sessions" / "one-session"
        session_root.mkdir(parents=True)
        payload = build_export_payload(
            self.report,
            self.demographics,
            source_inventory=self.inventory,
        )

        with self.assertRaisesRegex(ValueError, "outside"):
            write_export(
                payload,
                session_root / "report.json",
                session_root=session_root,
            )

        destination = self.base / "saved" / "report.json"
        written = write_export(payload, destination, session_root=session_root)
        self.assertEqual(written, destination.resolve())
        self.assertEqual(json.loads(destination.read_text()), payload)

    def test_all_session_exit_paths_erase_biometric_material(self) -> None:
        now = [1_700_000_000.0]
        store = SessionStore(
            self.base / "cleanup-sessions",
            ttl_seconds=10,
            clock=lambda: now[0],
        )

        ended = store.create(self.demographics)
        store.path_for(ended.session_id, "uploads", "photo.jpg").write_bytes(b"face")
        store.end(ended.session_id)
        store.end(ended.session_id)
        self.assertFalse(ended.root.exists())

        expired = store.create(self.demographics)
        store.path_for(expired.session_id, "uploads", "photo.jpg").write_bytes(b"face")
        now[0] += 11
        self.assertEqual(store.expire_stale(), [expired.session_id])
        self.assertFalse(expired.root.exists())

        failed = store.create(self.demographics)
        with self.assertRaisesRegex(RuntimeError, "simulated analysis failure"):
            with store.analysis_scope(failed.session_id) as workspace:
                (workspace / "face-template.bin").write_bytes(b"biometric")
                raise RuntimeError("simulated analysis failure")
        self.assertFalse(failed.root.exists())

        shutdown = store.create(self.demographics)
        store.path_for(shutdown.session_id, "uploads", "photo.jpg").write_bytes(b"face")
        self.assertEqual(store.cleanup_all(), 1)
        self.assertFalse(shutdown.root.exists())

    def test_server_close_performs_normal_shutdown_cleanup(self) -> None:
        class UnusedAnalysis:
            pass

        server = make_server(
            base_dir=self.base / "server-sessions",
            port=0,
            analysis_factory=lambda _store: UnusedAnalysis(),
        )
        session = server.store.create(self.demographics)
        server.store.path_for(session.session_id, "uploads", "photo.jpg").write_bytes(
            b"face"
        )

        server.server_close()

        self.assertFalse(session.root.exists())

    def test_each_request_expires_stale_session_material(self) -> None:
        class UnusedAnalysis:
            pass

        server = make_server(
            base_dir=self.base / "expiry-server-sessions",
            port=0,
            analysis_factory=lambda _store: UnusedAnalysis(),
        )
        session = server.store.create(self.demographics)
        server.store.path_for(session.session_id, "uploads", "photo.jpg").write_bytes(
            b"face"
        )
        server.store._clock = (
            lambda: session.last_activity + server.store.ttl_seconds + 1
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            connection.request("GET", "/")
            response = connection.getresponse()
            response.read()
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertFalse(session.root.exists())
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_unhandled_server_analysis_failure_erases_the_session(self) -> None:
        class FailingAnalysis:
            def analyze(self, _session_id, *, force=False):
                raise RuntimeError("simulated analysis failure")

        server = make_server(
            base_dir=self.base / "failure-server-sessions",
            port=0,
            analysis_factory=lambda _store: FailingAnalysis(),
        )
        session = server.store.create(self.demographics)
        server.store.path_for(session.session_id, "uploads", "photo.jpg").write_bytes(
            b"face"
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            connection.request(
                "GET",
                f"/sessions/{session.session_id}/results",
                headers={"Cookie": f"face_session={session.session_id}"},
            )
            response = connection.getresponse()
            response.read()
            set_cookie = response.getheader("Set-Cookie", "")
            connection.close()

            self.assertEqual(response.status, 500)
            self.assertIn("Max-Age=0", set_cookie)
            self.assertFalse(session.root.exists())
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_an_analysis_that_calls_sys_exit_still_answers_500_and_erases_the_session(
        self,
    ) -> None:
        # 2026-09-21: the packaged app's first real upload died exactly this way.
        # A research module reached sys.exit() while importing inside the
        # request thread; SystemExit is not an Exception, so the handler thread
        # ended silently, the browser saw the connection drop with no response,
        # and store.end() never ran.
        class ExitingAnalysis:
            def analyze(self, _session_id, *, force=False):
                raise SystemExit("movement needs at least 2 gate-passing sittings")

        server = make_server(
            base_dir=self.base / "exit-server-sessions",
            port=0,
            analysis_factory=lambda _store: ExitingAnalysis(),
        )
        session = server.store.create(self.demographics)
        server.store.path_for(session.session_id, "uploads", "photo.jpg").write_bytes(
            b"face"
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            connection.request(
                "GET",
                f"/sessions/{session.session_id}/results",
                headers={"Cookie": f"face_session={session.session_id}"},
            )
            response = connection.getresponse()
            body = response.read()
            set_cookie = response.getheader("Set-Cookie", "")
            connection.close()

            self.assertEqual(response.status, 500)
            self.assertIn(b"could not be read", body)
            self.assertIn("Max-Age=0", set_cookie)
            self.assertFalse(session.root.exists())
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_progressive_analysis_completes_with_outbound_network_disabled(
        self,
    ) -> None:
        store = SessionStore(self.base / "offline-sessions")
        session = store.create(self.demographics)
        catalog = PhotoCatalog(store, session.session_id)
        for index in range(7):
            catalog.add_bytes(f"private-{index}.jpg", jpeg_bytes(index * 31))
        service = ProgressiveAnalysisService(
            store,
            adapter=OfflineAdapter(),
            resolver=FixedResolver(self.cohort),
        )

        def network_forbidden(*_args, **_kwargs):
            raise AssertionError("outbound networking is forbidden during analysis")

        with patch.object(
            socket, "create_connection", side_effect=network_forbidden
        ), patch.object(
            socket.socket,
            "connect",
            side_effect=network_forbidden,
        ):
            report = service.analyze(session.session_id)

        self.assertEqual(set(report.axes), set(AXES))
        self.assertTrue(
            all(result.status == "measured" for result in report.axes.values())
        )


if __name__ == "__main__":
    unittest.main()
