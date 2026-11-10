"""Archive authorization is independent from temporary session ownership."""

import http.client
import json
import tempfile
import unittest
from pathlib import Path
from threading import Thread
from local_app.server import make_server
from local_app.analysis import ProgressiveAnalysisService
from tests.test_server import FakeAdapter, FakeResolver, jpeg_bytes
from local_app.photos import PhotoCatalog
from local_app.contracts import Demographics
from tests.test_history import Keys
from local_app.history import ArchiveStore


class HistoryRouteTests(unittest.TestCase):
    def test_restored_hair_coverage_excludes_automatic_readings_without_confirming_points(self):
        from dataclasses import replace
        from unittest.mock import patch
        from local_app.frontal_routes import _record, frontal_payloads
        from local_app.landmark_review import save_review
        from local_app.report import build_export_payload
        from tests.test_frontal_review import example_points

        original = FakeAdapter.measure_axis

        def with_forehead(adapter, axis, photos, cohort):
            result = original(adapter, axis, photos, cohort)
            if axis == "harmony":
                return replace(result, details={**result.details, "regional_measurements": [
                    dict(id="facial_index", value=1.25, label="Mesh ratio", unit="ratio",
                         evidence_photo_ids=[photos[0].entry.photo_id])
                ]})
            return result

        store = self.server.store
        session = store.create(Demographics(25, "male", ("white",)))
        entry = PhotoCatalog(store, session.session_id).add_bytes("source.jpg", jpeg_bytes(90))
        with patch("local_app.frontal_routes.detected_points", return_value=example_points()), patch.object(FakeAdapter, "measure_axis", with_forehead):
            session = store.load(session.session_id)
            record = _record(session, store, entry)
            session.state["frontal_reviews"][entry.photo_id] = save_review(
                record, record["points"], expected_revision=record["revision"],
                binding=record["binding"], forehead_visibility="forehead_hidden",
            )
            store.save(session)
            headers = {"X-Archive-Capability": self.server.history_capability,
                       "X-Archive-CSRF": self.server.history_csrf,
                       "Cookie": "face_session=" + session.session_id}
            status, body = self.post("/history/api", dict(
                operation="save", session=session.session_id, session_csrf=session.csrf_token,
                profile="Fixture", protocol="controlled", camera="Fixture",
                photos=[entry.photo_id],
            ), headers)
            self.assertEqual(status, 200)
            archive_id = json.loads(body)["id"]
            status, body = self.post("/history/api", dict(operation="restore", id=archive_id), headers)
            self.assertEqual(status, 200)
            restored = store.load(json.loads(body)["location"].split("/")[2])
            self.assertFalse(next(iter(restored.state["frontal_reviews"].values()))["reviewed"])
            frontal = frontal_payloads(restored, store)
            self.assertEqual(len(frontal), 1)
            self.assertTrue(frontal[0]["visibility_only"])
            self.assertEqual(frontal[0]["measurements"], [])
            result = self.server.analysis.analyze(restored.session_id)
            payload = build_export_payload(result, restored.demographics, frontal=frontal)
            self.assertNotIn("facial_index", {r["id"] for r in payload["regional_measurements"]})
            self.assertNotIn(entry.photo_id, str(payload))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.server = make_server(
            base_dir=Path(self.tmp.name) / "sessions",
            port=0,
            analysis_factory=lambda store: ProgressiveAnalysisService(
                store, adapter=FakeAdapter(), resolver=FakeResolver()
            ),
            history_dir=Path(self.tmp.name) / "history",
        )
        self.server.archive = ArchiveStore(Path(self.tmp.name) / "history", Keys())
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, path, data, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        c.request(
            "POST",
            path,
            json.dumps(data),
            {
                "Content-Type": "application/json",
                "Origin": self.origin,
                **(headers or {}),
            },
        )
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, body

    def test_root_never_discloses_capability_and_api_requires_both_tokens_exact_origin_host(
        self,
    ):
        c = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        c.request("GET", "/")
        r = c.getresponse()
        body = r.read()
        c.close()
        self.assertNotIn(self.server.history_capability.encode(), body)
        self.assertEqual(self.post("/history/api", {"operation": "list"})[0], 403)
        self.assertEqual(self.post("/history/auth", {"capability": "wrong"})[0], 403)
        status, body = self.post(
            "/history/auth", {"capability": self.server.history_capability}
        )
        self.assertEqual(status, 200)
        auth = json.loads(body)
        headers = {
            "X-Archive-Capability": auth["capability"],
            "X-Archive-CSRF": auth["csrf"],
        }
        self.assertEqual(
            self.post("/history/api", {"operation": "list"}, headers)[0], 200
        )
        self.assertEqual(
            self.post(
                "/history/api", {"operation": "list", "unexpected": "x"}, headers
            )[0],
            422,
        )
        for extra in ({"Origin": "null"}, {"Origin": ""}, {"Host": "127.0.0.1:1"}):
            self.assertIn(
                self.post("/history/api", {"operation": "list"}, {**headers, **extra})[
                    0
                ],
                (403, 421),
            )

    def test_auth_is_throttled_and_json_limit_enforced(self):
        for _ in range(5):
            self.post("/history/auth", {"capability": "wrong"})
        self.assertEqual(
            self.post("/history/auth", {"capability": self.server.history_capability})[
                0
            ],
            429,
        )
        self.assertEqual(
            self.post("/history/api", {"operation": "list", "padding": "x" * 100000})[
                0
            ],
            413,
        )

    def test_save_evidence_restore_new_session_backup_remove_and_session_isolation(
        self,
    ):
        store = self.server.store
        session = store.create(Demographics(25, "male", ("white",)))
        entry = PhotoCatalog(store, session.session_id).add_bytes(
            "source.jpg", jpeg_bytes(90)
        )
        headers = {
            "X-Archive-Capability": self.server.history_capability,
            "X-Archive-CSRF": self.server.history_csrf,
            "Cookie": "face_session=" + session.session_id,
        }
        data = dict(
            operation="save",
            session=session.session_id,
            session_csrf=session.csrf_token,
            profile="Test",
            protocol="controlled",
            camera="Fixture camera",
        )
        self.assertEqual(
            self.post("/history/api", {**data, "session_csrf": "wrong"}, headers)[0],
            422,
        )
        status, body = self.post("/history/api", data, headers)
        self.assertEqual(status, 200)
        first = json.loads(body)["id"]
        self.assertEqual(self.server.archive.read(first).evidence, [])
        status, body = self.post(
            "/history/api", {**data, "photos": [entry.photo_id]}, headers
        )
        self.assertEqual(status, 200)
        second = json.loads(body)["id"]
        original = self.server.archive.read(second)
        self.assertEqual(len(original.evidence), 1)
        status, body = self.post(
            "/history/api",
            dict(
                operation="preview",
                session=session.session_id,
                session_csrf=session.csrf_token,
                photo=entry.photo_id,
                original=True,
            ),
            headers,
        )
        self.assertEqual(status, 200)
        self.assertIn(b"Illustrative", body)
        status, body = self.post(
            "/history/api", dict(operation="restore", id=second), headers
        )
        self.assertEqual(status, 200)
        new_id = json.loads(body)["location"].split("/")[2]
        self.assertNotEqual(new_id, session.session_id)
        self.assertTrue(
            store.load(new_id).state["restored_archive"]["annotations_unconfirmed"]
        )
        status, _ = self.post(
            "/history/api", dict(operation="remove_evidence", id=second), headers
        )
        self.assertEqual(status, 200)
        changed = self.server.archive.read(second)
        self.assertEqual(changed.snapshot, original.snapshot)
        self.assertEqual(changed.evidence, [])
        store.end(session.session_id)
        self.assertTrue(self.server.archive.read(first))
        self.assertEqual(
            self.post("/history/api", dict(operation="restore", id=first), headers)[0],
            422,
        )

    def test_nested_json_and_locked_form_never_have_get_password_fallback(self):
        import http.client

        c = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        c.request(
            "POST",
            "/history/auth",
            "[" * 1100 + "0" + "]" * 1100,
            {"Content-Type": "application/json", "Origin": self.origin},
        )
        r = c.getresponse()
        self.assertEqual(r.status, 422)
        r.read()
        c.close()
        from local_app.history_page import render_history

        page = render_history()
        self.assertIn(
            'method="post" action="/history/locked" data-history-import', page
        )
        self.assertIn("fieldset disabled", page)
