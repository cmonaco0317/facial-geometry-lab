"""Actual HTTP review lifecycle, conflicts, safe exports and partial visibility."""

import json
import re
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from tests import test_server as fixtures
from tests.test_frontal_review import example_points


class FrontalHTTPTests(unittest.TestCase):
    _stop_server = fixtures.LocalServerTests._stop_server
    _landing = fixtures.LocalServerTests._landing
    _token = fixtures.LocalServerTests._token
    _create_session = fixtures.LocalServerTests._create_session

    def setUp(self):
        fixtures.LocalServerTests.setUp(self)
        self.addCleanup(patch.stopall)
        patch(
            "local_app.frontal_routes.detected_points", return_value=example_points()
        ).start()
        self.sid, self.csrf = self._create_session(photo_count=1)
        self.base = f"/sessions/{self.sid}"
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": f"http://127.0.0.1:{self.server.server_address[1]}",
        }

    def _open(self):
        status, _, body = self.client.request("GET", self.base + "/frontal")
        self.assertEqual(status, 200)
        page = body.decode()
        fields = {
            key: re.search(f'name="{key}" value="([^"]*)"', page)[1]
            for key in ("csrf_token", "photo", "revision", "binding")
        }
        return fields, page

    def _fields(self, fields):
        out = {**fields, "action": "save", "confirmed": "yes"}
        for key, pair in example_points().items():
            if pair is None:
                out["missing_" + key] = "yes"
            else:
                out[key + "_x"] = str(pair[0] * 100)
                out[key + "_y"] = str(pair[1] * 100)
        return out

    def _post(self, fields, route="/frontal"):
        return self.client.request(
            "POST", self.base + route, body=urlencode(fields), headers=self.headers
        )

    def test_preview_save_export_conflict_clear_and_delete(self):
        fields, page = self._open()
        values = self._fields(fields)
        self.assertIn("Unavailable / not visible", page)
        status, _, body = self._post({**values, "action": "preview"})
        self.assertEqual(status, 200)
        self.assertIn(b"Draft preview", body)
        self.assertFalse(
            self.server.store.load(self.sid).state["frontal_reviews"][fields["photo"]][
                "reviewed"
            ]
        )
        values["hairline_x"] = "50"
        values["hairline_y"] = "5"
        values["forehead_visibility"] = "visible"
        values.pop("missing_hairline")
        self.assertEqual(self._post(values)[0], 303)
        status, _, body = self.client.request("GET", self.base + "/report.json")
        self.assertEqual(status, 200)
        report = json.loads(body)
        self.assertEqual(report["format_version"], 6)
        reviewed = report["frontal_reviews"][0]
        self.assertEqual(reviewed["revision"], 2)
        self.assertIn(
            "hairline_upper_share", {r["id"] for r in reviewed["measurements"]}
        )
        self.assertTrue(all(r["percentile"] is None for r in reviewed["measurements"]))
        self.assertNotIn(fields["binding"], body.decode())
        self.assertNotIn(fields["photo"], body.decode())
        self.assertIn(
            b"Reviewed frontal measurements",
            self.client.request("GET", self.base + "/report.html")[2],
        )
        self.assertEqual(self._post(values)[0], 409)
        # Preview does not overwrite a newer revision either.
        self.assertEqual(self._post({**values, "action": "preview"})[0], 409)
        clear = {**fields, "revision": "2"}
        self.assertEqual(self._post(clear, "/frontal-clear")[0], 303)
        self.assertEqual(
            json.loads(self.client.request("GET", self.base + "/report.json")[2])[
                "frontal_reviews"
            ],
            [],
        )
        self.assertEqual(self._post(clear, "/frontal-clear")[0], 409)

    def test_validation_and_owned_session(self):
        fields, _ = self._open()
        values = self._fields(fields)
        for changed in (
            {"chin_x": "NaN"},
            {"chin_x": "101"},
            {"unknown": "true"},
            {"confirmed": "no"},
        ):
            self.assertEqual(self._post({**values, **changed})[0], 422)
        self.assertEqual(self._post({**values, "csrf_token": "wrong"})[0], 403)
        self.assertEqual(self._post({**values, "photo": "other-person"})[0], 404)
        stranger = fixtures.BrowserClient(*self.server.server_address)
        self.assertEqual(stranger.request("GET", self.base + "/frontal")[0], 404)
        self.assertEqual(
            self.client.request("GET", self.base + "/frontal?photo=../../outside")[0],
            404,
        )
        self.assertEqual(
            self.client.request("GET", self.base + "/frontal?photo=a&photo=b")[0], 422
        )

    def test_export_diagram_uses_reviewed_points_and_aspect(self):
        fields, _ = self._open()
        self.assertEqual(self._post(self._fields(fields))[0], 303)
        payload = json.loads(self.client.request("GET", self.base + "/report.json")[2])
        review = payload["frontal_reviews"][0]
        self.assertIn("diagram", review)
        self.assertIn(
            "<svg", self.client.request("GET", self.base + "/report.html")[2].decode()
        )

    def test_detector_failure_returns_sanitized_http_error_and_retains_session(self):
        with patch(
            "local_app.frontal_routes.detected_points",
            side_effect=RuntimeError("/private/hidden/model"),
        ):
            status, _, body = self.client.request("GET", self.base + "/frontal")
        self.assertEqual(status, 500)
        self.assertNotIn(b"/private/hidden", body)
        self.assertEqual(self.server.store.load(self.sid).session_id, self.sid)

    def test_hair_coverage_preview_save_and_exports_mask_obscured_points(self):
        fields, page = self._open()
        self.assertIn('name="forehead_visibility"', page)
        values = {**self._fields(fields), "forehead_visibility": "forehead_hidden",
                  "hairline_x": "50", "hairline_y": "5"}
        values.pop("missing_hairline")
        status, _, draft = self._post({**values, "action": "preview"})
        self.assertEqual(status, 200)
        self.assertIn(b'hair hides the forehead', draft.lower())
        record = self.server.store.load(self.sid).state["frontal_reviews"][fields["photo"]]
        self.assertFalse(record["reviewed"])
        self.assertEqual(self._post(values)[0], 303)
        payload = json.loads(self.client.request("GET", self.base + "/report.json")[2])
        review = payload["frontal_reviews"][0]
        self.assertEqual(review["forehead_visibility"], "forehead_hidden")
        self.assertIsNone(review["diagram"]["points"]["mesh_top"])
        self.assertNotIn("reviewed_facial_index", {r["id"] for r in review["measurements"]})
        self.assertIn("reviewed_eye_spacing", {r["id"] for r in review["measurements"]})
        self.assertIn(b"hair lifted", self.client.request("GET", self.base + "/report.html")[2])
