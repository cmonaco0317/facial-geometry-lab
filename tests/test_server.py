import http.client
import json
from pathlib import Path
import re
import socket
import tempfile
from threading import Event, RLock, Thread, current_thread
import unittest
from unittest.mock import patch
from urllib.parse import urlencode
from types import SimpleNamespace

import cv2
import numpy as np

from local_app.contracts import AxisResult, CohortUse, Demographics
from local_app.server import SlidingWindowRateLimiter, make_server, serve


def jpeg_bytes(value: int) -> bytes:
    image = np.full((20, 24, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("test JPEG encoding failed")
    return encoded.tobytes()


def multipart(
    fields: list[tuple[str, str]],
    files: list[tuple[str, str, str, bytes]],
) -> tuple[bytes, str]:
    boundary = "----face-analysis-test-boundary"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            )
        )
    for name, filename, content_type, content in files:
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            )
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class FakeResolver:
    def resolve(self, demographics) -> CohortUse:
        ethnicity = "+".join(demographics.ethnicities)
        cohort_id = f"{demographics.sex}_{ethnicity}_{demographics.age}"
        return CohortUse(
            requested_id=cohort_id,
            used_id=cohort_id,
            fallback_level="exact",
            sample_size=420,
            requested_label="requested test cohort",
            used_label="used test cohort",
            member_ids=(cohort_id,),
        )


class FakeAdapter:
    def fingerprint(self) -> str:
        return "server-test-instrument-v1"

    def route(self, photos, cohort):
        identifiers = tuple(photo.entry.photo_id for photo in photos)
        return {
            axis: identifiers
            for axis in ("harmony", "dimorphism", "symmetry", "averageness")
        }

    def measure_axis(self, axis, photos, cohort) -> AxisResult:
        count = len(photos)
        return AxisResult.measured(
            axis=axis,
            summary=f"{axis.title()} measured from {count} photos.",
            value=float(count),
            unit="test units",
            percentile=60.0,
            interval=(52.0, 68.0),
            cohort=cohort,
            photo_count=count,
            details={
                "test_adapter": True,
                "evidence_photo_ids": [photo.entry.photo_id for photo in photos],
            },
        )


class BrowserClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.cookies: dict[str, str] = {}

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        request_headers = dict(headers or {})
        if self.cookies:
            request_headers["Cookie"] = "; ".join(
                f"{name}={value}" for name, value in self.cookies.items()
            )
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10)
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        payload = response.read()
        response_headers: dict[str, str] = {}
        for key, value in response.getheaders():
            normalized = key.lower()
            if normalized in response_headers:
                response_headers[normalized] += f"\n{value}"
            else:
                response_headers[normalized] = value
        for key, value in response.getheaders():
            if key.lower() != "set-cookie":
                continue
            pair = value.split(";", 1)[0]
            name, cookie_value = pair.split("=", 1)
            if "max-age=0" in value.lower():
                self.cookies.pop(name, None)
            else:
                self.cookies[name] = cookie_value
        connection.close()
        return response.status, response_headers, payload


class BlockingHandlerLock:
    """Hold the first request handler before it acquires the server state lock."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._blocked_handler = False
        self.handler_waiting = Event()
        self.release_handler = Event()

    def __enter__(self) -> "BlockingHandlerLock":
        is_handler = "process_request_thread" in current_thread().name
        if is_handler and not self._blocked_handler:
            self._blocked_handler = True
            self.handler_waiting.set()
            self.release_handler.wait(timeout=5)
        self._lock.acquire()
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        self._lock.release()


class LocalServerTests(unittest.TestCase):
    def setUp(self) -> None:
        from local_app.analysis import ProgressiveAnalysisService

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        session_dir = Path(self.temp_dir.name) / "sessions"

        def analysis_factory(store):
            return ProgressiveAnalysisService(
                store,
                adapter=FakeAdapter(),
                resolver=FakeResolver(),
            )

        self.server = make_server(
            base_dir=session_dir,
            host="127.0.0.1",
            port=0,
            analysis_factory=analysis_factory,
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        host, port = self.server.server_address
        self.client = BrowserClient(host, port)

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def _landing(self, client: BrowserClient | None = None) -> str:
        target = client or self.client
        status, _, body = target.request("GET", "/")
        self.assertEqual(status, 200)
        page = body.decode()
        self.assertIn('name="csrf_token"', page)
        return page

    def _token(self, page: str) -> str:
        match = re.search(r'name="csrf_token" value="([^"]+)"', page)
        self.assertIsNotNone(match)
        return match.group(1)

    def _create_session(
        self,
        client: BrowserClient | None = None,
        *,
        photo_count: int = 8,
    ) -> tuple[str, str]:
        target = client or self.client
        bootstrap = self._token(self._landing(target))
        fields = [
            ("csrf_token", bootstrap),
            ("age", "31"),
            ("sex", "female"),
            ("ethnicity", "east_asian"),
            ("ethnicity", "white"),
        ]
        files = [
            ("photos", f"casual-{index}.jpg", "image/jpeg", jpeg_bytes(index * 20))
            for index in range(photo_count)
        ]
        body, content_type = multipart(fields, files)
        status, headers, _ = target.request(
            "POST",
            "/sessions",
            body=body,
            headers={
                "Content-Type": content_type,
                "Origin": f"http://127.0.0.1:{self.server.server_address[1]}",
            },
        )
        self.assertEqual(status, 303)
        location = headers["location"]
        match = re.fullmatch(r"/sessions/([A-Za-z0-9_-]{32,96})/results", location)
        self.assertIsNotNone(match)
        session_id = match.group(1)
        status, _, payload = target.request("GET", location)
        self.assertEqual(status, 200)
        page = payload.decode()
        return session_id, self._token(page)

    def test_landing_static_assets_and_security_headers(self) -> None:
        page = self._landing()
        self.assertIn("Choose 1 or more photos", page)

        status, headers, css = self.client.request("GET", "/static/app.css")
        self.assertEqual(status, 200)
        self.assertIn(b".axis-grid", css)
        for name in (
            "cache-control",
            "content-security-policy",
            "referrer-policy",
            "x-content-type-options",
        ):
            with self.subTest(header=name):
                self.assertIn(name, headers)
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertEqual(headers["referrer-policy"], "same-origin")
        self.assertEqual(headers["x-content-type-options"], "nosniff")

    def test_all_session_cookies_are_secure_http_only_and_same_site(self) -> None:
        status, headers, _ = self.client.request("GET", "/")
        self.assertEqual(status, 200)
        cookie_headers = headers["set-cookie"].splitlines()
        self.assertTrue(cookie_headers)
        for value in cookie_headers:
            self.assertIn("Secure", value)
            self.assertIn("HttpOnly", value)
            self.assertIn("SameSite=Strict", value)

    def test_every_endpoint_is_rate_limited_per_client(self) -> None:
        self.server.rate_limiter.request_limit = 2
        for path in ("/", "/static/app.css"):
            status, _, _ = self.client.request("GET", path)
            self.assertEqual(status, 200)
        status, headers, _ = self.client.request("GET", "/")
        self.assertEqual(status, 429)
        self.assertEqual(headers["retry-after"], "60")

    def test_mutations_have_a_stricter_rate_limit_and_window_expires(self) -> None:
        now = [100.0]
        limiter = SlidingWindowRateLimiter(
            request_limit=60,
            mutation_limit=2,
            window_seconds=60,
        )
        with patch("limits.storage.memory.time", SimpleNamespace(time=lambda: now[0])):
            self.assertTrue(limiter.allow("127.0.0.1", mutating=True))
            self.assertTrue(limiter.allow("127.0.0.1", mutating=True))
            self.assertFalse(limiter.allow("127.0.0.1", mutating=True))
            self.assertTrue(limiter.allow("127.0.0.1", mutating=False))
            now[0] += 61
            self.assertTrue(limiter.allow("127.0.0.1", mutating=True))

    def test_development_keep_alive_does_not_block_server_close(self) -> None:
        self.assertTrue(self.server.daemon_threads)
        persistent = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        self.addCleanup(persistent.close)
        persistent.request("GET", "/")
        response = persistent.getresponse()
        response.read()
        self.assertEqual(response.status, 200)
        self.assertIsNone(response.getheader("Connection"))

        close_complete = Event()

        def shutdown_and_close() -> None:
            self.server.shutdown()
            self.server.server_close()
            close_complete.set()

        close_thread = Thread(target=shutdown_and_close, name="development-close")
        close_thread.start()
        close_thread.join(timeout=2)
        stopped_while_client_open = not close_thread.is_alive()
        persistent.close()
        close_thread.join(timeout=5)
        self.thread.join(timeout=5)

        self.assertTrue(stopped_while_client_open)
        self.assertTrue(close_complete.is_set())
        self.assertFalse(self.thread.is_alive())

    def test_create_upload_analyze_add_export_and_end(self) -> None:
        session_id, csrf = self._create_session()
        results_path = f"/sessions/{session_id}/results"

        status, _, payload = self.client.request("GET", results_path)
        self.assertEqual(status, 200)
        self.assertEqual(payload.count(b'<article class="axis-card'), 4)
        self.assertIn(b"Four-pillar facial analysis", payload)
        self.assertIn(b"Your results at a glance", payload)
        self.assertIn(b"8 photos in this session", payload)

        upload, content_type = multipart(
            [("csrf_token", csrf)],
            [("photos", "extra.jpg", "image/jpeg", jpeg_bytes(177))],
        )
        status, headers, _ = self.client.request(
            "POST",
            f"/sessions/{session_id}/photos",
            body=upload,
            headers={"Content-Type": content_type},
        )
        self.assertEqual(status, 303)
        self.assertEqual(headers["location"], results_path)

        status, headers, report_body = self.client.request(
            "GET", f"/sessions/{session_id}/report.json"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json; charset=utf-8")
        report = json.loads(report_body)
        self.assertEqual(report["accepted_photo_count"], 9)
        self.assertEqual(report["demographics"]["age"], 31)
        self.assertEqual(report["demographics"]["sex"], "female")
        self.assertEqual(report["comparison"]["fallback_reason"], "exact")
        self.assertEqual(
            set(report["axes"]), {"harmony", "dimorphism", "symmetry", "averageness"}
        )
        self.assertNotIn("session_id", report)
        self.assertNotIn("photo_fingerprint", report)
        serialized = json.dumps(report)
        for forbidden in (
            "stored_name",
            "display_name",
            "source_bytes",
            "evidence_photo_ids",
            str(Path(self.temp_dir.name)),
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)

        status, headers, printable = self.client.request(
            "GET", f"/sessions/{session_id}/report.html"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "text/html; charset=utf-8")
        self.assertIn(b"Four-pillar facial analysis", printable)
        self.assertIn(b"Facial measurement report", printable)
        self.assertIn(b"@media print", printable)
        self.assertNotIn(str(Path(self.temp_dir.name)).encode(), printable)

        encoded = urlencode({"csrf_token": csrf})
        status, headers, _ = self.client.request(
            "POST",
            f"/sessions/{session_id}/end",
            body=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        self.assertEqual(status, 303)
        self.assertEqual(headers["location"], "/")
        self.assertNotIn("face_session", self.client.cookies)
        status, _, _ = self.client.request("GET", results_path)
        self.assertEqual(status, 404)

    def test_packaged_end_returns_goodbye_clears_cookies_and_requests_shutdown(
        self,
    ) -> None:
        from local_app.analysis import ProgressiveAnalysisService

        self._stop_server()
        self._cleanups.pop()
        session_dir = Path(self.temp_dir.name) / "packaged-sessions"
        self.server = make_server(
            base_dir=session_dir,
            host="127.0.0.1",
            port=0,
            analysis_factory=lambda store: ProgressiveAnalysisService(
                store,
                adapter=FakeAdapter(),
                resolver=FakeResolver(),
            ),
            shutdown_after_end=True,
        )
        self.assertFalse(self.server.daemon_threads)

        def serve_until_closed() -> None:
            try:
                self.server.serve_forever()
            finally:
                self.server.server_close()

        self.thread = Thread(target=serve_until_closed, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.client = BrowserClient(*self.server.server_address)
        session_id, csrf = self._create_session()
        session_root = self.server.store.load(session_id).root
        self.assertTrue(session_root.exists())

        persistent = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        self.addCleanup(persistent.close)
        persistent.request(
            "GET",
            f"/sessions/{session_id}/results",
            headers={"Cookie": f"face_session={session_id}"},
        )
        persistent_response = persistent.getresponse()
        persistent_response.read()
        self.assertEqual(persistent_response.status, 200)
        persistent_close = persistent_response.getheader("Connection")

        status, headers, body = self.client.request(
            "POST",
            f"/sessions/{session_id}/end",
            body=urlencode({"csrf_token": csrf}),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        self.assertEqual(status, 200)
        self.assertIn(b"Session erased", body)
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertEqual(headers["set-cookie"].count("Max-Age=0"), 2)
        self.assertIn("face_session=", headers["set-cookie"])
        self.assertIn("face_bootstrap=", headers["set-cookie"])
        self.assertNotIn("face_session", self.client.cookies)
        self.assertNotIn("face_bootstrap", self.client.cookies)
        self.assertTrue(self.server.shutdown_requested.wait(timeout=1.0))
        self.thread.join(timeout=2)
        stopped_while_client_open = not self.thread.is_alive()
        persistent.close()
        self.thread.join(timeout=5)

        self.assertEqual(persistent_close, "close")
        self.assertEqual(headers["connection"], "close")
        self.assertTrue(stopped_while_client_open)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse(session_root.exists())

    def test_server_close_drains_accepted_handlers_before_final_cleanup(self) -> None:
        from local_app.analysis import ProgressiveAnalysisService

        self._stop_server()
        self._cleanups.pop()
        self.server = make_server(
            base_dir=Path(self.temp_dir.name) / "drain-sessions",
            host="127.0.0.1",
            port=0,
            analysis_factory=lambda store: ProgressiveAnalysisService(
                store,
                adapter=FakeAdapter(),
                resolver=FakeResolver(),
            ),
            shutdown_after_end=True,
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.client = BrowserClient(*self.server.server_address)
        bootstrap = self._token(self._landing())
        body, content_type = multipart(
            [
                ("csrf_token", bootstrap),
                ("age", "31"),
                ("sex", "female"),
                ("ethnicity", "white"),
            ],
            [
                ("photos", f"photo-{index}.jpg", "image/jpeg", jpeg_bytes(index * 20))
                for index in range(7)
            ],
        )
        gate = BlockingHandlerLock()
        self.server.state_lock = gate
        response: list[tuple[int, dict[str, str], bytes]] = []
        request_errors: list[BaseException] = []
        shutdown_complete = Event()
        close_complete = Event()
        cleanup_started = Event()
        original_cleanup = self.server.store.cleanup_all

        def submit_request() -> None:
            try:
                response.append(
                    self.client.request(
                        "POST",
                        "/sessions",
                        body=body,
                        headers={"Content-Type": content_type},
                    )
                )
            except BaseException as error:
                request_errors.append(error)

        def shutdown_and_close() -> None:
            self.server.shutdown()
            shutdown_complete.set()
            self.server.server_close()
            close_complete.set()

        def tracked_cleanup() -> int:
            cleanup_started.set()
            return original_cleanup()

        request_thread = Thread(target=submit_request, name="submit-request")
        close_thread = Thread(target=shutdown_and_close, name="server-close")
        with patch.object(
            self.server.store, "cleanup_all", side_effect=tracked_cleanup
        ):
            request_thread.start()
            self.assertTrue(gate.handler_waiting.wait(timeout=1))
            close_thread.start()
            self.assertTrue(shutdown_complete.wait(timeout=2))
            try:
                self.assertFalse(cleanup_started.wait(timeout=0.25))
            finally:
                gate.release_handler.set()
            self.assertTrue(cleanup_started.wait(timeout=5))
            request_thread.join(timeout=5)
            close_thread.join(timeout=5)

        self.thread.join(timeout=5)
        self.assertFalse(request_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        self.assertFalse(self.thread.is_alive())
        self.assertTrue(close_complete.is_set())
        self.assertEqual(request_errors, [])
        self.assertEqual(response[0][0], 303)
        self.assertFalse(self.server.store.has_owned_sessions)

    def test_packaged_partial_request_cannot_block_final_cleanup(self) -> None:
        class UnusedAnalysis:
            pass

        server = make_server(
            base_dir=Path(self.temp_dir.name) / "partial-request-sessions",
            port=0,
            analysis_factory=lambda _store: UnusedAnalysis(),
            shutdown_after_end=True,
        )
        self.addCleanup(server.server_close)
        session = server.store.create(
            Demographics(age=31, sex="female", ethnicities=("white",))
        )
        accepted = Event()
        close_complete = Event()
        original_get_request = server.get_request

        def tracked_get_request() -> tuple[socket.socket, tuple[str, int]]:
            request = original_get_request()
            accepted.set()
            return request

        def shutdown_and_close() -> None:
            server.shutdown()
            server.server_close()
            close_complete.set()

        serve_thread = Thread(
            target=lambda: server.serve_forever(poll_interval=0.01),
            name="partial-request-server",
            daemon=True,
        )
        close_thread = Thread(target=shutdown_and_close, name="partial-request-close")
        with (
            patch(
                "local_app.server.PACKAGED_REQUEST_TIMEOUT_SECONDS",
                0.1,
            ),
            patch.object(server, "get_request", side_effect=tracked_get_request),
        ):
            serve_thread.start()
            partial = socket.create_connection(server.server_address, timeout=1)
            self.addCleanup(partial.close)
            partial.sendall(
                f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{server.server_address[1]}\r\n".encode()
            )
            self.assertTrue(accepted.wait(timeout=1))
            close_thread.start()
            close_thread.join(timeout=1)
            stopped_while_client_open = not close_thread.is_alive()
            partial.close()
            close_thread.join(timeout=5)
            serve_thread.join(timeout=5)

        self.assertTrue(stopped_while_client_open)
        self.assertTrue(close_complete.is_set())
        self.assertFalse(serve_thread.is_alive())
        self.assertFalse(session.root.exists())

    def test_serve_closes_bound_server_when_on_ready_raises(self) -> None:
        class UnusedAnalysis:
            pass

        server = make_server(
            base_dir=Path(self.temp_dir.name) / "callback-sessions",
            port=0,
            analysis_factory=lambda _store: UnusedAnalysis(),
        )
        self.addCleanup(server.server_close)
        session = server.store.create(
            Demographics(age=31, sex="female", ethnicities=("white",))
        )
        urls: list[str] = []

        def fail_after_ready(url: str) -> None:
            urls.append(url)
            raise RuntimeError("simulated callback failure")

        with patch("local_app.server.make_server", return_value=server):
            with self.assertRaisesRegex(RuntimeError, "simulated callback failure"):
                serve(
                    host="127.0.0.1",
                    port=0,
                    open_browser=False,
                    base_dir=Path(self.temp_dir.name) / "unused",
                    on_ready=fail_after_ready,
                )

        host, port = server.server_address
        self.assertEqual(urls, [f"http://{host}:{port}/#history={server.history_capability}"])
        self.assertEqual(server.socket.fileno(), -1)
        self.assertFalse(session.root.exists())

    def test_native_forms_retain_same_origin_without_cross_site_referrers(self) -> None:
        status, headers, _ = self.client.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(headers["referrer-policy"], "same-origin")

    def test_csrf_and_invalid_origin_are_rejected(self) -> None:
        session_id, _ = self._create_session()
        encoded = urlencode({"csrf_token": "wrong-token"})
        status, _, _ = self.client.request(
            "POST",
            f"/sessions/{session_id}/analyze",
            body=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        self.assertEqual(status, 403)

        separate = BrowserClient(*self.server.server_address)
        token = self._token(self._landing(separate))
        body, content_type = multipart(
            [
                ("csrf_token", token),
                ("age", "31"),
                ("sex", "female"),
                ("ethnicity", "white"),
            ],
            [],
        )
        status, _, _ = separate.request(
            "POST",
            "/sessions",
            body=body,
            headers={"Content-Type": content_type, "Origin": "http://evil.example"},
        )
        self.assertEqual(status, 403)

    def test_server_is_loopback_only_and_sessions_are_isolated(self) -> None:
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        first_id, _ = self._create_session(self.client)
        other_client = BrowserClient(*self.server.server_address)
        other_id, _ = self._create_session(other_client)
        self.assertNotEqual(first_id, other_id)

        status, _, _ = self.client.request("GET", f"/sessions/{other_id}/results")
        self.assertEqual(status, 404)

        with self.assertRaises(ValueError):
            make_server(
                base_dir=Path(self.temp_dir.name) / "unsafe",
                host="0.0.0.0",
                port=0,
            )

    def test_nonlocal_host_is_rejected(self) -> None:
        status, _, _ = self.client.request("GET", "/", headers={"Host": "evil.example"})
        self.assertEqual(status, 421)

    def test_intake_rejects_ethnicity_values_not_offered_by_the_form(self) -> None:
        bootstrap = self._token(self._landing())
        body, content_type = multipart(
            [
                ("csrf_token", bootstrap),
                ("age", "31"),
                ("sex", "female"),
                ("ethnicity", "injected_cohort_name"),
            ],
            [
                ("photos", f"photo-{index}.jpg", "image/jpeg", jpeg_bytes(index * 20))
                for index in range(7)
            ],
        )
        status, _, payload = self.client.request(
            "POST",
            "/sessions",
            body=body,
            headers={"Content-Type": content_type},
        )

        self.assertEqual(status, 422)
        self.assertIn(b"unsupported ethnicity selection", payload)
        self.assertEqual(list(self.server.store.base_dir.iterdir()), [])

    def test_multipart_part_count_is_bounded(self) -> None:
        bootstrap = self._token(self._landing())
        fields = [
            ("csrf_token", bootstrap),
            ("age", "31"),
            ("sex", "female"),
            ("ethnicity", "white"),
        ]
        fields.extend((f"ignored-{index}", "x") for index in range(600))
        body, content_type = multipart(fields, [])
        status, _, _ = self.client.request(
            "POST",
            "/sessions",
            body=body,
            headers={"Content-Type": content_type},
        )

        self.assertEqual(status, 400)
        self.assertEqual(list(self.server.store.base_dir.iterdir()), [])

    def test_below_minimum_session_keeps_an_upload_form_after_a_bad_addition(
        self,
    ) -> None:
        bootstrap = self._token(self._landing())
        body, content_type = multipart(
            [
                ("csrf_token", bootstrap),
                ("age", "31"),
                ("sex", "female"),
                ("ethnicity", "white"),
            ],
            [("photos", "broken-first.jpg", "image/jpeg", b"not an image")],
        )
        status, _, payload = self.client.request(
            "POST",
            "/sessions",
            body=body,
            headers={"Content-Type": content_type},
        )
        page = payload.decode()
        self.assertEqual(status, 422)
        session_match = re.search(r'action="/sessions/([^/]+)/photos"', page)
        self.assertIsNotNone(session_match)
        session_id = session_match.group(1)
        csrf = self._token(page)

        addition, addition_type = multipart(
            [("csrf_token", csrf)],
            [("photos", "broken.jpg", "image/jpeg", b"not an image")],
        )
        status, _, payload = self.client.request(
            "POST",
            f"/sessions/{session_id}/photos",
            body=addition,
            headers={"Content-Type": addition_type},
        )

        self.assertEqual(status, 422)
        self.assertIn(b"Add 1 more photo", payload)
        self.assertIn(b'name="photos"', payload)

    def test_oversized_body_is_rejected_before_session_creation(self) -> None:
        self._stop_server()
        self._cleanups.pop()

        session_dir = Path(self.temp_dir.name) / "small-sessions"
        self.server = make_server(
            base_dir=session_dir,
            host="127.0.0.1",
            port=0,
            max_request_bytes=256,
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.client = BrowserClient(*self.server.server_address)
        self._landing()

        status, _, _ = self.client.request(
            "POST",
            "/sessions",
            body=b"x" * 257,
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(status, 413)
        self.assertEqual(list(session_dir.iterdir()), [])

    def test_cached_analysis_cannot_overwrite_report_preference_update(self):
        session_id, csrf = self._create_session(photo_count=1)
        saving, release, updated = Event(), Event(), Event()
        original_save = self.server.store.save
        errors = []
        responses = []

        def delayed_save(session):
            if current_thread().name == "cached-analysis-check":
                saving.set()
                if not release.wait(5):
                    raise RuntimeError("test timed out releasing cached save")
            original_save(session)

        def analyze():
            try:
                self.server.analyze_session(session_id)
            except Exception as error:
                errors.append(error)

        def update():
            try:
                responses.append(
                    self.client.request(
                        "POST",
                        f"/sessions/{session_id}/preferences",
                        body=urlencode(
                            {
                                "csrf_token": csrf,
                                "goal": "balance",
                                "hair": "yes",
                                "eyewear": "no",
                            }
                        ),
                        headers={
                            "Content-Type": "application/x-www-form-urlencoded",
                            "Origin": f"http://127.0.0.1:{self.server.server_address[1]}",
                        },
                    )[0]
                )
            except Exception as error:
                errors.append(error)
            finally:
                updated.set()

        with patch.object(self.server.store, "save", side_effect=delayed_save):
            analysis = Thread(target=analyze, name="cached-analysis-check")
            mutation = Thread(target=update)
            analysis.start()
            try:
                self.assertTrue(saving.wait(5))
                mutation.start()
                self.assertFalse(updated.wait(0.15))
            finally:
                release.set()
                analysis.join(5)
                if mutation.ident is not None:
                    mutation.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(responses, [303])
        self.assertEqual(
            self.server.store.load(session_id).state["report_preferences"]["goal"],
            "balance",
        )

    def test_consultation_preferences_and_manual_profile_flow(self):
        from local_app.photos import PhotoCatalog

        session_id, csrf = self._create_session(photo_count=1)
        base = f"/sessions/{session_id}"
        entry = PhotoCatalog(self.server.store, session_id).entries()[0]
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": f"http://127.0.0.1:{self.server.server_address[1]}",
        }
        status, _, body = self.client.request("GET", base + "/profile")
        self.assertEqual(status, 200)
        self.assertIn(b"data:image/jpeg;base64,", body)
        self.assertIn(b"Review your profile landmarks", body)
        stranger = BrowserClient(*self.server.server_address)
        self.assertEqual(stranger.request("GET", base + "/profile")[0], 404)
        self.assertEqual(
            self.client.request("GET", base + "/profile?photo=../../private")[0], 404
        )
        fields = {"csrf_token": csrf, "goal": "balance", "hair": "yes", "eyewear": "no", "facial_hair": "yes", "preserve": "eyes", "hair_length": "long", "hair_texture": "wavy", "hair_change": "open", "maintenance": "flexible", "budget": "flexible"}
        self.assertEqual(
            self.client.request(
                "POST", base + "/preferences", body=urlencode(fields), headers=headers
            )[0],
            303,
        )
        self.assertEqual(
            self.client.request(
                "POST",
                base + "/preferences",
                body=urlencode({**fields, "unknown": "yes"}),
                headers=headers,
            )[0],
            422,
        )
        self.assertEqual(
            self.client.request(
                "POST",
                base + "/preferences",
                body=urlencode({**fields, "csrf_token": "wrong"}),
                headers=headers,
            )[0],
            403,
        )
        from tests.test_consultation import ProfileReviewTests

        fixture = ProfileReviewTests()
        fixture.setUp()
        points = fixture.points
        fields = {
            "csrf_token": csrf,
            "photo": entry.photo_id,
            "confirmed": "yes",
            "active_anchor": "tip",
        }
        for key, point in points.items():
            fields[key + "_x"] = str(point[0] * 100)
            fields[key + "_y"] = str(point[1] * 100)
        self.assertEqual(
            self.client.request(
                "POST",
                base + "/profile",
                body=urlencode({**fields, "tip_x": "nan"}),
                headers=headers,
            )[0],
            422,
        )
        self.assertEqual(
            self.client.request(
                "POST",
                base + "/profile",
                body=urlencode({**fields, "photo": "someone-else"}),
                headers=headers,
            )[0],
            422,
        )
        self.assertEqual(
            self.client.request(
                "POST", base + "/profile", body=urlencode(fields), headers=headers
            )[0],
            303,
        )
        status, _, body = self.client.request("GET", base + "/report.json")
        self.assertEqual(status, 200)
        report = json.loads(body)
        self.assertEqual(report["format_version"], 6)
        self.assertEqual(report["consultation"]["preferences"]["hair"], "yes")
        self.assertEqual(len(report["profile_review"]["measurements"]), 4)
        self.assertNotIn(entry.photo_id, body.decode())
        self.assertNotIn("data:image", body.decode())
        self.assertEqual(
            self.client.request(
                "POST",
                base + "/profile-clear",
                body=urlencode({"csrf_token": csrf}),
                headers=headers,
            )[0],
            303,
        )
        self.assertIsNone(
            json.loads(self.client.request("GET", base + "/report.json")[2])[
                "profile_review"
            ]
        )


class CompatibilityLauncherTests(unittest.TestCase):
    def test_local_page_launches_the_progressive_loopback_server(self) -> None:
        import local_page

        with patch("local_page.serve", return_value=0) as serve:
            result = local_page.main(["--port", "9123", "--no-browser"])

        self.assertEqual(result, 0)
        serve.assert_called_once_with(
            host="127.0.0.1",
            port=9123,
            open_browser=False,
        )

    def test_old_folder_and_cohort_post_flow_is_removed(self) -> None:
        import local_page

        source = Path(local_page.__file__).read_text(encoding="utf-8")
        self.assertNotIn('self.path != "/cohort"', source)
        self.assertNotIn("folder of your photographs", source.lower())


if __name__ == "__main__":
    unittest.main()
