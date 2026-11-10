"""Loopback-only HTTP adapter for private progressive facial analysis."""

from __future__ import annotations

import atexit
import cgi
from collections.abc import Callable
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import math

from limits import RateLimitItemPerSecond
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from pathlib import Path
import re
import secrets
import socket
import tempfile
from threading import Event, RLock, Thread
from typing import Any
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .analysis import AnalysisNotReady, ProgressiveAnalysisService
from .contracts import Demographics
from .cohorts import NoCompatibleCohort
from .photos import MAX_FILE_BYTES, MAX_SESSION_BYTES, PhotoCatalog, PhotoRejected
from .report import build_export_payload, render_html_export, render_json_export
from .render import ETHNICITY_OPTIONS, render_landing, render_results
from .sessions import SESSION_ID_PATTERN, Session, SessionStore

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_REQUEST_OVERHEAD = 16 * 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = MAX_SESSION_BYTES + DEFAULT_REQUEST_OVERHEAD
PACKAGED_REQUEST_TIMEOUT_SECONDS = 2.0
MAX_MULTIPART_FIELDS = 256
MAX_TEXT_FIELD_CHARS = 4_096
SESSION_COOKIE = "face_session"
BOOTSTRAP_COOKIE = "face_bootstrap"
LOCAL_HOST_PATTERN = re.compile(r"^(?:127\.0\.0\.1|localhost)(?::[0-9]{1,5})?$")
SESSION_ROUTE = re.compile(
    r"^/sessions/(?P<session>[A-Za-z0-9_-]{32,96})/"
    r"(?P<resource>photos|analyze|results|report\.(?:json|html)|preferences|ideas|profile|profile-clear|frontal|frontal-clear|end)$"
)
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'self'; script-src 'self'; "
        "img-src 'self' data: blob:; connect-src 'self'; form-action 'self'; base-uri 'none'; "
        "frame-ancestors 'none'"
    ),
    # Native same-origin forms need their origin preserved for the CSRF check.
    # Cross-origin destinations still receive no referrer.
    "Referrer-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
ALLOWED_ETHNICITIES = frozenset(value for value, _label in ETHNICITY_OPTIONS)
AnalysisFactory = Callable[[SessionStore], ProgressiveAnalysisService]


class SlidingWindowRateLimiter:
    """Maintained moving-window storage, serialized across request/mutation limits."""

    def __init__(
        self,
        *,
        request_limit: int = 60,
        mutation_limit: int = 20,
        window_seconds: float = 60.0,
    ) -> None:
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in (request_limit, mutation_limit)
        ):
            raise ValueError("rate counts must be positive integers")
        if (
            isinstance(window_seconds, bool)
            or not isinstance(window_seconds, (int, float))
            or not math.isfinite(window_seconds)
            or window_seconds <= 0
        ):
            raise ValueError("rate window must be positive and finite")
        self.request_limit = request_limit
        self.mutation_limit = mutation_limit
        self.window_seconds = math.ceil(window_seconds)
        self._storage = MemoryStorage()
        self._limiter = MovingWindowRateLimiter(self._storage)
        self._lock = RLock()

    def allow(self, client: str, *, mutating: bool) -> bool:
        requests = RateLimitItemPerSecond(self.request_limit, self.window_seconds)
        mutations = RateLimitItemPerSecond(self.mutation_limit, self.window_seconds)
        with self._lock:
            if not self._limiter.test(requests, client, "all"):
                return False
            if mutating and not self._limiter.test(mutations, client, "mutations"):
                return False
            if not self._limiter.hit(requests, client, "all"):
                return False
            return not mutating or self._limiter.hit(mutations, client, "mutations")


class LocalAnalysisHTTPServer(ThreadingHTTPServer):
    """Threaded loopback server carrying the local application's dependencies."""

    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        store: SessionStore,
        analysis: ProgressiveAnalysisService,
        max_request_bytes: int,
        shutdown_after_end: bool = False,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        history_dir: Path | None = None,
    ) -> None:
        from .history import ArchiveStore

        self.archive = ArchiveStore(history_dir) if history_dir is not None else None
        self.history_capability = secrets.token_urlsafe(32)
        self.history_csrf = secrets.token_urlsafe(32)
        self.history_auth_limiter = SlidingWindowRateLimiter(
            request_limit=5, mutation_limit=5, window_seconds=900
        )
        self.store = store
        self.analysis = analysis
        self.max_request_bytes = max_request_bytes
        self.shutdown_after_end = shutdown_after_end
        self.rate_limiter = rate_limiter or SlidingWindowRateLimiter()
        self.daemon_threads = not shutdown_after_end
        self.shutdown_requested = Event()
        self.static_dir = Path(__file__).with_name("static")
        self.state_lock = RLock()
        self._cleanup_complete = False
        super().__init__(address, LocalAnalysisHandler)
        self._atexit_callback = self._cleanup_sessions
        atexit.register(self._atexit_callback)

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        connection, address = super().get_request()
        if self.shutdown_after_end:
            connection.settimeout(PACKAGED_REQUEST_TIMEOUT_SECONDS)
        return connection, address

    def expire_stale_sessions(self) -> None:
        with self.state_lock:
            self.store.expire_stale()

    def analyze_session(self, session_id: str, *, force: bool = False):
        with self.state_lock:
            session = self.store.load(session_id)
            self.analysis.resolver.resolve(session.demographics)
            with self.store.analysis_scope(session_id):
                try:
                    return self.analysis.analyze(session_id, force=force)
                except SystemExit as error:
                    # A module exiting inside a request must still use the
                    # ordinary fatal-error response and session cleanup path.
                    raise RuntimeError(f"the analysis exited: {error}") from error

    def request_shutdown(self) -> None:
        if not self.shutdown_after_end:
            return
        self.shutdown_requested.set()
        Thread(
            target=self.shutdown,
            name="local-app-shutdown",
            daemon=True,
        ).start()

    def _cleanup_sessions(self) -> None:
        with self.state_lock:
            if self._cleanup_complete:
                return
            self.store.cleanup_all()
            self._cleanup_complete = not self.store.has_owned_sessions

    def server_close(self) -> None:
        super().server_close()
        self._cleanup_sessions()
        if self._cleanup_complete:
            atexit.unregister(self._atexit_callback)


class LocalAnalysisHandler(BaseHTTPRequestHandler):
    """Small endpoint adapter; all face processing remains in local services."""

    server: LocalAnalysisHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if not self._rate_allowed(mutating=False):
            return
        self.server.expire_stale_sessions()
        if not self._valid_host():
            return
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            token = secrets.token_urlsafe(24)
            page = render_landing(csrf_token=token)
            self._send(
                200,
                page.encode("utf-8"),
                "text/html; charset=utf-8",
                extra_headers=[
                    ("Set-Cookie", self._cookie_header(BOOTSTRAP_COOKIE, token))
                ],
            )
            return
        if path == "/history":
            from .history_page import render_history

            self._send(200, render_history().encode(), "text/html; charset=utf-8")
            return
        if path == "/static/history.js":
            self._serve_static("history.js", "text/javascript; charset=utf-8")
            return
        if path == "/static/app.css":
            self._serve_static("app.css", "text/css; charset=utf-8")
            return
        if path == "/static/frontal.js":
            self._serve_static("frontal.js", "text/javascript; charset=utf-8")
            return
        if path == "/static/review.js":
            self._serve_static("review.js", "text/javascript; charset=utf-8")
            return
        if path == "/static/app.js":
            self._serve_static("app.js", "text/javascript; charset=utf-8")
            return

        route = SESSION_ROUTE.fullmatch(path)
        if route is None:
            self._error(404, "Page not found")
            return
        session = self._owned_session(route.group("session"))
        if session is None:
            self._error(404, "Session not found")
            return
        resource = route.group("resource")
        if resource == "frontal":
            from .frontal_routes import get_frontal

            with self.server.state_lock:
                get_frontal(self, self.server.store.load(session.session_id))
        elif resource == "profile":
            from .review_routes import get_review

            with self.server.state_lock:
                get_review(self, self.server.store.load(session.session_id))
        elif resource == "results":
            with self.server.state_lock:
                self._get_results(session)
        elif resource in {"report.json", "report.html"}:
            with self.server.state_lock:
                self._get_report(session, format_name=resource.rsplit(".", 1)[1])
        else:
            self._error(405, "This endpoint requires POST", allow="POST")

    def do_POST(self) -> None:
        if not self._rate_allowed(mutating=True):
            return
        self.server.expire_stale_sessions()
        if not self._valid_host() or not self._valid_origin():
            return
        content_length = self._content_length()
        if content_length is None:
            return
        path = urlsplit(self.path).path
        if path in {"/history/auth", "/history/api", "/history/import"}:
            from .history_routes import handle

            handle(self, path, content_length)
            return
        if path == "/sessions":
            with self.server.state_lock:
                self._create_session(content_length)
            return

        route = SESSION_ROUTE.fullmatch(path)
        if route is None:
            self._error(404, "Page not found")
            return
        session = self._owned_session(route.group("session"))
        if session is None:
            self._error(404, "Session not found")
            return
        resource = route.group("resource")
        with self.server.state_lock:
            if resource in {"frontal", "frontal-clear"}:
                from .frontal_routes import update_frontal

                update_frontal(self, session, resource, content_length)
            elif resource in {"preferences", "ideas", "profile", "profile-clear"}:
                from .review_routes import update_review

                update_review(self, session, resource, content_length)
            elif resource == "photos":
                self._add_photos(session, content_length)
            elif resource == "analyze":
                self._analyze(session, content_length)
            elif resource == "end":
                self._end_session(session, content_length)
            else:
                self._error(405, "This endpoint requires GET", allow="GET")

    def _create_session(self, content_length: int) -> None:
        form = self._multipart_form(content_length)
        if form is None:
            return
        fields = self._text_fields(form)
        submitted = self._one(fields, "csrf_token")
        expected = self._cookie_value(BOOTSTRAP_COOKIE)
        if not self._tokens_match(submitted, expected):
            self._error(403, "The form expired. Reload the page and try again.")
            return
        try:
            ethnicities = tuple(fields.get("ethnicity", ()))
            if (
                not ethnicities
                or len(ethnicities) > len(ALLOWED_ETHNICITIES)
                or any(value not in ALLOWED_ETHNICITIES for value in ethnicities)
            ):
                raise ValueError("unsupported ethnicity selection")
            demographics = Demographics(
                age=int(self._one(fields, "age") or ""),
                sex=self._one(fields, "sex") or "",
                ethnicities=ethnicities,
            )
        except (TypeError, ValueError) as error:
            self._error(422, str(error))
            return

        try:
            self.server.analysis.resolver.resolve(demographics)
        except NoCompatibleCohort:
            self._reference_unavailable()
            return

        session = self.server.store.create(demographics)
        try:
            self._catalog_uploads(session, form)
            catalog = PhotoCatalog(self.server.store, session.session_id)
            readiness = catalog.readiness()
            if not readiness.ready:
                self._pending_response(session, readiness.message)
                return
            self.server.analyze_session(session.session_id)
        except Exception:
            self.server.store.end(session.session_id)
            self._failed_session_error(500, "The local analysis could not be started.")
            return

        location = f"/sessions/{session.session_id}/results"
        self._redirect_to_session(location, session)

    def _add_photos(self, session: Session, content_length: int) -> None:
        form = self._multipart_form(content_length)
        if form is None:
            return
        fields = self._text_fields(form)
        if not self._valid_csrf(session, self._one(fields, "csrf_token")):
            return
        try:
            self.server.analysis.resolver.resolve(session.demographics)
        except NoCompatibleCohort:
            self._reference_unavailable()
            return
        try:
            self._catalog_uploads(session, form)
            catalog = PhotoCatalog(self.server.store, session.session_id)
            readiness = catalog.readiness()
            if not readiness.ready:
                self._pending_response(session, readiness.message)
                return
            self.server.analyze_session(session.session_id)
        except Exception:
            self.server.store.end(session.session_id)
            self._failed_session_error(500, "The local analysis could not be updated.")
            return
        location = f"/sessions/{session.session_id}/results"
        self._redirect(location)

    def _analyze(self, session: Session, content_length: int) -> None:
        fields = self._simple_form(content_length)
        if fields is None or not self._valid_csrf(
            session, self._one(fields, "csrf_token")
        ):
            return
        try:
            self.server.analyze_session(session.session_id, force=True)
        except AnalysisNotReady as error:
            self._error(409, str(error))
            return
        except NoCompatibleCohort:
            self._reference_unavailable()
            return
        except Exception:
            self.server.store.end(session.session_id)
            self._failed_session_error(500, "The local analysis could not be retried.")
            return
        self._redirect(f"/sessions/{session.session_id}/results")

    def _end_session(self, session: Session, content_length: int) -> None:
        fields = self._simple_form(content_length)
        if fields is None or not self._valid_csrf(
            session, self._one(fields, "csrf_token")
        ):
            return
        self.server.store.end(session.session_id)
        if self.server.shutdown_after_end:
            body = (
                '<!doctype html><html lang="en"><meta charset="utf-8">'
                "<title>Session erased</title><h1>Session erased</h1>"
                "<p>Session erased. Facial Analysis has quit.</p></html>"
            ).encode("utf-8")
            self._send(
                200,
                body,
                "text/html; charset=utf-8",
                extra_headers=[
                    (
                        "Set-Cookie",
                        self._cookie_header(SESSION_COOKIE, "", max_age=0),
                    ),
                    (
                        "Set-Cookie",
                        self._cookie_header(BOOTSTRAP_COOKIE, "", max_age=0),
                    ),
                ],
            )
            self.server.request_shutdown()
            return
        self._redirect(
            "/",
            extra_headers=[
                (
                    "Set-Cookie",
                    self._cookie_header(SESSION_COOKIE, "", max_age=0),
                )
            ],
        )

    def _get_results(self, session: Session) -> None:
        try:
            report = self.server.analyze_session(session.session_id)
        except AnalysisNotReady as error:
            self._error(409, str(error))
            return
        except NoCompatibleCohort:
            self._reference_unavailable()
            return
        except Exception:
            self.server.store.end(session.session_id)
            self._failed_session_error(
                500, "The local analysis results could not be read."
            )
            return
        from .review_routes import profile_payload
        from .frontal_routes import frontal_payloads

        session = self.server.store.load(session.session_id)
        page = render_results(
            report,
            settings=session.state.get("report_preferences"),
            ideas=session.state.get("idea_choices"),
            profile=profile_payload(session, self.server.store),
            frontal=frontal_payloads(session, self.server.store),
            csrf_token=session.csrf_token,
            quit_after_end=self.server.shutdown_after_end,
        )
        self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")

    def _get_report(self, session: Session, *, format_name: str) -> None:
        try:
            report = self.server.analyze_session(session.session_id)
        except AnalysisNotReady as error:
            self._error(409, str(error))
            return
        except NoCompatibleCohort:
            self._reference_unavailable()
            return
        except Exception:
            self.server.store.end(session.session_id)
            self._failed_session_error(500, "The local report could not be created.")
            return
        from .review_routes import profile_payload
        from .frontal_routes import frontal_payloads

        session = self.server.store.load(session.session_id)
        payload = build_export_payload(
            report,
            session.demographics,
            settings=session.state.get("report_preferences"),
            ideas=session.state.get("idea_choices"),
            profile=profile_payload(session, self.server.store),
            frontal=frontal_payloads(session, self.server.store),
        )
        if format_name == "json":
            body = render_json_export(payload)
            content_type = "application/json; charset=utf-8"
            filename = "facial-measurements.json"
            security_headers = None
        else:
            body = render_html_export(payload).encode("utf-8")
            content_type = "text/html; charset=utf-8"
            filename = "facial-measurements.html"
            security_headers = {
                "Content-Security-Policy": (
                    "default-src 'none'; style-src 'unsafe-inline'; "
                    "img-src 'none'; form-action 'none'; base-uri 'none'; "
                    "frame-ancestors 'none'"
                )
            }
        self._send(
            200,
            body,
            content_type,
            extra_headers=[
                (
                    "Content-Disposition",
                    f'attachment; filename="{filename}"',
                )
            ],
            security_headers=security_headers,
        )

    def _catalog_uploads(self, session: Session, form: cgi.FieldStorage) -> None:
        catalog = PhotoCatalog(self.server.store, session.session_id)
        for item in self._file_fields(form, "photos"):
            filename = item.filename or "unnamed image"
            try:
                content = self._bounded_file_bytes(item.file, MAX_FILE_BYTES + 1)
                catalog.add_bytes(filename, content)
            except PhotoRejected:
                continue

    @staticmethod
    def _bounded_file_bytes(stream: Any, limit: int) -> bytes:
        temporary = tempfile.TemporaryFile(mode="w+b")
        try:
            remaining = limit
            while remaining > 0:
                chunk = stream.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                temporary.write(chunk)
                remaining -= len(chunk)
            temporary.seek(0)
            return temporary.read(limit)
        finally:
            temporary.close()

    def _multipart_form(self, content_length: int) -> cgi.FieldStorage | None:
        content_type = self.headers.get("Content-Type", "")
        media_type, parameters = cgi.parse_header(content_type)
        if media_type.lower() != "multipart/form-data" or not parameters.get(
            "boundary"
        ):
            self._error(415, "Expected a multipart form upload.")
            return None
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": str(content_length),
                },
                keep_blank_values=True,
                limit=content_length,
                max_num_fields=MAX_MULTIPART_FIELDS,
            )
            self._validate_text_parts(form)
            return form
        except (OSError, TypeError, ValueError):
            self._error(400, "The multipart form could not be read.")
            return None

    @staticmethod
    def _validate_text_parts(form: cgi.FieldStorage) -> None:
        for item in form.list or []:
            if item.filename is not None or item.file is None:
                continue
            position = item.file.tell()
            item.file.seek(0, 2)
            size = item.file.tell()
            item.file.seek(position)
            if size > MAX_TEXT_FIELD_CHARS:
                raise ValueError("multipart text field is too large")

    def _simple_form(self, content_length: int) -> dict[str, list[str]] | None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].lower()
        if content_type != "application/x-www-form-urlencoded":
            self._error(415, "Expected a form submission.")
            return None
        raw = self.rfile.read(content_length)
        if len(raw) != content_length:
            self._error(400, "The form body ended early.")
            return None
        try:
            return parse_qs(
                raw.decode("utf-8"),
                keep_blank_values=True,
                max_num_fields=128,
            )
        except (UnicodeDecodeError, ValueError):
            self._error(400, "The form body is invalid.")
            return None

    @staticmethod
    def _text_fields(form: cgi.FieldStorage) -> dict[str, list[str]]:
        fields: dict[str, list[str]] = {}
        items = form.list or []
        for item in items:
            if item.filename is not None:
                continue
            value = item.value
            if isinstance(value, bytes):
                value = value.decode("utf-8", "strict")
            fields.setdefault(item.name, []).append(str(value))
        return fields

    @staticmethod
    def _file_fields(form: cgi.FieldStorage, name: str) -> list[cgi.FieldStorage]:
        return [
            item
            for item in (form.list or [])
            if item.name == name and item.filename is not None and item.file is not None
        ]

    @staticmethod
    def _one(fields: dict[str, list[str]], name: str) -> str | None:
        values = fields.get(name, [])
        return values[0] if len(values) == 1 else None

    def _pending_response(self, session: Session, message: str) -> None:
        action = f"/sessions/{session.session_id}/photos"
        body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Add more photos</title><link rel="stylesheet" href="/static/app.css"></head>
<body><main class="pending-panel"><h1>{html.escape(message)}</h1>
<p>The session is saved locally. Add at least one valid JPEG or PNG photo to begin.</p>
<form method="post" action="{html.escape(action)}" enctype="multipart/form-data">
<input type="hidden" name="csrf_token" value="{html.escape(session.csrf_token)}">
<input type="file" name="photos" accept="image/jpeg,image/png" multiple required>
<button type="submit">Add photos</button></form></main></body></html>""".encode("utf-8")
        self._send(
            422,
            body,
            "text/html; charset=utf-8",
            extra_headers=[
                ("Set-Cookie", self._cookie_header(SESSION_COOKIE, session.session_id)),
                ("Set-Cookie", self._cookie_header(BOOTSTRAP_COOKIE, "", max_age=0)),
            ],
        )

    def _valid_csrf(self, session: Session, submitted: str | None) -> bool:
        if self._tokens_match(submitted, session.csrf_token):
            return True
        self._error(403, "The session form token is invalid. Reload and try again.")
        return False

    @staticmethod
    def _tokens_match(first: str | None, second: str | None) -> bool:
        return bool(first and second and secrets.compare_digest(first, second))

    def _owned_session(self, session_id: str) -> Session | None:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            return None
        owned = self._cookie_value(SESSION_COOKIE)
        if not self._tokens_match(owned, session_id):
            return None
        try:
            session = self.server.store.load(session_id)
            self.server.store.claim(session_id)
            return session
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return None

    def _content_length(self) -> int | None:
        if self.headers.get("Transfer-Encoding"):
            self._error(400, "Chunked request bodies are not accepted.")
            return None
        raw = self.headers.get("Content-Length")
        if raw is None:
            self._error(411, "Content-Length is required.")
            return None
        try:
            length = int(raw, 10)
        except ValueError:
            self._error(400, "Content-Length is invalid.")
            return None
        if length < 0:
            self._error(400, "Content-Length is invalid.")
            return None
        if length > self.server.max_request_bytes:
            self.close_connection = True
            self._error(413, "The request exceeds the local upload limit.")
            return None
        return length

    def _valid_host(self) -> bool:
        host = self.headers.get("Host", "")
        if LOCAL_HOST_PATTERN.fullmatch(host):
            return True
        self._error(421, "This local server accepts only loopback Host values.")
        return False

    def _rate_allowed(self, *, mutating: bool) -> bool:
        if self.server.rate_limiter.allow(self.client_address[0], mutating=mutating):
            return True
        self._error(
            429,
            "Too many local requests. Wait briefly and try again.",
            extra_headers=[("Retry-After", "60")],
        )
        return False

    def _valid_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        parsed = urlsplit(origin)
        host = self.headers.get("Host", "")
        if (
            parsed.scheme == "http"
            and not parsed.username
            and not parsed.password
            and not parsed.path.rstrip("/")
            and not parsed.query
            and not parsed.fragment
            and parsed.netloc == host
        ):
            return True
        self.close_connection = True
        self._error(403, "The request did not come from this local application.")
        return False

    def _serve_static(self, filename: str, content_type: str) -> None:
        path = self.server.static_dir / filename
        try:
            content = path.read_bytes()
        except OSError:
            self._error(404, "Asset not found")
            return
        self._send(200, content, content_type)

    def _cookie_value(self, name: str) -> str | None:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except CookieError:
            return None
        morsel = cookie.get(name)
        return morsel.value if morsel is not None else None

    @staticmethod
    def _cookie_header(name: str, value: str, *, max_age: int | None = None) -> str:
        parts = [
            f"{name}={value}",
            "Path=/",
            "Secure",
            "HttpOnly",
            "SameSite=Strict",
        ]
        if max_age is not None:
            parts.append(f"Max-Age={max_age}")
        return "; ".join(parts)

    def _redirect_to_session(self, location: str, session: Session) -> None:
        self._redirect(
            location,
            extra_headers=[
                ("Set-Cookie", self._cookie_header(SESSION_COOKIE, session.session_id)),
                ("Set-Cookie", self._cookie_header(BOOTSTRAP_COOKIE, "", max_age=0)),
            ],
        )

    def _redirect(
        self,
        location: str,
        *,
        extra_headers: list[tuple[str, str]] | None = None,
    ) -> None:
        headers = [("Location", location)]
        headers.extend(extra_headers or [])
        self._send(303, b"", "text/plain; charset=utf-8", extra_headers=headers)

    def _failed_session_error(self, status: int, message: str) -> None:
        self._error(
            status,
            message,
            extra_headers=[
                (
                    "Set-Cookie",
                    self._cookie_header(SESSION_COOKIE, "", max_age=0),
                )
            ],
        )

    def _reference_unavailable(self) -> None:
        self._error(422,
            "No matching reference is available for these comparison selections. "
            "This public edition currently supports male/female reference labels "
            "with a single listed ethnicity. No replacement population was inferred.")

    def _error(
        self,
        status: int,
        message: str,
        *,
        allow: str | None = None,
        extra_headers: list[tuple[str, str]] | None = None,
    ) -> None:
        body = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            f"<title>{status}</title><h1>{status}</h1><p>{html.escape(message)}</p>"
            "</html>"
        ).encode("utf-8")
        headers = list(extra_headers or [])
        if allow:
            headers.append(("Allow", allow))
        self._send(status, body, "text/html; charset=utf-8", extra_headers=headers)

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        extra_headers: list[tuple[str, str]] | None = None,
        security_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        headers = dict(SECURITY_HEADERS)
        headers.update(security_headers or {})
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if self.server.shutdown_after_end:
            self.send_header("Connection", "close")
        for name, value in extra_headers or []:
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(body)
        if self.server.shutdown_after_end:
            self.close_connection = True


def make_server(
    *,
    base_dir: Path,
    host: str = LOOPBACK_HOST,
    port: int = DEFAULT_PORT,
    analysis_factory: AnalysisFactory | None = None,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    shutdown_after_end: bool = False,
    rate_limiter: SlidingWindowRateLimiter | None = None,
    history_dir: Path | None = None,
) -> LocalAnalysisHTTPServer:
    """Construct, but do not start, a numeric-loopback application server."""
    if host != LOOPBACK_HOST:
        raise ValueError("the local analysis server binds only to 127.0.0.1")
    if isinstance(max_request_bytes, bool) or max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive")
    store = SessionStore(Path(base_dir))
    store.expire_stale()
    factory = analysis_factory or ProgressiveAnalysisService
    analysis = factory(store)
    return LocalAnalysisHTTPServer(
        (host, port),
        store=store,
        analysis=analysis,
        max_request_bytes=int(max_request_bytes),
        shutdown_after_end=shutdown_after_end,
        rate_limiter=rate_limiter,
        history_dir=history_dir,
    )


def serve(
    *,
    host: str = LOOPBACK_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    base_dir: Path | None = None,
    shutdown_after_end: bool = False,
    on_ready: Callable[[str], None] | None = None,
    history_dir: Path | None = None,
) -> int:
    """Run the local UI until interrupted."""
    session_dir = base_dir or Path(tempfile.gettempdir()) / "facial-analysis-sessions"
    server = make_server(
        base_dir=session_dir,
        host=host,
        port=port,
        shutdown_after_end=shutdown_after_end,
        history_dir=history_dir
        or Path.home() / "Library/Application Support/Facial Analysis/history",
    )
    actual_host, actual_port = server.server_address
    url = f"http://{actual_host}:{actual_port}/"
    try:
        print(f"Local facial analysis: {url} (Ctrl-C to stop)")
        if on_ready is not None:
            on_ready(url + "#history=" + server.history_capability)
        if open_browser:
            webbrowser.open(url + "#history=" + server.history_capability)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = [
    "LocalAnalysisHTTPServer",
    "SlidingWindowRateLimiter",
    "make_server",
    "serve",
]
