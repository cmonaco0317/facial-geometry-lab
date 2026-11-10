"""Origin-bound archive API. No history credentials are stored in cookies."""

from __future__ import annotations
import base64
import binascii
import json
from typing import Literal
from pydantic import Field, ValidationError
from .archive_schema import (
    Strict,
    Short,
    Evidence,
    Annotation,
    compare,
    freeze_report,
    reference_fingerprint,
)
from .history import ArchiveError
from .cohorts import NoCompatibleCohort


class Request(Strict):
    operation: Literal[
        "list",
        "read",
        "save",
        "rename",
        "delete",
        "remove_evidence",
        "backup",
        "import",
        "compare",
        "restore",
        "photos",
        "preview",
    ]
    id: Short = ""
    other_id: Short = ""
    name: Short = ""
    profile: Short = ""
    protocol: Literal["unspecified", "casual", "controlled", "mixed"] = "unspecified"
    camera: Short = ""
    captured_at: Short = ""
    session: Short = ""
    session_csrf: Short = ""
    photos: list[Short] = Field(default_factory=list, max_length=30)
    password: str = Field(default="", max_length=1024)
    backup: str = Field(default="", max_length=8_100_000)
    photo: Short = ""
    style: Literal["eyewear", "hair", "reviewed"] = "eyewear"
    original: bool = False
    x: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    y: float = Field(default=0.4, ge=0, le=1, allow_inf_nan=False)
    scale: float = Field(default=0.4, ge=0.1, le=0.8, allow_inf_nan=False)


ALLOWED = {
    "list": set(),
    "read": {"id"},
    "save": {
        "name",
        "profile",
        "protocol",
        "camera",
        "captured_at",
        "session",
        "session_csrf",
        "photos",
    },
    "rename": {"id", "name"},
    "delete": {"id"},
    "remove_evidence": {"id"},
    "backup": {"id", "password"},
    "import": {"backup", "password"},
    "compare": {"id", "other_id"},
    "restore": {"id"},
    "photos": {"session", "session_csrf"},
    "preview": {
        "session",
        "session_csrf",
        "photo",
        "style",
        "x",
        "y",
        "scale",
        "original",
    },
}


def _json(handler, status, value):
    handler._send(
        status,
        json.dumps(value, allow_nan=False).encode(),
        "application/json; charset=utf-8",
    )


def handle(handler, path, length):
    server = handler.server
    if length > (10_000_000 if path == "/history/import" else 100_000):
        handler.close_connection = True
        handler._error(413, "Archive request exceeds its size limit.")
        return
    expected = f"http://127.0.0.1:{server.server_port}"
    if (
        handler.headers.get("Host") != expected.removeprefix("http://")
        or handler.headers.get("Origin") != expected
    ):
        handler._error(403, "History requires this exact local application origin.")
        return
    if server.archive is None:
        handler._error(503, "History is unavailable for this launch.")
        return
    if path == "/history/auth" or path == "/history/import":
        if not server.history_auth_limiter.allow(
            handler.client_address[0], mutating=True
        ):
            handler._error(
                429, "Wait 15 minutes before another archive unlock or backup attempt."
            )
            return
    try:
        if handler.headers.get("Content-Type") != "application/json":
            handler._error(415, "Expected JSON.")
            return
        raw = handler.rfile.read(length)
        if len(raw) != length:
            raise ValueError("Incomplete request.")
        payload = json.loads(raw)
        if path == "/history/auth":
            if (
                not isinstance(payload, dict)
                or set(payload) != {"capability"}
                or not isinstance(payload["capability"], str)
                or not handler._tokens_match(
                    payload["capability"], server.history_capability
                )
            ):
                handler._error(403, "Open history from the application launch window.")
                return
            _json(
                handler,
                200,
                dict(capability=server.history_capability, csrf=server.history_csrf),
            )
            return
        if not handler._tokens_match(
            handler.headers.get("X-Archive-Capability"), server.history_capability
        ) or not handler._tokens_match(
            handler.headers.get("X-Archive-CSRF"), server.history_csrf
        ):
            handler._error(
                403, "History is locked in this browser tab. Reopen the app."
            )
            return
        request = Request.model_validate(payload)
        if request.model_fields_set - {"operation"} - ALLOWED[request.operation]:
            raise ValueError("Unsupported fields for this archive operation.")
        if (path == "/history/import") != (request.operation == "import"):
            raise ValueError("Use the dedicated backup import endpoint.")
        if request.operation == "backup" and not server.history_auth_limiter.allow(
            handler.client_address[0], mutating=True
        ):
            handler._error(429, "Backup attempts are limited. Wait 15 minutes.")
            return
        with server.state_lock:
            result = dispatch(handler, request)
        if result is not None:
            _json(handler, 200, result)
    except NoCompatibleCohort:
        handler._reference_unavailable()
    except ArchiveError as error:
        # ArchiveError messages are fixed local implementation text, never imported values.
        handler._error(422, str(error))
    except (
        ValueError,
        TypeError,
        KeyError,
        binascii.Error,
        ValidationError,
        RecursionError,
    ):
        # Imported strings and filesystem/crypto details never become error HTML or logs.
        handler._error(
            422,
            "Archive operation failed. Check the selected report, required fields, password, Keychain access and storage limits. Existing numerical snapshots are preserved.",
        )
    except RuntimeError:
        handler._error(
            503,
            "Local analysis or Keychain access is unavailable. Retry from the report; existing archives remain available.",
        )
    except OSError:
        handler._error(
            503, "Local history storage is unavailable. Existing files were preserved."
        )


def owned(handler, request):
    session = handler._owned_session(request.session)
    if session is None or not handler._tokens_match(
        request.session_csrf, session.csrf_token
    ):
        raise ArchiveError("Owned session required.")
    return session


def evidence_for(session, store, selected, frontal, profiles):
    from .photos import PhotoCatalog

    entries = PhotoCatalog(store, session.session_id).entries()
    if len(set(selected)) != len(selected) or set(selected) - {
        e.photo_id for e in entries
    }:
        raise ArchiveError("Select owned source photos.")
    result = []
    for ordinal, entry in enumerate(entries, 1):
        if entry.photo_id not in selected:
            continue
        annotations = []
        for kind, rows in (("frontal", frontal), ("profile", profiles)):
            for row in rows:
                if row.get("photo_number") != ordinal:
                    continue
                annotations.append(
                    Annotation(
                        kind=kind,
                        method=row["method"],
                        contract=row.get("contract", ""),
                        revision=row.get("revision", 1),
                        forehead_visibility=row.get("forehead_visibility", "unassessed"),
                        points=row["diagram"]["points"],
                        facing=row.get("facing", ""),
                        level_degrees=float(
                            session.state.get("profile_reviews", {})
                            .get(entry.photo_id, {})
                            .get("level_degrees", 0.0)
                        ),
                    )
                )
        raw = store.path_for(
            session.session_id, "uploads", entry.stored_name
        ).read_bytes()
        result.append(
            Evidence(
                ordinal=ordinal,
                jpeg_b64=base64.b64encode(raw).decode(),
                annotations=annotations,
            )
        )
    return result


def dispatch(handler, r):
    server = handler.server
    archive = server.archive
    if r.operation == "list":
        return dict(
            records=archive.list(),
            limits=dict(records=100, record_bytes=6_000_000, total_bytes=200_000_000),
        )
    if r.operation == "read":
        return archive.read(r.id).model_dump()
    if r.operation == "rename":
        archive.rename(r.id, r.name)
        return dict(saved=True)
    if r.operation == "delete":
        archive.delete(r.id)
        return dict(deleted=True)
    if r.operation == "remove_evidence":
        archive.remove_evidence(r.id)
        return dict(deleted=True)
    if r.operation == "backup":
        backup = archive.backup(r.id, r.password)
        handler._send(
            200,
            backup,
            "application/octet-stream",
            extra_headers=[
                ("Content-Disposition", 'attachment; filename="facial-history.fab"')
            ],
        )
        return None
    if r.operation == "import":
        return dict(
            id=archive.import_backup(
                base64.b64decode(r.backup, validate=True), r.password
            ).id
        )
    if r.operation == "compare":
        return dict(
            comparison=compare(
                archive.read(r.id).snapshot, archive.read(r.other_id).snapshot
            ),
            before=archive.read(r.id).model_dump(),
            after=archive.read(r.other_id).model_dump(),
        )
    if r.operation == "restore":
        return restore(handler, archive.read(r.id))
    session = owned(handler, r)
    from .photos import PhotoCatalog

    entries = PhotoCatalog(server.store, session.session_id).entries()
    if r.operation == "photos":
        return dict(
            photos=[
                dict(id=e.photo_id, ordinal=n, width=e.width, height=e.height)
                for n, e in enumerate(entries, 1)
            ]
        )
    if r.operation == "preview":
        from .previews import make_preview

        entry = next((e for e in entries if e.photo_id == r.photo), None)
        if entry is None:
            raise ValueError("Select a source photograph.")
        record = session.state.get("frontal_reviews", {}).get(entry.photo_id)
        content = server.store.path_for(
            session.session_id, "uploads", entry.stored_name
        ).read_bytes()
        value = make_preview(
            content,
            style=r.style,
            x=r.x,
            y=r.y,
            scale=r.scale,
            review=record,
            original=r.original,
        )
        handler._send(
            200,
            value.content,
            "image/svg+xml",
            extra_headers=[
                (
                    "Content-Disposition",
                    'attachment; filename="illustrative-preview.svg"',
                )
            ],
            security_headers={
                "Content-Security-Policy": "default-src 'none'; img-src data:; style-src 'none'; script-src 'none'; sandbox"
            },
        )
        return None
    if r.operation == "save":
        from .frontal_routes import frontal_payloads
        from .review_routes import profile_payload

        report = server.analyze_session(session.session_id)
        session = server.store.load(session.session_id)
        frontal = frontal_payloads(session, server.store)
        profiles = profile_payload(session, server.store)
        profiles = (
            profiles if isinstance(profiles, list) else [profiles] if profiles else []
        )
        snap = freeze_report(
            report,
            session.demographics,
            profile=r.profile,
            protocol=r.protocol,
            camera=r.camera,
            captured_at=r.captured_at,
            reference=reference_fingerprint(),
            frontal=frontal,
            profiles=profiles,
            settings=session.state.get("report_preferences"),
            ideas=session.state.get("idea_choices"),
        )
        evidence = evidence_for(session, server.store, r.photos, frontal, profiles)
        record = archive.save(snap, evidence=evidence, name=r.name or None)
        return dict(id=record.id, evidence_count=len(evidence))
    raise ValueError("Unsupported operation.")


def restore(handler, record):
    from .contracts import Demographics
    from .photos import PhotoCatalog
    from .frontal_routes import _record as frontal_record
    from .landmark_review import measurement_contract
    from .profile_routes import _record as profile_record
    from .profile_v2 import contract as profile_contract

    if not record.evidence:
        raise ArchiveError("This report has no saved evidence.")
    server = handler.server
    d = record.snapshot.demographics
    demographics = Demographics(d.age, d.sex, tuple(d.ethnicities))
    from .cohorts import NoCompatibleCohort

    try:
        server.analysis.resolver.resolve(demographics)
    except NoCompatibleCohort as error:
        raise ArchiveError(
            "This saved report has no matching comparison reference in the public edition. "
            "The saved archive is unchanged; no photos were restored."
        ) from error
    session = server.store.create(demographics)
    try:
        for item in record.evidence:
            entry = PhotoCatalog(server.store, session.session_id).add_bytes(
                f"Saved photo {item.ordinal}.jpg",
                base64.b64decode(item.jpeg_b64, validate=True),
            )
            session = server.store.load(session.session_id)
            for annotation in item.annotations:
                if (
                    annotation.kind == "frontal"
                    and annotation.contract == measurement_contract()
                ):
                    current = frontal_record(session, server.store, entry)
                    session.state["frontal_reviews"][entry.photo_id] = {
                        **current,
                        "points": annotation.points,
                        "forehead_visibility": annotation.forehead_visibility,
                        "visibility_pending": annotation.forehead_visibility in {
                            "hairline_hidden", "forehead_hidden", "uncertain"
                        },
                        "reviewed": False,
                        "revision": current["revision"] + 1,
                    }
                elif (
                    annotation.kind == "profile"
                    and annotation.contract == profile_contract()
                ):
                    current = profile_record(session, server.store, entry)
                    session.state["profile_reviews"][entry.photo_id] = {
                        **current,
                        "points": annotation.points,
                        "facing": annotation.facing,
                        "level_degrees": annotation.level_degrees,
                        "reviewed": False,
                        "revision": current["revision"] + 1,
                    }
            session.state["report_preferences"] = dict(record.snapshot.preferences)
            session.state["restored_archive"] = dict(
                id=record.id, annotations_unconfirmed=True
            )
            server.store.save(session)
        server.analyze_session(session.session_id)
        handler._send(
            200,
            json.dumps(
                dict(
                    location=f"/sessions/{session.session_id}/results",
                    notice="New temporary session. Saved point hints require fresh review.",
                )
            ).encode(),
            "application/json; charset=utf-8",
            extra_headers=[
                (
                    "Set-Cookie",
                    handler._cookie_header("face_session", session.session_id),
                )
            ],
        )
        return None
    except Exception:
        server.store.end(session.session_id)
        raise
