"""Owned-session frontal review with exact photo binding and stale-save detection."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .frontal_page import render_frontal
from .landmark_review import (
    ANCHORS,
    ReviewConflict,
    detected_points,
    observation_binding,
    public_review,
    save_review,
    validate_points,
    forehead_points,
)
from .photos import PhotoCatalog

STATE_KEY = "frontal_reviews"


def _instrument() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).with_name("landmark_review.py"),
        Path(__file__).with_name("measurement_registry.py"),
        Path(__file__).with_name("automatic_forehead.py"),
        Path(__file__).parent.parent / "face_landmarker.task",
        Path(__file__).parent.parent / "forehead_context.bin",
    ):
        if not path.exists():
            digest.update(b"missing:" + path.name.encode())
            continue
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _selection(session: Any, store: Any, photo_id: str | None) -> tuple:
    entries = PhotoCatalog(store, session.session_id).entries()
    selected = (
        next((e for e in entries if e.photo_id == photo_id), None)
        if photo_id is not None
        else next(iter(entries), None)
    )
    if selected is None:
        raise LookupError("Photograph not found in this session.")
    return entries, selected


def _automatic_status(session: Any, photo_id: str) -> str | None:
    from .automatic_forehead import EXPLANATIONS

    rows = session.state.get("analysis_report", {}).get("axes", {}).get(
        "harmony", {}).get("details", {}).get("automatic_forehead", [])
    return next((r["status"] for r in rows if r.get("photo_id") == photo_id
                 and r.get("status") in EXPLANATIONS), None)


def _record(session: Any, store: Any, entry: Any) -> dict:
    path = store.path_for(session.session_id, "uploads", entry.stored_name)
    # Bind normalized stored bytes as well as catalog identity and instrument.
    content = hashlib.sha256(path.read_bytes()).hexdigest()
    binding = observation_binding(entry, _instrument() + content)
    current = session.state.get(STATE_KEY, {}).get(entry.photo_id)
    automatic_status = _automatic_status(session, entry.photo_id)
    if current is not None and current.get("binding") == binding:
        if current.get("automatic_forehead_status") != automatic_status:
            current = {**current, "automatic_forehead_status": automatic_status,
                       "revision": current["revision"] + 1}
            session.state[STATE_KEY][entry.photo_id] = current
            store.save(session)
        return current
    points = validate_points(detected_points(path))
    record = dict(
        binding=binding,
        revision=(current["revision"] + 1 if current else 1),
        automatic=deepcopy(points),
        points=points,
        reviewed=False,
        forehead_visibility="unassessed",
        automatic_forehead_status=automatic_status,
    )
    session.state.setdefault(STATE_KEY, {})[entry.photo_id] = record
    store.save(session)
    return record


def _display(
    handler: Any,
    session: Any,
    entries: tuple,
    entry: Any,
    record: dict,
    *,
    draft: dict | None = None,
    draft_visibility: str | None = None,
) -> None:
    import cv2

    image = cv2.imread(
        str(
            handler.server.store.path_for(
                session.session_id, "uploads", entry.stored_name
            )
        )
    )
    if image is None:
        raise ValueError("This photograph could not be opened.")
    scale = min(1, 1200 / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("This photograph could not be displayed.")
    page = render_frontal(
        session, entries, entry, encoded.tobytes(), record, draft=draft,
        draft_visibility=draft_visibility,
    )
    handler._send(200, page.encode(), "text/html; charset=utf-8")


def get_frontal(handler: Any, session: Any) -> None:
    try:
        query = parse_qs(
            urlsplit(handler.path).query, keep_blank_values=True, max_num_fields=1
        )
        if set(query) - {"photo"} or any(len(v) != 1 for v in query.values()):
            raise ValueError("Choose one photograph.")
        entries, entry = _selection(
            session, handler.server.store, query.get("photo", [None])[0]
        )
        _display(
            handler,
            session,
            entries,
            entry,
            _record(session, handler.server.store, entry),
        )
    except LookupError as error:
        handler._error(404, str(error))
    except ValueError as error:
        handler._error(422, str(error))
    except (OSError, RuntimeError):
        handler._error(
            500,
            "The frontal review could not be prepared. Your session is still available; retry from the report.",
        )


def update_frontal(
    handler: Any, session: Any, resource: str, content_length: int
) -> None:
    if content_length > 100_000:
        handler._error(413, "This form is too large.")
        return
    fields = handler._simple_form(content_length)
    if fields is None or not handler._valid_csrf(
        session, handler._one(fields, "csrf_token")
    ):
        return
    try:
        allowed = {"csrf_token", "photo", "revision", "binding"}
        if resource == "frontal":
            allowed |= (
                {"action", "confirmed", "active_anchor", "acknowledged", "forehead_visibility"}
                | {key + "_" + axis for key in ANCHORS for axis in ("x", "y")}
                | {"missing_" + key for key in ANCHORS}
            )
        if set(fields) - allowed or any(len(v) != 1 for v in fields.values()):
            raise ValueError("Unsupported or repeated frontal review fields.")
        current = handler.server.store.load(session.session_id)
        entries, entry = _selection(
            current, handler.server.store, handler._one(fields, "photo") or ""
        )
        record = _record(current, handler.server.store, entry)
        revision = int(handler._one(fields, "revision") or "")
        binding = handler._one(fields, "binding") or ""
        if revision != record["revision"] or binding != record["binding"]:
            raise ReviewConflict(
                "This review changed or the photo changed. Reload before saving."
            )
        if resource == "frontal-clear":
            current.state[STATE_KEY][entry.photo_id] = {
                **deepcopy(record),
                "points": deepcopy(record["automatic"]),
                "revision": revision + 1,
                "reviewed": False,
                "forehead_visibility": "unassessed",
                "visibility_pending": False,
            }
        else:
            action = handler._one(fields, "action")
            if action not in {"save", "preview"}:
                raise ValueError("Choose preview or save.")
            if (
                "active_anchor" in fields
                and handler._one(fields, "active_anchor") not in ANCHORS
            ):
                raise ValueError("Unsupported selected point.")
            points = {}
            for key in ANCHORS:
                if "missing_" + key in fields:
                    if fields["missing_" + key] != ["yes"]:
                        raise ValueError("Invalid visibility field.")
                    points[key] = None
                else:
                    points[key] = [
                        float(handler._one(fields, key + "_" + axis) or "") / 100
                        for axis in ("x", "y")
                    ]
            points = validate_points(points)
            visibility = handler._one(fields, "forehead_visibility") or "unassessed"
            if points["hairline"] is not None and visibility == "unassessed":
                raise ValueError("Confirm that the central hairline is visible before measuring it.")
            points = forehead_points(points, visibility, record.get("automatic_forehead_status"))
            if action == "preview":
                _display(handler, current, entries, entry, record, draft=points,
                         draft_visibility=visibility)
                return
            if fields.get("confirmed") != ["yes"]:
                raise ValueError(
                    "Check the visible points and confirm review before saving."
                )
            current.state[STATE_KEY][entry.photo_id] = save_review(
                record,
                points,
                expected_revision=revision,
                binding=binding,
                acknowledged=fields.get("acknowledged") == ["yes"],
                forehead_visibility=visibility,
            )
        handler.server.store.save(current)
        handler._redirect(f"/sessions/{session.session_id}/results")
    except ReviewConflict as error:
        handler._error(409, str(error))
    except LookupError as error:
        handler._error(404, str(error))
    except (ValueError, TypeError) as error:
        handler._error(422, str(error))
    except (OSError, RuntimeError):
        handler._error(
            500,
            "The frontal review could not be updated. Your session is still available.",
        )


def frontal_payloads(session: Any, store: Any) -> list[dict]:
    records = session.state.get(STATE_KEY, {})
    result = []
    for index, entry in enumerate(PhotoCatalog(store, session.session_id).entries(), 1):
        record = records.get(entry.photo_id)
        if record and (record.get("reviewed") or record.get("visibility_pending")):
            content = hashlib.sha256(
                store.path_for(
                    session.session_id, "uploads", entry.stored_name
                ).read_bytes()
            ).hexdigest()
            if record.get("binding") != observation_binding(
                entry, _instrument() + content
            ):
                continue
            payload = public_review({**record, "automatic_forehead_status":
                                     _automatic_status(session, entry.photo_id)}, entry.width, entry.height)
            if payload:
                result.append({**payload, "photo_number": index,
                               "evidence_photo_id": entry.photo_id})
    return result
