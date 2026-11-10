"""Authenticated session extensions for report preferences and profile review."""

from __future__ import annotations

from typing import Any

from .consultation import PREFERENCE_OPTIONS, preferences
from .photos import PhotoCatalog
from .profile_review import ANCHORS, measure_profile


def profile_payload(session: Any, store: Any) -> dict | None:
    from .profile_routes import profile_payloads

    return profile_payloads(session, store)


def get_review(handler: Any, session: Any) -> None:
    from .profile_routes import get_profile

    get_profile(handler, session)


def update_review(
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
    if resource in {"profile", "profile-clear"} and (
        "method" in fields or "revision" in fields
    ):
        from .profile_routes import update_profile

        update_profile(handler, session, resource, fields)
        return
    try:
        if resource == "ideas":
            from .idea_choices import update

            if set(fields) != {"csrf_token", "revision", "idea", "action"} or any(
                len(v) != 1 for v in fields.values()
            ):
                raise ValueError("Unsupported or repeated idea fields.")
            current = handler.server.store.load(session.session_id)
            try:
                current.state["idea_choices"] = update(
                    current.state.get("idea_choices"),
                    handler._one(fields, "action"),
                    handler._one(fields, "idea"),
                    expected=int(handler._one(fields, "revision") or ""),
                )
            except RuntimeError as error:
                handler._error(409, str(error))
                return
            handler.server.store.save(current)
            handler._redirect(f"/sessions/{session.session_id}/results")
            return
        allowed = {"csrf_token"}
        if resource == "preferences":
            allowed |= set(PREFERENCE_OPTIONS)
        elif resource == "profile":
            allowed |= {"photo", "confirmed", "active_anchor"} | {
                f"{k}_{axis}" for k in ANCHORS for axis in ("x", "y")
            }
        if set(fields) - allowed or any(len(v) != 1 for v in fields.values()):
            raise ValueError("The form contains unsupported or repeated fields.")
        # Load under the handler lock so concurrent analysis cannot be overwritten.
        current = handler.server.store.load(session.session_id)
        if resource == "preferences":
            current.state["report_preferences"] = preferences(
                {k: v[0] for k, v in fields.items() if k != "csrf_token"}
            )
        elif resource == "profile-clear":
            current.state.pop("profile_review", None)
        else:
            if "active_anchor" in fields and fields["active_anchor"][0] not in ANCHORS:
                raise ValueError("Unsupported selected landmark.")
            if fields.get("confirmed") != ["yes"]:
                raise ValueError(
                    "Review the lateral photograph and confirm all landmark positions."
                )
            entries = PhotoCatalog(handler.server.store, session.session_id).entries()
            entry = next(
                (e for e in entries if e.photo_id == handler._one(fields, "photo")),
                None,
            )
            if entry is None:
                raise ValueError("Choose a photograph belonging to this session.")
            points = {
                key: [
                    float(handler._one(fields, f"{key}_{axis}") or "") / 100
                    for axis in ("x", "y")
                ]
                for key in ANCHORS
            }
            measure_profile(points, entry.width, entry.height)
            current.state["profile_review"] = {
                "photo_id": entry.photo_id,
                "points": points,
            }
        handler.server.store.save(current)
    except (TypeError, ValueError) as error:
        handler._error(422, str(error))
        return
    handler._redirect(f"/sessions/{session.session_id}/results")
