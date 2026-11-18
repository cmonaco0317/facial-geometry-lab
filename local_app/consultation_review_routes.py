"""Optional reviewer-note forms using the same current report binding as exports."""

from __future__ import annotations

import html
from typing import Any

from .consultation_review import render_review, update_review
from .frontal_routes import frontal_payloads
from .report import build_export_payload
from .review_routes import profile_payload


def _context(handler: Any, session: Any) -> tuple[dict, Any]:
    report = handler.server.analyze_session(session.session_id)
    store = handler.server.store
    current = store.load(session.session_id)
    payload = build_export_payload(
        report,
        current.demographics,
        adult_confirmed=current.adult_confirmed,
        settings=current.state.get("report_preferences"),
        ideas=current.state.get("idea_choices"),
        frontal=frontal_payloads(current, store),
        profile=profile_payload(current, store),
        appearance_review=current.state.get("appearance_review"),
        reviewer_state=current.state.get("consultation_reviewer"),
        experiment_state=current.state.get("photo_experiment"),
    )
    return payload["consultation"], store.load(session.session_id)


def get_review(handler: Any, session: Any) -> None:
    from .render import _document

    try:
        payload, current = _context(handler, session)
        state = current.state.get("consultation_reviewer", {})
        entry = state.get("entry") or {}
        esc = html.escape
        base = f"/sessions/{session.session_id}"
        fields = "".join(
            f'<label>{label}<input name="{key}" type="{kind}" maxlength="{limit}" value="{esc(entry.get(key,""),quote=True)}"></label>'
            for key, label, kind, limit in (
                ("reviewer_name", "Reviewer identity as supplied", "text", 100),
                ("reviewed_on", "Actual review date", "date", 10),
                ("scope", "What the reviewer examined", "text", 200),
            )
        )
        body = f'<main class="results-shell"><a href="{base}/results">Back to report</a><h1>Optional reviewer record</h1><p>Record a review only when it actually happened. Identity, qualifications and conclusions are supplied by you, not verified by the app. This record appears in exports and never changes measured values or creates new evidence.</p>'
        body += render_review(payload["reviewer_record"])
        body += f'<form class="preferences-form" method="post" action="{base}/consultation-review"><input type="hidden" name="csrf_token" value="{esc(session.csrf_token,quote=True)}"><input type="hidden" name="revision" value="{state.get("revision",0)}"><input type="hidden" name="binding" value="{payload["reviewer_context_binding"]}">{fields}<label>Reviewer note<textarea name="note" maxlength="1000">{esc(entry.get("note",""))}</textarea></label><label><input type="checkbox" name="confirmed" value="yes">I confirm this reviewer actually examined the stated scope of this current report.</label><button name="action" value="save">Record review</button><button name="action" value="reset">Remove current record</button></form><p>Changing a source photo, landmark, appearance finding, comparison or preference marks an older record stale. Review its scope again before recording it against changed inputs.</p></main>'
        page = _document(
            title="Optional reviewer record", body=body, page_class="results-page"
        )
        handler._send(200, page.encode(), "text/html; charset=utf-8")
    except ValueError as error:
        handler._error(422, str(error))
    except (OSError, RuntimeError):
        handler._error(
            500,
            "The reviewer form could not be opened. Your session is still available.",
        )


def save_review(handler: Any, session: Any, content_length: int) -> None:
    if content_length > 16000:
        handler._error(413, "This reviewer form is too large.")
        return
    fields = handler._simple_form(content_length)
    if fields is None or not handler._valid_csrf(
        session, handler._one(fields, "csrf_token")
    ):
        return
    required = {"csrf_token", "revision", "binding", "action"}
    allowed = required | {"reviewer_name", "reviewed_on", "scope", "note", "confirmed"}
    try:
        if (
            not required <= set(fields)
            or set(fields) - allowed
            or any(len(v) != 1 for v in fields.values())
        ):
            raise ValueError("Unsupported, missing or repeated reviewer fields.")
        payload, current = _context(handler, session)
        values = {k: v[0] for k, v in fields.items()}
        state = update_review(
            current.state.get("consultation_reviewer", {}),
            expected=int(values["revision"]),
            submitted=values["binding"],
            current=payload["reviewer_context_binding"],
            action=values["action"],
            reviewer_name=values.get("reviewer_name", ""),
            reviewed_on=values.get("reviewed_on", ""),
            scope=values.get("scope", ""),
            note=values.get("note", ""),
            confirmed=values.get("confirmed") == "yes",
        )
        current.state["consultation_reviewer"] = state
        handler.server.store.save(current)
        handler._redirect(
            f"/sessions/{session.session_id}/results#consultation-reviewer"
        )
    except ValueError as error:
        handler._error(
            409 if "stale" in str(error) or "earlier report" in str(error) else 422,
            str(error),
        )
    except (OSError, RuntimeError):
        handler._error(
            500,
            "The reviewer record could not be saved. Your session is still available.",
        )
