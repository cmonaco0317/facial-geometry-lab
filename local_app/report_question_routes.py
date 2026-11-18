"""Owned, CSRF-protected question forms bound to explicitly saved report revisions."""

from __future__ import annotations

import html
from typing import Any

from .frontal_routes import frontal_payloads
from .question_session import render_questions, update_questions
from .report import build_export_payload
from .report_questions import QUESTION_KINDS, available_topics, freeze_report
from .review_routes import profile_payload

STATE_KEY = "report_questions"


def _context(
    handler: Any, session: Any, *, include_experiment: bool = True
) -> tuple[dict, Any]:
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
        experiment_state=(
            current.state.get("photo_experiment") if include_experiment else None
        ),
    )
    return payload, store.load(session.session_id)


def get_questions(handler: Any, session: Any) -> None:
    from .render import _document

    try:
        current = handler.server.store.load(session.session_id)
        snapshot, current_error = None, ""
        try:
            payload, current = _context(handler, session)
            snapshot = freeze_report(payload)
        except ValueError as error:
            current_error = str(error)
        except (OSError, RuntimeError):
            current_error = "The current report cannot be prepared for saving. Existing saved questions are still available."
        state = current.state.get(STATE_KEY, {})
        base = f"/sessions/{session.session_id}"
        esc = html.escape
        binding = snapshot["revision"] if snapshot else ""
        hidden = f'<input type="hidden" name="csrf_token" value="{esc(session.csrf_token,quote=True)}"><input type="hidden" name="revision" value="{state.get("revision",0)}"><input type="hidden" name="binding" value="{binding}">'
        body = f'<main class="results-shell"><a href="{base}/results">Back to report</a><h1>Ask about a saved report</h1><p>Save the report you want to discuss. Answers use its stored readings and source records. Later edits stay separate until you save a new report. The last two saved reports support an arithmetic construction comparison when their photo and method bindings match; this does not establish physical progress.</p><p>Report exports include both saved contexts, their source-image crops, illustrations and your saved answers. Remove saved questions here to exclude those earlier contexts from subsequent exports.</p>'
        if current_error:
            body += f'<p role="status">{esc(current_error)}</p>'
        save_button = (
            '<button name="action" value="freeze">Save current report for questions</button>'
            if snapshot
            else '<button type="button" disabled>Current report cannot be saved for questions</button>'
        )
        body += f'<form method="post" action="{base}/questions">{hidden}{save_button}<button name="action" value="reset">Remove saved questions</button></form>'
        if state.get("current"):
            if snapshot and state["current"]["revision"] != snapshot["revision"]:
                body += '<p role="status">Your current report has changed. Questions below still use the earlier saved report. Save the current report to update the context.</p>'
            kinds = "".join(
                f'<option value="{key}">{esc(label)}</option>'
                for key, label in QUESTION_KINDS.items()
            )
            topics = "".join(
                f'<option value="{esc(row["id"],quote=True)}">{esc(row["label"])}</option>'
                for row in available_topics(state["current"])
            )
            body += f'<form class="preferences-form" method="post" action="{base}/questions">{hidden}<label>Question<select name="kind">{kinds}</select></label><label>Reading<select name="target">{topics}</select></label><label>Other feature, if absent from the list<input name="other_target" maxlength="160"></label><button name="action" value="ask">Answer from the saved report</button></form>'
        body += render_questions(state) + "</main>"
        handler._send(
            200,
            _document(
                title="Saved report questions", body=body, page_class="results-page"
            ).encode(),
            "text/html; charset=utf-8",
        )
    except ValueError as error:
        handler._error(422, str(error))
    except (OSError, RuntimeError):
        handler._error(
            500,
            "Saved report questions could not be opened. Your session is still available.",
        )


def save_question(handler: Any, session: Any, content_length: int) -> None:
    if content_length > 8000:
        handler._error(413, "This question form is too large.")
        return
    fields = handler._simple_form(content_length)
    if fields is None or not handler._valid_csrf(
        session, handler._one(fields, "csrf_token")
    ):
        return
    required = {"csrf_token", "revision", "binding", "action"}
    try:
        if (
            not required <= set(fields)
            or set(fields) - required - {"kind", "target", "other_target"}
            or any(len(v) != 1 for v in fields.values())
        ):
            raise ValueError("Unsupported, missing or repeated question fields.")
        values = {k: v[0] for k, v in fields.items()}
        if values["action"] == "freeze":
            payload, current = _context(handler, session)
        else:
            payload, current = {}, handler.server.store.load(session.session_id)
        current.state[STATE_KEY] = update_questions(
            current.state.get(STATE_KEY, {}),
            payload,
            expected=int(values["revision"]),
            binding=values["binding"],
            action=values["action"],
            kind=values.get("kind", "scope"),
            target=values.get("other_target", "").strip() or values.get("target", ""),
        )
        handler.server.store.save(current)
        handler._redirect(
            f"/sessions/{session.session_id}/questions#saved-report-questions"
        )
    except ValueError as error:
        handler._error(
            409 if "stale" in str(error) or "earlier report" in str(error) else 422,
            str(error),
        )
    except (OSError, RuntimeError):
        handler._error(
            500, "The question could not be saved. Your session is still available."
        )
