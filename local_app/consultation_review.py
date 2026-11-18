"""Optional user-attested reviewer metadata; never a substitute for measurement."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import html
import json
import re
from typing import Any

VERSION = "consultation-review-record-v1"


def analysis_context(report: Any, experiment_state: dict | None = None) -> dict:
    context = dict(
        photo_fingerprint=report.photo_fingerprint,
        instrument_version=report.instrument_fingerprint,
        reference=report.reference.to_dict(),
    )
    if experiment_state and experiment_state.get("entry"):
        from .photo_experiments import public_experiment

        context["photo_experiment_id"] = public_experiment(experiment_state)["entry"]["id"]
    return context


def report_binding(
    observations: list[dict], settings: dict, appearance: dict, context: dict
) -> str:
    payload = dict(
        observations=observations,
        preferences=settings,
        appearance=appearance,
        context=context,
    )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _text(value: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        raise ValueError("Provide a bounded reviewer name, date, scope and note.")
    return value.strip()


def update_review(
    state: dict,
    *,
    expected: int,
    submitted: str,
    current: str,
    action: str,
    reviewer_name: str = "",
    reviewed_on: str = "",
    scope: str = "",
    note: str = "",
    confirmed: bool = False,
) -> dict:
    if type(expected) is not int or expected != state.get("revision", 0):
        raise ValueError("This reviewer form is stale. Reload the current report.")
    if (
        not isinstance(current, str)
        or not re.fullmatch("[a-f0-9]{64}", current)
        or submitted != current
    ):
        raise ValueError(
            "This reviewer form describes an earlier report. Reload before recording a review."
        )
    if action not in {"save", "reset"}:
        raise ValueError("Choose save or reset for the reviewer record.")
    updated = deepcopy(state) if state else dict(revision=0, entry=None, history=[])
    if len(updated["history"]) >= 100:
        raise ValueError(
            "The reviewer-record revision limit was reached. Save this report before starting a new session."
        )
    entry = None
    if action == "save":
        if confirmed is not True:
            raise ValueError(
                "Record a reviewer only when you confirm that this review actually occurred."
            )
        reviewer_name, scope, note = (
            _text(reviewer_name, 100),
            _text(scope, 200),
            _text(note, 1000),
        )
        reviewed_on = _text(reviewed_on, 10)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", reviewed_on):
            raise ValueError("Use the actual review date in YYYY-MM-DD form.")
        try:
            date.fromisoformat(reviewed_on)
        except ValueError as error:
            raise ValueError(
                "Use a valid calendar date for the actual review."
            ) from error
        entry = dict(
            version=VERSION,
            reviewer_name=reviewer_name,
            reviewed_on=reviewed_on,
            scope=scope,
            note=note,
            binding=current,
            user_attested=True,
        )
    updated["revision"] += 1
    updated["history"].append(
        dict(
            revision=updated["revision"],
            action=action,
            previous=deepcopy(updated.get("entry")),
            current=deepcopy(entry),
        )
    )
    updated["entry"] = entry
    return updated


def public_review(state: dict | None, binding: str) -> dict:
    state = state or {}
    entry = state.get("entry")
    if not entry:
        return dict(status="not_supplied", revision=state.get("revision", 0))
    if entry.get("version") != VERSION or entry.get("user_attested") is not True:
        raise ValueError("The reviewer record is unsupported; preserve it separately.")
    return dict(
        status="current" if entry.get("binding") == binding else "stale",
        revision=state.get("revision", 0),
        reviewer_name=_text(entry.get("reviewer_name"), 100),
        reviewed_on=_text(entry.get("reviewed_on"), 10),
        scope=_text(entry.get("scope"), 200),
        note=_text(entry.get("note"), 1000),
        attestation="User reports that this review occurred. Identity, qualifications and conclusions are not independently verified.",
    )


def render_review(record: dict, *, action: str | None = None) -> str:
    esc = html.escape
    if record["status"] == "not_supplied":
        body = "<p>No external reviewer record was supplied.</p>"
    else:
        body = (
            f'<p><strong>{esc(record["reviewer_name"])}</strong> · {esc(record["reviewed_on"])}</p>'
            f'<p>Recorded scope: {esc(record["scope"])}</p><p>{esc(record["note"])}</p>'
            f'<p>{esc(record["attestation"])}</p>'
        )
        if record["status"] == "stale":
            body = (
                "<p><strong>This record describes an earlier report and does not apply to current priorities. It is retained as historical context.</strong></p>"
                + body
            )
    link = (
        f'<p><a href="{esc(action,quote=True)}">Record or update an optional reviewer note</a></p>'
        if action
        else ""
    )
    return (
        '<section id="consultation-reviewer"><h2>Optional reviewer record</h2>'
        + body
        + link
        + "</section>"
    )
