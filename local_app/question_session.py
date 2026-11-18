"""Bounded saved-report questions and their self-contained evidence rendering."""

from __future__ import annotations

from copy import deepcopy
import html
import json

from .consultation_findings import evidence_anchor
from .observations import Observation, public_observations
from .report_questions import answer_question, available_topics, freeze_report

MAX_QUESTIONS = 20


def update_questions(
    state: dict,
    payload: dict,
    *,
    expected: int,
    binding: str,
    action: str,
    kind: str = "scope",
    target: str = "",
) -> dict:
    revision = state.get("revision", 0)
    if type(expected) is not int or expected != revision:
        raise ValueError("This question form is stale; reload it before saving.")
    if revision >= 1000:
        raise ValueError("This session has reached its question revision limit.")
    if action == "reset":
        return {
            "revision": revision + 1,
            "current": None,
            "previous": None,
            "answers": [],
        }
    result = deepcopy(state)
    if action == "freeze":
        snapshot = freeze_report(payload)
        if binding != snapshot["revision"]:
            raise ValueError("This form describes an earlier report; reload it first.")
        result = {"current": snapshot, "previous": state.get("current"), "answers": []}
    elif action == "ask":
        if not state.get("current"):
            raise ValueError("Save a report revision before asking a question.")
        answer = answer_question(
            state["current"], kind, target, previous=state.get("previous")
        )
        result["answers"] = (list(state.get("answers", [])) + [answer])[-MAX_QUESTIONS:]
    else:
        raise ValueError("Choose save, ask or remove saved questions.")
    result["revision"] = revision + 1
    return result


def public_questions(state: dict | None) -> dict:
    if not state or not state.get("current"):
        return {
            "revision": (state or {}).get("revision", 0),
            "current": None,
            "previous": None,
            "answers": [],
        }
    # Validate the exact context before exporting or rendering saved answers.
    available_topics(state["current"])
    if state.get("previous"):
        available_topics(state["previous"])
    if len(state.get("answers", [])) > MAX_QUESTIONS:
        raise ValueError("Saved questions exceed the session limit.")
    for answer in state.get("answers", []):
        expected = answer_question(
            state["current"],
            answer["kind"],
            answer["target"],
            previous=state.get("previous"),
        )
        if answer != expected:
            raise ValueError("A saved answer does not match its report revision.")
    return deepcopy(state)


def evidence_id(revision: str, anchor: str) -> str:
    return "qa-" + revision + "-" + anchor


def render_questions(state: dict | None, *, action: str | None = None) -> str:
    payload = public_questions(state)
    esc = html.escape
    output = (
        '<section id="saved-report-questions"><h2>Questions about a saved report</h2>'
    )
    if action:
        output += f'<p><a href="{esc(action,quote=True)}">Save a report revision and ask questions</a></p>'
    if not payload["current"]:
        return (
            output
            + "<p>No saved question context. Answers require an explicitly saved report revision.</p></section>"
        )
    output += f'<p>Saved report {payload["current"]["revision"][7:19]}. Answers use this saved report; later photo or landmark edits do not silently replace it. The latest {MAX_QUESTIONS} answers are retained. Exports include both saved contexts, source-image crops and illustrations. Remove saved questions from the local session to exclude those earlier contexts from subsequent exports. Physical progress and benefit require separate evidence.</p>'
    contexts = [p for p in (payload.get("previous"), payload["current"]) if p]
    for answer in payload.get("answers", []):
        output += f'<article class="saved-answer"><h3>{esc(answer["question"])} {esc(answer.get("label",answer["target"]))}</h3><p>{esc(answer["text"])}</p>'
        for revision, refs in answer["evidence_reports"].items():
            for ref in refs:
                output += f'<a href="#{evidence_id(revision,ref)}">Inspect saved evidence · {esc(ref)}</a> '
        for field in answer.get("report_refs", []):
            target = "photo-experiment" if field == "photo_experiment" else field
            output += f'<a href="#{evidence_id(answer["report_revision"],target)}">Inspect saved report context</a> '
        output += "</article>"
    rendered = set()
    for saved in contexts:
        if saved["revision"] in rendered:
            continue
        rendered.add(saved["revision"])
        report = json.loads(saved["report_json"])
        records = public_observations(
            tuple(
                Observation.model_validate_json(json.dumps(record))
                for record in report["observations"]
            )
        )
        needed = {
            ref
            for a in payload.get("answers", [])
            for ref in a["evidence_reports"].get(saved["revision"], [])
        }
        output += f'<details><summary>Evidence from saved revision {saved["revision"]}</summary>'
        for field in ("observations", "consultation"):
            if field not in report:
                continue
            content = (
                [
                    {k: r[k] for k in ("label", "definition_key", "status", "method")}
                    for r in records
                ]
                if field == "observations"
                else {"preferences": report[field].get("preferences", {})}
            )
            output += f'<section id="{evidence_id(saved["revision"],field)}"><h3>Saved {field}</h3><pre>{esc(json.dumps(content,indent=2))}</pre></section>'
        for record in records:
            anchor = evidence_anchor(record)
            if anchor not in needed:
                continue
            output += f'<article id="{evidence_id(saved["revision"],anchor)}"><h3>{esc(record["label"])}</h3><p>{esc(str(record["value"]))} {esc(record["unit"])} · {esc(record["status"])} · {esc(record["method"])} · {esc(record["definition_version"])}</p><p>{esc(record["withheld_reason"] or "")}</p><pre>{esc(json.dumps(record["sources"],indent=2))}</pre></article>'
        output += "</details>"
        if "photo_experiment" in report:
            from .photo_experiments import render_experiment

            output += render_experiment(
                report["photo_experiment"], prefix="qa-" + saved["revision"] + "-"
            )
    return output + "</section>"
