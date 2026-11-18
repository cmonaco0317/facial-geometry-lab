"""Deterministic answers from a checksum-bound saved report, with no model calls."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from math import isfinite
import re

from .consultation_findings import current_rows, evidence_anchor

VERSION = "saved-report-questions-v1"
MAX_REPORT_BYTES = 16_000_000
QUESTION_KINDS = {
    "value": "What is the current reading?",
    "method": "How was this reading constructed?",
    "missing": "What evidence is missing?",
    "source": "Which photo and revision support it?",
    "change": "What changed between these saved constructions?",
    "experiment": "What changes in the saved geometry illustration?",
    "scope": "What can these photos and experiments establish?",
}


def _revision(encoded: str) -> str:
    return "report-" + hashlib.sha256(encoded.encode()).hexdigest()


def freeze_report(payload: dict) -> dict:
    """Copy exact JSON bytes so subsequent mutable-session edits cannot alter answers."""
    if not isinstance(payload, dict) or not isinstance(
        payload.get("observations"), list
    ):
        raise ValueError("Save an exported report with observation records first.")
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(encoded.encode()) > MAX_REPORT_BYTES:
        raise ValueError("This report is too large for a saved question context.")
    return {"revision": _revision(encoded), "report_json": encoded}


def _load(saved: dict) -> dict:
    if not isinstance(saved, dict):
        raise ValueError("A saved report revision is required.")
    encoded = saved.get("report_json")
    if (
        not isinstance(encoded, str)
        or len(encoded.encode()) > MAX_REPORT_BYTES
        or saved.get("revision") != _revision(encoded)
    ):
        raise ValueError("The saved report revision does not match its content.")
    payload = json.loads(encoded)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("observations"), list
    ):
        raise ValueError("The saved report has no observation collection.")
    return payload


def available_topics(saved: dict) -> list[dict]:
    """Return current construction IDs, including unavailable cards for missingness."""
    payload = _load(saved)
    topics = [
        {"id": row["id"], "label": row["label"], "status": row["status"]}
        for row in current_rows(payload["observations"])
    ]
    from .photo_experiments import public_experiment

    entry = public_experiment(payload.get("photo_experiment"))["entry"]
    if entry and entry.get("kind") == "geometry":
        existing = {t["id"] for t in topics}
        for row in entry["measurements"]:
            key = row["id"].removeprefix("reviewed_")
            if key not in existing:
                topics.append(
                    dict(
                        id=key,
                        label=row["label"] + " (illustration)",
                        status=(
                            "simulated" if row["after"] is not None else "unavailable"
                        ),
                    )
                )
    return topics


def _experiment_answer(saved: dict, payload: dict, answer: dict, target: str) -> dict:
    from .photo_experiments import public_experiment

    experiment = public_experiment(payload.get("photo_experiment"))
    entry = experiment["entry"]
    refs = ["photo_experiment"] if "photo_experiment" in payload else ["observations"]
    if entry is None or entry["kind"] != "geometry":
        return {
            **answer,
            "status": "unavailable",
            "text": "No numeric geometry experiment is stored in this saved report. Hair and eyewear overlays do not remeasure geometry.",
            "report_refs": refs,
        }
    row = next(
        (r for r in entry["measurements"] if r["id"] in {target, "reviewed_" + target}),
        None,
    )
    if row is None:
        return {
            **answer,
            "status": "unsupported",
            "text": "That feature is not a supported construction in this saved geometry illustration. No unseen anatomy or effect is inferred.",
            "report_refs": refs,
        }
    ref = entry["id"] + "-" + row["id"]
    text = (
        f"Illustrative construction: {row['before']} → {row['after']} {row['unit']}; stored arithmetic difference {row['delta']} {row['unit']}. This is a simulated point warp, not measured anatomy, physical progress or predicted benefit. Original measured readings and scores are unchanged."
        if row["after"] is not None
        else "This simulated construction is unavailable. " + row["withheld_reason"]
    )
    return {
        **answer,
        "status": "simulated" if row["after"] is not None else "unavailable",
        "label": row["label"],
        "before": row["before"],
        "value": row["after"],
        "delta": row["delta"],
        "unit": row["unit"],
        "method": row["method"],
        "provenance": "illustrative_simulation",
        "evidence_refs": [ref],
        "evidence_reports": {saved["revision"]: [ref]},
        "photo_numbers": [entry["photo_number"]],
        "text": text,
    }


def _row(payload: dict, target: str) -> dict | None:
    return next(
        (r for r in current_rows(payload["observations"]) if r["id"] == target), None
    )


def _source_signature(record: dict) -> list[tuple]:
    return sorted(
        (s["photo_number"], s["view"], s["revision"], s["width"], s["height"])
        for s in record["sources"]
    )


def _source_binding(payload: dict, record: dict) -> tuple | None:
    """Bind same-photo edits to retained source bytes, never ordinal coincidence."""
    instrument = payload.get("instrument_version")
    if not isinstance(instrument, str) or not instrument:
        return None
    hashes: dict[int, set[str]] = {}
    appearance = payload.get("appearance", {})
    families = appearance.get("families", []) if isinstance(appearance, dict) else []
    for family in families if isinstance(families, list) else []:
        findings = family.get("findings", []) if isinstance(family, dict) else []
        for finding in findings if isinstance(findings, list) else []:
            if not isinstance(finding, dict):
                continue
            number, digest = finding.get("photo_number"), finding.get("source_sha256")
            if (
                type(number) is int
                and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                hashes.setdefault(number, set()).add(digest)
    numbers = sorted({s["photo_number"] for s in record["sources"]})
    if not numbers or any(len(hashes.get(n, set())) != 1 for n in numbers):
        return None
    return instrument, tuple((n, next(iter(hashes[n]))) for n in numbers)


def _change(answer: dict, row: dict, previous: dict | None, payload: dict) -> dict:
    if previous is None:
        return {
            **answer,
            "status": "unavailable",
            "text": "No previous saved report was supplied; no edit delta is available.",
        }
    old_payload = _load(previous)
    old = _row(old_payload, row["id"])
    answer = {**answer, "previous_report_revision": previous["revision"]}
    if old is None:
        return {
            **answer,
            "status": "unavailable",
            "text": "The previous report has no corresponding construction.",
        }
    answer["before"] = old["value"]
    answer["evidence_refs"] = list(
        dict.fromkeys(old["evidence_refs"] + row["evidence_refs"])
    )
    answer["evidence_reports"] = {
        previous["revision"]: old["evidence_refs"],
        answer["report_revision"]: row["evidence_refs"],
    }
    binding = _source_binding(payload, row["current_observation"])
    compatible = (
        binding is not None
        and binding == _source_binding(old_payload, old["current_observation"])
        and all(
            old[key] == row[key]
            for key in ("definition_key", "definition_version", "method", "unit")
        )
        and _source_signature(old["current_observation"])
        == _source_signature(row["current_observation"])
    )
    if not compatible:
        return {
            **answer,
            "status": "unavailable",
            "text": "Definition, method, unit, instrument or source-image binding differs or is unavailable. No edit delta is substituted across these constructions.",
        }
    if old["value"] is None or row["value"] is None:
        return {
            **answer,
            "status": "unavailable",
            "text": "One saved construction is unavailable; a numeric edit delta cannot be reconstructed. "
            + (row["withheld_reason"] or old["withheld_reason"]),
        }
    delta = float(Decimal(str(row["value"])) - Decimal(str(old["value"])))
    if not isfinite(delta):
        return {
            **answer,
            "status": "unavailable",
            "text": "The stored values do not permit a finite numeric delta.",
        }
    answer["delta"] = delta
    answer["text"] = (
        f"Stored construction: {old['value']} → {row['value']} {row['unit']}; "
        f"arithmetic difference {answer['delta']} {row['unit']}. "
        "This is not physical progress or validated benefit. Matching recorded context "
        "does not establish repeatability, identity or comparable real captures."
    )
    return answer


def answer_question(
    saved: dict, kind: str, target: str = "", *, previous: dict | None = None
) -> dict:
    """Answer a supported question from this saved revision, never the live session."""
    if kind not in QUESTION_KINDS:
        raise ValueError("Choose a supported report question.")
    if not isinstance(target, str) or len(target) > 160:
        raise ValueError("Choose a bounded report topic.")
    payload = _load(saved)
    answer = dict(
        version=VERSION,
        question=QUESTION_KINDS[kind],
        kind=kind,
        target=target,
        report_revision=saved["revision"],
        status="supported",
        value=None,
        before=None,
        delta=None,
        method=None,
        provenance=None,
        evidence_refs=[],
        evidence_reports={},
        photo_numbers=[],
    )
    if kind == "scope":
        return {
            **answer,
            "text": "Answers use only the stored report revision. An unseen feature is not established by a related proxy. Anatomy, identity, calibrated uncertainty and physical repeatability need separate evidence; styling or geometry benefits are not validated by an illustration.",
            "report_refs": [
                key for key in ("observations", "consultation") if key in payload
            ],
        }
    if kind == "experiment":
        return _experiment_answer(saved, payload, answer, target)
    row = _row(payload, target)
    if row is None:
        return {
            **answer,
            "status": "unsupported",
            "text": "That feature is not present as a current construction in this saved report. No value or unseen anatomy can be inferred from it.",
            "report_refs": ["observations"],
        }
    record = row["current_observation"]
    answer.update(
        label=row["label"],
        value=row["value"],
        unit=row["unit"],
        method=row["method"],
        provenance=row["status"],
        definition_key=row["definition_key"],
        definition_version=row["definition_version"],
        evidence_refs=[evidence_anchor(record)],
        photo_numbers=row["photo_numbers"],
    )
    answer["evidence_reports"] = {saved["revision"]: answer["evidence_refs"]}
    if kind == "change":
        return _change(answer, row, previous, payload)
    if kind == "source":
        answer["sources"] = record["sources"]
        answer["text"] = (
            f"Stored source photos {row['photo_numbers']}; {row['status']} via {row['method']}. Inspect each source's image and review revision and dependency points. A workflow label does not prove human participation."
        )
    elif kind == "method":
        answer["definition"] = record["definition"]
        answer["formula"] = record["formula"]
        answer["text"] = (
            f"{row['label']}: {row['status']} via {row['method']}, definition {row['definition_version']}. {record['definition']} A proxy or reviewed construction keeps its own method; reference validity is not inferred."
        )
    elif row["value"] is None:
        answer["status"] = "unavailable"
        answer["text"] = (
            f"{row['label']} is unavailable. {row['withheld_reason']} No older automatic number is substituted for a missing current reviewed input."
        )
    elif kind == "missing":
        answer["text"] = (
            "This construction has a stored numeric value; no missing dependency is recorded for it. Numeric availability alone does not prove visible anatomy or accuracy."
        )
    else:
        answer["text"] = (
            f"{row['label']}: {row['value']} {row['unit']}; {row['status']} via {row['method']}. This is the saved reading, not an attractiveness or improvement claim."
        )
    return answer
