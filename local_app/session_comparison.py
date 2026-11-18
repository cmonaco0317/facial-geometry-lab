"""Construction arithmetic with explicit capture and empirical-evidence limits."""

from __future__ import annotations

from datetime import datetime
from collections import defaultdict
from decimal import Decimal
from math import isfinite
from typing import TYPE_CHECKING

from .archive_schema import compare

if TYPE_CHECKING:
    from .archive_schema import Record
    from .observations import Observation


PROVENANCE = {
    "automatic": "Automatic image construction; not direct anatomical observation",
    "proxy": "Estimated target via a proxy; not direct anatomical observation",
    "estimated": "Estimated reading; evaluation is not a physical repeatability limit",
    "user_reviewed": "User-reviewed image construction, possibly from detector proposals; not direct anatomical validation or verified qualifications",
    "unavailable": "Unavailable; no direct or estimated reading supplied",
}
REASONS = {
    "repeatability_unavailable": "No empirical repeat-capture limit is available for this construction and capture condition; physical progress cannot be interpreted, even for a nonzero difference.",
    "capture_context_unavailable": "Pose, expression, lighting and camera-distance context was not retained for both reports.",
    "same_inputs": "The same input set is reused; this is not an independent repeat capture.",
    "chronology_unavailable": "Capture dates are missing, equal or reversed; chronological progress is not established.",
    "source_not_visible": "A source is occluded, uncertain or unassessed; directly visible anatomy is not established.",
    "capture_ineligible": "A source is not admitted by an automatic capture gate; user review alone does not establish capture compatibility.",
    "view_changed": "Source views differ; frontal, profile and smile constructions cannot be interchanged.",
    "view_unknown": "A source view is unknown.",
    "reading_unavailable": "One report has no numeric reading; its saved missing-evidence reason is retained.",
    "reading_absent": "The construction is absent from one report.",
    "profile_changed": "Local profile names differ; same-person identity is not established.",
    "protocol_unknown": "A controlled repeat setup is not recorded for both sessions.",
    "camera_unknown": "The camera/setup label is missing from one report.",
    "source_capture_context_unavailable": "Per-source capture metadata was not retained for every contributing source.",
    "pose_estimate_changed": "Recorded pose estimates differ. No validated pose-match tolerance or physical change interpretation is available.",
    "pose_estimate_unavailable": "At least one source has no complete pose estimate; pose comparability is unverified.",
    "expression_label_changed": "Source expression/view labels differ; a neutral, smile or profile label is not interchangeable.",
    "expression_label_unknown": "A source expression/view label is unknown. Labels and detector candidates are unverified.",
    "source_count_changed": "The construction uses different numbers of source photos; aggregation conditions differ.",
    "frame_dimensions_changed": "Original image frame dimensions differ; crop, scale or framing comparability is unverified.",
}


def _explain(code: str) -> str:
    if code in REASONS:
        return REASONS[code]
    return code.replace("_", " ").capitalize() + "; no compatible change is inferred."


def _conditions(a: Record, b: Record) -> list[str]:
    reasons = []
    for key in ("profile", "camera", "protocol", "instrument"):
        if getattr(a.snapshot, key) != getattr(b.snapshot, key):
            reasons.append(key + "_changed")
    if a.snapshot.protocol != "controlled" or b.snapshot.protocol != "controlled":
        reasons.append("protocol_unknown")
    if not a.snapshot.camera or not b.snapshot.camera:
        reasons.append("camera_unknown")
    if a.context is None or b.context is None:
        return reasons + ["capture_context_unavailable"]
    if a.context.report_version != b.context.report_version:
        reasons.append("report_version_changed")
    for key in ("pose", "expression", "lighting", "distance"):
        x, y = getattr(a.context.capture, key), getattr(b.context.capture, key)
        if x != y:
            reasons.append("capture_" + key + "_changed")
        if x in {"unknown", "mixed", ""} or y in {"unknown", "mixed", ""}:
            reasons.append("capture_" + key + "_unknown")
    return reasons


def _row(
    before: Observation | None,
    after: Observation | None,
    conditions: list[str],
    notices: list[str],
) -> dict:
    current = after or before
    reasons = list(conditions)
    arithmetic_reasons = [
        r for r in conditions if r in {"ambiguous_source_pairing", "instrument_changed"}
    ]
    if before is None or after is None:
        arithmetic_reasons.append("reading_absent")
    else:
        for field in ("definition_version", "method", "unit", "status", "formula"):
            if getattr(before, field) != getattr(after, field):
                arithmetic_reasons.append(field + "_changed")
        if before.value is None or after.value is None:
            arithmetic_reasons.append("reading_unavailable")
        views = [sorted(s.view for s in r.sources) for r in (before, after)]
        if views[0] != views[1]:
            reasons.append("view_changed")
        if any("unknown" in view for view in views):
            reasons.append("view_unknown")
        if any(not s.capture_eligible for r in (before, after) for s in r.sources):
            reasons.append("capture_ineligible")
        if any(s.visibility != "visible" for r in (before, after) for s in r.sources):
            reasons.append("source_not_visible")
    delta = (
        float(Decimal(str(after.value)) - Decimal(str(before.value)))
        if not arithmetic_reasons
        else None
    )

    if delta is not None and not isfinite(delta):
        delta = None
        arithmetic_reasons.append("nonfinite_difference")
    codes = list(
        dict.fromkeys(
            reasons + arithmetic_reasons + notices + ["repeatability_unavailable"]
        )
    )
    return dict(
        definition_key=current.definition_key,
        label=current.label,
        before=before.value if before else None,
        after=after.value if after else None,
        before_method=before.method if before else None,
        after_method=after.method if after else None,
        before_unit=before.unit if before else None,
        after_unit=after.unit if after else None,
        before_status=before.status if before else None,
        after_status=after.status if after else None,
        before_provenance=PROVENANCE[before.status] if before else "Absent",
        after_provenance=PROVENANCE[after.status] if after else "Absent",
        before_sources=(
            [s.model_dump(mode="json", exclude={"points"}) for s in before.sources]
            if before
            else []
        ),
        after_sources=(
            [s.model_dump(mode="json", exclude={"points"}) for s in after.sources]
            if after
            else []
        ),
        before_missing=before.withheld_reason if before else "Construction absent.",
        after_missing=after.withheld_reason if after else "Construction absent.",
        delta=delta,
        arithmetic_compatible=not arithmetic_reasons,
        capture_compatible=not reasons,
        repeatability_limit=None,
        progress_interpretable=False,
        reason_codes=codes,
        reasons=[_explain(code) for code in codes],
    )


def _captures(record: Record, row: dict, side: str) -> list[dict]:
    slots = (
        {s.photo_number: s for s in record.context.sources} if record.context else {}
    )
    return [
        slots[s["photo_number"]].model_dump(mode="json", exclude={"photo_id"})
        for s in sorted(row[side + "_sources"], key=lambda r: r["photo_number"])
        if s["photo_number"] in slots
    ]


def _capture_details(row: dict, a: Record, b: Record) -> dict:
    before, after = _captures(a, row, "before"), _captures(b, row, "after")
    reasons = []
    if (
        not before
        or not after
        or len(before) != len(row["before_sources"])
        or len(after) != len(row["after_sources"])
    ):
        reasons.append("source_capture_context_unavailable")
    if len(row["before_sources"]) != len(row["after_sources"]):
        reasons.append("source_count_changed")
    # These are ordered source slots, never a claim of identity correspondence.
    for left, right in zip(before, after):
        if (left["width"], left["height"]) != (right["width"], right["height"]):
            reasons.append("frame_dimensions_changed")
        if any(value is None for p in (left, right) for value in p["pose"].values()):
            reasons.append("pose_estimate_unavailable")
        if left["pose"] != right["pose"]:
            reasons.append("pose_estimate_changed")
        if "unknown" in (left["role"], right["role"]):
            reasons.append("expression_label_unknown")
        if left["role"] != right["role"]:
            reasons.append("expression_label_changed")
        if left["views"] != right["views"]:
            reasons.append("view_changed")
    codes = list(dict.fromkeys(row["reason_codes"] + reasons))
    return {
        **row,
        "before_capture": before,
        "after_capture": after,
        "capture_compatible": row["capture_compatible"] and not reasons,
        "source_capture_reason_codes": list(dict.fromkeys(reasons)),
        "reason_codes": codes,
        "reasons": [_explain(code) for code in codes],
    }


def compare_records(a: Record, b: Record) -> dict:
    """Both reports remain visible; no estimate is upgraded by a matching number."""
    result = compare(a.snapshot, b.snapshot)
    source_sets = [
        {s.source_sha256 for s in record.context.sources} if record.context else set()
        for record in (a, b)
    ]
    normalized_match = bool(source_sets[0]) and source_sets[0] == source_sets[1]
    result["same_normalized_source_set"] = (
        normalized_match if all(source_sets) else None
    )
    if normalized_match:
        result["same_inputs"] = True
        result["observed_range"] = None
        result["warnings"].append(
            "The same normalized source image set is reused, even if catalog IDs, order or slot counts differ. These are not independent retakes."
        )
    elif source_sets[0] & source_sets[1]:
        result["warnings"].append(
            "At least one normalized source image is reused; the source sets are not fully independent."
        )
    if any(
        record.context
        and len({s.source_sha256 for s in record.context.sources})
        < len(record.context.sources)
        for record in (a, b)
    ):
        result["warnings"].append(
            "A report contains duplicate normalized source bytes in distinct slots. Those slots are not independent images."
        )
    conditions = _conditions(a, b)
    notices = ["same_inputs"] if result["same_inputs"] else []
    dates = [s.captured_at for s in (a.snapshot, b.snapshot)]
    if not all(dates) or datetime.fromisoformat(
        dates[0].replace("Z", "+00:00")
    ) >= datetime.fromisoformat(dates[1].replace("Z", "+00:00")):
        notices.append("chronology_unavailable")
    before, after = defaultdict(list), defaultdict(list)
    for r in a.snapshot.observations:
        before[r.definition_key].append(r)
    for r in b.snapshot.observations:
        after[r.definition_key].append(r)
    rows = []
    for key in sorted(before.keys() | after.keys()):
        left, right = before[key], after[key]
        if len(left) <= 1 and len(right) <= 1:
            rows.append(
                _row(
                    left[0] if left else None,
                    right[0] if right else None,
                    conditions,
                    notices,
                )
            )
        else:
            # A photo ordinal is a slot, not correspondence between independent captures.
            # Keep every candidate; do not guess identity from order or last insertion.
            for old, new in [(r, None) for r in left] + [(None, r) for r in right]:
                rows.append(
                    _row(old, new, conditions + ["ambiguous_source_pairing"], notices)
                )
    rows = [_capture_details(row, a, b) for row in rows]
    source_differences = sorted(
        {code for row in rows for code in row["source_capture_reason_codes"]}
    )
    if conditions:
        result.update(
            status=(
                "method_changed"
                if "instrument_changed" in conditions
                else "capture_mismatch"
            ),
            score_delta=None,
            observed_range=None,
            readings=[],
        )
    elif rows and result["status"] == "compatible":
        result["status"] = (
            "arithmetic_only"
            if all(r["arithmetic_compatible"] for r in rows)
            else "partial_arithmetic_only"
        )
    elif not rows:
        result.update(
            status="insufficient_evidence",
            score_delta=None,
            observed_range=None,
            readings=[],
        )
    return {
        **result,
        "observations": rows,
        "capture_differences": conditions,
        "source_capture_differences": source_differences,
        "repeatability_limit": None,
        "progress_interpretable": False,
        "repeatability_status": "unavailable_no_empirical_repeat_captures",
        "warnings": result["warnings"]
        + [_explain(c) for c in conditions + ["repeatability_unavailable"]],
    }
