"""Explicit, path-free exports for the local progressive analysis workflow."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import html
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any

from .analysis import AXIS_ORDER, AnalysisReport, ProgressiveAnalysisService
from .contracts import AxisResult, CohortUse, Demographics
from .sessions import SessionStore
from .consultation import consultation_payload, render_consultation
from .illustrations import public_diagram, render_diagrams
from .landmark_review import render_reviewed, safe_review, render_forehead_warning
from .profile_review import public_profile, render_profile
from .product_report import composite_score, public_regions, render_product_summary
from .score_audit import audit
from .automatic_forehead import public_forehead, render_automatic_forehead

EXPORT_FORMAT_VERSION = 6
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")

CAPTURE_REASONS = {
    "face_not_detected": "No detected face",
    "too_small": "Limited image detail",
    "blurred": "Blur",
    "uneven_light": "Uneven lighting",
    "occluded": "Obstruction",
    "expression": "Mouth posture not cleared",
    "pose_unknown": "Unknown pose",
    "yaw": "Head turn",
    "pitch": "Head pitch",
    "roll": "Head roll",
}


def public_capture_summary(raw: Any) -> dict[str, Any] | None:
    """Export aggregate counts only; never paths, identifiers or model outputs."""
    fields = ("submitted_photos", "detected_photos", "eligible_photos", "selected_axis_photos")
    if not isinstance(raw, dict) or any(
        type(raw.get(key)) is not int or not 0 <= raw[key] <= 10000
        for key in fields
    ):
        return None
    if not (
        raw["eligible_photos"] <= raw["detected_photos"] <= raw["submitted_photos"]
        and raw["selected_axis_photos"] <= raw["detected_photos"]
    ):
        return None
    reasons = raw.get("failure_counts", {})
    return {
        **{key: raw[key] for key in fields},
        "failure_counts": {
            key: count for key, count in (reasons.items() if isinstance(reasons, dict) else [])
            if key in CAPTURE_REASONS and type(count) is int
            and 0 < count <= raw["submitted_photos"]
        },
    }


def render_capture_summary(raw: Any, *, expanded: bool = False) -> str:
    summary = public_capture_summary(raw)
    if summary is None:
        return ""
    reasons = "; ".join(
        f"{CAPTURE_REASONS[key]}: {count}"
        for key, count in summary["failure_counts"].items()
    )
    return (
        f'<details class="capture-summary"{" open" if expanded else ""}><summary>How photos were used</summary>'
        f'<p>{summary["submitted_photos"]} assessed; {summary["detected_photos"]} with a detected face; '
        f'{summary["eligible_photos"]} meet all strict checks for this category. '
        f'{summary["selected_axis_photos"]} used for its axis reading. '
        'Regional rows show their own photo counts.</p>'
        f'<p>{reasons}. A photo can have several flags. Flags are automated checks, '
        'not verified judgments about expression or anatomy.</p></details>'
    )


def _public_text(value: object) -> str:
    """Return internal/user text while refusing strings that are local paths."""
    text = str(value)
    if "\x00" in text or text.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(text):
        return "[local path redacted]"
    return text


def _default_source_inventory() -> Mapping[str, Mapping[str, Any]]:
    try:
        import bank_reference

        return bank_reference.cohort_inventory()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return {}


def source_disclosures(
    cohort: CohortUse,
    source_inventory: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    inventory = (
        source_inventory
        if source_inventory is not None
        else _default_source_inventory()
    )
    grouped: dict[tuple[str, str], list[str]] = {}
    members = cohort.member_ids or (cohort.used_id,)
    for member_id in members:
        record = inventory.get(member_id, {})
        source = record.get("source")
        meaning = record.get("source_label_kind")
        if not isinstance(source, str) or not source.strip():
            source = "Source metadata unavailable in the installed reference bank"
        if not isinstance(meaning, str) or not meaning.strip():
            meaning = "Label meaning unavailable in the installed reference bank"
        key = (_public_text(source.strip()), _public_text(meaning.strip()))
        grouped.setdefault(key, []).append(_public_text(member_id))
    return [
        {
            "source": source,
            "label_meaning": meaning,
            "cohort_ids": sorted(ids),
        }
        for (source, meaning), ids in sorted(grouped.items())
    ]


def _comparison_payload(
    cohort: CohortUse,
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "requested": {
            "id": _public_text(cohort.requested_id),
            "label": _public_text(cohort.requested_label or cohort.requested_id),
        },
        "used": {
            "id": _public_text(cohort.used_id),
            "label": _public_text(cohort.used_label or cohort.used_id),
        },
        "fallback_reason": _public_text(cohort.fallback_level),
        "exact_match": cohort.fallback_level == "exact",
        "sample_size": cohort.sample_size,
        "reference_sources": source_disclosures(cohort, inventory),
    }


def _profile_payload(result: AxisResult) -> dict[str, Any] | None:
    profile = result.details.get("profile")
    if not isinstance(profile, dict) or not profile:
        return None
    exported: dict[str, Any] = {}
    for name, raw in profile.items():
        if not isinstance(raw, dict):
            continue
        interval = raw.get("interval")
        if interval is not None and (
            not isinstance(interval, (list, tuple)) or len(interval) != 2
        ):
            continue
        exported[_public_text(name)] = {
            "label": _public_text(
                raw.get("label", str(name).replace("_", " ").title())
            ),
            "value": raw.get("value"),
            "percentile": raw.get("percentile"),
            "uncertainty_percentile": list(interval) if interval is not None else None,
        }
    return exported or None


def _axis_payload(
    result: AxisResult,
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summary = _public_text(result.summary)
    if (
        result.axis == "harmony"
        and result.status == "measured"
        and not result.headline_available
    ):
        summary = (
            "Measured component geometry is shown below; no validated overall "
            "harmony score is available."
        )
    payload: dict[str, Any] = {
        "status": result.status,
        "summary": summary,
        "evidence_photo_count": result.photo_count,
    }
    capture = public_capture_summary(result.details.get("capture_diagnostics"))
    if capture is not None:
        payload["capture_diagnostics"] = capture
    if result.cohort is not None:
        payload["comparison"] = _comparison_payload(result.cohort, inventory)

    if result.status in {"measured", "estimated"}:
        payload["headline_available"] = result.headline_available
        if result.headline_available:
            payload.update(
                {
                    "value": result.value,
                    "unit": _public_text(result.unit or ""),
                    "percentile": result.percentile,
                    "uncertainty_percentile": list(result.interval)
                    if result.interval is not None
                    else None,
                }
            )
        if result.status == "estimated":
            payload["capture_issues"] = [
                _public_text(key) for key in result.details.get("capture_issues", {})
            ]
            payload["estimate_note"] = _public_text(
                result.details.get("estimate_note", "Uncertainty is not calibrated.")
            )
        profile = _profile_payload(result)
        if profile is not None:
            payload["profile"] = profile
        if result.axis == "averageness":
            placement = result.details.get("placement")
            if isinstance(placement, str) and placement:
                payload["placement"] = {
                    "status": (
                        "unresolved" if placement.startswith("withheld") else "resolved"
                    ),
                    "explanation": _public_text(placement),
                }
    elif result.status == "needs_photo" and result.photo_request is not None:
        request = result.photo_request
        payload["requested_photo"] = {
            "view": _public_text(request.view),
            "expression": _public_text(request.expression),
            "lighting": _public_text(request.lighting),
            "framing": _public_text(request.framing),
            "reason": _public_text(request.reason),
        }
    elif result.status == "processing_error":
        payload["error"] = {
            "code": _public_text(result.error_code or "processing_error"),
            "message": _public_text(result.error_message or result.summary),
            "retryable": result.retryable,
        }
    return payload


def build_export_payload(
    report: AnalysisReport,
    demographics: Demographics,
    *,
    source_inventory: Mapping[str, Mapping[str, Any]] | None = None,
    settings: dict[str, str] | None = None,
    profile: dict[str, Any] | list[dict] | None = None,
    ideas: dict | None = None,
    frontal: list[dict] | None = None,
) -> dict[str, Any]:
    """Build the durable report; photo/session identifiers are omitted."""
    inventory = (
        source_inventory
        if source_inventory is not None
        else _default_source_inventory()
    )
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "automatic_forehead": public_forehead(report.axes),
        "frontal_reviews": [
            p for raw in frontal or [] if (p := safe_review(raw)) is not None
        ],
        "consultation": consultation_payload(
            public_regions(report.axes, frontal=frontal),
            settings,
            frontal=frontal,
            ideas=ideas,
            profile=profile,
        ),
        "illustration": public_diagram(
            report.axes["harmony"].details.get("illustration")
        ),
        "profile_review": public_profile(profile)
        if isinstance(profile, dict) and profile.get("method") == "manual-profile-v1"
        else None,
        "profile_reviews": [
            p
            for raw in (profile if isinstance(profile, list) else [profile])
            if (p := public_profile(raw)) is not None
        ],
        "composite": composite_score(report.axes),
        "score_audit": audit(report.axes),
        "regional_measurements": public_regions(report.axes, frontal=frontal),
        "demographics": {
            **demographics.to_dict(),
            "entered_by_user": True,
        },
        "accepted_photo_count": report.accepted_photos,
        "instrument_version": _public_text(report.instrument_fingerprint),
        "comparison": _comparison_payload(report.cohort, inventory),
        "axes": {
            axis: _axis_payload(report.axes[axis], inventory) for axis in AXIS_ORDER
        },
    }


def render_json_export(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _h(value: object) -> str:
    return html.escape(str(value), quote=True)


def _ordinal(value: float) -> str:
    rounded = int(round(float(value)))
    if 10 <= rounded % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rounded % 10, "th")
    return f"{rounded}{suffix}"


def _plain_label(value: object) -> str:
    return " ".join(str(value).replace("_", " ").split())


def _method_html(
    comparison: Mapping[str, Any],
    instrument_version: object,
    harmony: Mapping[str, Any],
) -> str:
    requested = comparison["requested"]
    used = comparison["used"]
    sources = "".join(
        "<li><strong>{}</strong><span>{}</span><small>{}</small></li>".format(
            _h(source["source"]),
            _h(source["label_meaning"]),
            _h(", ".join(source["cohort_ids"])),
        )
        for source in comparison["reference_sources"]
    )
    if harmony["status"] == "estimated":
        harmony_note = "<p>Best-effort proportions use the available photos; uncertainty is not calibrated.</p>"
    elif harmony["status"] == "measured":
        if harmony.get("headline_available"):
            harmony_note = (
                "<p><strong>Harmony scalar status.</strong> The harmony headline is "
                "a validated geometric measure and is shown with its underlying "
                "proportions; it is not an attractiveness score.</p>"
            )
        else:
            harmony_note = (
                "<p><strong>How harmony contributes.</strong> The available "
                "component proportions feed the separate product score. Its weights "
                "are design choices; the score is not a validated attractiveness model.</p>"
            )
    else:
        harmony_note = (
            "<p><strong>Harmony status.</strong> A harmony result is not available yet; "
            "its section shows the current photo request or processing status.</p>"
        )
    return f"""
    <section class="method" aria-labelledby="method-title">
      <div class="section-heading">
        <p class="section-number">Evidence note</p>
        <h2 id="method-title">Method and limits</h2>
      </div>
      <div class="method-grid">
        <dl class="comparison">
          <div><dt>Requested group</dt><dd>{_h(requested["label"])}</dd></div>
          <div><dt>Comparison group used</dt><dd>{_h(used["label"])}</dd></div>
          <div><dt>Cohort fallback</dt><dd>{_h(_plain_label(comparison["fallback_reason"]))}</dd></div>
          <div><dt>Reference sample</dt><dd>{int(comparison["sample_size"]):,} reference identities</dd></div>
        </dl>
        <div class="limits-copy">
          {harmony_note}
          <p>Percentiles describe position within the comparison group, not quality,
          health, identity, or attractiveness. Ranges show measured uncertainty;
          wider ranges should be read more cautiously.</p>
          <p><strong>Human-rater check.</strong> In the existing SCUT-FBP5500
          analysis, only nose–mouth–eye cleared the preregistered association bar
          in every measured cohort; the other five remain descriptive geometry.
          The panel was 60 Chinese volunteers aged 18–27, so it is not universal.</p>
        </div>
      </div>
      <div class="source-block">
        <h3>Reference sources and label meaning</h3>
        <ul class="source-list">{sources}</ul>
      </div>
      <p class="instrument"><strong>Instrument fingerprint</strong>
      <span>{_h(instrument_version)}</span></p>
    </section>"""


def _profile_html(profile: Mapping[str, Any]) -> str:
    rows = "".join(
        '<tr><th scope="row">{}</th><td>{}</td><td>{} percentile</td>'
        "<td>{}–{}</td></tr>".format(
            _h(row["label"]),
            _h(row["value"]),
            _h(
                _ordinal(row["percentile"])
                if row.get("percentile") is not None
                else "No reference"
            ),
            _h(
                _ordinal(row["uncertainty_percentile"][0])
                if row.get("uncertainty_percentile")
                else "Not calibrated"
            ),
            _h(
                _ordinal(row["uncertainty_percentile"][1])
                if row.get("uncertainty_percentile")
                else ""
            ),
        )
        for row in profile.values()
    )
    return (
        '<div class="table-wrap"><table><caption>Harmony component profile</caption>'
        "<thead><tr><th>Profile measure</th><th>Observed value</th>"
        "<th>Cohort position</th><th>Uncertainty range</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _summary_item(name: str, result: Mapping[str, Any]) -> str:
    state = result["status"]
    if state == "estimated":
        reading = (
            f"{result['value']:.2f} {result['unit']}"
            if result.get("headline_available")
            else "Proportion estimates"
        )
        detail = "Best effort · capture quality below"
    elif state == "measured" and result.get("headline_available"):
        interval = result["uncertainty_percentile"]
        unresolved = (
            name == "averageness"
            and result.get("placement", {}).get("status") == "unresolved"
        )
        if unresolved:
            reading = f"{_ordinal(interval[0])}–{_ordinal(interval[1])} percentile"
            detail = "Placement unresolved"
        else:
            reading = f"{_ordinal(result['percentile'])} percentile"
            detail = f"Range {_ordinal(interval[0])}–{_ordinal(interval[1])}"
    elif state == "measured" and name == "harmony" and result.get("profile"):
        reading = "Harmony profile"
        detail = "Proportion balance contributes to the overall score"
    elif state == "measured":
        reading = "Profile measured"
        detail = "No validated single-number summary"
    elif state == "needs_photo":
        reading = "More evidence needed"
        detail = "See the capture request below"
    else:
        reading = "Not completed"
        detail = "See the processing note below"
    return f"""
      <article class="snapshot-item snapshot-item--{_h(name)}">
        <p class="snapshot-item__label">{_h(name.title())}</p>
        <p class="snapshot-item__reading">{_h(reading)}</p>
        <p class="snapshot-item__detail">{_h(detail)}</p>
      </article>"""


def _range_track(point: object, interval: Sequence[object]) -> str:
    try:
        value = min(100.0, max(0.0, float(point)))
        low = min(100.0, max(0.0, float(interval[0])))
        high = min(100.0, max(0.0, float(interval[1])))
    except (IndexError, TypeError, ValueError):
        return ""
    if high < low:
        low, high = high, low
    label = (
        f"Point estimate {_ordinal(value)} percentile; uncertainty "
        f"{_ordinal(low)} to {_ordinal(high)} percentile"
    )
    return f"""
      <div class="range-track" role="img" aria-label="{_h(label)}">
        <span class="range-track__band" style="left:{low:.2f}%;width:{max(1.0, high - low):.2f}%"></span>
        <span class="range-track__point" style="left:{value:.2f}%"></span>
      </div>
      <div class="range-scale" aria-hidden="true"><span>0</span><span>50</span><span>100</span></div>"""


def _axis_html(position: int, name: str, result: Mapping[str, Any]) -> str:
    measurement = ""
    if result["status"] == "estimated":
        if result.get("headline_available"):
            measurement = (
                f'<p class="reading">{_h(result["value"])} {_h(result["unit"])}</p>'
            )
        if result.get("percentile") is not None:
            measurement += f"<p>Approximate reference position: {_ordinal(result['percentile'])} percentile.</p>"
        if result.get("profile"):
            measurement += _profile_html(result["profile"])
        issues = ", ".join(
            str(issue).replace("_", " ") for issue in result.get("capture_issues", [])
        )
        measurement += (
            f"<p>Capture quality: {_h(issues or 'limited calibration')}.</p>"
            f"<p>{_h(result.get('estimate_note', 'Uncertainty is not calibrated.'))}</p>"
        )
    elif result["status"] == "measured":
        if result.get("headline_available"):
            interval = result["uncertainty_percentile"]
            placement_unresolved = (
                name == "averageness"
                and result.get("placement", {}).get("status") == "unresolved"
            )
            if placement_unresolved:
                measurement = (
                    f'<p class="reading">Estimated range: {_ordinal(interval[0])}–'
                    f"{_ordinal(interval[1])} percentile</p>"
                    "<p><strong>Placement unresolved.</strong> "
                    f"Point estimate: {_ordinal(result['percentile'])} percentile.</p>"
                    f"<p>{_h(result['value'])} {_h(result['unit'])}</p>"
                    f"{_range_track(result['percentile'], interval)}"
                )
            else:
                measurement = (
                    f'<p class="reading">{_ordinal(result["percentile"])} percentile '
                    f"<span>({_ordinal(interval[0])}–{_ordinal(interval[1])})</span></p>"
                    f"<p>{_h(result['value'])} {_h(result['unit'])}</p>"
                    f"{_range_track(result['percentile'], interval)}"
                )
        elif name == "harmony" and "profile" in result:
            measurement = (
                '<div class="score-status">'
                '<p class="score-status__label">Proportion balance contributes to the overall score</p>'
                "<p>The component measurements below feed the balance component "
                "of the separate product score.</p></div>"
            )
        if "profile" in result:
            measurement += _profile_html(result["profile"])
    elif result["status"] == "needs_photo":
        request = result["requested_photo"]
        measurement = "".join(
            (
                f"<p><strong>Needed photo:</strong> {_h(request['reason'])}</p>",
                '<dl class="capture">',
                *(
                    f"<div><dt>{_h(key.title())}</dt><dd>{_h(request[key])}</dd></div>"
                    for key in ("view", "expression", "lighting", "framing")
                ),
                "</dl>",
            )
        )
    else:
        error = result["error"]
        measurement = (
            f"<p><strong>Processing error:</strong> {_h(error['message'])}</p>"
        )
    return f"""
    <section class="axis axis--{_h(name)}">
      <div class="section-heading">
        <p class="section-number">0{position}</p>
        <h2>{_h(name.title())}</h2>
      </div>
      <p class="status">{_h(result["status"].replace("_", " ").title())} ·
      {int(result["evidence_photo_count"]):,} photos used</p>
      <p>{_h(result["summary"])}</p>
      {measurement}
      {render_capture_summary(result.get("capture_diagnostics"), expanded=True)}
    </section>"""


def render_html_export(payload: Mapping[str, Any]) -> str:
    """Render a self-contained, script-free HTML report suitable for printing."""
    demographics = payload["demographics"]
    summary = "".join(_summary_item(axis, payload["axes"][axis]) for axis in AXIS_ORDER)
    axes = "".join(
        _axis_html(index, axis, payload["axes"][axis])
        for index, axis in enumerate(AXIS_ORDER, start=1)
    )
    method = _method_html(
        payload["comparison"],
        payload["instrument_version"],
        payload["axes"]["harmony"],
    )
    ethnicities = ", ".join(
        _plain_label(value).title() for value in demographics["ethnicities"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Four-pillar facial analysis · Facial measurement report</title>
  <style>
    :root {{ color-scheme: light; --paper: #f6f4ef; --surface: #fffdfc;
      --ink: #18201e; --evergreen: #163e36; --blue: #54788a;
      --amber: #b57436; --muted: #60706b; --rule: #d7ddd8;
      font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ background: var(--paper); color: var(--ink); margin: 0; line-height: 1.55; }}
    .report {{ background: var(--surface); margin: 0 auto; max-width: 1080px; min-height: 100vh; padding: 56px 64px 72px; }}
    .composite-panel, .regional-section {{ border: 1px solid var(--rule); padding: 28px; margin-top: 32px; }}
    .composite-value {{ font-size: 4rem; color: var(--evergreen); margin: 16px 0; }}
    .reviewed-diagram {{ width:240px; max-width:100%; height:auto; display:block; margin:16px auto; }}
    .composite-value small {{ font-size: 1.2rem; }}
    .composite-panel details {{ margin-top: 18px; }}
    .report-cover {{ border-top: 8px solid var(--evergreen); padding-top: 28px; }}
    .cover-row {{ align-items: start; display: flex; gap: 28px; justify-content: space-between; }}
    .kicker, .section-number {{ color: var(--amber); font-size: .74rem; font-weight: 800; letter-spacing: .12em; margin: 0 0 8px; text-transform: uppercase; }}
    h1, h2, h3 {{ color: var(--evergreen); font-weight: 650; letter-spacing: -.025em; line-height: 1.12; }}
    h1 {{ font-size: clamp(2.5rem, 6vw, 4.8rem); margin: 0; max-width: 760px; }}
    h2 {{ font-size: 2rem; margin: 0; }} h3 {{ font-size: 1.05rem; }}
    .private-mark {{ border: 1px solid var(--rule); color: var(--muted); font-size: .78rem; padding: 8px 12px; white-space: nowrap; }}
    .cover-deck {{ font-size: 1.16rem; margin: 22px 0 28px; max-width: 720px; }}
    .cover-meta {{ border-bottom: 1px solid var(--rule); border-top: 1px solid var(--rule); display: flex; flex-wrap: wrap; gap: 10px 28px; padding: 14px 0; }}
    .cover-meta span {{ color: var(--muted); }} .cover-meta strong {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
    .snapshot, .reading-guide, .method {{ margin-top: 52px; }}
    .snapshot-grid {{ border-bottom: 1px solid var(--rule); border-top: 1px solid var(--rule); display: grid; grid-template-columns: repeat(4, 1fr); margin-top: 20px; }}
    .snapshot-item {{ min-height: 150px; padding: 22px 22px 18px; position: relative; }}
    .snapshot-item + .snapshot-item {{ border-left: 1px solid var(--rule); }}
    .snapshot-item::before {{ background: var(--blue); content: ""; height: 7px; left: 22px; position: absolute; top: -4px; width: 7px; }}
    .snapshot-item__label {{ color: var(--muted); font-size: .82rem; font-weight: 700; letter-spacing: .04em; margin: 0 0 22px; text-transform: uppercase; }}
    .snapshot-item__reading {{ color: var(--evergreen); font-size: 1.36rem; font-weight: 650; line-height: 1.2; margin: 0; }}
    .snapshot-item__detail {{ color: var(--muted); font-size: .88rem; margin: 7px 0 0; }}
    .reading-guide {{ background: #edf2ef; border-left: 4px solid var(--blue); display: grid; gap: 28px; grid-template-columns: minmax(190px, .65fr) 2fr; padding: 26px 30px; }}
    .reading-guide h2 {{ font-size: 1.45rem; }} .guide-copy {{ display: grid; gap: 18px; grid-template-columns: repeat(3, 1fr); }}
    .guide-copy p {{ font-size: .9rem; margin: 0; }} .guide-copy strong {{ display: block; margin-bottom: 4px; }}
    .axis {{ border-top: 1px solid var(--rule); break-inside: avoid; display: grid; grid-template-columns: minmax(180px, .58fr) minmax(0, 1.7fr); column-gap: 48px; margin-top: 60px; padding-top: 30px; }}
    .section-heading {{ align-self: start; }}
    .axis > .status, .axis > p, .axis > .reading, .axis > .score-status, .axis > .table-wrap, .axis > .range-track, .axis > .range-scale, .axis > .capture {{ grid-column: 2; }}
    .status {{ color: var(--muted); font-size: .82rem; font-weight: 700; letter-spacing: .03em; margin: 2px 0 0; text-transform: uppercase; }}
    .reading {{ color: var(--evergreen); font-size: 1.55rem; font-weight: 700; margin-bottom: 0; }}
    .reading span {{ color: var(--muted); font-size: .92rem; font-weight: 600; }}
    .score-status {{ border-left: 4px solid var(--amber); margin: 18px 0 26px; padding: 2px 0 2px 18px; }}
    .score-status p {{ margin: 4px 0; }} .score-status__label {{ color: var(--evergreen); font-size: 1.2rem; font-weight: 700; }}
    .range-track {{ background: #dfe5e1; height: 5px; margin-top: 18px; position: relative; }}
    .range-track__band {{ background: var(--blue); height: 5px; position: absolute; top: 0; }}
    .range-track__point {{ background: var(--evergreen); border: 3px solid var(--surface); border-radius: 50%; height: 15px; margin-left: -7px; position: absolute; top: -5px; width: 15px; }}
    .range-scale {{ color: var(--muted); display: flex; font-size: .72rem; justify-content: space-between; margin-top: 5px; }}
    .table-wrap {{ margin-top: 20px; overflow-x: auto; }} table {{ border-collapse: collapse; font-size: .88rem; width: 100%; }}
    caption {{ color: var(--evergreen); font-size: 1rem; font-weight: 700; padding: 0 0 10px; text-align: left; }}
    th, td {{ border-bottom: 1px solid var(--rule); padding: 11px 8px; text-align: left; vertical-align: top; }}
    thead th {{ color: var(--muted); font-size: .72rem; letter-spacing: .04em; text-transform: uppercase; }}
    tbody th {{ color: var(--ink); font-weight: 650; }} td {{ font-variant-numeric: tabular-nums; }}
    .capture {{ display: grid; gap: 8px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .capture div {{ border-left: 3px solid var(--rule); padding-left: 10px; }}
    dt {{ color: var(--muted); font-size: .76rem; font-weight: 700; text-transform: uppercase; }} dd {{ margin: 3px 0 0; }}
    .method {{ border-top: 4px solid var(--evergreen); padding-top: 28px; }}
    .method-grid {{ display: grid; gap: 48px; grid-template-columns: 1fr 1fr; margin-top: 24px; }}
    .comparison {{ display: grid; gap: 14px; margin: 0; }} .comparison div {{ border-bottom: 1px solid var(--rule); padding-bottom: 12px; }}
    .limits-copy p {{ margin: 0 0 16px; }} .source-block {{ border-top: 1px solid var(--rule); margin-top: 28px; padding-top: 18px; }}
    .source-list {{ display: grid; gap: 12px; list-style: none; margin: 0; padding: 0; }}
    .source-list li {{ display: grid; gap: 4px; grid-template-columns: .6fr 1fr; }} .source-list small {{ color: var(--muted); grid-column: 2; }}
    .instrument {{ color: var(--muted); font-size: .75rem; margin-top: 24px; overflow-wrap: anywhere; }} .instrument strong {{ color: var(--ink); margin-right: 8px; }}
    .disclaimer {{ border-top: 1px solid var(--rule); color: var(--muted); font-size: .78rem; margin: 34px 0 0; padding-top: 18px; }}
    .chapter-nav {{ display:flex; flex-wrap:wrap; gap:12px; margin:24px 0; }}
    .feature-chapter {{ border-top:1px solid var(--rule); padding:28px 0; display:flow-root; }}
    .face-diagram {{ float:right; width:160px; margin:0 0 18px 22px; }}
    .face-diagram svg {{ width:100%; height:auto; }}
    .profile-diagram {{ display:block; width:100%; max-width:400px; height:auto; margin:18px auto; }}
    .face-diagram figcaption {{ font-size:.7rem; }}
    @media screen and (max-width: 720px) {{
      .face-diagram {{ float:none; margin:0 auto 18px; }} .report {{ padding: 32px 22px 48px; }} .cover-row, .cover-meta {{ align-items: flex-start; flex-direction: column; }}
      .private-mark {{ white-space: normal; }} .snapshot-grid {{ grid-template-columns: 1fr 1fr; }}
      .snapshot-item:nth-child(3) {{ border-left: 0; border-top: 1px solid var(--rule); }} .snapshot-item:nth-child(4) {{ border-top: 1px solid var(--rule); }}
      .reading-guide, .axis, .method-grid {{ grid-template-columns: 1fr; }} .guide-copy {{ grid-template-columns: 1fr; }} .axis {{ row-gap: 14px; }}
      .axis > .status, .axis > p, .axis > .reading, .axis > .score-status, .axis > .table-wrap, .axis > .range-track, .axis > .range-scale, .axis > .capture {{ grid-column: 1; }}
      .source-list li {{ grid-template-columns: 1fr; }} .source-list small {{ grid-column: 1; }}
    }}
    @media print {{ @page {{ margin: 15mm; }} body, .report {{ background: white; }}
      .report {{ max-width: none; min-height: 0; padding: 0; }} .chapter-nav {{ display:none; }} .feature-chapter {{ break-before:auto; }} .feature-chapter {{ break-inside:avoid; }} .report {{ font-size:13px; line-height:1.45; }} .feature-chapter table {{ font-size:12px; }} .composite-panel {{ break-inside:auto; padding:16px; margin-top:20px; }} .composite-panel p {{ margin:8px 0; }} .composite-panel .composite-value {{ font-size:3rem; }} .composite-panel details {{ margin-top:10px; }} h2,h3 {{ break-after:avoid; }} tr,.face-diagram {{ break-inside:avoid; }} .axis, .method, .snapshot-item {{ break-inside: avoid; }} }}
  </style>
</head>
<body>
  <div class="report">
    <header class="report-cover">
      <div class="cover-row">
        <div><p class="kicker">Evidence-led private report</p><h1>Four-pillar facial analysis</h1></div>
        <p class="private-mark">Private local export · Format {_h(payload["format_version"])}</p>
      </div>
      <p class="cover-deck">Your geometry overview, illustrated regional measurements, optional profile readings
      and goal-based styling experiments. Definitions and evidence limits stay beside each result.</p>
      <p class="cover-meta"><span><strong>Age {_h(demographics["age"])}</strong></span>
        <span>{_h(_plain_label(demographics["sex"]).title())}</span>
        <span>{_h(ethnicities)}</span>
        <span><strong>{int(payload["accepted_photo_count"]):,}</strong> accepted photos</span></p>
    </header>
    <main>
      {render_forehead_warning(payload.get("frontal_reviews"))}
      {render_automatic_forehead(payload.get("automatic_forehead", []))}
      {render_product_summary(payload.get("composite") or composite_score({}), payload.get("regional_measurements", []), show_regions=False, expanded=True, score_audit=payload.get("score_audit"))}
      <section class="snapshot" aria-labelledby="snapshot-title">
        <p class="section-number">Executive summary</p><h2 id="snapshot-title">Your results at a glance</h2>
        <div class="snapshot-grid">{summary}</div>
      </section>
      <section class="reading-guide" aria-labelledby="guide-title">
        <div><p class="section-number">Reading guide</p><h2 id="guide-title">How to read this report</h2></div>
        <div class="guide-copy">
          <p><strong>Percentiles are positions.</strong> They compare a measurement with the stated cohort; they are not grades.</p>
          <p><strong>Ranges carry the caution.</strong> A wider interval means the result moved more across photo conditions.</p>
          <p><strong>The breakdown explains the total.</strong> The composite summarizes geometry using disclosed design weights.</p>
        </div>
      </section>
      {render_consultation(payload.get("consultation") or consultation_payload(payload.get("regional_measurements", [])), render_diagrams(payload.get("illustration")))}
      {"".join(render_reviewed(p) for p in payload.get("frontal_reviews", []))}
      {render_profile(payload.get("profile_reviews") or payload.get("profile_review"))}
      {axes}
      {method}
    </main>
    <p class="disclaimer">Demographics were entered by the user. This export contains
    measurements, derived landmark diagrams, preferences and cohort provenance; it does not contain source photos,
    filenames, local paths, photo identifiers, or session credentials.</p>
  </div>
</body>
</html>"""


def write_export(
    payload: Mapping[str, Any],
    destination: Path,
    *,
    session_root: Path,
    overwrite: bool = False,
) -> Path:
    """Atomically save an explicit export outside the temporary session tree."""
    target = Path(destination).expanduser().resolve()
    private_root = Path(session_root).expanduser().resolve()
    if target == private_root or private_root in target.parents:
        raise ValueError(
            "exports must be written outside the temporary session workspace"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(f"export destination already exists: {target}")

    suffix = target.suffix.lower()
    if suffix == ".json":
        content = render_json_export(payload)
    elif suffix in {".html", ".htm"}:
        content = render_html_export(payload).encode("utf-8")
    else:
        raise ValueError("export destination must end in .json, .html, or .htm")

    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export one completed local analysis report."
    )
    parser.add_argument("--session-base", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    options = parser.parse_args(argv)

    store = SessionStore(options.session_base)
    session = store.load(options.session_id)
    report = ProgressiveAnalysisService(store).load_report(options.session_id)
    if report is None:
        parser.error("the selected session has no completed analysis report")
    from .review_routes import profile_payload
    from .frontal_routes import frontal_payloads

    payload = build_export_payload(
        report,
        session.demographics,
        settings=session.state.get("report_preferences"),
        ideas=session.state.get("idea_choices"),
        profile=profile_payload(session, store),
        frontal=frontal_payloads(session, store),
    )
    destination = write_export(
        payload,
        options.output,
        session_root=session.root,
        overwrite=options.overwrite,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_export_payload",
    "render_html_export",
    "render_json_export",
    "source_disclosures",
    "write_export",
]
