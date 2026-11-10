"""Server-rendered landing and result pages for the private local workflow."""

from __future__ import annotations

import html
import json
from typing import Any
from urllib.parse import quote

from .analysis import AXIS_ORDER, AnalysisReport
from .contracts import AxisResult, CohortUse
from .report import render_capture_summary, source_disclosures
from .consultation import (
    consultation_payload,
    preference_form,
    preferences,
    render_consultation,
)
from .illustrations import render_diagrams
from .profile_review import render_profile
from .landmark_review import render_reviewed, render_forehead_warning
from .product_report import composite_score, public_regions, render_product_summary
from .score_audit import audit
from .automatic_forehead import public_forehead, render_automatic_forehead

AXIS_TITLES = {
    "harmony": "Harmony",
    "dimorphism": "Dimorphism",
    "symmetry": "Symmetry",
    "averageness": "Averageness",
}

ETHNICITY_OPTIONS = (
    ("white", "White"),
    ("black", "Black"),
    ("east_asian", "East Asian"),
    ("southeast_asian", "Southeast Asian"),
    ("indian", "Indian"),
    ("middle_eastern", "Middle Eastern"),
    ("latino_hispanic", "Latino / Hispanic"),
    ("mixed", "Multiple ethnicities"),
    ("another_or_not_listed", "Another or not listed"),
)

SEX_OPTIONS = (
    ("female", "Female"),
    ("male", "Male"),
    ("another", "Another"),
    ("intersex", "Intersex"),
    ("prefer_not_to_say", "Prefer not to say"),
)

FALLBACK_LABELS = {
    "exact": "Exact demographic match",
    "same_age_sex_broader_ethnicity": "Broader ethnicity fallback",
    "nearest_age_same_sex_ethnicity": "Nearest age fallback",
    "nearest_age_same_sex_broader_ethnicity": "Age and ethnicity fallback",
    "all_sex_same_age_ethnicity": "All-sex comparison",
    "all_sex_same_age_broader_ethnicity": "All-sex, broader ethnicity fallback",
    "all_sex_nearest_age_ethnicity": "All-sex, nearest age fallback",
    "all_sex_nearest_age_broader_ethnicity": "All-sex, age and ethnicity fallback",
}


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _ordinal(value: float) -> str:
    rounded = int(round(float(value)))
    if 10 <= rounded % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rounded % 10, "th")
    return f"{rounded}{suffix}"


def _number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _document(*, title: str, body: str, page_class: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>{_escape(title)}</title>
  <link rel="stylesheet" href="/static/app.css">
  <script src="/static/app.js" defer></script>
  <script src="/static/history.js" defer></script>
</head>
<body class="{_escape(page_class)}">
{body}
</body>
</html>"""


def _evidence_strip() -> str:
    frames = "".join(
        f'<span class="evidence-frame"><i>{index}</i></span>' for index in range(1, 5)
    )
    return (
        '<div class="evidence-strip" aria-label="Start with one photo; add more views">'
        f"{frames}</div>"
    )


def render_landing(
    *,
    action: str = "/sessions",
    csrf_token: str | None = None,
) -> str:
    """Render one form that collects adult demographics and one or more photos together."""
    sex_controls = "".join(
        f"""<label class="choice">
  <input type="radio" name="sex" value="{_escape(value)}" required>
  <span>{_escape(label)}</span>
</label>"""
        for value, label in SEX_OPTIONS
    )
    ethnicity_controls = "".join(
        f"""<label class="choice">
  <input type="checkbox" name="ethnicity" value="{_escape(value)}">
  <span>{_escape(label)}</span>
</label>"""
        for value, label in ETHNICITY_OPTIONS
    )
    csrf_control = ""
    if csrf_token is not None:
        csrf_control = (
            f'<input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">'
        )
    body = f"""
<p class="history-navigation"><a href="/history">Private report history</a></p>
<p data-history-status role="status" class="history-navigation"></p>
<main class="landing-shell">
  <aside class="evidence-rail" aria-labelledby="local-title">
    <p class="local-mark"><span aria-hidden="true"></span> Local analysis</p>
    <h1 id="local-title">Your facial analysis.<br>One clear overview.</h1>
    {_evidence_strip()}
    <p class="rail-copy">Start with 1 or more photos. Extra photos remain
    useful when they add a clearer angle or stronger evidence.</p>
    <dl class="privacy-facts">
      <div><dt>Where it runs</dt><dd>Everything stays on this computer.</dd></div>
      <div><dt>What it returns</dt><dd>A composite overview, illustrated regional chapters and optional profile review.</dd></div>
      <div><dt>When files leave</dt><dd>Temporary files are deleted when you end the session. Explicitly saved history stays until you delete it.</dd></div>
    </dl>
  </aside>

  <section class="intake-panel" aria-labelledby="intake-title">
    <header class="panel-heading">
      <p>Adult, local, cohort-matched</p>
      <h2 id="intake-title">Create a measurement session</h2>
      <p>Your age, sex, and ethnicity select the comparison group. They are never
      guessed from your face.</p>
    </header>

    <form class="intake-form" method="post" action="{_escape(action)}" enctype="multipart/form-data">
      {csrf_control}
      <div class="age-field">
        <label for="age">Age</label>
        <input id="age" name="age" type="number" min="18" step="1" inputmode="numeric" required>
        <p>Adults 18 and older.</p>
      </div>

      <fieldset>
        <legend>Sex</legend>
        <p class="field-help">Used only to select a comparison group.</p>
        <div class="choice-grid choice-grid--sex">{sex_controls}</div>
      </fieldset>

      <fieldset>
        <legend>Ethnicity</legend>
        <p class="field-help">Choose every option that applies. If an exact cohort
        is unavailable, the result names the closest group actually used.</p>
        <div class="choice-grid">{ethnicity_controls}</div>
      </fieldset>

      <div class="photo-field">
        <label class="photo-drop" for="photos">
          <span class="photo-drop__title">Choose 1 or more photos</span>
          <span class="photo-drop__copy">Casual JPEG or PNG photos are fine. Different
          angles help; include at least a few straight-on, neutral views in even light.</span>
          <span class="photo-drop__button">Choose photos</span>
        </label>
        <input id="photos" type="file" name="photos" accept="image/jpeg,image/png" multiple required data-photo-input>
        <p class="file-count" data-file-count aria-live="polite">No photos selected yet.</p>
      </div>

      <button class="primary-action" type="submit">Create local analysis</button>
      <p class="form-note">No account, cloud upload, or background network request.</p>
    </form>
  </section>
</main>"""
    return _document(
        title="Local facial measurements", body=body, page_class="landing-page"
    )


def _fallback_copy(cohort: CohortUse) -> str:
    label = FALLBACK_LABELS.get(
        cohort.fallback_level,
        cohort.fallback_level.replace("_", " ").capitalize(),
    )
    requested = cohort.requested_label or cohort.requested_id
    used = cohort.used_label or cohort.used_id
    if cohort.fallback_level == "exact":
        return f"{_escape(label)}: {_escape(used)}."
    return (
        f"<strong>{_escape(label)}</strong>. Requested {_escape(requested)}; "
        f"compared with {_escape(used)}."
    )


def _axis_evidence(result: AxisResult) -> str:
    photo_word = "photo" if result.photo_count == 1 else "photos"
    return f"""
<dl class="evidence-note">
  <div><dt>Evidence</dt><dd>{result.photo_count:,} eligible {photo_word}</dd></div>
</dl>"""


def _technical_details(details: dict[str, Any]) -> str:
    public_keys = {
        "alignment_probability",
        "features_used",
        "held_out_calibration_error",
        "moves_percentile_points",
        "n_cohort_faces",
        "n_cohort_faces_near_frontal",
        "n_photographs_used",
        "n_photos_used",
        "p_male",
        "percentile_direction",
        "percentile_interval",
        "photos_in_gate",
        "placement",
        "probability_interval",
    }
    visible = {key: value for key, value in details.items() if key in public_keys}
    if not visible:
        return ""
    encoded = json.dumps(
        visible, ensure_ascii=False, indent=2, sort_keys=True, default=str
    )
    return (
        '<details class="technical"><summary>Measurement details</summary>'
        f"<pre>{_escape(encoded)}</pre></details>"
    )


def _measurement_track(
    point: float, interval: tuple[float, float] | list[float]
) -> str:
    value = min(100.0, max(0.0, float(point)))
    low = min(100.0, max(0.0, float(interval[0])))
    high = min(100.0, max(0.0, float(interval[1])))
    if high < low:
        low, high = high, low
    label = (
        f"Point estimate {_ordinal(value)} percentile; uncertainty "
        f"{_ordinal(low)} to {_ordinal(high)} percentile"
    )
    return f"""<svg class="measurement-track" viewBox="0 0 100 14"
  preserveAspectRatio="none" role="img" aria-label="{_escape(label)}">
  <line class="measurement-track__rail" x1="0" y1="7" x2="100" y2="7"></line>
  <line class="measurement-track__band" x1="{low:.2f}" y1="7" x2="{high:.2f}" y2="7"></line>
  <line class="measurement-track__point-outline" x1="{value:.2f}" y1="2" x2="{value:.2f}" y2="12"
    vector-effect="non-scaling-stroke"></line>
  <line class="measurement-track__point" x1="{value:.2f}" y1="2" x2="{value:.2f}" y2="12"
    vector-effect="non-scaling-stroke"></line>
</svg>"""


def _render_scalar(result: AxisResult) -> str:
    if result.percentile is None or result.interval is None:
        raise ValueError("scalar result is missing its percentile interval")
    low, high = result.interval
    value = _number(result.value) if result.value is not None else ""
    profile = result.details.get("profile", {})
    profile_details = (
        '<details class="profile-details"><summary>Underlying proportions</summary>'
        f"{_profile_list(profile)}</details>"
        if profile
        else ""
    )
    placement_unresolved = result.axis == "averageness" and str(
        result.details.get("placement", "")
    ).startswith("withheld")
    if placement_unresolved:
        headline = (
            f'<p class="percentile-reading">Estimated range: '
            f"{_ordinal(low)}–{_ordinal(high)} percentile</p>"
        )
        interval_reading = (
            '<p class="interval-reading"><strong>Placement unresolved</strong> · '
            f"Point estimate: {_ordinal(result.percentile)} percentile</p>"
        )
    else:
        headline = f'<p class="percentile-reading">{_ordinal(result.percentile)} percentile</p>'
        interval_reading = (
            f'<p class="interval-reading">Uncertainty: {_ordinal(low)}–'
            f"{_ordinal(high)} percentile</p>"
        )
    return f"""
<p class="axis-status axis-status--measured">Measured</p>
{headline}
{interval_reading}
{_measurement_track(result.percentile, result.interval)}
<p class="axis-summary">{_escape(result.summary)}</p>
<p class="raw-reading"><span>{_escape(value)}</span> {_escape(result.unit or "")}</p>
{profile_details}
{_axis_evidence(result)}
{_technical_details(result.details)}"""


def _profile_list(profile: dict[str, Any]) -> str:
    rows = []
    for key, row in profile.items():
        label = row.get("label") or key.replace("_", " ").capitalize()
        percentile = row.get("percentile")
        interval = row.get("interval")
        comparison = (
            f"{_ordinal(percentile)} reference percentile"
            if percentile is not None
            else "Reference unavailable"
        )
        if interval:
            comparison += f" · {_ordinal(interval[0])}–{_ordinal(interval[1])}"
        rows.append(
            f"""<li>
  <div><strong>{_escape(label)}</strong><span>{_number(row["value"])}</span></div>
  <p>{comparison}</p>
  {_measurement_track(percentile, interval) if interval and percentile is not None else ""}
</li>"""
        )
    return f'<ul class="profile-list">{"".join(rows)}</ul>'


def _render_profile(result: AxisResult) -> str:
    profile = result.details.get("profile", {})
    score_status = ""
    if result.axis == "harmony":
        score_status = """<div class="score-status">
  <p class="score-status__label">Proportion balance contributes to the overall score</p>
  <p>The available component profile shows the individual readings behind
  the balance component of the overall score.</p>
</div>"""
    summary = (
        "Measured component geometry is shown below."
        if result.axis == "harmony"
        else result.summary
    )
    return f"""
<p class="axis-status axis-status--measured">Profile measured</p>
{score_status}
<p class="axis-summary">{_escape(summary)}</p>
{_profile_list(profile)}
{_axis_evidence(result)}
{_technical_details(result.details)}"""


def _render_photo_request(result: AxisResult) -> str:
    request = result.photo_request
    if request is None:
        raise ValueError("photo request result has no request")
    return f"""
<p class="axis-status axis-status--request">Needs another photo</p>
<p class="request-reason">{_escape(request.reason)}</p>
<dl class="capture-spec">
  <div><dt>View</dt><dd>{_escape(request.view)}</dd></div>
  <div><dt>Expression</dt><dd>{_escape(request.expression)}</dd></div>
  <div><dt>Light</dt><dd>{_escape(request.lighting)}</dd></div>
  <div><dt>Frame</dt><dd>{_escape(request.framing)}</dd></div>
</dl>
{_axis_evidence(result)}"""


def _render_processing_error(
    result: AxisResult,
    *,
    csrf_token: str,
    analyze_action: str,
) -> str:
    retry = ""
    if result.retryable:
        retry = f"""<form method="post" action="{_escape(analyze_action)}" class="inline-action">
  <input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
  <button type="submit">Retry {_escape(AXIS_TITLES[result.axis].lower())}</button>
</form>"""
    return f"""
<p class="axis-status axis-status--error">Could not finish this measurement</p>
<p class="error-copy">{_escape(result.error_message or result.summary)}</p>
{retry}
{_axis_evidence(result)}"""


def _render_estimate(result: AxisResult) -> str:
    reading = (
        f'<p class="percentile-reading">{_number(result.value)} {_escape(result.unit)}</p>'
        if result.headline_available
        else _profile_list(result.details.get("profile", {}))
    )
    rank = (
        f"<p>Approximate reference position: {_ordinal(result.percentile)} percentile</p>"
        if result.percentile is not None
        else ""
    )
    issues = ", ".join(
        str(key).replace("_", " ") for key in result.details.get("capture_issues", {})
    )
    return f"""<p class="axis-status">Estimate · {result.photo_count} photo(s)</p>
{reading}{rank}<p>{_escape(result.summary)}</p>
<p>Capture quality: {_escape(issues or "limited calibration")}.</p>
<p>{_escape(result.details.get("estimate_note", "Uncertainty is not calibrated for this estimate."))}</p>"""


def _render_axis_card(
    axis: str,
    result: AxisResult,
    *,
    csrf_token: str,
    analyze_action: str,
) -> str:
    if result.status == "estimated":
        content = _render_estimate(result)
    elif result.status == "measured":
        content = (
            _render_scalar(result)
            if result.headline_available
            else _render_profile(result)
        )
    elif result.status == "needs_photo":
        content = _render_photo_request(result)
    else:
        content = _render_processing_error(
            result,
            csrf_token=csrf_token,
            analyze_action=analyze_action,
        )
    return f"""<article class="axis-card axis-card--{_escape(axis)} axis-card--{_escape(result.status)}" aria-labelledby="{axis}-title">
  <header class="axis-card__header"><h2 id="{axis}-title">{_escape(AXIS_TITLES[axis])}</h2></header>
  <div class="axis-card__body">{content}{render_capture_summary(result.details.get("capture_diagnostics"))}</div>
</article>"""


def _overview_item(axis: str, result: AxisResult) -> str:
    if result.status == "estimated":
        reading = (
            f"{_number(result.value)} {result.unit}"
            if result.headline_available
            else "Proportion estimates"
        )
        detail = "Best effort · capture quality below"
    elif result.status == "measured" and result.headline_available:
        if result.interval is None or result.percentile is None:
            reading = "Measured"
            detail = "No percentile interval available"
        elif axis == "averageness" and str(
            result.details.get("placement", "")
        ).startswith("withheld"):
            reading = (
                f"{_ordinal(result.interval[0])}–{_ordinal(result.interval[1])} range"
            )
            detail = "Placement unresolved"
        else:
            reading = f"{_ordinal(result.percentile)} percentile"
            detail = (
                f"Range {_ordinal(result.interval[0])}–{_ordinal(result.interval[1])}"
            )
    elif result.status == "measured" and axis == "harmony":
        reading = "Harmony profile"
        detail = "Proportion balance contributes to the overall score"
    elif result.status == "measured":
        reading = "Profile measured"
        detail = "No validated scalar headline"
    elif result.status == "needs_photo":
        reading = "More evidence needed"
        detail = f"{result.photo_count:,} eligible photos so far"
    else:
        reading = "Not completed"
        detail = (
            "A retry is available below" if result.retryable else "See details below"
        )
    return f"""<li class="overview-item overview-item--{_escape(result.status)}">
  <span class="overview-item__label">{_escape(AXIS_TITLES[axis])}</span>
  <strong class="overview-item__reading">{_escape(reading)}</strong>
  <span class="overview-item__detail">{_escape(detail)}</span>
</li>"""


def _results_overview(report: AnalysisReport) -> str:
    items = "".join(_overview_item(axis, report.axes[axis]) for axis in AXIS_ORDER)
    return f"""<section class="results-overview" aria-labelledby="overview-title">
  <div class="overview-heading">
    <p class="report-kicker">Executive summary</p>
    <h2 id="overview-title">Your results at a glance</h2>
    <p>Explore the measurements behind your overall score.</p>
  </div>
  <ol class="overview-spine">{items}</ol>
</section>"""


def _reference_disclosure(
    cohort: CohortUse,
    instrument_fingerprint: str,
    harmony: AxisResult,
) -> str:
    items = "".join(
        f"<li><strong>{_escape(item['source'])}</strong>: "
        f"{_escape(item['label_meaning'])}.</li>"
        for item in source_disclosures(cohort)
    )
    requested = cohort.requested_label or cohort.requested_id
    used = cohort.used_label or cohort.used_id
    identity_word = "identity" if cohort.sample_size == 1 else "identities"
    if harmony.status == "estimated":
        harmony_note = "Best-effort proportions are shown with capture-quality notes."
    elif harmony.status == "measured":
        if harmony.headline_available:
            harmony_note = (
                "The harmony headline is a validated geometric measure and is shown "
                "with its underlying proportions; it is not an attractiveness score."
            )
        else:
            harmony_note = (
                "Harmony components feed the separate product score. That aggregation "
                "is a design choice, not a validated attractiveness model."
            )
    else:
        harmony_note = (
            "A harmony result is not available yet; its card shows the current photo "
            "request or processing status."
        )
    return f"""<section class="method-and-limits" aria-labelledby="method-title">
  <div class="method-heading">
    <p class="report-kicker">Evidence note</p>
    <h2 id="method-title">Method and limits</h2>
  </div>
  <div class="method-grid">
    <dl class="method-facts">
      <div><dt>Requested group</dt><dd>{_escape(requested)}</dd></div>
      <div><dt>Comparison group used</dt><dd>{_escape(used)}</dd></div>
      <div><dt>Cohort selection</dt><dd>{_fallback_copy(cohort)}</dd></div>
      <div><dt>Reference sample</dt><dd>{cohort.sample_size:,} reference {identity_word}</dd></div>
    </dl>
    <div class="method-copy">
      <h3>How to read the numbers</h3>
      <p>Percentiles describe position within the comparison cohort, not quality.
      Uncertainty ranges show how much the measurement can move with photo evidence.</p>
      <p><strong>The overall score uses an explicit geometric formula.</strong>
      {_escape(harmony_note)}</p>
      <p><strong>Human-rater check.</strong> In the existing SCUT-FBP5500 analysis,
      only nose–mouth–eye cleared the preregistered association bar in every measured
      cohort; the other five remain descriptive geometry. The panel was 60 Chinese
      volunteers aged 18–27, so it is not universal.</p>
      <h3>How the reference labels were assigned</h3>
      <ul>{items}</ul>
    </div>
  </div>
  <p class="instrument-line"><strong>Instrument fingerprint</strong>
  {_escape(instrument_fingerprint)}</p>
</section>"""


def render_results(
    report: AnalysisReport,
    *,
    csrf_token: str,
    quit_after_end: bool = False,
    settings: dict[str, str] | None = None,
    profile: dict[str, Any] | list[dict] | None = None,
    ideas: dict | None = None,
    frontal: list[dict] | None = None,
) -> str:
    """Render the same typed four-axis payload used by persisted/exported reports."""
    session = quote(report.session_id, safe="")
    base = f"/sessions/{session}"
    upload_action = f"{base}/photos"
    analyze_action = f"{base}/analyze"
    end_action = f"{base}/end"
    json_export_action = f"{base}/report.json"
    html_export_action = f"{base}/report.html"
    end_label = (
        "End session and quit" if quit_after_end else "End session and delete files"
    )
    cards = "".join(
        _render_axis_card(
            axis,
            report.axes[axis],
            csrf_token=csrf_token,
            analyze_action=analyze_action,
        )
        for axis in AXIS_ORDER
    )
    photo_word = "photo" if report.accepted_photos == 1 else "photos"
    body = f"""
<header class="results-header">
  <div>
    <p class="local-mark"><span aria-hidden="true"></span> Analysis stays local</p>
    <h1>Four-pillar facial analysis</h1>
    <p>{report.accepted_photos:,} {photo_word} in this session. Each pillar uses
    the clearest available photos. Estimates remain visible when capture conditions are imperfect.</p>
  </div>
  <nav class="save-actions" aria-label="Save report">
    <a class="save-link" href="{
        _escape(html_export_action)
    }" download>Save printable report</a>
    <a class="save-link" href="{_escape(json_export_action)}" download>Save JSON</a>
  </nav>
</header>

<main class="results-shell">
  {render_automatic_forehead(public_forehead(report.axes))}
  {render_forehead_warning(frontal)}
  {
        render_product_summary(
            composite_score(report.axes),
            public_regions(report.axes, frontal=frontal),
            show_regions=False,
            score_audit=audit(report.axes),
        )
    }
  {preference_form(preferences(settings or {}), base + "/preferences", csrf_token)}
  <p class="review-link"><a href="{
        base
    }/frontal">Review frontal landmarks and hairline</a> · <a href="{
        base
    }/profile">Review profile landmarks</a></p>
  {"".join(render_reviewed(p) for p in frontal or [])}
  {
        render_consultation(
            consultation_payload(
                public_regions(report.axes, frontal=frontal),
                settings,
                frontal=frontal,
                ideas=ideas,
                profile=profile,
            ),
            render_diagrams(report.axes["harmony"].details.get("illustration")),
            action=base + "/ideas",
            csrf=csrf_token,
        )
    }
  {render_profile(profile)}
  {_results_overview(report)}

  <section class="axis-grid" aria-label="Facial measurement results">{cards}</section>

  {
        _reference_disclosure(
            report.cohort,
            report.instrument_fingerprint,
            report.axes["harmony"],
        )
    }

  <section class="continue-panel" aria-labelledby="continue-title">
    <div>
      <h2 id="continue-title">Add more photos</h2>
      <p>A new usable angle can complete a card
      or refine its uncertainty; unusable additions leave measured cards intact.</p>
    </div>
    <form method="post" action="{
        _escape(upload_action)
    }" enctype="multipart/form-data" class="add-photo-form">
      <input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
      <label for="more-photos">Choose additional photos</label>
      <input id="more-photos" type="file" name="photos" accept="image/jpeg,image/png" multiple data-photo-input>
      <p class="file-count" data-file-count aria-live="polite">No additional photos selected.</p>
      <button type="submit">Add photos and update</button>
    </form>
  </section>
</main>

<footer class="session-footer">
  <p>Photos and session measurements are deleted when you end this session.</p>
  <form method="post" action="{_escape(end_action)}">
    <input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
    <button class="quiet-action" type="submit">{end_label}</button>
  </form>
</footer>"""
    from .history_page import session_panel

    body += session_panel(report.session_id, csrf_token or "")
    return _document(
        title="Four-pillar facial analysis", body=body, page_class="results-page"
    )
