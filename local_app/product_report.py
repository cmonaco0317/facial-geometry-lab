"""Versioned product summary. Design weights are not an empirical beauty model."""

from __future__ import annotations

from collections.abc import Mapping
import html
from math import isfinite
import re
from typing import Any

from .contracts import AxisResult

SCORE_VERSION = "geometry-composite-v1"
SCORE_WEIGHTS = {"harmony": 0.4, "symmetry": 0.3, "averageness": 0.3}
SCORE_LABELS = {
    "harmony": "Proportion balance",
    "symmetry": "Symmetry",
    "averageness": "Shape typicality",
}
SCORE_EXPLANATION = (
    "A design-weighted summary of measured geometry, not a population percentile "
    "or a validated attractiveness rating. Higher means more central proportions, "
    "less measured asymmetry and greater reference-shape typicality. These components "
    "overlap. Dimorphism is descriptive and does not change the total."
)


def _number(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        return None
    return float(value)


def composite_score(axes: Mapping[str, AxisResult]) -> dict[str, Any]:
    """Reweight available components explicitly; absent values never become zero."""
    components = []
    for name, weight in SCORE_WEIGHTS.items():
        result = axes.get(name)
        if result is None or result.status not in {"measured", "estimated"}:
            continue
        component_coverage = None
        if name == "harmony":
            rows = result.details.get("profile", {})
            ranks = [
                _number(row.get("percentile"))
                for row in rows.values()
                if isinstance(row, dict)
            ]
            ranks = [v for v in ranks if v is not None and 0 <= v <= 100]
            if not ranks:
                continue
            value = sum(100 - 2 * abs(v - 50) for v in ranks) / len(ranks)
            component_coverage = len(ranks)
        else:
            value = _number(result.percentile)
        if value is None or not 0 <= value <= 100:
            continue
        components.append(
            dict(
                id=name,
                label=SCORE_LABELS[name],
                value=round(value, 1),
                weight=weight,
                status=result.status,
                observed_proportions=component_coverage,
            )
        )
    coverage = round(sum(c["weight"] for c in components), 2)
    value = (
        round(sum(c["value"] * c["weight"] for c in components) / coverage, 1)
        if coverage
        else None
    )
    for component in components:
        component["effective_weight"] = component["weight"] / coverage
    return dict(
        version=SCORE_VERSION,
        value=value,
        scale=100,
        coverage=coverage,
        status="unavailable"
        if not coverage
        else "complete"
        if coverage == 1
        else "partial",
        components=components,
        missing=[
            name for name in SCORE_WEIGHTS if name not in {c["id"] for c in components}
        ],
        explanation=SCORE_EXPLANATION,
        uncertainty="Not calibrated; photo quality is separate from the score.",
    )


def public_regions(axes: Mapping[str, AxisResult]) -> list[dict[str, Any]]:
    """Allowlisted numerical rows only; no image identifiers or arbitrary details."""
    harmony = axes.get("harmony")
    rows = harmony.details.get("regional_measurements", []) if harmony else []
    output = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or _number(row.get("value")) is None:
            continue

        # Labels and explanations come from the local catalogue but still escape at rendering.
        def safe_text(value: Any) -> str:
            text = value if isinstance(value, str) else ""
            if (
                "\x00" in text
                or text.startswith("/")
                or re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", text)
            ):
                return "[local path redacted]"
            return text

        percentile = _number(row.get("percentile"))
        output.append(
            {
                **{
                    key: safe_text(row.get(key))
                    for key in ("id", "label", "unit", "note")
                },
                "method": row.get("method", "automatic-regional-v1")
                if isinstance(row.get("method", "automatic-regional-v1"), str)
                and row.get("method", "automatic-regional-v1") in {
                    "automatic-regional-v1", "automatic-regional-v2"
                }
                else "incompatible",
                **({"evidence_photo_count": row["evidence_photo_count"]}
                   if type(row.get("evidence_photo_count")) is int
                   and 0 < row["evidence_photo_count"] <= 10000 else {}),
                "value": float(row["value"]),
                "percentile": percentile
                if percentile is not None and 0 <= percentile <= 100
                else None,
            }
        )
    return output


def render_product_summary(
    composite: Mapping[str, Any],
    regions: list[dict[str, Any]],
    *,
    show_regions: bool = True,
    expanded: bool = False,
    score_audit: dict | None = None,
) -> str:
    """Shared script-free markup for the application and printable export."""

    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    value = composite.get("value")
    reading = (
        f"{float(value):.0f}<small> / 100</small>"
        if value is not None
        else "No reading yet"
    )
    partial = (
        "Partial score — available components are reweighted. Compare only reports with "
        "the same components and formula."
        if composite["status"] == "partial"
        else "All three component groups contribute."
        if value is not None
        else "No comparison-based score yet. Available measurements remain below."
    )
    rows = "".join(
        f"<tr><th>{esc(c['label'])}</th><td>{c['value']:.1f}</td>"
        f"<td>{c['effective_weight']:.0%} ({c['value'] * c['effective_weight']:.2f} points)</td><td>{esc(c['status'].title())}</td></tr>"
        for c in composite["components"]
    )
    region_rows = "".join(
        f"<tr><th>{esc(row['label'])}</th><td>{float(row['value']):.3f} {esc(row.get('unit', ''))}</td>"
        f"<td>{'—' if row.get('percentile') is None else format(float(row['percentile']), '.0f')}</td>"
        f"<td>{esc(row.get('note', ''))}</td></tr>"
        for row in regions
    )
    region_section = (
        f"""<section class="regional-section"><h2>Regional measurements</h2>
      <p>Observed proportions and angles from the same photographs. Reference positions describe size or shape,
      not a grade; an absent reference does not remove the measurement.</p>
      <div class="table-wrap"><table><thead><tr><th>Measurement</th><th>Reading</th>
      <th>Reference percentile</th><th>Interpretation</th></tr></thead><tbody>{region_rows}</tbody></table></div>
      </section>"""
        if regions and show_regions
        else ""
    )
    from .score_audit import render_audit

    audit_markup = render_audit(score_audit, expanded=expanded)
    return f"""<section class="composite-panel" aria-labelledby="composite-title">
      <p class="report-kicker">Overview · {esc(composite["version"])}</p>
      <h2 id="composite-title">Overall geometry score</h2>
      <p>Experimental geometry summary · not an attractiveness percentile.</p>
      <p class="composite-value">{reading}</p><p>{esc(partial)}</p>
      <p>Component coverage: {composite["coverage"]:.0%}. No photo-quality penalty is applied.</p>
      <details{" open" if expanded else ""}><summary>Scoring breakdown</summary>
      <table><thead><tr><th>Component</th><th>Score</th><th>Weight used</th><th>Evidence</th></tr></thead>
      <tbody>{rows}</tbody></table>
      <p>{esc(composite["explanation"])}</p>
      <p>Proportion balance averages 100 − 2 × |reference percentile − 50| across available
      proportions. Full weights: balance 40%, symmetry 30%, typicality 30%.</p>
      <p>{esc(composite["uncertainty"])}</p></details>{audit_markup}</section>{region_section}"""
