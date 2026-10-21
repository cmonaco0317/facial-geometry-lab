"""Disclosed v1 dependencies and frozen shadow candidates; no model promotion."""

from __future__ import annotations
from types import SimpleNamespace
import hashlib
import inspect
import json
from math import isfinite
from .product_report import SCORE_VERSION, SCORE_WEIGHTS, composite_score

VERSION = "score-audit-v1"
CANDIDATES = {
    "baseline-v1": dict(SCORE_WEIGHTS),
    "equal-groups": dict(harmony=1 / 3, symmetry=1 / 3, averageness=1 / 3),
    "symmetry-downweighted": dict(harmony=0.45, symmetry=0.1, averageness=0.45),
    "without-typicality": dict(harmony=0.6, symmetry=0.4, averageness=0.0),
}
RATIO_IDS = {
    "vertical_balance",
    "horizontal_balance",
    "midface_proportion",
    "nose_mouth_eye",
    "lip_balance",
    "lower_face_split",
}


def contract():
    body = {
        "version": SCORE_VERSION,
        "weights": SCORE_WEIGHTS,
        "implementation": inspect.getsource(composite_score),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def ratio_id(value):
    return (
        value
        if value in RATIO_IDS
        else "unrecognized-" + hashlib.sha256(str(value).encode()).hexdigest()
    )


def candidate_score(components, candidate):
    if candidate not in CANDIDATES:
        raise ValueError("Unsupported diagnostic candidate.")
    weights = CANDIDATES[candidate]
    rows = [
        r
        for r in components
        if r["id"] in weights
        and weights[r["id"]] > 0
        and type(r["value"]) in (int, float)
        and isfinite(r["value"])
        and 0 <= r["value"] <= 100
    ]
    total = sum(weights[r["id"]] for r in rows)
    return (
        round(sum(weights[r["id"]] * r["value"] for r in rows) / total, 1)
        if total
        else None
    )


def audit(axes):
    score = composite_score(axes)
    components = [
        {**r, "contribution": r["value"] * r["effective_weight"]}
        for r in score["components"]
    ]
    ratios = []
    leave_ratios = []
    harmony = axes.get("harmony")
    profile = harmony.details.get("profile", {}) if harmony else {}
    ranks = (
        [
            (key, row)
            for key, row in profile.items()
            if isinstance(row, dict)
            and type(row.get("percentile")) in (int, float)
            and isfinite(row["percentile"])
            and 0 <= row["percentile"] <= 100
        ]
        if harmony and harmony.status in ("measured", "estimated")
        else []
    )
    weight = next(
        (r["effective_weight"] for r in components if r["id"] == "harmony"), 0
    )
    for key, row in sorted(ranks):
        percentile = float(row["percentile"])
        centrality = 100 - 2 * abs(percentile - 50)
        ratios.append(
            dict(
                id=ratio_id(key),
                percentile=percentile,
                centrality=centrality,
                within_group_weight=1 / len(ranks),
                effective_weight=weight / len(ranks),
            )
        )
        reduced_harmony = SimpleNamespace(
            status=harmony.status,
            details={
                **harmony.details,
                "profile": {k: v for k, v in profile.items() if k != key},
            },
        )
        reduced = {**axes, "harmony": reduced_harmony}
        leave_ratios.append(
            dict(id=ratio_id(key), value=composite_score(reduced)["value"])
        )
    leave_components = [
        dict(
            id=r["id"],
            value=candidate_score(
                [c for c in components if c["id"] != r["id"]], "baseline-v1"
            ),
        )
        for r in components
    ]
    return dict(
        version=VERSION,
        contract=contract(),
        headline=score,
        components=components,
        ratios=ratios,
        harmony_ratio_ids=[r["id"] for r in ratios],
        harmony_ratio_count=len(ratios),
        leave_one_ratio_out=leave_ratios,
        leave_one_component_out=leave_components,
        shadow_candidates={key: candidate_score(components, key) for key in CANDIDATES},
        decision="Retain geometry-composite-v1. Independent development/held-out evidence does not yet justify a replacement.",
        replacement_justified=False,
        interpretation="Component centrality overlaps. Leave-out results describe formula sensitivity, not expected personal change. No photo-quality penalty or profile contribution is added.",
    )


def render_audit(value, *, expanded=False):
    """Read-only audit shared by the live report and printable export."""
    import html

    def esc(item):
        return html.escape(str(item), quote=True)

    if not isinstance(value, dict) or value.get("version") != VERSION:
        return ""
    rows = "".join(
        f"<tr><th>{esc(r['id'])}</th><td>{r['percentile']:.2f}</td><td>{r['centrality']:.2f}</td><td>{r['effective_weight']:.2%}</td></tr>"
        for r in value["ratios"]
    )
    leave = "".join(
        f"<li>{esc(r['id'])}: {'unavailable' if r['value'] is None else format(r['value'], '.1f')}</li>"
        for r in value["leave_one_component_out"] + value["leave_one_ratio_out"]
    )
    return f"""<details class="score-audit"{" open" if expanded else ""}><summary>Which measurements influence this score?</summary>
    <p>Proportion coverage: {value["harmony_ratio_count"]} available ratio inputs. Group coverage and ratio coverage are different.</p>
    <p>Historical arithmetic: {esc(value["headline"]["value"])}; the headline is rounded for display, without implying tenth-point certainty.</p>
    <div class="table-wrap"><table><thead><tr><th>Ratio input</th><th>Reference percentile</th><th>Centrality</th><th>Effective weight</th></tr></thead><tbody>{rows}</tbody></table></div>
    <p>Leave-one-input-out diagnostic scores (each remaining group is reweighted):</p><ul>{leave}</ul>
    <p>{esc(value["interpretation"])}</p><p>{esc(value["decision"])}</p></details>"""
