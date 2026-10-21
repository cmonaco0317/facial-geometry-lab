#!/usr/bin/env python3
"""Render the offline synthetic portfolio demo; never opens photos or networks."""

from __future__ import annotations

import json
from pathlib import Path

from examples.synthetic import frontal_points, illustrative_axes, profile_points, region_diagram
from local_app.illustrations import render_diagrams
from local_app.landmark_review import public_review, render_reviewed
from local_app.product_report import composite_score, render_product_summary
from local_app.profile_review import render_profile
from local_app.profile_v2 import measure
from local_app.score_audit import audit

ROOT = Path(__file__).resolve().parent


def demo_payload() -> dict:
    """Compute actual geometry on fictional coordinates and a separate score fixture."""
    frontal = public_review(dict(reviewed=True, revision=1, points=frontal_points()), 400, 600)
    profile = {**measure(profile_points(), 400, 600, facing="right"), "revision": 1}
    axes = illustrative_axes()
    return dict(
        provenance="SYNTHETIC: fictional coordinates and independently invented rank inputs; no person or dataset",
        frontal=frontal, profile=profile, composite=composite_score(axes), score_audit=audit(axes),
    )


def render_demo() -> str:
    """Assemble real component renderers in a standalone, script-free report."""
    data = demo_payload()
    diagrams = render_diagrams(region_diagram())
    figures = "".join(
        f'<article class="region"><p class="eyebrow">{name}</p>{diagrams[name]}</article>'
        for name in ("proportions", "eyes", "jaw")
    )
    summary = render_product_summary(data["composite"], [], score_audit=data["score_audit"])
    body = f'''<header><a class="wordmark" href="#top">FG / LAB</a><span>GEOMETRY · EVIDENCE · REVIEW</span><a href="#methods">Read the method ↗</a></header>
    <main id="top"><section class="hero"><div><p class="eyebrow">FACIAL GEOMETRY LAB / PUBLIC EDITION</p>
    <h1>Make the measurement<br><em>visible.</em></h1><p class="intro">An inspectable geometry report, from landmark placement to the arithmetic behind every reading.</p>
    <div class="chips"><span>Offline by design</span><span>Script-free export</span><span>Revision-bound geometry</span></div></div>
    <aside class="specimen"><p class="eyebrow">SYNTHETIC SPECIMEN / 01</p>{diagrams['proportions']}</aside></section>
    <section class="notice"><strong>Entirely synthetic demonstration.</strong> Every coordinate is a newly authored schematic; every score input is invented. No photograph, personal measurement, dataset or model is included. The score fixture is independent of the landmark calculations.</section>
    <section class="section-heading"><p class="eyebrow">01 / READING THE STRUCTURE</p><h2>Every line has a definition.</h2><p>Semantic anchors, explicit units and original canvas proportions. Geometry below is calculated by the shipped Python components.</p></section>
    <div class="region-grid">{figures}</div>
    <section class="section-heading"><p class="eyebrow">02 / TRANSPARENT ARITHMETIC</p><h2>A score you can take apart.</h2><p>Fictional percentile inputs demonstrate reweighting and leave-one-input-out sensitivity. They do not describe a real reference population.</p></section>{summary}
    <section class="section-heading"><p class="eyebrow">03 / REVIEWED LANDMARKS</p><h2>Inspect the geometry.</h2><p>The following frontal and profile readings are computed directly from the synthetic schematic. No percentile is assigned.</p></section>
    {render_reviewed(data['frontal'])}{render_profile(data['profile'])}
    <section id="methods" class="methods"><p class="eyebrow">04 / THE ENGINEERING</p><h2>Small contracts.<br>Honest boundaries.</h2><div class="method-grid"><div><h3>Validate at the boundary</h3><p>Finite, bounded coordinates. Explicit missing anchors. Invalid dimensions and stale revisions are rejected.</p></div><div><h3>Preserve the evidence</h3><p>Manual readings have versioned definitions and source contracts. Missing anchors disable only dependent measurements.</p></div><div><h3>Separate the claims</h3><p>Two-dimensional geometry is descriptive. This public edition includes no automatic photo analysis, clinical diagnosis or calibrated beauty model.</p></div></div></section>
    <footer><strong>Facial Geometry Lab</strong><span>Fictional schematic · local computation · no external resources</span></footer></main>'''
    # Shared component copy normally describes photo inputs. This fixture-only shell
    # makes the different provenance explicit without changing production definitions.
    body = body.replace("Observed landmark spans from one selected photograph. Measurements may summarize several photos.", "Fictional schematic landmark spans; no source photograph.")
    body = body.replace("Saved reviewed points on one photograph; original aspect ratio preserved. No source image is included.", "Fictional schematic anchors; original canvas aspect ratio preserved.")
    css = (ROOT / "docs" / "demo.css").read_text(encoding="utf-8")
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Facial Geometry Lab — synthetic demonstration</title><style>{css}</style></head><body>{body}</body></html>'


def main() -> None:
    destination = ROOT / "docs"
    destination.mkdir(exist_ok=True)
    (destination / "demo.html").write_text(render_demo(), encoding="utf-8")
    (destination / "synthetic-report.json").write_text(
        json.dumps(demo_payload(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Generated docs/demo.html and docs/synthetic-report.json (synthetic only).")


if __name__ == "__main__":
    main()
