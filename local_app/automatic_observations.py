"""Bind existing regional arithmetic to its actual detector inputs, without estimates."""

from __future__ import annotations

from math import isfinite

from .anatomical_observations import PROXY_LABELS
from .observations import Observation, Point, SourceEvidence, public_observations
from .reading_registry import VERSION, reading_registry


def automatic_observations(
    rows, context, photos, *, forehead_statuses=None, forehead_overrides=None
) -> tuple[Observation, ...]:
    entries = {p.entry.photo_id: (n, p.entry) for n, p in enumerate(photos, 1)}
    detected = {p["photo"]: p for p in context.kept}
    facts = {p.photo_id: p for p in context.photo_facts}
    by_id = {row["id"]: row for row in rows}
    result = []
    for definition in reading_registry()["readings"]:
        if definition["family"] not in {"automatic", "iris"}:
            continue
        row = by_id.get(definition["id"], {})
        sources = []
        valid = True
        for photo_id in row.get("evidence_photo_ids", ()):
            if photo_id not in entries:
                valid = False
                break
            ordinal, entry = entries[photo_id]
            points = row.get("source_geometry", {}).get(photo_id)
            if points is None:
                points = detected.get(photo_id, {}).get("landmarks")
            try:
                if type(entry.width) is not int or type(entry.height) is not int:
                    raise ValueError("Image dimensions are missing.")
                anchors = tuple(
                    Point(
                        name=f"p{i}",
                        x=float(points[i][0]) / entry.width,
                        y=float(points[i][1]) / entry.height,
                    )
                    for i in definition["inputs"]
                )
                sources.append(
                    SourceEvidence(
                        photo_id=photo_id,
                        photo_number=ordinal,
                        view="unknown",
                        width=entry.width,
                        height=entry.height,
                        revision=1,
                        points=anchors,
                        visibility="unassessed",
                        automatic_forehead_status=(forehead_statuses or {}).get(
                            photo_id
                        ),
                        forehead_visibility=(forehead_overrides or {}).get(photo_id),
                        capture_eligible=bool(
                            facts.get(photo_id)
                            and facts[photo_id].eligible_for("harmony")
                        ),
                    )
                )
            except (TypeError, IndexError, KeyError, ValueError, ZeroDivisionError):
                valid = False
                break
        value = row.get("value")
        available = (
            valid and bool(sources) and type(value) in (int, float) and isfinite(value)
        )
        proxy = definition["id"] in PROXY_LABELS
        result.append(
            Observation(
                definition_key=definition["key"],
                definition_version=VERSION,
                label=definition["label"],
                definition=definition["definition"],
                formula="median over contributing photos: " + definition["formula"],
                method=definition["method"],
                unit=definition["unit"],
                status=(
                    ("proxy" if proxy else "automatic") if available else "unavailable"
                ),
                value=float(value) if available else None,
                revision=1,
                sources=tuple(sources) if valid else (),
                proxy_for=(
                    "anatomical:" + definition["id"] if proxy and available else None
                ),
                withheld_reason=(
                    None
                    if available
                    else "No compatible finite reading with complete source points. No evaluated same-quantity estimator is available."
                ),
                score_eligible=False,
                score_exclusion_reason="Regional observations do not replace the legacy composite's separately versioned inputs.",
            )
        )
    public_observations(tuple(result))
    return tuple(result)
