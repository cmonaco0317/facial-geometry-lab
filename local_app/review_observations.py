"""Rebuild current reviewed observations using the unchanged numerical instruments."""

from __future__ import annotations

from typing import Any

from .landmark_review import safe_review
from .observations import Observation, Point, SourceEvidence, public_observations
from .profile_v2 import safe_profile
from .reading_registry import reading_registry
from .profile_lineage import proposal_origin


def _visibility(payload: dict, definition: dict) -> str:
    """Preserve declared coverage; finite arithmetic does not establish visibility."""
    declared = payload.get("forehead_visibility", "unassessed")
    requires = set(definition["inputs"])
    if "hairline" in requires:
        return {
            "visible": "visible",
            "hairline_hidden": "occluded",
            "forehead_hidden": "occluded",
            "uncertain": "uncertain",
        }.get(declared, "unassessed")
    if requires & {"mesh_top", "forehead_a", "forehead_b"}:
        return {
            "visible": "visible",
            "hairline_hidden": "visible",
            "forehead_hidden": "occluded",
        }.get(declared, "unassessed")
    return "unassessed"


def _source(raw: dict, payload: dict, definition: dict) -> SourceEvidence:
    number = raw.get("photo_number")
    if type(number) is not int or not 1 <= number <= 10000:
        raise ValueError(
            "Reviewed observations need the source photo's report ordinal."
        )
    diagram = payload["diagram"]
    points = diagram["points"]
    return SourceEvidence(
        photo_id=raw.get("evidence_photo_id", f"photo-{number}"),
        photo_number=number,
        view=(
            "frontal"
            if definition["family"] == "reviewed_frontal"
            else f"profile_{payload['facing']}"
        ),
        width=diagram["width"],
        height=diagram["height"],
        revision=raw.get("image_revision", 1),
        review_revision=payload["revision"],
        points=tuple(
            Point(name=name, x=points[name][0], y=points[name][1])
            for name in definition["inputs"]
            if points[name] is not None
        ),
        visibility=_visibility(payload, definition),
        forehead_visibility=payload.get("forehead_visibility"),
        automatic_forehead_status=payload.get("automatic_forehead_status"),
        capture_eligible=False,
    )


def _observations(raw: dict, family: str) -> tuple[Observation, ...]:
    payload = safe_review(raw) if family == "reviewed_frontal" else safe_profile(raw)
    if payload is None:
        raise ValueError(
            "Unsupported or incompatible saved review; preserve its legacy snapshot separately."
        )
    definitions = [r for r in reading_registry()["readings"] if r["family"] == family]
    values = {r["id"]: r["value"] for r in payload["measurements"]}
    result = []
    origin = (
        proposal_origin(raw.get("proposal_ancestry"))
        if family == "reviewed_profile"
        else None
    )
    for definition in definitions:
        value = values.get(definition["id"])
        missing = [
            name
            for name in definition["inputs"]
            if payload["diagram"]["points"][name] is None
        ]
        reason = (
            None
            if value is not None
            else (
                "Missing visible inputs: " + ", ".join(missing)
                if missing
                else "The saved reference points do not define a finite measurement."
            )
        )
        result.append(
            Observation(
                definition_key=definition["key"],
                definition_version=payload["contract"],
                label=definition["label"],
                definition=definition["definition"]
                + (" " + origin.description() if origin else ""),
                formula=definition["formula"],
                method=payload["method"],
                unit=definition["unit"],
                value=value,
                status="user_reviewed" if value is not None else "unavailable",
                revision=payload["revision"],
                withheld_reason=reason,
                sources=(_source(raw, payload, definition),),
                score_eligible=False,
                score_exclusion_reason="Reviewed 2D points do not replace the legacy composite's original instrument inputs."
                + (
                    " Started from model proposals; unchanged positions are not independent verification."
                    if origin
                    else ""
                ),
            )
        )
    return tuple(result)


def reviewed_observations(
    frontal: tuple[dict[str, Any], ...] = (),
    profiles: tuple[dict[str, Any], ...] = (),
) -> tuple[Observation, ...]:
    """Retain every definition and unavailable card; never borrow automatic norms."""
    result = tuple(
        record for raw in frontal for record in _observations(raw, "reviewed_frontal")
    )
    result += tuple(
        record for raw in profiles for record in _observations(raw, "reviewed_profile")
    )
    public_observations(result)  # Validate source aliases/frames across both views.
    return result
