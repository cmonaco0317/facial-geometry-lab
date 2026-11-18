"""Current-source interpretation order, without an attractiveness or benefit rank."""

from __future__ import annotations

import hashlib
import html
import json

from .observations import Observation, public_observations
from .reading_registry import current_reading_registry

VERSION = "consultation-findings-v1"
GOAL_REGIONS = {
    "understand": (
        "proportions",
        "eyes",
        "jaw",
        "nose",
        "lips",
        "forehead",
        "profile",
        "frontal",
    ),
    "balance": (
        "proportions",
        "jaw",
        "eyes",
        "forehead",
        "nose",
        "lips",
        "profile",
        "frontal",
    ),
    "emphasize": (
        "eyes",
        "jaw",
        "proportions",
        "nose",
        "lips",
        "forehead",
        "profile",
        "frontal",
    ),
}
SELECTION_POLICY = (
    "Priorities organize current, traceable evidence for your selected goal. "
    "A current reviewed construction takes precedence over its automatic counterpart, "
    "including when the review leaves a required point missing. Multiple reviewed photos "
    "stay separate; one available construction from the lowest photo ordinal is selected. "
    "Other method/source alternatives remain in Measurement evidence. Order reflects "
    "inspectability and editorial relevance, not anatomical confidence, quality or predicted benefit."
)


def evidence_anchor(record: dict) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "evidence-" + hashlib.sha256(encoded.encode()).hexdigest()[:24]


def current_rows(records: list[dict]) -> list[dict]:
    """Select interpretations without overwriting or merging original instruments."""
    validated = tuple(Observation.model_validate_json(json.dumps(r)) for r in records)
    if any(not r.geometry_retained for r in validated):
        raise ValueError(
            "Remeasure archived numerical records before current interpretation."
        )
    public = public_observations(validated)
    definitions = {d["key"]: d for d in current_reading_registry()["readings"]}
    groups: dict[str, list[tuple[dict, dict]]] = {}
    for record in public:
        definition = definitions.get(record["definition_key"])
        if definition is None:
            continue
        family = definition["family"]
        key = definition["id"]
        if family == "reviewed_frontal":
            key = key.removeprefix("reviewed_")
        if family not in {"automatic", "reviewed_frontal", "iris"}:
            key = definition["key"]
        groups.setdefault(key, []).append((record, definition))
    result = []
    for key, candidates in groups.items():
        reviewed = [
            pair
            for pair in candidates
            if pair[1]["family"] == "reviewed_frontal" and pair[0]["sources"]
        ]
        pool = (
            reviewed
            or [pair for pair in candidates if pair[1]["family"] != "reviewed_frontal"]
            or candidates
        )
        record, definition = min(
            pool,
            key=lambda pair: (
                pair[0]["value"] is None,
                min((s["photo_number"] for s in pair[0]["sources"]), default=10001),
                pair[0]["definition_key"],
            ),
        )
        result.append(
            dict(
                id=key,
                label=record["label"],
                value=record["value"],
                unit=record["unit"],
                percentile=(
                    record["reference"]["percentile"] if record["reference"] else None
                ),
                method=record["method"],
                status=record["status"],
                definition=record["definition"],
                definition_key=record["definition_key"],
                definition_version=record["definition_version"],
                revision=record["revision"],
                withheld_reason=record["withheld_reason"],
                region=definition.get("region") or "profile",
                photo_numbers=sorted(s["photo_number"] for s in record["sources"]),
                evidence_refs=[evidence_anchor(record)],
                current_observation=record,
                note="Current source selection; method alternatives are retained separately.",
                source_selection="current-construction",
            )
        )
    return result


def priorities(
    rows: list[dict], settings: dict[str, str], suggestions: list[dict] | None = None
) -> list[dict]:
    order = GOAL_REGIONS[settings["goal"]]

    def linked_suggestion(row: dict) -> dict | None:
        return next(
            (
                s
                for s in suggestions or []
                if set(s.get("evidence_refs", ())) & set(row["evidence_refs"])
            ),
            None,
        )

    eligible = [
        row
        for row in rows
        if row["value"] is not None
        and row["current_observation"]["sources"]
        and not any(
            s["visibility"] in {"occluded", "uncertain"}
            for s in row["current_observation"]["sources"]
        )
    ]
    status_order = {"user_reviewed": 0, "automatic": 1, "proxy": 2, "estimated": 3}
    eligible.sort(
        key=lambda r: (
            status_order[r["status"]],
            order.index(r["region"]) if r["region"] in order else len(order),
            linked_suggestion(r) is None,
            rows.index(r),
        )
    )
    selected = []
    regions = set()
    for row in eligible:
        if row["region"] in regions:
            continue
        record = row["current_observation"]
        source_text = ", ".join(
            f'Photo {s["photo_number"]}, {s["view"]}, review {s["review_revision"]}'
            for s in record["sources"]
        )
        suggestion = linked_suggestion(row)
        selected.append(
            dict(
                priority=len(selected) + 1,
                region=row["region"],
                title=row["label"],
                statement=f'{row["label"]}: {row["value"]:.{record["display_decimals"]}f} {row["unit"]}. '
                f'{row["status"].replace("_","-")} via {row["method"]}. {source_text}.',
                reason=f'Your goal is {settings["goal"]}. This is a current, inspectable {row["region"]} construction. '
                "The ordering is editorial relevance and source traceability, not an anatomical quality ranking.",
                next_step=(
                    suggestion["action"]
                    if suggestion
                    else "Inspect the linked endpoints and definition against your photograph. Correct a misplaced point or leave a hidden input unavailable."
                ),
                alternative=(
                    suggestion["alternative"]
                    if suggestion
                    else "Keep the current appearance and save the report; editing or changing a feature is optional."
                ),
                suggestion_id=suggestion["id"] if suggestion else None,
                definition_key=row["definition_key"],
                definition_version=row["definition_version"],
                revision=row["revision"],
                method=row["method"],
                status=row["status"],
                photo_numbers=row["photo_numbers"],
                evidence_refs=sorted(
                    set(row["evidence_refs"])
                    | set(suggestion.get("evidence_refs", ()) if suggestion else ())
                ),
                observation=record,
            )
        )
        regions.add(row["region"])
        if len(selected) == 3:
            break
    return selected


def coherent_profile_rows(records: list[dict]) -> list[dict]:
    """Keep the profile experiment's two inputs in one view/version/source frame."""
    wanted = {
        "reviewed_profile:lower_face_ratio",
        "reviewed_profile:nasal_length_ratio",
    }
    groups: dict[tuple, dict] = {}
    for raw in records:
        if raw.get("definition_key") not in wanted:
            continue
        record = Observation.model_validate_json(json.dumps(raw)).to_public_dict()
        if (
            record["status"] != "user_reviewed"
            or record["method"] != "manual-profile-v2"
            or record["unit"] != "ratio"
            or record["value"] is None
            or record["value"] <= 0
            or len(record["sources"]) != 1
        ):
            continue
        source = record["sources"][0]
        if source["view"] not in {"profile_left", "profile_right"}:
            continue
        signature = (
            source["photo_number"],
            source["view"],
            record["definition_version"],
            source["revision"],
            source["review_revision"],
        )
        group = groups.setdefault(signature, {})
        key = record["definition_key"]
        group[key] = record if key not in group else None
    for signature, group in sorted(groups.items()):
        if set(group) != wanted or any(v is None for v in group.values()):
            continue
        return [
            dict(
                id="profile_" + key.split(":")[1],
                label=record["label"],
                value=record["value"],
                unit="ratio",
                percentile=None,
                method=record["method"],
                status=record["status"],
                facing=signature[1].removeprefix("profile_"),
                photo_numbers=[signature[0]],
                evidence_refs=[evidence_anchor(record)],
                source_selection="current-construction",
            )
            for key, record in group.items()
        ]
    return []


def point_map(record: dict) -> str:
    record = Observation.model_validate_json(json.dumps(record)).to_public_dict()
    if not record["sources"]:
        return ""
    source = record["sources"][0]
    width, height = source["width"], source["height"]
    scale = max(width, height)
    marks = "".join(
        f'<circle cx="{p["x"]*width}" cy="{p["y"]*height}" r="{scale*.007}" fill="#147566"/>'
        f'<text x="{min(width*.94,p["x"]*width+scale*.01)}" y="{max(scale*.025,p["y"]*height-scale*.01)}" font-size="{scale*.025}" fill="#34483f">{html.escape(p["name"])}</text>'
        for p in source["points"]
    )
    return f'<figure class="face-diagram"><svg viewBox="0 0 {width} {height}" width="280" role="img" aria-label="Current dependency point map for photo {source["photo_number"]}">{marks}</svg><figcaption>Current dependency points from Photo {source["photo_number"]}; {len(record["sources"])} source(s) contribute to this construction. A point map does not certify visible anatomy.</figcaption></figure>'


def render_priorities(payload: dict) -> str:
    esc = html.escape
    cards = []
    for item in payload.get("priorities", []):
        links = " · ".join(
            f'<a href="#{esc(ref, quote=True)}">Inspect current measurement evidence</a>'
            for ref in item["evidence_refs"]
        )
        cards.append(
            f'<article class="feature-chapter" id="priority-{item["priority"]}"><h3>{item["priority"]}. {esc(item["title"])}</h3>{point_map(item["observation"])}'
            f'<p>{esc(item["statement"])}</p><p>{esc(item["reason"])}</p><p>{links}</p><p><strong>Optional next step:</strong> {esc(item["next_step"])}</p>'
            f'<p><strong>Alternative:</strong> {esc(item["alternative"])}</p></article>'
        )
    content = (
        "".join(cards)
        or "<p>No current source-backed construction supports a priority. The available report and missing-input cards remain usable.</p>"
    )
    return f'<section id="consultation-priorities"><h2>Your current priorities</h2><p>{esc(SELECTION_POLICY)}</p>{content}</section>'
