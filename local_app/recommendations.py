"""Small, deterministic editorial catalogue. No causal improvement predictions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Any

VERSION = "styling-catalogue-v2"
REVIEWED_AT = "2026-10-18"
SOURCES = {
    "editorial-experiment": {
        "title": "Local editorial comparison protocol",
        "url": None,
        "kind": "styling convention or experiment",
        "scope": "Compare reversible options under similar lighting; no established improvement effect.",
    },
    "loreal-volume": {
        "title": "L’Oréal Paris: hairstyles and styling for round faces",
        "url": "https://www.lorealparisusa.com/beauty-magazine/hair-style/hairstyle-trends/hairstyles-for-round-faces",
        "kind": "vendor styling guide",
        "scope": "Describes crown/side volume and texture as styling choices; not a controlled intervention study.",
    },
    "loreal-part": {
        "title": "L’Oréal Paris: parts and hair type",
        "url": "https://www.lorealparisusa.com/beauty-magazine/hair-care/all-hair-types/best-part-for-face-shape-and-hair-type",
        "kind": "vendor styling guide",
        "scope": "Describes alternate parts and hair-type considerations; does not validate metric thresholds.",
    },
    "rayban-shape": {
        "title": "Ray-Ban: frame style guide",
        "url": "https://india.ray-ban.com/face-guide",
        "kind": "vendor styling guide",
        "scope": "Describes contrasting frame silhouettes as a styling convention; does not establish optical fit or a jaw-ratio rule.",
    },
}
LABELS = {
    "facial_index": "Mesh facial height / cheek width",
    "forehead_ratio": "Mesh forehead width / cheek width",
    "jaw_ratio": "Lower-jaw landmark width / cheek width",
    "eye_spacing": "Inner-eye distance / outer-eye span",
    "profile_lower_face_ratio": "Reviewed profile lower-face chord / face height",
    "profile_nasal_length_ratio": "Reviewed profile nasal chord / face height",
}


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    slot: str
    required: tuple[str, ...]
    criterion: str
    goal: str
    title: str
    action: str
    alternative: str
    effect: str
    sources: tuple[str, ...] = ("editorial-experiment",)
    maintenance: str = "low"
    cost: str = "none"
    priority: int = 1
    version: str = VERSION


RULES = (
    Rule(
        "hair.volume.width",
        "hair",
        "hair_volume",
        ("facial_index",),
        "tall",
        "balance",
        "Hair framing experiment",
        "Compare more width at the sides with less height on top, using your current hair length.",
        "Keep your current silhouette and compare a different part instead.",
        "side_width",
        ("editorial-experiment", "loreal-volume"),
    ),
    Rule(
        "hair.volume.height",
        "hair",
        "hair_volume",
        ("facial_index",),
        "short",
        "balance",
        "Hair framing experiment",
        "Compare a little height on top with less width at the sides, using your current hair length.",
        "Keep your current volume and compare a different part instead.",
        "crown_height",
        ("loreal-volume",),
    ),
    Rule(
        "hair.volume.vertical",
        "hair",
        "hair_volume",
        ("facial_index",),
        "tall",
        "emphasize",
        "Vertical silhouette experiment",
        "Compare extra height on top to emphasize the vertical outline.",
        "Keep the current silhouette if additional volume is impractical.",
        "crown_height",
    ),
    Rule(
        "hair.volume.horizontal",
        "hair",
        "hair_volume",
        ("facial_index",),
        "short",
        "emphasize",
        "Horizontal silhouette experiment",
        "Compare a fuller side silhouette to emphasize the horizontal outline.",
        "Keep the current silhouette and change only your part.",
        "side_width",
    ),
    Rule(
        "hair.part.compare",
        "hair",
        "hair_volume",
        ("facial_index",),
        "middle",
        "either",
        "Part comparison",
        "Compare a side part and a centre part at the same volume. Keep both options open near a selection boundary.",
        "Keep your natural part if your hair does not hold either option comfortably.",
        "compare_parts",
        ("loreal-part",),
    ),
    Rule(
        "hair.raw.compare",
        "hair",
        "hair_volume",
        ("facial_index",),
        "unranked",
        "either",
        "Hair framing comparison",
        "Compare your current part with a side or centre part while keeping volume unchanged.",
        "Keep your current part if either alternative is uncomfortable.",
        "compare_parts",
        ("editorial-experiment", "loreal-part"),
    ),
    Rule(
        "hair.fringe.consult",
        "hair",
        "hair_frame",
        ("facial_index", "forehead_ratio"),
        "forehead",
        "balance",
        "Fringe and open-forehead comparison",
        "With a stylist, compare a temporary forward-framing arrangement against an open forehead before considering a cut.",
        "Use your current length and compare a side part without cutting.",
        "forehead_frame",
        ("editorial-experiment", "loreal-part"),
        "flexible",
        "appointment",
        2,
    ),
    Rule(
        "facial_hair.outline.balance",
        "facial_hair",
        "facial_hair_outline",
        ("jaw_ratio",),
        "any",
        "balance",
        "Facial-hair outline comparison",
        "If you already have facial hair, compare its current outline with a tidier side contour using a temporary sketch before trimming.",
        "Keep the current outline; do not grow or remove facial hair solely for a ratio.",
        "compare_outline",
    ),
    Rule(
        "facial_hair.outline.emphasize",
        "facial_hair",
        "facial_hair_outline",
        ("jaw_ratio",),
        "any",
        "emphasize",
        "Keep the jaw outline visible",
        "If you already have facial hair, compare a contour that follows your current jaw outline with the present style in matched photographs.",
        "Keep the current style if contouring adds unwanted upkeep.",
        "follow_outline",
    ),
    Rule(
        "facial_hair.profile.compare",
        "facial_hair",
        "facial_hair_outline",
        ("profile_lower_face_ratio", "profile_nasal_length_ratio"),
        "any",
        "either",
        "Profile framing comparison",
        "If you already have facial hair, compare its current side outline with a temporary sketch that keeps the nose, lips and chin contour visible. Use the same lateral view for both options.",
        "Keep your current outline; these photo-space ratios do not establish an ideal beard or chin position.",
        "compare_profile_outline",
    ),
    Rule(
        "eyewear.silhouette",
        "eyewear",
        "eyewear",
        ("eye_spacing", "jaw_ratio"),
        "jaw_distinct",
        "either",
        "Frame silhouette comparison",
        "Compare rounded and angular frames at similar overall width. Choose the outline you prefer, then check bridge and temple comfort in person.",
        "Keep your current frames and compare only frame colour or weight at a free try-on.",
        "compare_frames",
        ("rayban-shape",),
        priority=1,
    ),
    Rule(
        "eyewear.fit",
        "eyewear",
        "eyewear",
        ("eye_spacing",),
        "any",
        "either",
        "Eyewear try-on",
        "Compare two frame widths and check pupil position, bridge comfort and temple fit in person. A facial ratio cannot select a frame size.",
        "Keep comfortable existing frames; a new pair is optional.",
        "compare_fit",
        priority=3,
    ),
)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _observations(rows: list[dict]) -> dict[str, dict]:
    counts = Counter(
        r.get("id")
        for r in rows
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    )
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("id")
        if not isinstance(key, str) or key not in LABELS or counts[key] != 1:
            continue
        value, rank = row.get("value"), row.get("percentile")
        if (
            not _finite(value)
            or value <= 0
            or (rank is not None and (not _finite(rank) or not 0 <= rank <= 100))
        ):
            continue
        if row.get("unit", "ratio") != "ratio":
            continue
        method = row.get("method", "automatic-regional-v1")
        if (
            not isinstance(method, str)
            or method
            not in {
                "automatic-regional-v1",
                "automatic-regional-v2",
                "reviewed-frontal-v1",
                "manual-profile-v2",
            }
            or (
                method in {"reviewed-frontal-v1", "manual-profile-v2"}
                and rank is not None
            )
        ):
            continue
        if key.startswith("profile_") != (method == "manual-profile-v2"):
            continue
        result[key] = dict(
            id=key,
            label=LABELS[key],
            value=float(value),
            percentile=float(rank) if rank is not None else None,
            method=method,
            facing=row.get("facing")
            if row.get("facing") in ("left", "right")
            else None,
            photo_numbers=[
                v
                for v in row.get("photo_numbers", [])
                if type(v) is int and 1 <= v <= 1000000
            ],
        )
    return result


def _matches(criterion: str, obs: dict[str, dict]) -> bool:
    face = obs.get("facial_index", {}).get("percentile")
    if criterion == "tall":
        return face is not None and face > 78
    if criterion == "short":
        return face is not None and face < 22
    if criterion == "middle":
        return face is not None and 22 <= face <= 78
    if criterion == "unranked":
        return face is None
    if criterion == "forehead":
        rank = obs["forehead_ratio"]["percentile"]
        return rank is not None and rank > 78
    if criterion == "jaw_distinct":
        rank = obs["jaw_ratio"]["percentile"]
        return rank is not None and (rank < 22 or rank > 78)
    return criterion == "any"


def select_recommendations(
    rows: list[dict], settings: dict[str, str], *, dismissed: list[str] | None = None
) -> dict:
    """Call with validated preferences; no user text is a rule or a source."""
    observed = _observations(rows)
    selected, excluded = [], []
    occupied = set()
    for rule in sorted(
        RULES,
        key=lambda r: (
            r.priority,
            ("hair", "facial_hair", "eyewear").index(r.category),
            -len(r.required),
            RULES.index(r),
        ),
    ):
        reason = None
        if rule.id in (dismissed or []):
            reason = "user_dismissed"
        elif settings["goal"] == "understand":
            reason = "understanding_only"
        elif settings[rule.category] != "yes":
            reason = "category_opt_out"
        elif (
            rule.category == "hair"
            and settings["preserve"] == "outline"
            or rule.category == "eyewear"
            and settings["preserve"] == "eyes"
            or rule.category == "facial_hair"
            and settings["preserve"] == "jaw"
        ):
            reason = "preserved_region"
        elif rule.goal not in ("either", settings["goal"]):
            reason = "different_goal"
        elif any(key not in observed for key in rule.required):
            reason = "missing_compatible_observation"
        elif not _matches(rule.criterion, observed):
            reason = "different_observation_band"
        elif rule.cost != "none" and settings["budget"] == "no_purchase":
            reason = "budget_constraint"
        elif rule.maintenance == "flexible" and settings["maintenance"] == "low":
            reason = "maintenance_constraint"
        elif rule.id == "hair.fringe.consult" and (
            settings["hair_change"] == "keep"
            or settings["hair_length"] not in ("medium", "long")
            or settings["hair_texture"] == "unspecified"
        ):
            reason = "hair_options_unconfirmed"
        elif rule.slot in occupied:
            reason = "alternative_in_same_region"
        elif len(selected) >= 5:
            reason = "shortlist_limit"
        if reason:
            excluded.append(dict(id=rule.id, reason=reason))
            continue
        facts = [observed[k].copy() for k in rule.required]
        basis = "; ".join(
            f"{o['label']} is {o['value']:.3f}"
            + (
                f" (reference percentile {o['percentile']:.0f})"
                if o["percentile"] is not None
                else " (reference comparison unavailable)"
            )
            for o in facts
        )
        reviewed_photos = sorted({n for o in facts for n in o["photo_numbers"]})
        if reviewed_photos:
            basis += ". Reviewed-point median from " + ", ".join(
                f"Photo {n}" for n in reviewed_photos
            )
        profile_facing = {o["facing"] for o in facts if o.get("facing")}
        if profile_facing:
            basis += (
                ". Nose points "
                + "/".join(sorted(profile_facing))
                + " in the reviewed image; no opposite-view averaging"
            )
        basis += f". Your goal is {settings['goal']}."
        if rule.category == "hair":
            basis += f" Hair texture: {settings['hair_texture']}; length: {settings['hair_length']} (your selections)."
        action = rule.action
        if rule.category == "eyewear" and settings.get("style") == "subtle":
            action += " Your preference is subtle framing: keep one comparison light or close to your current frame."
        elif rule.category == "eyewear" and settings.get("style") == "defined":
            action += " Your preference is defined framing: include one bolder outline in the comparison."
        if rule.category == "eyewear" and settings.get("eyewear_use") == "prescription":
            action += " Keep your prescription requirements and have an optician check any proposed fit."
        selected.append(
            dict(
                id=rule.id,
                version=rule.version,
                category=rule.category,
                priority=len(selected) + 1,
                title=rule.title,
                action=action,
                alternative=rule.alternative,
                basis=basis,
                observations=facts,
                intended_effect=rule.effect,
                cost=rule.cost,
                maintenance=rule.maintenance,
                evidence="Styling convention or experiment, not a tested improvement prediction. These ratios do not establish a face-shape category, hair suitability or optical fit.",
                sources=[dict(id=s, **SOURCES[s]) for s in rule.sources],
                reviewed_at=REVIEWED_AT,
            )
        )
        occupied.add(rule.slot)
    return dict(
        version=VERSION,
        suggestions=selected,
        excluded=excluded,
        selection_note="Ordinal priorities reflect relevance and feasibility, not predicted benefit. The 22–28 and 72–78 bands include a three-percentile editorial buffer around quarter boundaries; this is not measured uncertainty.",
    )


def recommendation_rows(
    rows: list[dict],
    frontal: list[dict] | None = None,
    profile: dict | list[dict] | None = None,
) -> list[dict]:
    """Prefer explicitly reviewed raw definitions, without reusing automatic ranks."""
    from statistics import median

    from .landmark_review import safe_review

    grouped = {}
    for raw in frontal or []:
        review = safe_review(raw)
        if review is None:
            continue
        for row in review["measurements"]:
            key = row["id"].removeprefix("reviewed_")
            if row["id"].startswith("reviewed_") and key in LABELS:
                grouped.setdefault(key, []).append(
                    (row["value"], review.get("photo_number"))
                )
    result = [dict(r) for r in rows if r.get("id") not in grouped]
    for key, values in grouped.items():
        result.append(
            dict(
                id=key,
                value=median(v for v, _ in values),
                unit="ratio",
                percentile=None,
                method="reviewed-frontal-v1",
                photo_numbers=sorted({p for _, p in values if p is not None}),
            )
        )
    from .profile_v2 import safe_profile

    for raw in profile if isinstance(profile, list) else [profile]:
        view = safe_profile(raw)
        if view is None:
            continue
        values = {r["id"]: r for r in view["measurements"]}
        if not all(k in values for k in ("lower_face_ratio", "nasal_length_ratio")):
            continue
        for key in ("lower_face_ratio", "nasal_length_ratio"):
            result.append(
                dict(
                    id="profile_" + key,
                    value=values[key]["value"],
                    unit="ratio",
                    percentile=None,
                    method="manual-profile-v2",
                    facing=view["facing"],
                    photo_numbers=[view["photo_number"]]
                    if "photo_number" in view
                    else [],
                )
            )
        break
    return result
