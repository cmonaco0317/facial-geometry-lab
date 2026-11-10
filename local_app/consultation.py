"""Observation-bound report chapters and explicitly optional styling experiments."""

from __future__ import annotations

import html
from math import isfinite
from typing import Any

PREFERENCE_OPTIONS = {
    "goal": {
        "understand": "Understand my proportions",
        "balance": "Explore visual balance",
        "emphasize": "Emphasize my existing features",
    },
    "hair": {"no": "No hair suggestions", "yes": "Include hair styling"},
    "eyewear": {"no": "No eyewear suggestions", "yes": "Include eyewear styling"},
    "facial_hair": {
        "no": "No facial-hair suggestions",
        "yes": "Include existing facial-hair styling",
    },
    "preserve": {
        "none": "No preserved region selected",
        "outline": "Keep my hair outline",
        "eyes": "Keep my eye framing",
        "jaw": "Keep my jaw framing",
    },
    "hair_length": {
        "unspecified": "Length not specified",
        "short": "Short",
        "medium": "Medium",
        "long": "Long",
    },
    "hair_texture": {
        "unspecified": "Texture not specified",
        "straight": "Straight",
        "wavy": "Wavy",
        "curly": "Curly",
        "coily": "Coily",
    },
    "hair_change": {
        "keep": "Keep my current length",
        "open": "Open to a length change",
    },
    "maintenance": {"low": "Low maintenance", "flexible": "Flexible maintenance"},
    "style": {
        "compare": "Compare different styles",
        "subtle": "Subtle framing",
        "defined": "Defined framing",
    },
    "eyewear_use": {
        "unspecified": "Eyewear use not specified",
        "fashion": "Styling only",
        "prescription": "Prescription eyewear",
    },
    "budget": {
        "no_purchase": "Try without a purchase",
        "flexible": "Open to an appointment or purchase",
    },
}
DEFAULT_PREFERENCES = {
    "goal": "understand",
    "hair": "no",
    "eyewear": "no",
    "facial_hair": "no",
    "preserve": "none",
    "hair_length": "unspecified",
    "hair_texture": "unspecified",
    "hair_change": "keep",
    "maintenance": "low",
    "budget": "no_purchase",
    "style": "compare",
    "eyewear_use": "unspecified",
}
# Descriptions refer to the existing catalogue's exact image-space definitions.
DEFINITIONS = {
    "facial_index": "Mesh forehead-to-chin height divided by cheek width. The top point is not the hairline.",
    "fwhr": "Cheek width divided by the vertical distance from the medial brows to the upper lip.",
    "upper_third": "Mesh forehead-to-medial-brow distance as a share of the three measured vertical segments.",
    "middle_third": "Medial-brow-to-nose-base distance as a share of the three measured vertical segments.",
    "lower_third": "Nose-base-to-chin distance as a share of the three measured vertical segments.",
    "middle_lower_ratio": "Medial-brow-to-nose-base distance divided by nose-base-to-chin distance.",
    "forehead_ratio": "Mesh forehead width divided by cheek width.",
    "jaw_ratio": "Lower-jaw landmark width divided by cheek width.",
    "chin_jaw_ratio": "Chin landmark width divided by lower-jaw landmark width.",
    "eye_spacing": "Inner-eye-corner distance divided by full outer-eye-corner span.",
    "eye_almondness": "Average eye-opening height divided by average eye-opening width.",
    "canthal_tilt": "Mean outer-corner elevation relative to the inner corner in image coordinates; head roll affects it.",
    "brow_position": "Average brow-peak elevation above eye centres divided by mesh forehead-to-chin height.",
    "iris_spacing_ratio": "Euclidean distance between model-detected iris centres divided by the outer-eye-corner span. No matched reference percentile; separate from the older eyelid-centre proxy.",
    "iris_brow_elevation": "Mean vertical elevation from model-detected iris centres to brow peaks divided by the outer-eye span. This avoids eyelid-centre and forehead-height normalization proxies; pose and detector placement still affect it.",
    "alar_ratio": "Nose-wing width divided by cheek width.",
    "naso_oral": "Nose-wing width divided by mouth-corner width.",
    "mouth_ratio": "Mouth-corner width divided by cheek width.",
    "lip_lower_face": "Upper-to-lower lip height divided by nose-base-to-chin distance.",
    "philtrum_ratio": "Nose-base-to-upper-lip distance divided by nose-base-to-chin distance.",
}
CHAPTERS = (
    (
        "proportions",
        "Overall proportions",
        (
            "facial_index",
            "fwhr",
            "upper_third",
            "middle_third",
            "lower_third",
            "middle_lower_ratio",
        ),
        "Vertical and horizontal relationships describe shape; an equal-thirds drawing is not a universal target.",
    ),
    (
        "forehead",
        "Forehead and frame",
        ("forehead_ratio",),
        "The mesh measures the visible forehead boundary. Hairline height, recession and scalp coverage are not measured.",
    ),
    (
        "eyes",
        "Eyes and brows",
        ("eye_spacing", "eye_almondness", "canthal_tilt", "brow_position",
         "iris_spacing_ratio", "iris_brow_elevation"),
        "Expression, eyelid movement and head roll can change these readings. They do not measure eyesight or personality.",
    ),
    (
        "nose",
        "Nose proportions",
        ("alar_ratio", "naso_oral"),
        "Frontal widths describe the visible nose. Projection and the lateral contour need a profile photograph.",
    ),
    (
        "lips",
        "Lips and philtrum",
        ("mouth_ratio", "lip_lower_face", "philtrum_ratio"),
        "These are visible soft-tissue proportions. Keep the lips relaxed; smiling or pursing changes the measurements.",
    ),
    (
        "jaw",
        "Cheeks, jaw and chin",
        ("jaw_ratio", "chin_jaw_ratio"),
        "Surface landmarks describe width relationships. They do not establish bone position, bite or body-fat percentage.",
    ),
)


def preferences(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) - set(PREFERENCE_OPTIONS):
        raise ValueError("Unsupported report preference field.")
    result = {**DEFAULT_PREFERENCES, **payload}
    for key, value in result.items():
        if not isinstance(value, str) or value not in PREFERENCE_OPTIONS[key]:
            raise ValueError("Unsupported report preference selection.")
    return result


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def build_chapters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {
        row["id"]: row
        for row in rows
        if row.get("id") in DEFINITIONS and _finite(row.get("value"))
    }
    chapters = []
    from .anatomical_observations import PROXY_LABELS, IRIS_LABELS

    for key, title, ids, note in CHAPTERS:
        measures = [
            {**by_id[i], "definition": DEFINITIONS[i]} for i in ids if i in by_id
        ]
        observations = []
        for row in measures:
            rank = row.get("percentile")
            if _finite(rank) and 0 <= rank <= 100:
                placement = (
                    "lower quarter"
                    if rank < 25
                    else "upper quarter"
                    if rank > 75
                    else "middle half"
                )
                observations.append(
                    f"{row.get('label', row['id'])} falls in the {placement} of the comparison distribution ({rank:.0f}th percentile)."
                )
        if not measures:
            observations = [
                "No usable measurement for this region in the current photographs."
            ]
        elif not observations:
            observations = [
                "Observed values are available; a matching reference comparison is not available for these rows."
            ]
        chapters.append(
            dict(
                id=key,
                title=title,
                measurements=measures,
                unavailable_measurements=[dict(
                    id=i, label=PROXY_LABELS.get(i, IRIS_LABELS.get(i, i.replace("_", " "))),
                    definition=DEFINITIONS[i],
                ) for i in ids if i not in by_id],
                observations=observations,
                note=note,
            )
        )
    return chapters


def styling_suggestions(
    rows: list[dict[str, Any]], settings: dict[str, str]
) -> list[dict[str, str]]:
    from .recommendations import select_recommendations

    return select_recommendations(rows, preferences(settings))["suggestions"]


def consultation_payload(
    rows: list[dict[str, Any]],
    settings: dict[str, str] | None = None,
    *,
    frontal: list[dict] | None = None,
    ideas: dict | None = None,
    profile: dict | list[dict] | None = None,
) -> dict[str, Any]:
    from .idea_choices import normalize
    from .recommendations import RULES, recommendation_rows, select_recommendations

    selected = preferences(settings or {})
    choices = normalize(ideas)
    evidence = recommendation_rows(rows, frontal, profile)
    recommendations = select_recommendations(
        evidence, selected, dismissed=choices["dismissed"]
    )
    applicable = {
        r["id"] for r in select_recommendations(evidence, selected)["suggestions"]
    }
    by_id = {r.id: r for r in RULES}
    return dict(
        version="consultation-v3",
        preferences=selected,
        idea_choices=choices,
        saved_ideas=[
            dict(id=key, title=by_id[key].title, currently_applicable=key in applicable)
            for key in choices["saved"]
        ],
        dismissed_ideas=[
            dict(id=key, title=by_id[key].title) for key in choices["dismissed"]
        ],
        chapters=build_chapters(rows),
        suggestions=recommendations["suggestions"],
        recommendation_version=recommendations["version"],
        exclusions=recommendations["excluded"],
        selection_note=recommendations["selection_note"],
    )


def preference_form(settings: dict[str, str], action: str, csrf: str) -> str:
    settings = preferences(settings)
    controls = ""
    optional = ""
    labels = {
        "goal": "Your goal",
        "hair": "Hair",
        "eyewear": "Eyewear",
        "facial_hair": "Facial hair",
        "preserve": "Preserve",
        "hair_length": "Hair length",
        "hair_texture": "Hair texture",
        "hair_change": "Length changes",
        "maintenance": "Maintenance",
        "budget": "Budget",
        "style": "Preferred style",
        "eyewear_use": "Eyewear use",
    }
    for key, options in PREFERENCE_OPTIONS.items():
        options_html = "".join(
            f'<option value="{value}"{" selected" if settings[key] == value else ""}>{html.escape(label)}</option>'
            for value, label in options.items()
        )
        control = (
            f'<label>{labels[key]}<select name="{key}">{options_html}</select></label>'
        )
        if key in {"goal", "hair", "eyewear"}:
            controls += control
        else:
            optional += control
    return f'<form method="post" action="{html.escape(action, quote=True)}" class="preferences-form"><input type="hidden" name="csrf_token" value="{html.escape(csrf, quote=True)}">{controls}<details class="optional-preferences"><summary>Optional styling preferences</summary><div class="preferences-extra">{optional}</div></details><button type="submit">Update report preferences</button></form>'


def render_consultation(
    payload: dict[str, Any],
    diagrams: dict[str, str] | None = None,
    *,
    action: str | None = None,
    csrf: str | None = None,
) -> str:
    esc = html.escape
    diagrams = diagrams or {}
    nav = "".join(
        f'<a href="#region-{c["id"]}">{esc(c["title"])}</a>'
        for c in payload["chapters"]
    )
    sections = []
    for chapter in payload["chapters"]:
        rows = "".join(
            f"<tr><th>{esc(str(r['label']))}</th><td>{r['value']:.3f} {esc(r['unit'])}"
            + (f" · {r['evidence_photo_count']} photo(s)"
               if type(r.get('evidence_photo_count')) is int
               and 0 < r['evidence_photo_count'] <= 10000 else "")
            + f"</td><td>{esc(r['definition'])}</td></tr>"
            for r in chapter["measurements"]
        )
        table = (
            f'<div class="table-wrap"><table><thead><tr><th>Measurement</th><th>Reading</th><th>Definition</th></tr></thead><tbody>{rows}</tbody></table></div>'
            if rows
            else ""
        )
        obs = "".join(f"<li>{esc(o)}</li>" for o in chapter["observations"])
        unavailable = "".join(
            f'<li><strong>{esc(r["label"])}</strong>: Not available from the current '
            f'visible/detected geometry. {esc(r["definition"])}</li>'
            for r in chapter.get("unavailable_measurements", [])
        )
        sections.append(
            f'<section class="feature-chapter" id="region-{chapter["id"]}"><h2>{esc(chapter["title"])}</h2>{diagrams.get(chapter["id"], "")}<p>{esc(chapter["note"])}</p>{table}<ul>{obs}</ul><ul>{unavailable}</ul></section>'
        )
    from .recommendations import SOURCES

    def source_item(source: dict) -> str:
        title = esc(source["title"])
        if source["url"]:
            title = f'<a href="{esc(source["url"], quote=True)}" rel="noreferrer noopener">{title}</a>'
        return f"<li>{title} — {esc(source['kind'])}. {esc(source['scope'])}</li>"

    choices = payload.get("idea_choices", {"revision": 1, "saved": [], "dismissed": []})

    def control(idea: str, verb: str, label: str) -> str:
        if action is None or csrf is None:
            return ""
        return f'<form method="post" action="{esc(action, quote=True)}" class="idea-action"><input type="hidden" name="csrf_token" value="{esc(csrf, quote=True)}"><input type="hidden" name="revision" value="{choices["revision"]}"><input type="hidden" name="idea" value="{esc(idea, quote=True)}"><button name="action" value="{verb}">{label}</button></form>'

    suggestions = ""
    for s in payload["suggestions"]:
        source_rows = "".join(
            source_item(SOURCES[row["id"]])
            for row in s.get("sources", [])
            if row.get("id") in SOURCES
        )
        suggestions += (
            f"<article><h3>{esc(s['title'])}</h3><p>{esc(s['action'])}</p>"
            f"<p><strong>Alternative:</strong> {esc(s.get('alternative', 'Keep your current style.'))}</p>"
            f"<details><summary>Why this was selected</summary><p>{esc(s['basis'])}</p>"
            f"<p>Cost: {esc(s.get('cost', 'none'))}. Maintenance: {esc(s.get('maintenance', 'low'))}.</p>"
            f"<ul>{source_rows}</ul></details><small>{esc(s['evidence'])}</small>"
            + control(
                s["id"],
                "unsave" if s["id"] in choices["saved"] else "save",
                "Unsave idea" if s["id"] in choices["saved"] else "Save idea",
            )
            + control(s["id"], "dismiss", "Dismiss idea")
            + "</article>"
        )
    if not suggestions:
        suggestions = "<p>No styling suggestion is selected or supported by the available measurements. Choose a goal and relevant styling categories to explore options.</p>"
    saved = "".join(
        f"<li>{esc(r['title'])} — {'currently selected' if r['currently_applicable'] else 'not selected under current preferences/evidence'}{control(r['id'], 'unsave', 'Unsave idea')}</li>"
        for r in payload.get("saved_ideas", [])
    )
    dismissed = "".join(
        f"<li>{esc(r['title'])}{control(r['id'], 'restore', 'Restore idea')}</li>"
        for r in payload.get("dismissed_ideas", [])
    )
    collections = (f"<h3>Saved ideas</h3><ul>{saved}</ul>" if saved else "") + (
        f"<details><summary>Dismissed ideas</summary><ul>{dismissed}</ul></details>"
        if dismissed
        else ""
    )
    goal = PREFERENCE_OPTIONS["goal"][payload["preferences"]["goal"]]
    return f'<p>Regional percentiles describe relative size or shape, not quality.</p><nav class="chapter-nav" aria-label="Report chapters">{nav}<a href="#profile-chapter">Profile</a><a href="#style-chapter">Styling</a></nav>{"".join(sections)}<section class="feature-chapter" id="style-chapter"><h2>Styling experiments</h2><p>Goal: {esc(goal)}. These are reversible choices to try, with no predicted score increase.</p>{suggestions}{collections}<p class="muted">{esc(payload.get("selection_note", ""))}</p></section>'
