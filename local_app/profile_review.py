"""Explicit manual soft-tissue profile measurements; no radiographic inference."""

from __future__ import annotations

import html
import math
from typing import Any

ANCHORS = {
    "nasion": "Nasion — deepest bridge point between forehead and nose",
    "tip": "Nose tip — most forward visible point",
    "subnasale": "Subnasale — junction of nose base and upper lip",
    "upper_lip": "Upper lip — most forward vermilion point",
    "lower_lip": "Lower lip — most forward vermilion point",
    "pogonion": "Pogonion — most forward soft-tissue chin point",
    "menton": "Menton — lowest soft-tissue chin point",
}


PROFILE_ROWS = [
    {
        "id": "convexity",
        "label": "Soft-tissue profile angle",
        "unit": "deg",
        "definition": "Internal angle at the nose base between bridge and forward chin (N–Sn–Pog).",
    },
    {
        "id": "total_convexity",
        "label": "Profile angle including nose",
        "unit": "deg",
        "definition": "Internal angle at the nose tip between bridge and forward chin (N–Prn–Pog).",
    },
    {
        "id": "upper_eline",
        "label": "Upper-lip offset from nose–chin line",
        "unit": "% N–Me",
        "definition": "Signed perpendicular distance from upper lip to tip–chin line, divided by bridge-to-bottom-chin distance. Positive is forward.",
    },
    {
        "id": "lower_eline",
        "label": "Lower-lip offset from nose–chin line",
        "unit": "% N–Me",
        "definition": "Signed perpendicular distance from lower lip to tip–chin line, divided by bridge-to-bottom-chin distance. Positive is forward.",
    },
]
PROFILE_NOTE = "Two-dimensional soft-tissue readings from user-placed landmarks on one lateral photograph. Placement, head posture and perspective affect them. No calibrated uncertainty or bone-position diagnosis. These do not change the frontal geometry score."


def validate_points(raw: Any) -> dict[str, list[float]]:
    if not isinstance(raw, dict) or set(raw) != set(ANCHORS):
        raise ValueError(
            "Place all seven profile landmarks; unknown landmarks are not accepted."
        )
    result = {}
    for name, point in raw.items():
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0 <= v <= 1
                for v in point
            )
        ):
            raise ValueError(
                "Landmarks must be finite coordinates within the photograph."
            )
        result[name] = [float(v) for v in point]
    return result


def _angle(a: list[float], b: list[float], c: list[float]) -> float:
    u, v = [a[i] - b[i] for i in (0, 1)], [c[i] - b[i] for i in (0, 1)]
    denominator = math.hypot(*u) * math.hypot(*v)
    if denominator < 1e-8:
        raise ValueError("Angle landmarks must be distinct.")
    return math.degrees(
        math.acos(max(-1, min(1, sum(x * y for x, y in zip(u, v)) / denominator)))
    )


def measure_profile(raw: Any, width: int, height: int) -> dict[str, Any]:
    points = validate_points(raw)
    if any(
        isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)
    ):
        raise ValueError("The photograph dimensions are invalid.")
    p = {k: [v[0] * width, v[1] * height] for k, v in points.items()}
    n, t, s, u, lower_lip, c, m = (p[k] for k in ANCHORS)
    scale = math.dist(n, m)
    if scale < 1 or math.dist(t, c) < 1 or abs(t[0] - n[0]) < 1:
        raise ValueError(
            "The profile landmarks do not define a usable lateral outline."
        )
    if not n[1] < s[1] < m[1]:
        raise ValueError(
            "Place the bridge above the nose base and the chin below it on a level profile."
        )
    # Positive = in front of tip-to-chin line, irrespective of left/right orientation.
    outward = 1 if t[0] > n[0] else -1

    def line_offset(q: list[float]) -> float:
        dx, dy = c[0] - t[0], c[1] - t[1]
        if abs(dy) < 1:
            raise ValueError(
                "The nose-to-chin line is too horizontal; check the landmarks."
            )
        signed = (dy * (q[0] - t[0]) - dx * (q[1] - t[1])) / math.hypot(dx, dy)
        return signed * (1 if dy > 0 else -1) * outward * 100 / scale

    values = (_angle(n, s, c), _angle(n, t, c), line_offset(u), line_offset(lower_lip))
    rows = [
        {**template, "value": round(value, 2)}
        for template, value in zip(PROFILE_ROWS, values)
    ]
    return dict(
        method="manual-profile-v1",
        status="user_reviewed",
        measurements=rows,
        note=PROFILE_NOTE,
    )


def public_profile(raw: Any) -> dict[str, Any] | None:
    """Rebuild the export from fixed definitions and finite values, never raw details."""
    if isinstance(raw,dict) and raw.get("method") == "manual-profile-v2":
        from .profile_v2 import safe_profile
        return safe_profile(raw)
    if not isinstance(raw, dict) or raw.get("method") != "manual-profile-v1":
        return None
    rows = raw.get("measurements")
    if not isinstance(rows, list) or len(rows) != len(PROFILE_ROWS):
        return None
    output = []
    for template, row in zip(PROFILE_ROWS, rows):
        if not isinstance(row, dict) or row.get("id") != template["id"]:
            return None
        value = row.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return None
        if template["unit"] == "deg" and not 0 <= value <= 180:
            return None
        output.append({**template, "value": float(value)})
    return dict(
        method="manual-profile-v1",
        status="user_reviewed",
        measurements=output,
        note=PROFILE_NOTE,
    )


def render_profile(payload: dict[str, Any] | None) -> str:
    if isinstance(payload,list):
        items=''.join(render_profile(p).replace('id="profile-chapter"','data-profile-view="yes"') for p in payload)
        return f'<section id="profile-chapter"><h2>Separate profile views</h2>{items}</section>'
    if isinstance(payload,dict) and payload.get("method")=="manual-profile-v2":
        from .profile_v2 import ROWS, diagram, safe_profile
        value=safe_profile(payload)
        if value is None:return ''
        rows=''.join(f'<tr><th>{html.escape(r["label"])}</th><td>{r["value"]:.4f} {html.escape(r["unit"])}</td><td>{html.escape(r["definition"])}</td></tr>' for r in value['measurements'])
        labels={r['id']:r['label'] for r in ROWS}
        missing=''.join(f'<li>{html.escape(labels[k])}: {"required points unavailable" if reason=="missing_anchors" else "reference points coincide or do not define this reading"}</li>' for k,reason in value['unavailable'].items())
        warning=''.join(f'<li>{html.escape(w)}</li>' for w in value['warnings'])
        photo=f' · Photo {value["photo_number"]}' if 'photo_number' in value else ''
        return f'<section class="feature-chapter" id="profile-chapter"><h2>Profile view{photo}</h2><p>User-reviewed · manual-profile-v2 · Nose points {value["facing"]} in this image · revision {value["revision"]}</p><p>{html.escape(value["note"])}</p><ul>{warning}</ul>{diagram(value)}<div class="table-wrap"><table><thead><tr><th>Reading</th><th>Value</th><th>Definition</th></tr></thead><tbody>{rows}</tbody></table></div><details><summary>Unavailable readings</summary><ul>{missing}</ul></details></section>'
    if not payload:
        return '<section class="feature-chapter" id="profile-chapter"><h2>Profile view</h2><p>Add a level lateral photograph and review its visible landmarks for up to eight independently available soft-tissue profile readings. Either facing direction works. A frontal photograph cannot supply these readings.</p></section>'
    rows = "".join(
        f"<tr><th>{html.escape(r['label'])}</th><td>{r['value']:.2f} {html.escape(r['unit'])}</td><td>{html.escape(r['definition'])}</td></tr>"
        for r in payload["measurements"]
    )
    return f'<section class="feature-chapter" id="profile-chapter"><h2>Profile view</h2><p>User-reviewed landmarks · {html.escape(payload["method"])}</p><p>{html.escape(payload["note"])}</p><div class="table-wrap"><table><thead><tr><th>Measurement</th><th>Reading</th><th>Definition</th></tr></thead><tbody>{rows}</tbody></table></div></section>'
