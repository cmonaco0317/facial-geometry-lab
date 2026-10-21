"""Versioned, independently available soft-tissue photo geometry. No clinical norms."""

from __future__ import annotations

import hashlib
import html
import math
from pathlib import Path
from typing import Any

from .profile_review import ANCHORS, PROFILE_NOTE, PROFILE_ROWS, _angle

METHOD = "manual-profile-v2"
EXTRA_ROWS = (
    dict(
        id="nasal_projection_ratio",
        label="Custom nasal projection / face height",
        unit="ratio",
        definition="Unsigned perpendicular distance from Prn to the infinite N–Sn line / Euclidean N–Me distance.",
        requires=("nasion", "tip", "subnasale", "menton"),
    ),
    dict(
        id="nasal_length_ratio",
        label="Nasal chord / face height",
        unit="ratio",
        definition="Euclidean N–Prn distance / Euclidean N–Me distance; a photo-space chord, not physical nose length.",
        requires=("nasion", "tip", "menton"),
    ),
    dict(
        id="lower_face_ratio",
        label="Lower-face chord / face height",
        unit="ratio",
        definition="Euclidean Sn–Me distance / Euclidean N–Me distance, not a projected vertical-third measure.",
        requires=("nasion", "subnasale", "menton"),
    ),
    dict(
        id="lip_offset_difference",
        label="Upper-minus-lower lip offset",
        unit="% N–Me",
        definition="Upper minus lower signed offset from the same Prn–Pog line, using Euclidean N–Me normalization. Derived contrast, not independent evidence.",
        requires=("nasion", "tip", "menton", "pogonion", "upper_lip", "lower_lip"),
    ),
)
DEPS = (
    ("nasion", "subnasale", "pogonion"),
    ("nasion", "tip", "pogonion"),
    ("nasion", "tip", "pogonion", "menton", "upper_lip"),
    ("nasion", "tip", "pogonion", "menton", "lower_lip"),
)
ROWS = (
    tuple({**r, "requires": deps} for r, deps in zip(PROFILE_ROWS, DEPS)) + EXTRA_ROWS
)
NOTE = (
    PROFILE_NOTE
    + " Facing is user-declared in the displayed image. Opposite-side views are separate observations, not a calibrated asymmetry test. All reference lengths are Euclidean; none uses a vertical world axis."
)


def contract() -> str:
    return hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("profile_review.py").read_bytes()
    ).hexdigest()


def validate_points(raw: Any) -> dict:
    if not isinstance(raw, dict) or set(raw) != set(ANCHORS):
        raise ValueError(
            "Use exactly the seven named points; mark unseen points unavailable."
        )
    result = {}
    for key, value in raw.items():
        if value is None:
            result[key] = None
        elif (
            isinstance(value, list)
            and len(value) == 2
            and all(
                type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                for v in value
            )
        ):
            result[key] = [float(v) for v in value]
        else:
            raise ValueError(
                "Profile coordinates must be finite, within the photo, or unavailable."
            )
    return result


def _dimensions(width: int, height: int, facing: str):
    if any(
        type(v) is not int or not 1 <= v <= 100000 for v in (width, height)
    ) or facing not in ("left", "right"):
        raise ValueError("Choose valid photo dimensions and a facing direction.")


def _perpendicular(q, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-8:
        raise ValueError("degenerate_reference")
    return (dy * (q[0] - a[0]) - dx * (q[1] - a[1])) / length


def _value(key, p):
    n, t, s, c, m = (
        p.get(k) for k in ("nasion", "tip", "subnasale", "pogonion", "menton")
    )
    if key == "convexity":
        return _angle(n, s, c)
    if key == "total_convexity":
        return _angle(n, t, c)
    height = math.dist(n, m)
    if height < 1:
        raise ValueError("degenerate_reference")
    if key == "nasal_projection_ratio":
        return abs(_perpendicular(t, n, s)) / height
    if key == "nasal_length_ratio":
        return math.dist(n, t) / height
    if key == "lower_face_ratio":
        return math.dist(s, m) / height
    # Orient the line normal toward the nose from the bridge; unlike a screen-X
    # sign, this convention rotates with the actual reviewed geometry.
    forward = _perpendicular(n, t, c)
    if abs(forward) < 1e-8:
        raise ValueError("degenerate_reference")
    direction = -1 if forward > 0 else 1

    def offset(point):
        return _perpendicular(point, t, c) * direction * 100 / height

    if key == "upper_eline":
        return offset(p["upper_lip"])
    if key == "lower_eline":
        return offset(p["lower_lip"])
    return offset(p["upper_lip"]) - offset(p["lower_lip"])


def warnings(points, width, height, *, facing):
    points = validate_points(points)
    _dimensions(width, height, facing)
    result = []
    n, t, s, m = (points[k] for k in ("nasion", "tip", "subnasale", "menton"))
    if n and t:
        if (t[0] - n[0]) * (1 if facing == "right" else -1) <= 0:
            result.append(
                "Nose-tip placement disagrees with the declared facing direction. Check the view and points."
            )
        if abs(t[0] - n[0]) * width < 0.04 * height:
            result.append(
                "The visible nose projection is small. Check that this is a true lateral view; frontal views do not establish profile measurements."
            )
    if (
        n
        and m
        and math.dist((n[0] * width, n[1] * height), (m[0] * width, m[1] * height)) < 80
    ):
        result.append(
            "The reviewed face has limited image detail. Check placement at full size."
        )
    if n and s and m and not n[1] < s[1] < m[1]:
        result.append(
            "Bridge, nose base and chin are not in their usual top-to-bottom order. Check posture and point placement."
        )
    return result


def measure(raw, width, height, *, facing):
    points = validate_points(raw)
    _dimensions(width, height, facing)
    p = {k: [v[0] * width, v[1] * height] if v else None for k, v in points.items()}
    rows = []
    unavailable = {}
    for template in ROWS:
        key = template["id"]
        if any(p[k] is None for k in template["requires"]):
            unavailable[key] = "missing_anchors"
            continue
        try:
            value = _value(key, p)
            if not math.isfinite(value):
                raise ValueError("degenerate_reference")
        except ValueError:
            unavailable[key] = "degenerate_reference"
            continue
        rows.append({**template, "value": round(value, 6)})
    return dict(
        method=METHOD,
        contract=contract(),
        status="user_reviewed",
        facing=facing,
        measurements=rows,
        unavailable=unavailable,
        diagram=dict(points=points, width=width, height=height),
        warnings=warnings(points, width, height, facing=facing),
        note=NOTE,
    )


def safe_profile(raw: Any) -> dict | None:
    if (
        not isinstance(raw, dict)
        or raw.get("method") != METHOD
        or raw.get("contract") != contract()
    ):
        return None
    revision = raw.get("revision")
    photo = raw.get("photo_number")
    if (
        type(revision) is not int
        or revision < 1
        or (photo is not None and (type(photo) is not int or not 1 <= photo <= 1000000))
    ):
        return None
    diagram = raw.get("diagram")
    if not isinstance(diagram, dict) or set(diagram) != {"points", "width", "height"}:
        return None
    try:
        result = measure(
            diagram["points"],
            diagram["width"],
            diagram["height"],
            facing=raw.get("facing"),
        )
    except (ValueError, TypeError):
        return None
    return {
        **result,
        "revision": revision,
        **({"photo_number": photo} if photo is not None else {}),
    }


def diagram(payload):
    """Script-free observed points, angle arcs and true projection feet; no photo."""
    d = payload["diagram"]
    w, h = d["width"], d["height"]
    p = {k: [v[0] * w, v[1] * h] for k, v in d["points"].items() if v is not None}

    def line(a, b, style=""):
        return f'<line x1="{a[0]:.4f}" y1="{a[1]:.4f}" x2="{b[0]:.4f}" y2="{b[1]:.4f}" {style}/>'

    content = ""
    for a, b in (
        ("nasion", "menton"),
        ("nasion", "subnasale"),
        ("nasion", "tip"),
        ("subnasale", "menton"),
        ("tip", "pogonion"),
        ("subnasale", "pogonion"),
        ("nasion", "pogonion"),
    ):
        if a in p and b in p:
            content += line(p[a], p[b])
    for name, a, b in (
        ("tip", "nasion", "subnasale"),
        ("upper_lip", "tip", "pogonion"),
        ("lower_lip", "tip", "pogonion"),
    ):
        if all(k in p for k in (name, a, b)):
            q, u, v = p[name], p[a], p[b]
            dx, dy = v[0] - u[0], v[1] - u[1]
            den = dx * dx + dy * dy
            if den > 1e-8:
                projection = ((q[0] - u[0]) * dx + (q[1] - u[1]) * dy) / den
                foot = [u[0] + projection * dx, u[1] + projection * dy]
                content += line(q, foot, 'stroke-dasharray="4 3"')
    for a, b, c in (("nasion", "subnasale", "pogonion"), ("nasion", "tip", "pogonion")):
        if all(k in p for k in (a, b, c)):
            u, v, q = p[a], p[b], p[c]
            radius = min(math.dist(u, v), math.dist(q, v)) * 0.18
            start = math.atan2(u[1] - v[1], u[0] - v[0])
            end = math.atan2(q[1] - v[1], q[0] - v[0])
            delta = (end - start + math.pi) % (2 * math.pi) - math.pi
            if radius > 1e-8:
                x, y = v[0] + radius * math.cos(start), v[1] + radius * math.sin(start)
                ex, ey = (
                    v[0] + radius * math.cos(start + delta),
                    v[1] + radius * math.sin(start + delta),
                )
                content += f'<path d="M{x:.4f},{y:.4f} A{radius:.4f},{radius:.4f} 0 0 {1 if delta > 0 else 0} {ex:.4f},{ey:.4f}"/>'
    radius = max(w, h) * 0.009
    marks = "".join(
        f'<circle cx="{xy[0]:.4f}" cy="{xy[1]:.4f}" r="{radius:.4f}"/><text x="{xy[0] + radius * 1.5:.4f}" y="{xy[1] - radius * 1.5:.4f}">{html.escape(key)}</text>'
        for key, xy in p.items()
    )
    return f'<svg class="profile-diagram" viewBox="{-w * 0.1} {-h * 0.1} {w * 1.2} {h * 1.2}" role="img" aria-label="Reviewed profile: reference lines, angle arcs and perpendicular offsets"><g fill="none" stroke="#18705e" stroke-width="{max(w, h) * 0.003}">{content}</g><g fill="#163e36" font-size="{max(w, h) * 0.021}">{marks}</g></svg>'
