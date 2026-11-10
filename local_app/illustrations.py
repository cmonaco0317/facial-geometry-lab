"""Bounded, script-free diagrams from one observed face; never a generated face."""

from __future__ import annotations

import html
from math import isfinite
from typing import Any

OUTLINE = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
    10,
)
REGION_LINES = {
    "proportions": ((10, 152), (234, 454), (55, 285), (2, 152)),
    "forehead": ((54, 284),),
    "eyes": (
        (33, 133),
        (362, 263),
        (159, 145),
        (386, 374),
        (70, 63, 105, 66, 107),
        (300, 293, 334, 296, 336),
    ),
    "nose": ((129, 358), (4, 2)),
    "lips": ((61, 291), (0, 17), (2, 0)),
    "jaw": ((172, 397), (32, 262), (234, 454)),
}
ALL_IDS = sorted(
    set(OUTLINE)
    | {i for lines in REGION_LINES.values() for line in lines for i in line}
)


def observed_diagram(context: Any, photo_ids: tuple[str, ...]) -> dict[str, Any] | None:
    import numpy as np

    candidates = [p for p in context.kept if p["photo"] in photo_ids]
    if not candidates:
        return None
    chosen = min(candidates, key=lambda p: (abs(p.get("yaw", 0)), str(p["photo"])))
    points = np.asarray(chosen["landmarks"], dtype=float)
    if points.shape != (468, 2) or not np.isfinite(points).all():
        return None
    selected = points[ALL_IDS]
    lo = selected.min(axis=0)
    span = np.ptp(selected, axis=0)
    scale = max(float(span[0]) / 280, float(span[1]) / 340)
    if scale <= 0:
        return None
    shift = np.array([(320 - span[0] / scale) / 2, (380 - span[1] / scale) / 2])
    normalized = (selected - lo) / scale + shift
    return {
        "points": {
            str(i): [round(float(v), 3) for v in p] for i, p in zip(ALL_IDS, normalized)
        },
        "note": "One selected photograph, normalized without changing aspect ratio. Lines show the landmark spans used in this region; table values may summarize several photographs. The top mesh point is not the hairline.",
    }


def public_diagram(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("points"), dict):
        return None
    source = raw["points"]
    if set(source) != {str(i) for i in ALL_IDS}:
        return None
    points = {}
    for key, pair in source.items():
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not isfinite(v)
                or not 0 <= v <= 380
                for v in pair
            )
        ):
            return None
        points[key] = [float(v) for v in pair]
    return {"points": points}


def render_diagrams(raw: Any) -> dict[str, str]:
    validated = public_diagram(raw)
    if not validated:
        return {}
    p = validated["points"]

    def coords(ids: tuple[int, ...]) -> str:
        return " ".join(f"{p[str(i)][0]:.3f},{p[str(i)][1]:.3f}" for i in ids)

    result = {}
    for key, lines in REGION_LINES.items():
        paths = "".join(
            f'<polyline points="{coords(line)}" fill="none" stroke="#147566" stroke-width="2.5"/>'
            for line in lines
        )
        circles = "".join(
            f'<circle cx="{p[str(i)][0]:.3f}" cy="{p[str(i)][1]:.3f}" r="3" fill="#a5632b"/>'
            for i in sorted({i for line in lines for i in line})
        )
        result[key] = (
            f'<figure class="face-diagram"><svg viewBox="0 0 320 380" role="img" aria-label="Observed {html.escape(key)} landmark spans"><polyline points="{coords(OUTLINE)}" fill="#f1f4f0" stroke="#a7b6af" stroke-width="1.5"/>{paths}{circles}</svg><figcaption>Observed landmark spans from one selected photograph. Measurements may summarize several photos. The top mesh point is not the hairline.</figcaption></figure>'
        )
    return result
