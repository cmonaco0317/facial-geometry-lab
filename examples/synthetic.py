"""Fictional geometry authored for this public demo; never photo-derived."""

from __future__ import annotations

import math

from local_app.contracts import AxisResult, CohortUse
from local_app.illustrations import ALL_IDS, OUTLINE
from local_app.landmark_review import ANCHORS


def frontal_points() -> dict:
    """Regular bilateral schematic on a normalized canvas, not a human scan."""
    central = {
        "mesh_top": [0.5, 0.10], "hairline": [0.5, 0.16],
        "chin": [0.5, 0.90], "nose_base": [0.5, 0.61],
        "upper_lip": [0.5, 0.67], "lower_lip": [0.5, 0.72],
    }
    pairs = {
        "forehead": (0.23, 0.24), "brow_head": (0.42, 0.32),
        "brow_peak": (0.30, 0.29), "eye_outer": (0.23, 0.39),
        "eye_inner": (0.42, 0.41), "eye_top": (0.32, 0.38),
        "eye_bottom": (0.32, 0.43), "eye_centre": (0.32, 0.405),
        "cheek": (0.16, 0.53), "jaw": (0.24, 0.75),
        "chin": (0.42, 0.86), "nose": (0.43, 0.59),
        "mouth": (0.36, 0.69),
    }
    bilateral = {
        f"{name}_{side}": [x if side == "a" else 1 - x, y]
        for name, (x, y) in pairs.items() for side in ("a", "b")
    }
    return {key: (central | bilateral).get(key) for key in ANCHORS}


def profile_points() -> dict:
    """Seven hand-authored schematic anchors facing right."""
    return {
        "nasion": [0.48, 0.22], "tip": [0.79, 0.43],
        "subnasale": [0.56, 0.51], "upper_lip": [0.64, 0.58],
        "lower_lip": [0.62, 0.65], "pogonion": [0.61, 0.79],
        "menton": [0.51, 0.87],
    }


def region_diagram() -> dict:
    """Ellipse plus semantic anchors for the existing regional SVG renderer."""
    points = {str(i): [160.0, 190.0] for i in ALL_IDS}
    for position, landmark in enumerate(OUTLINE[:-1]):
        angle = -math.pi / 2 + position * 2 * math.pi / (len(OUTLINE) - 1)
        points[str(landmark)] = [160 + 115 * math.cos(angle), 190 + 165 * math.sin(angle)]
    for name, (_label, index, _region) in ANCHORS.items():
        if index is not None and str(index) in points:
            x, y = frontal_points()[name]
            points[str(index)] = [x * 320, y * 380]
    for ids, center in (((70, 63, 105, 66, 107), 0.32),
                        ((300, 293, 334, 296, 336), 0.68)):
        for step, index in enumerate(ids):
            points[str(index)] = [(center + (step - 2) * 0.04) * 320,
                                  (0.30 + abs(step - 2) * 0.01) * 380]
    points["4"] = [160.0, 200.0]
    return {"points": points}


def illustrative_axes() -> dict[str, AxisResult]:
    """Invented ranks to exercise score arithmetic, not inferred from landmarks."""
    cohort = CohortUse("synthetic", "synthetic", "synthetic", 0,
                       "Fictional display fixture", "Fictional display fixture")
    common = dict(status="estimated", summary="Synthetic display fixture only",
                  cohort=cohort, photo_count=1)
    return {
        "harmony": AxisResult(axis="harmony", headline_available=False,
            details={"profile": {
                "vertical_balance": {"value": 1.0, "percentile": 45.0},
                "horizontal_balance": {"value": 1.0, "percentile": 60.0},
                "midface_proportion": {"value": 1.0, "percentile": 40.0},
            }}, **common),
        "symmetry": AxisResult(axis="symmetry", value=1.0, unit="fixture",
                               percentile=78.0, **common),
        "averageness": AxisResult(axis="averageness", value=1.0, unit="fixture",
                                  percentile=62.0, **common),
    }
