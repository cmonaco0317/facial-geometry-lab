"""Inventory of the v0.4.7 regional instruments; does not replace their contracts.

Formula notation is documentation, never evaluated as code. Automatic and reviewed
definitions intentionally have separate keys even where their arithmetic overlaps.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .anatomical_observations import IRIS_LABELS, IRIS_METHOD, PROXY_LABELS
from .measurement_registry import METRICS, REGISTRY_VERSION
from .profile_v2 import METHOD, ROWS

VERSION = "regional-reading-inventory-v1"

PROFILE_FORMULAS = {
    "convexity": "degrees(acos(dot(N-Sn,Pog-Sn) / (D(N,Sn)*D(Pog,Sn))))",
    "total_convexity": "degrees(acos(dot(N-Prn,Pog-Prn) / (D(N,Prn)*D(Pog,Prn))))",
    "upper_eline": "100 * offset(upper_lip,Prn,Pog) / D(N,Me)",
    "lower_eline": "100 * offset(lower_lip,Prn,Pog) / D(N,Me)",
    "nasal_projection_ratio": "abs(perp(Prn,N,Sn)) / D(N,Me)",
    "nasal_length_ratio": "D(N,Prn) / D(N,Me)",
    "lower_face_ratio": "D(Sn,Me) / D(N,Me)",
    "lip_offset_difference": "100 * (offset(upper_lip,Prn,Pog)-offset(lower_lip,Prn,Pog)) / D(N,Me)",
}

FORMULAS = {
    "facial_index": "V(mesh_top,chin) / H(cheek_a,cheek_b)",
    "fwhr": "H(cheek_a,cheek_b) / abs(BY - y(upper_lip))",
    "upper_third": "U / (U + M + L)",
    "middle_third": "M / (U + M + L)",
    "lower_third": "L / (U + M + L)",
    "middle_lower_ratio": "M / L",
    "forehead_ratio": "H(forehead_a,forehead_b) / H(cheek_a,cheek_b)",
    "jaw_ratio": "H(jaw_a,jaw_b) / H(cheek_a,cheek_b)",
    "chin_jaw_ratio": "H(chin_a,chin_b) / H(jaw_a,jaw_b)",
    "eye_spacing": "H(eye_inner_a,eye_inner_b) / H(eye_outer_a,eye_outer_b)",
    "eye_almondness": "(V(eye_top_a,eye_bottom_a) + V(eye_top_b,eye_bottom_b)) / (H(eye_outer_a,eye_inner_a) + H(eye_outer_b,eye_inner_b))",
    "canthal_tilt": "mean_s(degrees(atan2(y(eye_inner_s)-y(eye_outer_s), H(eye_outer_s,eye_inner_s))))",
    "brow_position": "mean_s(y(eye_centre_s)-y(brow_peak_s)) / V(mesh_top,chin)",
    "alar_ratio": "H(nose_a,nose_b) / H(cheek_a,cheek_b)",
    "naso_oral": "H(nose_a,nose_b) / H(mouth_a,mouth_b)",
    "mouth_ratio": "H(mouth_a,mouth_b) / H(cheek_a,cheek_b)",
    "lip_lower_face": "V(upper_lip,lower_lip) / L",
    "philtrum_ratio": "V(nose_base,upper_lip) / L",
    "hairline_facial_index": "V(hairline,chin) / H(cheek_a,cheek_b)",
    "hairline_upper_share": "HU / (HU + M + L)",
    "hairline_middle_share": "M / (HU + M + L)",
    "hairline_lower_share": "L / (HU + M + L)",
}

# Explicit detector indices, including inputs hidden inside means and minima.
MESH_INPUTS = {
    "mesh_top": [10],
    "chin": [152],
    "cheek_a": [234],
    "cheek_b": [454],
    "brow_head_a": [55],
    "brow_head_b": [285],
    "upper_lip": [0],
    "lower_lip": [17],
    "nose_base": [2],
    "forehead_a": [54],
    "forehead_b": [284],
    "jaw_a": [172],
    "jaw_b": [397],
    "chin_a": [32],
    "chin_b": [262],
    "eye_outer_a": [33],
    "eye_inner_a": [133],
    "eye_inner_b": [362],
    "eye_outer_b": [263],
    "eye_top_a": [159],
    "eye_bottom_a": [145],
    "eye_top_b": [386],
    "eye_bottom_b": [374],
    "nose_a": [129],
    "nose_b": [358],
    "mouth_a": [61],
    "mouth_b": [291],
    "eye_centre_a": [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    ],
    "eye_centre_b": [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ],
    "brow_peak_a": [70, 63, 105, 66, 107, 55, 65, 52, 46],
    "brow_peak_b": [300, 293, 334, 296, 336, 285, 295, 282, 276],
}


def _entry(family: str, metric: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"key": f"{family}:{metric['id']}", "family": family, **metric, **extra}


def reading_registry() -> dict[str, Any]:
    """Return fresh metadata; no face model, reference bank or detector is loaded."""
    readings = []
    for metric in METRICS:
        base = metric.id.removeprefix("reviewed_")
        fields = asdict(metric)
        fields["inputs"] = fields.pop("requires")
        readings.append(
            _entry(
                "reviewed_frontal",
                fields,
                source_view="neutral frontal",
                owner="local_app/landmark_review.py:calculate",
                method=REGISTRY_VERSION,
                formula=FORMULAS[base],
                display_decimals=4,
                reference_policy="No matched reviewed-reference percentiles.",
            )
        )
        if metric.id.startswith("reviewed_"):
            indices = sorted({i for k in metric.requires for i in MESH_INPUTS[k]})
            definition = metric.definition
            formula = FORMULAS[base]
            if base == "brow_position":
                definition = "Mean eyelid-ring-centroid to minimum-y brow elevation / mesh-top-to-chin vertical height. Not an iris measurement."
                formula = "mean_s(y(mean(eyelid_ring_s))-min(y(brow_set_s))) / V(mesh_top,chin)"
            readings.append(
                _entry(
                    "automatic",
                    {**fields, "id": base},
                    label=PROXY_LABELS.get(
                        base, metric.label.removeprefix("Reviewed ")
                    ),
                    definition=definition,
                    inputs=indices,
                    formula=formula,
                    source_view="Intended neutral frontal; existing best-effort route may retain the least-compromised photo even when it fails pose gates.",
                    selection_owners=[
                        "local_app/analysis.py",
                        "local_app/regional_evidence.py:metric_evidence",
                        "local_app/estimates.py:regional_measurements",
                        "local_app/automatic_forehead.py:coverage_exclusions",
                    ],
                    owner="facial_catalogue.py:base,measure_all",
                    method=(
                        "automatic-regional-v2"
                        if base in {"eye_spacing", "eye_almondness", "canthal_tilt"}
                        else "automatic-regional-v1"
                    ),
                    display_decimals=4,
                    reference_policy="Only same-definition automatic reference; aggregate median over eligible finite observations, then round to 4 decimals.",
                )
            )
    eye_check = [33, 133, 362, 263, 159, 145, 386, 374, 468, 473]
    for key, label in IRIS_LABELS.items():
        brow = key == "iris_brow_elevation"
        readings.append(
            _entry(
                "iris",
                dict(
                    id=key,
                    label=label,
                    region="eyes",
                    unit="ratio",
                    inputs=sorted(
                        set(
                            eye_check
                            + (
                                MESH_INPUTS["brow_peak_a"] + MESH_INPUTS["brow_peak_b"]
                                if brow
                                else []
                            )
                        )
                    ),
                    definition=label
                    + "; iris centres must lie inside their detected eye boxes.",
                    formula=(
                        "mean_s(y(iris_s)-min(y(brow_set_s))) / D(33,263)"
                        if brow
                        else "D(468,473) / D(33,263)"
                    ),
                ),
                source_view="Intended neutral frontal; follows eye-region selection, including a possible out-of-pose primary fallback when no compatible candidate exists.",
                selection_owners=[
                    "local_app/analysis.py",
                    "local_app/regional_evidence.py:metric_evidence",
                    "local_app/anatomical_observations.py:observed_iris_rows",
                ],
                method=IRIS_METHOD,
                owner="local_app/anatomical_observations.py:iris_metrics,observed_iris_rows",
                display_decimals=4,
                reference_policy="Unranked; never substitute eyelid centroids for missing irises.",
            )
        )
    for row in ROWS:
        fields = dict(row)
        fields["inputs"] = fields.pop("requires")
        readings.append(
            _entry(
                "reviewed_profile",
                fields,
                region="profile",
                source_view="Intended true lateral; facing user-declared in displayed image, with placement warnings rather than independently verified pose.",
                method=METHOD,
                owner="local_app/profile_v2.py:measure,_value",
                formula=PROFILE_FORMULAS[row["id"]],
                display_decimals=4,
                reference_policy="Unranked 2D soft-tissue geometry; not bone position. Stored values round to 6 decimals before 4-decimal display.",
            )
        )
    return dict(
        version=VERSION,
        baseline="v0.4.7",
        readings=readings,
        notation={
            "coordinates": "normalized image x/y; multiply x by original width and y by original height before lengths or angles; y increases downward",
            "H,V,D": "absolute horizontal span, absolute vertical span, Euclidean distance in pixels",
            "BY": "mean y of brow_head_a,brow_head_b",
            "U,M,L,HU": "abs(BY-y(mesh_top)), abs(y(nose_base)-BY), V(nose_base,chin), abs(BY-y(hairline))",
            "s": "each image side a,b; no silent mirroring",
            "brow_set_s,eyelid_ring_s": "explicit detector index lists in mesh_input_groups",
            "N,Prn,Sn,Pog,Me": "nasion, tip, subnasale, pogonion, menton",
            "perp(q,a,b)": "((by-ay)*(qx-ax)-(bx-ax)*(qy-ay)) / D(a,b); infinite-line signed perpendicular distance",
            "offset(q,Prn,Pog)": "perp(q,Prn,Pog) * (-1 if perp(N,Prn,Pog)>0 else 1); positive away from the nasion side; undefined when nasion is collinear with the line",
        },
        mesh_input_groups={key: list(indices) for key, indices in MESH_INPUTS.items()},
        scope="50 regional definition paths, with overlapping inputs. Pillars, intermediate legacy CLI outputs, physical-scale research and composite arithmetic are separate, not 50 independent anatomical observations.",
        score_dependencies={
            "version": "geometry-composite-v1",
            "owner": "local_app/product_report.py:composite_score",
            "weights": {"harmony": 0.4, "symmetry": 0.3, "averageness": 0.3},
            "harmony_inputs": [
                "vertical_balance",
                "horizontal_balance",
                "midface_proportion",
                "nose_mouth_eye",
                "lip_balance",
                "lower_face_split",
            ],
            "harmony_owner": "harmony_axes.py:magnitudes (mesh-iris substitution on the legacy path)",
            "forehead_dependency": "vertical_balance uses upper/lower shares with mesh_top in their common denominator; symmetry and typicality use mesh shape, including predicted forehead points",
            "review_policy": "v0.4.7 reviewed edits and regional visibility exclusions do not recompute the original pillar inputs. No validated composite error bound. Preserve historical instrument values; reconciliation is checkpoint 2.",
        },
    )


def current_reading_registry() -> dict[str, Any]:
    """Extend the frozen inventory with independently named unranked constructions."""
    from .profile_proposals import profile_proxy_definitions
    from .expanded_measurements import expanded_definitions

    original = reading_registry()
    return {
        **original,
        "version": "regional-reading-inventory-v3",
        "readings": original["readings"]
        + profile_proxy_definitions()
        + expanded_definitions(),
        "scope": "50 preserved legacy paths, 8 profile mesh proxies and 33 expanded frontal/profile construction paths. Overlapping definitions and reviewed/proxy alternatives are not independent anatomical measurements.",
    }
