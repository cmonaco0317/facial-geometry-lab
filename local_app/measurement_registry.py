"""Versioned reviewed-photo definitions; no automatic-bank compatibility implied."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

REGISTRY_VERSION = "reviewed-frontal-v1"


@dataclass(frozen=True)
class Metric:
    id: str
    label: str
    region: str
    requires: tuple[str, ...]
    definition: str
    unit: str = "ratio"


# Dependencies are semantic anchors, not arbitrary detector array indices.
FACE = ("mesh_top", "chin")
CHEEKS = ("cheek_a", "cheek_b")
BROWS = ("brow_head_a", "brow_head_b")
THIRDS = (*FACE, *BROWS, "nose_base")
EYES = ("eye_outer_a", "eye_inner_a", "eye_inner_b", "eye_outer_b")

METRICS = (
    Metric(
        "reviewed_facial_index",
        "Reviewed mesh forehead-to-chin height / cheek span",
        "proportions",
        (*FACE, *CHEEKS),
        "Mesh-top-to-chin vertical height / horizontal cheek span.",
    ),
    Metric(
        "reviewed_fwhr",
        "Reviewed facial width-to-height ratio",
        "proportions",
        (*CHEEKS, *BROWS, "upper_lip"),
        "Horizontal cheek span / medial-brow-to-upper-lip vertical distance.",
    ),
    Metric(
        "reviewed_upper_third",
        "Reviewed mesh upper segment share",
        "proportions",
        THIRDS,
        "Mesh-top-to-medial-brow segment / sum of three vertical segments.",
    ),
    Metric(
        "reviewed_middle_third",
        "Reviewed middle segment share",
        "proportions",
        THIRDS,
        "Medial-brow-to-nose-base segment / sum of three vertical segments.",
    ),
    Metric(
        "reviewed_lower_third",
        "Reviewed lower segment share",
        "proportions",
        THIRDS,
        "Nose-base-to-chin segment / sum of three vertical segments.",
    ),
    Metric(
        "reviewed_middle_lower_ratio",
        "Reviewed middle / lower segment",
        "proportions",
        (*BROWS, "nose_base", "chin"),
        "Medial-brow-to-nose-base distance / nose-base-to-chin distance.",
    ),
    Metric(
        "reviewed_forehead_ratio",
        "Reviewed forehead width / cheek width",
        "forehead",
        ("forehead_a", "forehead_b", *CHEEKS),
        "Horizontal mesh-forehead span / horizontal cheek span; not the hairline.",
    ),
    Metric(
        "reviewed_jaw_ratio",
        "Reviewed jaw width / cheek width",
        "jaw",
        ("jaw_a", "jaw_b", *CHEEKS),
        "Horizontal visible lower-jaw landmark span / horizontal cheek span.",
    ),
    Metric(
        "reviewed_chin_jaw_ratio",
        "Reviewed chin width / jaw width",
        "jaw",
        ("chin_a", "chin_b", "jaw_a", "jaw_b"),
        "Horizontal chin landmark span / horizontal lower-jaw span.",
    ),
    Metric(
        "reviewed_eye_spacing",
        "Reviewed eye spacing",
        "eyes",
        EYES,
        "Horizontal inner-corner span / horizontal outer-corner span.",
    ),
    Metric(
        "reviewed_eye_almondness",
        "Reviewed eye opening height / width",
        "eyes",
        (*EYES, "eye_top_a", "eye_bottom_a", "eye_top_b", "eye_bottom_b"),
        "Mean vertical eye opening / mean horizontal eye width.",
    ),
    Metric(
        "reviewed_canthal_tilt",
        "Reviewed canthal tilt",
        "eyes",
        EYES,
        "Mean outer-corner elevation angle relative to inner corners; head roll affects it.",
        "deg",
    ),
    Metric(
        "reviewed_brow_position",
        "Reviewed brow elevation / face height",
        "eyes",
        (*FACE, "eye_centre_a", "eye_centre_b", "brow_peak_a", "brow_peak_b"),
        "Mean eye-centre-to-brow-peak vertical elevation / mesh-top-to-chin height.",
    ),
    Metric(
        "reviewed_alar_ratio",
        "Reviewed nose width / cheek width",
        "nose",
        ("nose_a", "nose_b", *CHEEKS),
        "Horizontal nose-wing landmark span / horizontal cheek span.",
    ),
    Metric(
        "reviewed_naso_oral",
        "Reviewed nose width / mouth width",
        "nose",
        ("nose_a", "nose_b", "mouth_a", "mouth_b"),
        "Horizontal nose-wing landmark span / horizontal mouth-corner span.",
    ),
    Metric(
        "reviewed_mouth_ratio",
        "Reviewed mouth width / cheek width",
        "lips",
        ("mouth_a", "mouth_b", *CHEEKS),
        "Horizontal mouth-corner span / horizontal cheek span.",
    ),
    Metric(
        "reviewed_lip_lower_face",
        "Reviewed lip height / lower segment",
        "lips",
        ("upper_lip", "lower_lip", "nose_base", "chin"),
        "Visible upper-to-lower lip vertical height / nose-base-to-chin distance.",
    ),
    Metric(
        "reviewed_philtrum_ratio",
        "Reviewed philtrum / lower segment",
        "lips",
        ("nose_base", "upper_lip", "chin"),
        "Nose-base-to-upper-lip vertical distance / nose-base-to-chin distance.",
    ),
    Metric(
        "hairline_facial_index",
        "Hairline-to-chin height / cheek width",
        "forehead",
        ("hairline", "chin", *CHEEKS),
        "Visible central hairline-to-chin vertical height / horizontal cheek span. Separate from mesh height.",
    ),
    Metric(
        "hairline_upper_share",
        "Hairline upper segment share",
        "forehead",
        ("hairline", *BROWS, "nose_base", "chin"),
        "Hairline-to-medial-brow segment / sum of hairline, middle and lower vertical segments.",
    ),
    Metric(
        "hairline_middle_share",
        "Hairline-defined middle segment share",
        "forehead",
        ("hairline", *BROWS, "nose_base", "chin"),
        "Medial-brow-to-nose-base segment / sum of hairline, middle and lower vertical segments.",
    ),
    Metric(
        "hairline_lower_share",
        "Hairline-defined lower segment share",
        "forehead",
        ("hairline", *BROWS, "nose_base", "chin"),
        "Nose-base-to-chin segment / sum of hairline, middle and lower vertical segments.",
    ),
)


def measurement_contract() -> str:
    payload = {
        "version": REGISTRY_VERSION,
        "metrics": [asdict(m) for m in METRICS],
        "coordinates": "normalized image x/y; pixel conversion preserves original aspect",
    }
    payload["calculation_source"] = hashlib.sha256(
        Path(__file__).with_name("landmark_review.py").read_bytes()
    ).hexdigest()
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
