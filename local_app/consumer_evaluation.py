"""Numerical consumer-output sensitivity; no photos or unreviewed accuracy claims."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import AXIS_ORDER, AnalysisReport
from .consultation import DEFINITIONS, consultation_payload
from .contracts import SYMMETRY_UNIT
from .landmark_review import safe_review
from .product_report import composite_score, public_regions

VERSION = "consumer-evaluation-v1"
ISSUES = frozenset(
    {
        "yaw",
        "pitch",
        "roll",
        "too_small",
        "blurred",
        "uneven_light",
        "occluded",
        "expression",
        "pose_unknown",
        "face_not_detected",
        "limited_photo_count",
    }
)
HARMONY_IDS = frozenset(
    {
        "vertical_balance",
        "horizontal_balance",
        "midface_proportion",
        "nose_mouth_eye",
        "lip_balance",
        "lower_face_split",
    }
)
UNITS = frozenset(
    {
        "ratio",
        "deg",
        "% IPD asymmetry",
        SYMMETRY_UNIT,
        "% alignment probability",
        "typicality percentile",
        "geometric harmony percentile",
    }
)


def number(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        else None
    )


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def reference_fingerprint() -> str | None:
    root = Path(__file__).resolve().parent.parent / "bank"
    required = [root / "derived.npz", root / "provenance.json"]
    paths = required + sorted(root.glob("*/measurements.npz"))
    result = hashlib.sha256()
    try:
        for path in paths:
            result.update(str(path.relative_to(root)).encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    result.update(chunk)
            result.update(b"\0")
    except OSError:
        return None
    return result.hexdigest()


def _reviewed_snapshot(reviewed: list[tuple[dict, str]]) -> list[dict]:
    groups = defaultdict(list)
    for raw, kind in reviewed:
        safe = safe_review(raw)
        if safe is None or kind not in {"self", "trained", "other"}:
            continue
        for row in safe["measurements"]:
            groups[(row["id"], row["unit"], safe["contract"])].append(
                (row["value"], kind)
            )
    return [
        dict(
            id=key[0],
            unit=key[1],
            contract=key[2],
            method="reviewed-frontal-v1",
            value=float(np.median([v for v, _ in rows])),
            photo_count=len(rows),
            reviewers=sorted({kind for _, kind in rows}),
        )
        for key, rows in sorted(groups.items())
    ]


def _profile_snapshot(records):
    from .profile_v2 import safe_profile

    groups = defaultdict(list)
    for raw, kind in records:
        value = safe_profile(raw)
        if value is None or kind not in {"self", "trained", "other"}:
            continue
        for row in value["measurements"]:
            groups[
                (value["facing"] + ":" + row["id"], row["unit"], value["contract"])
            ].append((row["value"], kind))
    return [
        dict(
            id=key[0],
            unit=key[1],
            contract=key[2],
            method="manual-profile-v2",
            value=float(np.median([v for v, _ in values])),
            photo_count=len(values),
            reviewers=sorted({kind for _, kind in values}),
        )
        for key, values in sorted(groups.items())
    ]


def snapshot(
    report: AnalysisReport,
    *,
    reference: str | None,
    reviewed: list[tuple[dict, str]] | None = None,
    profiles: list[tuple[dict, str]] | None = None,
    settings: dict[str, str] | None = None,
) -> dict:
    axes = {}
    for axis, result in report.axes.items():
        issues = result.details.get("capture_issues", {})
        axes[axis] = dict(
            status=result.status,
            value=number(result.value),
            percentile=number(result.percentile),
            unit=result.unit if result.unit in UNITS else None,
            photo_count=result.photo_count,
            reported_interval=list(result.interval)
            if result.status == "measured" and result.interval is not None
            else None,
            capture_issues={
                k: v
                for k, v in issues.items()
                if k in ISSUES and type(v) is int and v >= 0
            },
        )
    profile = report.axes["harmony"].details.get("profile", {})
    ratios = sorted(
        key if key in HARMONY_IDS else "unrecognized-" + digest(key)
        for key, row in profile.items()
        if isinstance(row, dict)
        and number(row.get("percentile")) is not None
        and 0 <= row["percentile"] <= 100
    )
    composite = composite_score(report.axes)
    composite["inputs"] = dict(
        harmony_ratios=ratios,
        components=[
            dict(id=c["id"], weight=c["weight"], effective_weight=c["effective_weight"])
            for c in composite["components"]
        ],
        routes={c["id"]: c["status"] for c in composite["components"]},
    )
    harmony_components = [
        dict(
            id=key if key in HARMONY_IDS else "unrecognized-" + digest(key),
            value=number(row.get("value")),
            percentile=number(row.get("percentile")),
        )
        for key, row in sorted(profile.items())
        if isinstance(row, dict)
    ]
    # Source text, labels, image IDs, names, raw points and capture hashes do not
    # enter the research snapshot. A digest describes the common method only.
    regions = [
        dict(
            id=r["id"],
            value=r["value"],
            unit="deg" if r["id"] == "canthal_tilt" else "ratio",
            percentile=number(r.get("percentile")),
            method=r["method"],
        )
        for r in public_regions(report.axes)
        if r["id"] in DEFINITIONS
        and r.get("method") in {"automatic-regional-v1", "automatic-regional-v2"}
        and r.get("unit") == ("deg" if r["id"] == "canthal_tilt" else "ratio")
    ]
    recommendations = consultation_payload(
        regions,
        settings,
        frontal=[raw for raw, _ in reviewed or []],
        profile=[raw for raw, _ in profiles or []],
    )
    return dict(
        version=VERSION,
        instrument=digest(report.instrument_fingerprint),
        reference=reference,
        cohort=digest(
            {
                k: v
                for k, v in report.cohort.to_dict().items()
                if k
                in {
                    "requested_id",
                    "used_id",
                    "fallback_level",
                    "member_ids",
                    "sample_size",
                }
            }
        ),
        axes=axes,
        harmony_components=harmony_components,
        composite={
            k: composite[k]
            for k in ("version", "value", "status", "coverage", "inputs")
        },
        score_components=[
            dict(
                id=c["id"],
                value=c["value"],
                effective_weight=c["effective_weight"],
                contribution=c["value"] * c["effective_weight"],
            )
            for c in composite["components"]
        ],
        regions=regions,
        reviewed=_reviewed_snapshot(reviewed or []),
        profiles=_profile_snapshot(profiles or []),
        recommendations=dict(
            version=recommendations["recommendation_version"],
            preferences=recommendations["preferences"],
            ids=sorted(r["id"] for r in recommendations["suggestions"]),
            effects={
                r["id"]: r["intended_effect"] for r in recommendations["suggestions"]
            },
        ),
    )


def stats(changes: list[float]) -> dict:
    absolute = np.abs(changes)
    mean = float(np.mean(changes)) if changes else None
    sd = float(np.std(changes, ddof=1)) if len(changes) > 1 else None
    return dict(
        n_identities=len(changes),
        median=float(np.median(absolute)) if changes else None,
        p90=float(np.percentile(absolute, 90)) if changes else None,
        signed_bias=mean,
        descriptive_agreement_limits=[mean - 1.96 * sd, mean + 1.96 * sd]
        if sd is not None
        else None,
    )


def score_reasons(a: dict, b: dict) -> list[str]:
    reasons = []
    for key, reason in (
        ("instrument", "instrument_changed"),
        ("reference", "reference_changed"),
        ("cohort", "cohort_changed"),
    ):
        if not a.get(key) or not b.get(key):
            reasons.append(key + "_unavailable")
        elif a[key] != b[key]:
            reasons.append(reason)
    ca, cb = a["composite"], b["composite"]
    if ca["version"] != cb["version"]:
        reasons.append("formula_changed")
    for key, reason in (
        ("harmony_ratios", "harmony_ratios_changed"),
        ("components", "components_changed"),
        ("routes", "routes_changed"),
    ):
        if ca["inputs"][key] != cb["inputs"][key]:
            reasons.append(reason)
    if number(ca["value"]) is None or number(cb["value"]) is None:
        reasons.append("score_unavailable")
    return reasons


def _pair(row: dict) -> tuple[dict, dict] | None:
    sets = row.get("sets", [])
    if (
        row.get("status") != "evaluated"
        or len(sets) != 2
        or any(
            s.get("status") != "analyzed" or not isinstance(s.get("consumer"), dict)
            for s in sets
        )
    ):
        return None
    return sets[0]["consumer"], sets[1]["consumer"]


def _axis_summary(pairs: list[tuple[dict, dict]], axis: str, total: int) -> dict:
    strata = defaultdict(lambda: dict(values=[], percentiles=[], centres=[], count=0))
    values = []
    percentiles = []
    centres = []
    for a, b in pairs:
        if not a["instrument"] or a["instrument"] != b["instrument"]:
            continue
        left, right = a["axes"][axis], b["axes"][axis]
        if any(r["status"] not in {"measured", "estimated"} for r in (left, right)):
            continue
        key = left["status"] + "/" + right["status"]
        group = strata[key]
        group["count"] += 1
        va, vb = number(left["value"]), number(right["value"])
        reference_matches = bool(
            a["reference"]
            and a["reference"] == b["reference"]
            and a["cohort"]
            and a["cohort"] == b["cohort"]
        )
        if (
            left["unit"] is not None
            and left["unit"] == right["unit"]
            and va is not None
            and vb is not None
            and (axis == "symmetry" or reference_matches)
        ):
            group["values"].append(vb - va)
            values.append(vb - va)
        ra, rb = number(left["percentile"]), number(right["percentile"])
        if reference_matches and ra is not None and rb is not None:
            group["percentiles"].append(rb - ra)
            percentiles.append(rb - ra)
            centres.append((ra + rb) / 2)
    count = sum(g["count"] for g in strata.values())
    return dict(
        pairs=count,
        coverage=count / total if total else 0,
        scalar_pairs=len(values),
        scalar_coverage=len(values) / total if total else 0,
        value_movement=stats(values),
        percentile_movement=stats(percentiles),
        between_identity_percentile_sd=float(np.std(centres, ddof=1))
        if len(centres) > 1
        else None,
        strata={
            key: dict(
                pairs=g["count"],
                value_movement=stats(g["values"]),
                percentile_movement=stats(g["percentiles"]),
            )
            for key, g in sorted(strata.items())
        },
    )


def _metric_summary(pairs: list[tuple[dict, dict]], key: str, total: int) -> dict:
    changes = defaultdict(list)
    counts = Counter()
    strata = defaultdict(Counter)
    for a, b in pairs:
        left = {r["id"]: r for r in a[key]}
        right = {r["id"]: r for r in b[key]}
        for metric in left.keys() & right.keys():
            x, y = left[metric], right[metric]
            if x["unit"] != y["unit"] or x["method"] != y["method"]:
                continue
            if key in {"reviewed", "profiles"} and x["contract"] != y["contract"]:
                continue
            if key == "regions" and (
                not a["instrument"] or a["instrument"] != b["instrument"]
            ):
                continue
            va, vb = number(x["value"]), number(y["value"])
            if va is None or vb is None:
                continue
            changes[metric].append(vb - va)
            counts[metric] += 1
            if key in {"reviewed", "profiles"}:
                strata[metric][
                    "+".join(x["reviewers"]) + "/" + "+".join(y["reviewers"])
                ] += 1
    return {
        metric: dict(
            pairs=counts[metric],
            coverage=counts[metric] / total if total else 0,
            value_movement=stats(values),
            reviewer_pairs=dict(strata[metric]),
        )
        for metric, values in sorted(changes.items())
    }


def _recommendation_summary(pairs: list[tuple[dict, dict]]) -> dict:
    eligible, changed, active, opposing = 0, 0, 0, 0
    for a, b in pairs:
        left, right = a.get("recommendations"), b.get("recommendations")
        if (
            not left
            or not right
            or any(a[k] != b[k] for k in ("reference", "cohort", "instrument"))
        ):
            continue
        if (
            left["version"] != right["version"]
            or left["preferences"] != right["preferences"]
        ):
            continue
        eligible += 1
        active += bool(left["ids"] or right["ids"])
        changed += left["ids"] != right["ids"]
        effects = [set(r["effects"].values()) for r in (left, right)]
        opposing += ("side_width" in effects[0] and "crown_height" in effects[1]) or (
            "crown_height" in effects[0] and "side_width" in effects[1]
        )
    return dict(
        pairs=eligible,
        active_identity_pairs=active,
        changed_identity_pairs=changed,
        opposing_volume_identity_pairs=opposing,
    )


def summarize_consumer(rows: list[dict], *, study_kind: str = "capture_retake") -> dict:
    if not isinstance(study_kind, str) or study_kind not in {
        "capture_retake",
        "annotation_repeat",
    }:
        raise ValueError("Unsupported consumer study kind")
    pairs = [p for row in rows if (p := _pair(row)) is not None]
    score_changes = []
    transition_changes = []
    reasons = Counter()
    if study_kind == "capture_retake":
        for a, b in pairs:
            incompatible = score_reasons(a, b)
            if not incompatible:
                score_changes.append(b["composite"]["value"] - a["composite"]["value"])
            else:
                reasons.update(incompatible)
                if incompatible == ["routes_changed"]:
                    transition_changes.append(
                        b["composite"]["value"] - a["composite"]["value"]
                    )
    axes = {
        axis: _axis_summary(
            pairs if study_kind == "capture_retake" else [], axis, len(rows)
        )
        for axis in AXIS_ORDER
    }
    return dict(
        scope=study_kind,
        input_identities=len(rows),
        eligible_identity_pairs=len(pairs),
        identity_status_counts=dict(Counter(r.get("status", "unknown") for r in rows)),
        axes=axes,
        composite=dict(
            pairs=len(score_changes),
            coverage=len(score_changes) / len(rows) if rows else 0,
            movement=stats(score_changes),
            route_transition=stats(transition_changes),
            incompatibilities=dict(reasons),
        ),
        regions=_metric_summary(
            pairs if study_kind == "capture_retake" else [], "regions", len(rows)
        ),
        reviewed=_metric_summary(pairs, "reviewed", len(rows)),
        profiles=_metric_summary(pairs, "profiles", len(rows)),
        recommendations=_recommendation_summary(
            pairs if study_kind == "capture_retake" else []
        ),
        interval_note="Agreement limits are descriptive normal approximations across independent people, not confidence intervals or anatomical accuracy bounds.",
        annotation_labels_verified=False,
        correctness_validated=False,
    )
