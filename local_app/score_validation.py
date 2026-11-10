"""Score real app-path pairs, including the withheld harmony candidate.

This is a shadow validation bridge, never an authorization to expose a score.
The comparison population and independent reference evidence remain separate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from validate_harmony_score import (
    FORMULA_VERSION,
    HARMONY_AXES,
    MINIMUM_VALIDATION_IDENTITIES,
    build_cohort_models,
    gate_decision,
    score_candidate_from_artifact,
)

from .analysis import AXIS_ORDER, SetMeasurementAdapter
from .evaluation import _stats, read_regular_file


def verify_source_instrument(report: dict, artifact: dict) -> None:
    """Verify detector graph, models, runtime and candidate bank provenance."""
    import bank_instrument as bi

    recorded = report.get("execution_fingerprint")
    if not isinstance(recorded, dict):
        raise ValueError("missing source execution evidence")
    current = bi.measurement_fingerprint(force=True)
    if artifact.get("instrument_fingerprint") != current:
        raise ValueError("candidate bank instrument is incompatible")
    modules = recorded.get("modules")
    required = set(current["modules"]) | {
        "assess_set.py",
        "local_app/evaluation.py",
        "validate_harmony_score.py",
    }
    if (
        not isinstance(modules, dict)
        or not required <= modules.keys()
        or len(modules) > 500
        or recorded.get("n_modules") != len(modules)
        or recorded.get("models") != current["models"]
        or recorded.get("packages") != current["packages"]
    ):
        raise ValueError("source measurement instrument is incompatible")
    root = Path(__file__).resolve().parent.parent
    for name, digest in modules.items():
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{16}", digest)
        ):
            raise ValueError("invalid source module evidence")
        path = root / name
        if (
            Path(name).is_absolute()
            or Path(name).suffix != ".py"
            or not path.resolve().is_relative_to(root)
            or path.is_symlink()
        ):
            raise ValueError("invalid source module path")
        if (
            hashlib.sha256(read_regular_file(path, 2 * 1024 * 1024)).hexdigest()[:16]
            != digest
        ):
            raise ValueError("source execution module changed")


def _validate_structure(report: dict, artifact: dict) -> None:
    if not isinstance(report, dict) or not isinstance(artifact, dict):
        raise ValueError("expected object reports")
    digest = report.get("manifest_digest")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid source manifest digest")
    rows = report.get("identities")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 5000:
        raise ValueError("invalid source identities")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("sets"), list):
            raise ValueError("invalid source identity structure")
        for side in row["sets"]:
            if not isinstance(side, dict) or not isinstance(side.get("axes"), dict):
                raise ValueError("invalid source set structure")
            if any(not isinstance(value, dict) for value in side["axes"].values()):
                raise ValueError("invalid source axis structure")


def _number(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError("nonfinite or nonnumeric measurement")
    return float(value)


def _pair_scores(sets: list[dict], artifact: dict, axis: str) -> list[float] | None:
    if axis != "harmony":
        units = [s.get("axes", {}).get(axis, {}).get("unit") for s in sets]
        if not units[0] or units[0] != units[1]:
            return None
    scores = []
    for side in sets:
        reading = side.get("axes", {}).get(axis, {})
        if reading.get("status") != "measured":
            return None
        if axis == "harmony":
            components = reading.get("components", {})
            if not all(name in components for name in HARMONY_AXES):
                return None
            value = score_candidate_from_artifact(
                artifact,
                side["cohort_id"],
                {name: _number(components[name]["value"]) for name in HARMONY_AXES},
            )
            if value is None:
                return None
        else:
            if reading.get("headline_available") is not True:
                return None
            value = _number(reading.get("percentile"))
        if not 0 <= value <= 100:
            raise ValueError("percentile outside its declared range")
        scores.append(float(value))
    return scores


def validate_scores(report: dict, artifact: dict, expected_instrument: str) -> dict:
    """One independent pair per input identity, never manufacture missing scores."""
    if isinstance(report, dict) and (
        "consumer_version" in report
        or report.get("version", 1) != 1
        or report.get("study_kind", "capture_retake") != "capture_retake"
    ):
        raise ValueError("consumer evaluation is not a legacy score-validation input")
    _validate_structure(report, artifact)
    verify_source_instrument(report, artifact)
    if (
        report.get("protocol")
        not in {"fixed_reference_stability", "labelled_repeatability"}
        or not isinstance(report.get("identities"), list)
        or not 1 <= len(report["identities"]) <= 5000
        or not expected_instrument
        or report.get("expected_app_instrument") != expected_instrument
    ):
        raise ValueError("incompatible evaluation report or changed app instrument")
    rows = report["identities"]
    indices = [r.get("index") for r in rows]
    if (
        any(type(index) is not int or index < 1 for index in indices)
        or len(set(indices)) != len(indices)
        or artifact.get("formula_version") != FORMULA_VERSION
    ):
        raise ValueError("invalid identity indices or candidate formula")
    axes: dict[str, Any] = {}
    for axis in AXIS_ORDER:
        movements, centers, by_cohort = [], [], defaultdict(list)
        statuses = Counter()
        for row in rows:
            sides = row.get("sets", [])
            if row.get("status") != "evaluated":
                statuses["excluded_identity"] += 1
                continue
            if len(sides) != 2 or any(s.get("status") != "analyzed" for s in sides):
                statuses["incomplete_pair"] += 1
                continue
            cohort = sides[0].get("cohort_id")
            if (
                not cohort
                or cohort != sides[1].get("cohort_id")
                or any(
                    s.get("instrument_fingerprint") != expected_instrument
                    for s in sides
                )
            ):
                statuses["incompatible_pair"] += 1
                continue
            try:
                scores = _pair_scores(sides, artifact, axis)
            except (ValueError, KeyError, TypeError):
                statuses["invalid_measurement"] += 1
                continue
            if scores is None:
                statuses["unavailable_measurement"] += 1
                continue
            movement = abs(scores[0] - scores[1])
            movements.append(movement)
            centers.append(float(np.mean(scores)))
            # Only known model keys may be emitted as reference population names.
            name = (
                cohort
                if cohort in artifact.get("training_cohorts", {})
                else "unresolved"
            )
            by_cohort[name].append(movement)
            statuses["measured"] += 1
        axes[axis] = {
            "input_identities": len(rows),
            "paired_identities": len(movements),
            "coverage": len(movements) / len(rows),
            "status_counts": dict(statuses),
            "percentile_movement": _stats(movements),
            "between_identity_percentile_sd": float(np.std(centers, ddof=1))
            if len(centers) > 1
            else None,
            "cohort_movements": {
                key: _stats(v) for key, v in sorted(by_cohort.items())
            },
        }
        if axis == "harmony":
            axes[axis]["repeatability_gate"] = gate_decision(
                movements=movements,
                eligible_identities=len(rows),
                scored_identities=len(movements),
                minimum_identities=MINIMUM_VALIDATION_IDENTITIES,
            ).to_dict()
    return {
        "protocol": "four_axis_shadow_validation",
        "version": 1,
        "source_manifest_digest": report.get("manifest_digest"),
        "comparison_protocol": report["protocol"],
        "candidate_digest": hashlib.sha256(
            json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "app_instrument": expected_instrument,
        "source_execution_verified": True,
        "axes": axes,
        "product_available": False,
        "release_gate_modified": False,
        "correctness_validated": False,
        "bank_identity_overlap_verified": False,
        "independent_reference_validated": False,
        "limitations": [
            "Harmony is the existing six-proportion geometric-center candidate, not attractiveness.",
            "Repeatability is not correctness; no independent reference agreement is established.",
            "Source/bank identity overlap and source identity labels need independent verification.",
            "Fixed-reference demographics are comparison settings, not subject labels.",
            "Candidate gate describes pooled repeatability only; cohort-specific release evidence remains required.",
            "No new symmetry, dimorphism or averageness algorithm was substituted for its bank instrument.",
        ],
    }


def validate_report_file(source: Path, destination: Path, bank_dir: Path) -> None:
    from evaluate_casual_photos import write_report

    root = Path(__file__).resolve().parent.parent
    destination = destination.resolve()
    if destination.suffix.lower() != ".json" or any(
        destination.is_relative_to(area.resolve())
        for area in (
            bank_dir,
            root / "bank",
            root / "captures",
            root / "_datasets",
            root / "_scans",
            root / "dist",
        )
    ):
        raise ValueError(
            "shadow report must be a new JSON outside source/bank/release areas"
        )
    if bank_dir.resolve() != (root / "bank").resolve():
        raise ValueError("report validation must use the app's actual bank")
    report = json.loads(read_regular_file(source.resolve(), 64 * 1024 * 1024))
    before = SetMeasurementAdapter().fingerprint()
    provenance = json.loads(
        read_regular_file(bank_dir / "provenance.json", 16 * 1024 * 1024)
    )
    # Refit the declared candidate from the current bank, never restamp a stale model.
    models = build_cohort_models(bank_dir)
    artifact = {
        "formula_version": FORMULA_VERSION,
        "available": False,
        "instrument_fingerprint": provenance.get("instrument"),
        "training_cohorts": {
            name: model.to_storage() for name, model in models.items()
        },
    }
    result = validate_scores(report, artifact, before)
    verify_source_instrument(report, artifact)
    if SetMeasurementAdapter().fingerprint() != before:
        raise ValueError("bank changed during validation")
    result["candidate_rebuilt_from_current_bank"] = True
    write_report(destination, result)
