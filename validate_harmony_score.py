#!/usr/bin/env python3
"""Validate a cohort-relative harmony scalar before the UI may expose it.

The candidate is a robust, correlation-adjusted distance from the center of the
six already-declared frontal harmony axes. A higher percentile means closer to
that multivariate center. The scalar remains unavailable unless identity-level
set-to-set validation clears every pre-registered repeatability threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import harmony_axes as HA


HERE = Path(__file__).resolve().parent
HARMONY_AXES = tuple(HA.AXES)
FORMULA_VERSION = "robust-mahalanobis-v1"
SPLIT_SEED = 20260918
REFERENCE_FRACTION = 0.25
MEDIAN_MOVEMENT_LIMIT = 15.0
P90_MOVEMENT_LIMIT = 25.0
AVAILABILITY_FLOOR = 0.80
MINIMUM_VALIDATION_IDENTITIES = 20
MINIMUM_TRAINING_IDENTITIES = 12
MINIMUM_REFERENCE_IDENTITIES = 5
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class IdentitySplit:
    training: tuple[str, ...]
    reference: tuple[str, ...]


@dataclass(frozen=True)
class HarmonyCandidate:
    axes: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    inverse_correlation: np.ndarray
    reference_distances: np.ndarray
    training_identity_ids: tuple[str, ...]
    reference_identity_ids: tuple[str, ...]

    def to_storage(self) -> dict[str, Any]:
        probabilities = np.linspace(0.0, 1.0, 101)
        return {
            "axes": list(self.axes),
            "center": np.round(self.center, 10).tolist(),
            "scale": np.round(self.scale, 10).tolist(),
            "inverse_correlation": np.round(self.inverse_correlation, 10).tolist(),
            "reference_distance_quantiles": np.round(
                np.quantile(self.reference_distances, probabilities),
                10,
            ).tolist(),
            "n_training_identities": len(self.training_identity_ids),
            "n_reference_identities": len(self.reference_identity_ids),
            "training_identity_digest": _identity_digest(self.training_identity_ids),
            "reference_identity_digest": _identity_digest(self.reference_identity_ids),
        }


@dataclass(frozen=True)
class ValidationPair:
    identity_id: str
    first: np.ndarray | Sequence[float] | None
    second: np.ndarray | Sequence[float] | None


@dataclass(frozen=True)
class GateDecision:
    movements: tuple[float, ...]
    eligible_identities: int
    scored_identities: int
    coverage: float
    median_movement: float | None
    p90_movement: float | None
    median_passed: bool
    p90_passed: bool
    coverage_passed: bool
    sample_size_passed: bool
    available: bool
    minimum_identities: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_identities": self.eligible_identities,
            "scored_identities": self.scored_identities,
            "coverage": self.coverage,
            "median_movement": self.median_movement,
            "p90_movement": self.p90_movement,
            "median_passed": self.median_passed,
            "p90_passed": self.p90_passed,
            "coverage_passed": self.coverage_passed,
            "sample_size_passed": self.sample_size_passed,
            "available": self.available,
        }


def identity_split(
    identity_ids: Iterable[str],
    *,
    seed: int = SPLIT_SEED,
    reference_fraction: float = REFERENCE_FRACTION,
) -> IdentitySplit:
    """Return an order-independent, fixed-seed identity-level split."""
    identities = tuple(sorted(str(value) for value in identity_ids))
    if len(identities) != len(set(identities)):
        raise ValueError("identity ids must be unique before splitting")
    if len(identities) < 2:
        raise ValueError("at least two identities are required for a split")
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("reference_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(identities))
    reference_count = min(
        len(identities) - 1,
        max(1, int(round(len(identities) * reference_fraction))),
    )
    reference_indexes = set(int(value) for value in order[:reference_count])
    training = tuple(
        identity
        for index, identity in enumerate(identities)
        if index not in reference_indexes
    )
    reference = tuple(
        identity
        for index, identity in enumerate(identities)
        if index in reference_indexes
    )
    return IdentitySplit(training=training, reference=reference)


def _matrix(values: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(HARMONY_AXES):
        raise ValueError(f"{name} must have one column per declared harmony axis")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _identities(values: Sequence[str], rows: int, name: str) -> tuple[str, ...]:
    identities = tuple(str(value) for value in values)
    if len(identities) != rows:
        raise ValueError(f"{name} identity count must match its matrix")
    if len(identities) != len(set(identities)):
        raise ValueError(f"{name} identities must be unique")
    return identities


def _robust_scale(training: np.ndarray, center: np.ndarray) -> np.ndarray:
    scale = 1.4826 * np.median(np.abs(training - center), axis=0)
    q25, q75 = np.percentile(training, [25.0, 75.0], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    standard_scale = np.std(training, axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, iqr_scale)
    scale = np.where(scale > 1e-12, scale, standard_scale)
    return np.where(scale > 1e-12, scale, 1.0)


def _distance(
    center: np.ndarray,
    scale: np.ndarray,
    inverse_correlation: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    standardized = (values - center) / scale
    squared = np.einsum(
        "...i,ij,...j->...",
        standardized,
        inverse_correlation,
        standardized,
    )
    return np.sqrt(np.maximum(0.0, squared))


def fit_candidate(
    training_values: np.ndarray | Sequence[Sequence[float]],
    training_identity_ids: Sequence[str],
    reference_values: np.ndarray | Sequence[Sequence[float]],
    reference_identity_ids: Sequence[str],
) -> HarmonyCandidate:
    """Fit standardization only on training identities, then rank on held-out rows."""
    training = _matrix(training_values, "training matrix")
    reference = _matrix(reference_values, "reference matrix")
    training_ids = _identities(training_identity_ids, len(training), "training")
    reference_ids = _identities(reference_identity_ids, len(reference), "reference")
    if set(training_ids) & set(reference_ids):
        raise ValueError("identity leakage between training and reference splits")
    if len(training) < MINIMUM_TRAINING_IDENTITIES:
        raise ValueError("too few training identities for a six-axis candidate")
    if len(reference) < MINIMUM_REFERENCE_IDENTITIES:
        raise ValueError("too few reference identities for an empirical percentile")

    center = np.median(training, axis=0)
    scale = _robust_scale(training, center)
    standardized = (training - center) / scale
    correlation = np.corrcoef(standardized, rowvar=False)
    correlation = np.asarray(correlation, dtype=np.float64)
    if correlation.shape != (len(HARMONY_AXES), len(HARMONY_AXES)):
        raise ValueError("candidate correlation matrix has the wrong shape")
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    regularized = 0.9 * correlation + 0.1 * np.eye(len(HARMONY_AXES))
    inverse = np.linalg.pinv(regularized, hermitian=True)
    reference_distances = _distance(center, scale, inverse, reference)
    return HarmonyCandidate(
        axes=HARMONY_AXES,
        center=center,
        scale=scale,
        inverse_correlation=inverse,
        reference_distances=np.sort(reference_distances),
        training_identity_ids=training_ids,
        reference_identity_ids=reference_ids,
    )


def score_percentile(
    candidate: HarmonyCandidate,
    values: np.ndarray | Sequence[float],
) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(HARMONY_AXES),) or not np.all(np.isfinite(vector)):
        raise ValueError("harmony score requires one finite value per declared axis")
    distance = float(
        _distance(
            candidate.center,
            candidate.scale,
            candidate.inverse_correlation,
            vector[None, :],
        )[0]
    )
    return float(100.0 * np.mean(candidate.reference_distances >= distance))


def _candidate_from_storage(payload: Mapping[str, Any]) -> HarmonyCandidate:
    axes = tuple(payload.get("axes", ()))
    if axes != HARMONY_AXES:
        raise ValueError("stored harmony model does not use the declared axes")
    center = np.asarray(payload.get("center"), dtype=np.float64)
    scale = np.asarray(payload.get("scale"), dtype=np.float64)
    inverse = np.asarray(payload.get("inverse_correlation"), dtype=np.float64)
    distances = np.asarray(
        payload.get("reference_distance_quantiles"),
        dtype=np.float64,
    )
    axis_count = len(HARMONY_AXES)
    if center.shape != (axis_count,) or scale.shape != (axis_count,):
        raise ValueError("stored harmony standardization has the wrong shape")
    if (
        inverse.shape != (axis_count, axis_count)
        or distances.ndim != 1
        or len(distances) < 2
    ):
        raise ValueError("stored harmony distribution has the wrong shape")
    if (
        not np.all(np.isfinite(center))
        or not np.all(np.isfinite(scale))
        or not np.all(scale > 0)
        or not np.all(np.isfinite(inverse))
        or not np.all(np.isfinite(distances))
        or not np.all(np.diff(distances) >= 0)
    ):
        raise ValueError("stored harmony model contains invalid numbers")
    return HarmonyCandidate(
        axes=axes,
        center=center,
        scale=scale,
        inverse_correlation=inverse,
        reference_distances=np.sort(distances),
        training_identity_ids=(),
        reference_identity_ids=(),
    )


def score_from_artifact(
    artifact: Mapping[str, Any],
    cohort_id: str,
    values: Mapping[str, float],
) -> float | None:
    """Score only a validator-enabled model; unavailable or absent stays silent."""
    if artifact.get("available") is not True:
        return None
    return score_candidate_from_artifact(artifact, cohort_id, values)


def score_candidate_from_artifact(
    artifact: Mapping[str, Any],
    cohort_id: str,
    values: Mapping[str, float],
) -> float | None:
    """Research score before availability; never call this from a product card."""
    models = artifact.get("training_cohorts")
    if not isinstance(models, Mapping):
        raise ValueError("harmony artifact has no cohort models")
    stored = models.get(cohort_id)
    if not isinstance(stored, Mapping):
        return None
    if any(axis not in values for axis in HARMONY_AXES):
        return None
    try:
        vector = np.asarray([values[axis] for axis in HARMONY_AXES], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("harmony profile contains an invalid axis value") from error
    return score_percentile(_candidate_from_storage(stored), vector)


def gate_decision(
    *,
    movements: np.ndarray | Sequence[float],
    eligible_identities: int,
    scored_identities: int,
    minimum_identities: int = MINIMUM_VALIDATION_IDENTITIES,
) -> GateDecision:
    observed = np.asarray(movements, dtype=np.float64)
    if observed.ndim != 1 or not np.all(np.isfinite(observed)) or np.any(observed < 0):
        raise ValueError("movements must be a finite, nonnegative vector")
    if min(eligible_identities, scored_identities, minimum_identities) < 0:
        raise ValueError("identity counts cannot be negative")
    if scored_identities > eligible_identities or len(observed) != scored_identities:
        raise ValueError("scored identities and movements must align")
    coverage = (
        float(scored_identities / eligible_identities) if eligible_identities else 0.0
    )
    median = float(np.median(observed)) if len(observed) else None
    p90 = float(np.percentile(observed, 90.0)) if len(observed) else None
    median_passed = median is not None and median <= MEDIAN_MOVEMENT_LIMIT
    p90_passed = p90 is not None and p90 <= P90_MOVEMENT_LIMIT
    coverage_passed = coverage >= AVAILABILITY_FLOOR
    sample_size_passed = eligible_identities >= minimum_identities
    available = all((median_passed, p90_passed, coverage_passed, sample_size_passed))
    return GateDecision(
        movements=tuple(float(value) for value in observed),
        eligible_identities=int(eligible_identities),
        scored_identities=int(scored_identities),
        coverage=coverage,
        median_movement=median,
        p90_movement=p90,
        median_passed=median_passed,
        p90_passed=p90_passed,
        coverage_passed=coverage_passed,
        sample_size_passed=sample_size_passed,
        available=available,
        minimum_identities=int(minimum_identities),
    )


def evaluate_candidate(
    candidate: HarmonyCandidate,
    pairs: Sequence[ValidationPair],
    *,
    minimum_identities: int = MINIMUM_VALIDATION_IDENTITIES,
) -> GateDecision:
    pair_ids = tuple(pair.identity_id for pair in pairs)
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("validation identities must be unique")
    fitted_ids = set(candidate.training_identity_ids) | set(
        candidate.reference_identity_ids
    )
    if fitted_ids & set(pair_ids):
        raise ValueError("identity leakage between model fit and validation")
    movements: list[float] = []
    for pair in pairs:
        if pair.first is None or pair.second is None:
            continue
        try:
            first = score_percentile(candidate, pair.first)
            second = score_percentile(candidate, pair.second)
        except ValueError:
            continue
        movements.append(abs(first - second))
    return gate_decision(
        movements=np.asarray(movements, dtype=np.float64),
        eligible_identities=len(pairs),
        scored_identities=len(movements),
        minimum_identities=minimum_identities,
    )


def _identity_digest(identity_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for identity_id in sorted(identity_ids):
        digest.update(identity_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _availability_reason(decision: GateDecision) -> str:
    failures = []
    if not decision.sample_size_passed:
        failures.append(
            "validation sample is smaller than the declared minimum of "
            f"{decision.minimum_identities} identities"
        )
    if not decision.coverage_passed:
        failures.append(
            f"two-subset score coverage is {decision.coverage:.1%}, below 80%"
        )
    if not decision.median_passed:
        failures.append(
            "median movement was not shown to be at most 15 percentile points"
        )
    if not decision.p90_passed:
        failures.append(
            "90th-percentile movement was not shown to be at most 25 points"
        )
    return "All scalar gates passed." if not failures else "; ".join(failures) + "."


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{secrets.token_hex(8)}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_validation_artifact(
    destination: Path | str,
    *,
    cohort_models: Mapping[str, HarmonyCandidate | Mapping[str, Any]],
    validation_manifest: Sequence[Mapping[str, Any]],
    decision: GateDecision,
    instrument_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    stored_models = {
        cohort_id: (
            candidate.to_storage()
            if isinstance(candidate, HarmonyCandidate)
            else dict(candidate)
        )
        for cohort_id, candidate in sorted(cohort_models.items())
    }
    payload = {
        "schema_version": 1,
        "formula_version": FORMULA_VERSION,
        "construct": (
            "cohort-relative geometric harmony; higher means closer to the "
            "six-axis multivariate center, not attractiveness"
        ),
        "axes": list(HARMONY_AXES),
        "split": {
            "unit": "identity_cluster_id",
            "seed": SPLIT_SEED,
            "training_fraction": 1.0 - REFERENCE_FRACTION,
            "reference_fraction": REFERENCE_FRACTION,
        },
        "thresholds": {
            "median_movement_max_percentile_points": MEDIAN_MOVEMENT_LIMIT,
            "p90_movement_max_percentile_points": P90_MOVEMENT_LIMIT,
            "two_subset_availability_min": AVAILABILITY_FLOOR,
        },
        "minimum_validation_identities": decision.minimum_identities,
        "instrument_fingerprint": dict(instrument_fingerprint),
        "training_cohorts": stored_models,
        "validation_set_manifest": [dict(item) for item in validation_manifest],
        "achieved_metrics": decision.to_dict(),
        "mutation_checks": {
            "status": "not_run"
            if not decision.available
            else "required_before_release",
            "reason": (
                "The repeatability gate did not make the scalar available."
                if not decision.available
                else "An available candidate still requires release-time mutation checks."
            ),
        },
        "available": decision.available,
        "availability_reason": _availability_reason(decision),
    }
    _atomic_json(Path(destination), payload)
    return payload


def build_cohort_models(bank_dir: Path | str) -> dict[str, HarmonyCandidate]:
    bank_dir = Path(bank_dir)
    with (bank_dir / "provenance.json").open("r", encoding="utf-8") as stream:
        provenance = json.load(stream)
    models: dict[str, HarmonyCandidate] = {}
    with np.load(bank_dir / "derived.npz", allow_pickle=False) as derived:
        for cohort_id in sorted(provenance.get("cohorts", {})):
            keys = [f"{cohort_id}|harm__{axis}" for axis in HARMONY_AXES]
            if any(key not in derived.files for key in keys):
                continue
            arrays = [np.asarray(derived[key], dtype=np.float64) for key in keys]
            row_count = min(len(values) for values in arrays)
            values = np.column_stack([array[:row_count] for array in arrays])
            with np.load(
                bank_dir / cohort_id / "measurements.npz",
                allow_pickle=False,
            ) as measurements:
                identity_ids = np.asarray(
                    measurements["identity_cluster_id"][:row_count]
                ).astype(str)
            finite = np.all(np.isfinite(values), axis=1)
            values = values[finite]
            identities = tuple(identity_ids[finite].tolist())
            if len(values) < MINIMUM_TRAINING_IDENTITIES + MINIMUM_REFERENCE_IDENTITIES:
                continue
            split = identity_split(
                identities,
                seed=SPLIT_SEED,
                reference_fraction=REFERENCE_FRACTION,
            )
            row_by_identity = {
                identity_id: values[index]
                for index, identity_id in enumerate(identities)
            }
            training = np.stack([row_by_identity[value] for value in split.training])
            reference = np.stack([row_by_identity[value] for value in split.reference])
            if (
                len(training) < MINIMUM_TRAINING_IDENTITIES
                or len(reference) < MINIMUM_REFERENCE_IDENTITIES
            ):
                continue
            models[cohort_id] = fit_candidate(
                training,
                split.training,
                reference,
                split.reference,
            )
    return models


def validation_manifest(captures_dir: Path | str) -> tuple[dict[str, Any], ...]:
    """Inventory approved-name fixtures without processing any face unnecessarily."""
    captures_dir = Path(captures_dir)
    rows: list[dict[str, Any]] = []
    for directory in sorted(captures_dir.glob("_val_*")):
        if not directory.is_dir():
            continue
        images = tuple(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        source_file = directory / "SOURCE.txt"
        source_text = (
            source_file.read_text(encoding="utf-8", errors="replace")
            if source_file.is_file()
            else ""
        )
        license_documented = (
            "license" in source_text.lower() or "licence" in source_text.lower()
        )
        consent_documented = any(
            marker in source_text.lower()
            for marker in (
                "informed consent",
                "consented donation",
                "biometric consent",
            )
        )
        enough_raw_photos = len(images) >= 14
        if not enough_raw_photos:
            reason = (
                "fewer than 14 source photos; two disjoint seven-photo subsets "
                "are impossible"
            )
        else:
            reason = (
                "no declared comparison cohort and two-subset eligibility measurement"
            )
        rows.append(
            {
                "identity_id": directory.name,
                "source_documented": source_file.is_file(),
                "license_status": "documented"
                if license_documented
                else "not documented",
                "biometric_consent_status": (
                    "documented" if consent_documented else "not documented"
                ),
                "source_photo_count": len(images),
                "two_disjoint_seven_photo_subsets_possible": enough_raw_photos,
                "eligible_for_repeatability_gate": False,
                "ineligibility_reason": reason,
            }
        )
    return tuple(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", required=True, type=Path)
    parser.add_argument("--bank-dir", type=Path, default=HERE / "bank")
    parser.add_argument("--captures-dir", type=Path, default=HERE / "captures")
    options = parser.parse_args()

    models = build_cohort_models(options.bank_dir)
    manifest = validation_manifest(options.captures_dir)
    eligible = sum(bool(row["eligible_for_repeatability_gate"]) for row in manifest)
    decision = gate_decision(
        movements=np.array([], dtype=np.float64),
        eligible_identities=eligible,
        scored_identities=0,
        minimum_identities=MINIMUM_VALIDATION_IDENTITIES,
    )
    with (options.bank_dir / "provenance.json").open("r", encoding="utf-8") as stream:
        provenance = json.load(stream)
    payload = write_validation_artifact(
        options.write,
        cohort_models=models,
        validation_manifest=manifest,
        decision=decision,
        instrument_fingerprint=provenance.get("instrument", {}),
    )
    print(
        f"harmony scalar available: {str(payload['available']).lower()} — "
        f"{payload['availability_reason']}"
    )
    print(
        f"fitted {len(models)} cohort candidates; "
        f"{eligible} validation identities eligible"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
