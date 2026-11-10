"""Shadow repeatability evaluation through the supported application's pipeline.

This produces evidence, never a release decision. Fixed reference demographics
are comparison settings, not inferred labels or proof of cohort correctness.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .analysis import (
    AXIS_ORDER,
    AnalysisNotReady,
    ProgressiveAnalysisService,
    SetMeasurementAdapter,
)
from .contracts import AxisResult, Demographics
from .photos import MAX_FILE_BYTES, PhotoCatalog, PhotoRejected
from .render import ETHNICITY_OPTIONS
from .sessions import SessionStore


class InputReadError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def read_regular_file(path: Path, limit: int) -> bytes:
    """Check and bound a single descriptor, including files that grow during read."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise InputReadError("input_not_regular")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise InputReadError("file_too_large")
        return content
    finally:
        if descriptor != -1:
            os.close(descriptor)


@dataclass(frozen=True)
class EvaluationCase:
    identity_id: str
    demographics: Demographics
    sets: tuple[tuple[Path, ...], tuple[Path, ...]]
    reviews: tuple[tuple[Any, ...], tuple[Any, ...]] = ((), ())
    profiles: tuple[tuple[Any, ...], tuple[Any, ...]] = ((), ())


@dataclass(frozen=True)
class EvaluationManifest:
    protocol: str
    dataset_name: str
    digest: str
    cases: tuple[EvaluationCase, ...]
    study_kind: str = "capture_retake"
    report_preferences: dict[str, str] | None = None


def _fields(value: Any, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError("manifest is missing required fields")
    if value.keys() - required - (optional or set()):
        raise ValueError("manifest contains unknown fields")
    return value


def _text(value: Any, maximum: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("manifest text is empty or too long")
    return value


def _demographics(value: Any) -> Demographics:
    fields = _fields(value, {"age", "sex", "ethnicities"})
    if not isinstance(fields["ethnicities"], list):
        raise ValueError("ethnicities must be a list")
    demographics = Demographics(
        fields["age"], fields["sex"], tuple(fields["ethnicities"])
    )
    allowed = {value for value, _ in ETHNICITY_OPTIONS}
    if not set(demographics.ethnicities) <= allowed:
        raise ValueError("unsupported ethnicity selection")
    return demographics


def load_manifest(path: Path) -> EvaluationManifest:
    """Read an explicit two-set protocol; never assign labels from images."""
    path = Path(path).resolve()
    raw = read_regular_file(path, 2 * 1024 * 1024)
    payload = _fields(
        json.loads(raw),
        {"version", "protocol", "dataset", "identities"},
        {"comparison_demographics", "study_kind", "report_preferences"},
    )
    if type(payload["version"]) is not int or payload["version"] not in {1, 2}:
        raise ValueError("unsupported manifest version")
    study_kind = payload.get("study_kind", "capture_retake")
    if not isinstance(study_kind, str) or study_kind not in {
        "capture_retake",
        "annotation_repeat",
    }:
        raise ValueError("unsupported study kind")
    if payload["version"] == 1 and "study_kind" in payload:
        raise ValueError("study kind requires manifest version2")
    selected_preferences = None
    if "report_preferences" in payload:
        from .consultation import preferences

        if payload["version"] != 2:
            raise ValueError("report preferences require manifest version2")
        selected_preferences = preferences(payload["report_preferences"])
    protocol = payload["protocol"]
    if protocol not in ("fixed_reference_stability", "labelled_repeatability"):
        raise ValueError("unsupported evaluation protocol")
    dataset = _fields(payload["dataset"], {"name", "source", "license_note"})
    for value in dataset.values():
        _text(value)
    identities = payload["identities"]
    if not isinstance(identities, list) or not 1 <= len(identities) <= 5000:
        raise ValueError("manifest requires 1 to 5000 identities")
    fixed = None
    if protocol == "fixed_reference_stability":
        fixed = _demographics(payload.get("comparison_demographics"))
    elif "comparison_demographics" in payload:
        raise ValueError(
            "labelled protocol requires each identity’s documented demographics"
        )
    cases = []
    seen = set()
    for raw_case in identities:
        case = _fields(
            raw_case,
            {"id", "sets"} if fixed else {"id", "sets", "demographics"},
            {"review_annotations", "profile_annotations"}
            if payload["version"] == 2
            else set(),
        )
        identity = _text(case["id"], 200)
        if identity in seen:
            raise ValueError("identity IDs must be unique")
        seen.add(identity)
        sets = case["sets"]
        if not isinstance(sets, list) or len(sets) != 2:
            raise ValueError("each identity needs exactly two independent photo sets")
        paths = []
        for photo_set in sets:
            if not isinstance(photo_set, list) or len(photo_set) > 200:
                raise ValueError("each photo set must contain at most 200 paths")
            resolved = []
            for item in photo_set:
                item = _text(item)
                image = (path.parent / item).resolve()
                if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    raise ValueError("evaluation accepts JPEG and PNG only")
                resolved.append(image)
            paths.append(tuple(resolved))
        reviews = ((), ())
        if "review_annotations" in case:
            from .review_annotations import parse_annotations

            reviews = parse_annotations(case["review_annotations"], sets)
        profiles = ((), ())
        if "profile_annotations" in case:
            from .review_annotations import parse_annotations

            profiles = parse_annotations(
                case["profile_annotations"], sets, profile=True
            )
        if study_kind == "annotation_repeat" and (
            any(len(p) != 1 for p in paths)
            or any(len(reviews[i]) + len(profiles[i]) < 1 for i in (0, 1))
        ):
            raise ValueError(
                "annotation-repeat needs one photo and one confirmed annotation per side"
            )
        cases.append(
            EvaluationCase(
                identity,
                fixed or _demographics(case["demographics"]),
                (paths[0], paths[1]),
                reviews,
                profiles,
            )
        )
    return EvaluationManifest(
        protocol,
        dataset["name"],
        hashlib.sha256(raw).hexdigest(),
        tuple(cases),
        study_kind,
        selected_preferences,
    )


def _duplicate_cases(
    cases: tuple[EvaluationCase, ...], *, annotation_repeat: bool = False
) -> set[int]:
    """Disallow exact-byte reuse anywhere, including across labelled identities."""
    owners: dict[str, int] = {}
    invalid = set()
    for index, case in enumerate(cases):
        for photo_set in case.sets:
            for path in photo_set:
                try:
                    digest = hashlib.sha256(
                        read_regular_file(path, MAX_FILE_BYTES)
                    ).hexdigest()
                except (OSError, InputReadError):
                    continue  # Missing/unreadable inputs remain in the coverage denominator.
                if digest in owners and (
                    owners[digest] != index or not annotation_repeat
                ):
                    invalid.update((owners[digest], index))
                owners[digest] = index
    return invalid


def _snapshot(result: AxisResult) -> dict[str, Any]:
    output = {
        name: getattr(result, name)
        for name in (
            "status",
            "headline_available",
            "value",
            "unit",
            "percentile",
            "interval",
            "photo_count",
            "error_code",
        )
    }
    # Never copy details: they can contain IDs, filenames or machine paths.
    if result.axis == "harmony":
        output["components"] = {
            name: {key: row[key] for key in ("value", "percentile", "interval")}
            for name, row in result.details.get("profile", {}).items()
        }
    return output


def analyze_evaluation_session(service, session_id: str, adapter: Any) -> dict:
    """Safe numerical snapshot of the real service, shared by perturbation studies."""
    attempted = None
    try:
        attempted = adapter.fingerprint()
        if not isinstance(attempted, str) or not attempted.strip():
            raise ValueError("invalid attempted instrument fingerprint")
        report = service.analyze(session_id)
    except AnalysisNotReady:
        return {
            "status": "intake_incomplete",
            "axes": {},
            "attempted_instrument_fingerprint": attempted,
        }
    except (Exception, SystemExit) as error:
        return {
            "status": "processing_error",
            "error_type": type(error).__name__,
            "axes": {},
            "attempted_instrument_fingerprint": attempted
            if isinstance(attempted, str) and attempted.strip()
            else None,
        }
    result = {
        "status": "analyzed",
        "cohort_id": report.cohort.used_id,
        "requested_cohort_id": report.cohort.requested_id,
        "fallback_level": report.cohort.fallback_level,
        "instrument_fingerprint": report.instrument_fingerprint,
        "attempted_instrument_fingerprint": attempted,
        "axes": {axis: _snapshot(value) for axis, value in report.axes.items()},
    }
    if hasattr(adapter, "routing_diagnostics"):
        result["routing"] = adapter.routing_diagnostics()
    return result


def _evaluate_case(
    case: EvaluationCase,
    store: SessionStore,
    adapter: Any,
    resolver: Any,
    *,
    skip_scoring: bool = False,
    consumer: bool = False,
    reference: str | None = None,
    annotation_repeat: bool = False,
    settings: dict[str, str] | None = None,
) -> dict:
    sessions = []
    records = []
    normalized_hashes = []
    accepted_entries = []
    try:
        for paths in case.sets:
            session = store.create(case.demographics)
            sessions.append(session)
            catalog = PhotoCatalog(store, session.session_id)
            rejected: Counter[str] = Counter()
            source_entries = {}
            for ordinal, path in enumerate(paths, 1):
                try:
                    source_entries[ordinal] = catalog.add_bytes(
                        path.name,
                        read_regular_file(path, catalog.limits.max_file_bytes),
                    )
                except PhotoRejected as error:
                    rejected[error.code] += 1
                except OSError:
                    rejected["input_unreadable"] += 1
                except InputReadError as error:
                    rejected[error.code] += 1
            accepted_entries.append(source_entries)
            hashes = [
                hashlib.sha256(
                    store.path_for(
                        session.session_id, "uploads", entry.stored_name
                    ).read_bytes()
                ).hexdigest()
                for entry in catalog.entries()
            ]
            normalized_hashes.extend(hashes)
            ready = catalog.readiness()
            records.append(
                {
                    "source_photos": len(paths),
                    "accepted_photos": ready.accepted,
                    "minimum_photos": ready.minimum,
                    "upload_rejections": dict(rejected),
                    "status": "pending",
                    "axes": {},
                }
            )
        if skip_scoring or (
            not annotation_repeat
            and len(set(normalized_hashes)) != len(normalized_hashes)
        ):
            return {
                "status": "duplicate_evidence",
                "sets": [],
                "_normalized_hashes": normalized_hashes,
            }
        if annotation_repeat and (
            len(normalized_hashes) != 2 or normalized_hashes[0] != normalized_hashes[1]
        ):
            return {
                "status": "annotation_image_mismatch",
                "sets": records,
                "_normalized_hashes": normalized_hashes,
            }
        service = ProgressiveAnalysisService(store, adapter=adapter, resolver=resolver)
        for side, (session, record) in enumerate(zip(sessions, records)):
            record.update(
                analyze_evaluation_session(service, session.session_id, adapter)
            )
            if consumer and record["status"] == "analyzed":
                from .consumer_evaluation import snapshot
                from .review_annotations import (
                    apply_annotations,
                    apply_profile_annotations,
                )

                reviewed, rejected_annotations = apply_annotations(
                    session, store, case.reviews[side], accepted_entries[side]
                )
                profiles, profile_rejections = apply_profile_annotations(
                    session, store, case.profiles[side], accepted_entries[side]
                )
                record["profile_annotation_rejections"] = profile_rejections
                record["annotation_rejections"] = rejected_annotations
                record["consumer"] = snapshot(
                    service.load_report(session.session_id),
                    reference=reference,
                    reviewed=reviewed,
                    profiles=profiles,
                    settings=settings,
                )
        return {
            "status": "evaluated",
            "sets": records,
            "_normalized_hashes": normalized_hashes,
        }
    finally:
        for session in sessions:
            store.end(session.session_id)


def _stats(values: list[float]) -> dict:
    return {
        "n_identities": len(values),
        "median": float(np.median(values)) if values else None,
        "p90": float(np.percentile(values, 90)) if values else None,
    }


def summarize(rows: list[dict]) -> dict:
    """One independent pair per identity; failures stay in every denominator."""
    output: dict[str, Any] = {
        "input_identities": len(rows),
        "all_four_measured_pairs": 0,
        "all_four_scalar_pairs": 0,
        "axes": {},
    }
    comparable = []
    for row in rows:
        sets = row["sets"]
        match = (
            row["status"] == "evaluated"
            and len(sets) == 2
            and all(s["status"] == "analyzed" for s in sets)
        )
        if match:
            match = sets[0]["cohort_id"] == sets[1]["cohort_id"] and (
                sets[0].get("instrument_fingerprint")
                == sets[1].get("instrument_fingerprint")
            )
        comparable.append(match)
        if match and all(
            s["axes"][a]["status"] == "measured" for s in sets for a in AXIS_ORDER
        ):
            output["all_four_measured_pairs"] += 1
            if all(
                s["axes"][a]["headline_available"] for s in sets for a in AXIS_ORDER
            ):
                output["all_four_scalar_pairs"] += 1
    for axis in AXIS_ORDER:
        status_counts: Counter[str] = Counter()
        percentile_moves, value_moves, centers = [], [], []
        units = set()
        components: dict[str, list[float]] = defaultdict(list)
        paired = 0
        for row, match in zip(rows, comparable):
            sets = row["sets"]
            for side in range(2):
                item = sets[side] if len(sets) == 2 else {}
                status_counts[
                    item.get("axes", {})
                    .get(axis, {})
                    .get("status", item.get("status", row["status"]))
                ] += 1
            if not match:
                continue
            a, b = (s["axes"][axis] for s in sets)
            if a["status"] != "measured" or b["status"] != "measured":
                continue
            paired += 1
            if (
                a["headline_available"]
                and b["headline_available"]
                and a["unit"] == b["unit"]
            ):
                percentile_moves.append(abs(a["percentile"] - b["percentile"]))
                value_moves.append(abs(a["value"] - b["value"]))
                centers.append((a["percentile"] + b["percentile"]) / 2)
                units.add(a["unit"])
            for name in a.get("components", {}).keys() & b.get("components", {}).keys():
                components[name].append(
                    abs(
                        a["components"][name]["percentile"]
                        - b["components"][name]["percentile"]
                    )
                )
        output["axes"][axis] = {
            "paired_measured": paired,
            "paired_coverage": paired / len(rows) if rows else 0.0,
            "set_status_counts": dict(status_counts),
            "percentile_movement": _stats(percentile_moves),
            "value_movement": _stats(value_moves) if len(units) <= 1 else _stats([]),
            "value_unit": next(iter(units)) if len(units) == 1 else None,
            "between_identity_percentile_sd": (
                float(np.std(centers, ddof=1)) if len(centers) >= 2 else None
            ),
            "components": {
                name: {"percentile_movement": _stats(values)}
                for name, values in sorted(components.items())
            },
        }
    return output


def run_evaluation(
    manifest: EvaluationManifest,
    *,
    adapter_factory: Callable[[], Any] = SetMeasurementAdapter,
    resolver: Any = None,
    progress: Callable[[int, int], None] | None = None,
    consumer: bool = False,
) -> dict:
    if not consumer and (
        manifest.study_kind == "annotation_repeat"
        or any(any(c.reviews) or any(c.profiles) for c in manifest.cases)
        or manifest.report_preferences is not None
    ):
        raise ValueError("review annotations require consumer evaluation mode")
    reference = None
    if consumer:
        from .consumer_evaluation import reference_fingerprint

        reference = reference_fingerprint()
    invalid = _duplicate_cases(
        manifest.cases, annotation_repeat=manifest.study_kind == "annotation_repeat"
    )
    rows = []
    normalized_owners: dict[str, int] = {}
    expected_instrument = None
    with tempfile.TemporaryDirectory(prefix="facial-analysis-evaluation-") as temporary:
        store = SessionStore(Path(temporary) / "sessions")
        for index, case in enumerate(manifest.cases):
            # Even raw-invalid cases must contribute normalized hashes: otherwise
            # a third metadata-distinct copy could escape the duplicate index.
            row = _evaluate_case(
                case,
                store,
                adapter_factory(),
                resolver,
                skip_scoring=index in invalid,
                consumer=consumer,
                reference=reference,
                annotation_repeat=manifest.study_kind == "annotation_repeat",
                settings=manifest.report_preferences,
            )
            normalized = row.pop("_normalized_hashes", ())
            for digest in normalized:
                owner = normalized_owners.get(digest)
                if owner is not None and owner != index:
                    rows[owner].update(status="duplicate_evidence", sets=[])
                    row.update(status="duplicate_evidence", sets=[])
                normalized_owners[digest] = index
            fingerprints = {
                s["instrument_fingerprint"]
                for s in row["sets"]
                if s["status"] == "analyzed"
            }
            if expected_instrument is None and fingerprints:
                expected_instrument = next(
                    s["instrument_fingerprint"]
                    for s in row["sets"]
                    if s["status"] == "analyzed"
                )
            if fingerprints - {expected_instrument}:
                row["status"] = "instrument_changed"
            rows.append({"index": index + 1, **row})
            if progress:
                progress(index + 1, len(manifest.cases))
        if store.has_owned_sessions or any(store.base_dir.iterdir()):
            raise RuntimeError("evaluation temporary sessions were not deleted")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["sets"][0].get("cohort_id", "unresolved")
            if row["sets"]
            else "unresolved"
        )
        groups[key].append(row)
    import bank_instrument

    consumer_data = {}
    if consumer:
        from .consumer_evaluation import reference_fingerprint, summarize_consumer

        if reference_fingerprint() != reference:
            for row in rows:
                if row["status"] == "evaluated":
                    row["status"] = "reference_changed"
        consumer_data = {
            "consumer_version": 1,
            "study_kind": manifest.study_kind,
            "consumer_summary": summarize_consumer(
                rows, study_kind=manifest.study_kind
            ),
        }

    def applicable_summary(values):
        if manifest.study_kind == "annotation_repeat":
            return {
                "input_identities": len(values),
                "automatic_movement_applicable": False,
                "all_four_measured_pairs": None,
                "all_four_scalar_pairs": None,
                "axes": {},
            }
        return {**summarize(values), "automatic_movement_applicable": True}

    return {
        **consumer_data,
        "version": 2 if consumer else 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": manifest.protocol,
        "manifest_digest": manifest.digest,
        "correctness_validated": False,
        "release_gate_modified": False,
        "bank_identity_overlap_verified": False,
        "near_duplicate_detection": False,
        "temporary_sessions_deleted": True,
        "expected_app_instrument": expected_instrument,
        "instrument_mismatch_identities": sum(
            r["status"] == "instrument_changed" for r in rows
        ),
        "execution_fingerprint": bank_instrument.fingerprint(),
        "summary": applicable_summary(rows),
        "cohorts": {
            key: applicable_summary(values) for key, values in sorted(groups.items())
        },
        "identities": rows,
    }
