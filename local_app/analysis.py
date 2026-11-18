"""Persistent, progressive orchestration for the four independent face axes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from .cohorts import NoCompatibleCohort
from .contracts import AxisResult, CohortUse, ReferenceAvailability
from .photos import PhotoCatalog, PhotoEntry
from .sessions import SessionStore

AXIS_ORDER = ("harmony", "dimorphism", "symmetry", "averageness")
ANALYSIS_STATE_KEY = "analysis_report"
ANALYSIS_STATE_VERSION = 1


class AnalysisNotReady(RuntimeError):
    """The session has no accepted photo to analyze."""


class AnalysisStateError(ValueError):
    """Persisted analysis state or an adapter response violates its contract."""


@dataclass(frozen=True)
class PhotoInput:
    entry: PhotoEntry
    path: Path


class CohortResolverLike(Protocol):
    def resolve(self, demographics) -> CohortUse: ...


class MeasurementAdapter(Protocol):
    def fingerprint(self) -> str: ...

    def route(
        self,
        photos: tuple[PhotoInput, ...],
        cohort: CohortUse | None,
    ) -> dict[str, tuple[str, ...]]: ...

    def measure_axis(
        self,
        axis: str,
        photos: tuple[PhotoInput, ...],
        cohort: CohortUse | None,
    ) -> AxisResult: ...


@dataclass(frozen=True)
class AnalysisReport:
    session_id: str
    cohort: CohortUse | None
    axes: dict[str, AxisResult]
    accepted_photos: int
    instrument_fingerprint: str
    photo_fingerprint: str
    reference: ReferenceAvailability = field(
        default_factory=ReferenceAvailability.not_requested
    )

    def __post_init__(self) -> None:
        if set(self.axes) != set(AXIS_ORDER):
            raise ValueError("an analysis report requires exactly four named axes")
        if self.accepted_photos < 0:
            raise ValueError("accepted photo count cannot be negative")
        if not self.instrument_fingerprint or not self.photo_fingerprint:
            raise ValueError("analysis fingerprints cannot be empty")
        if self.cohort is not None and self.reference.status == "not_requested":
            object.__setattr__(
                self, "reference", ReferenceAvailability.available(self.cohort)
            )
        if self.reference.status == "available" and self.cohort is None:
            raise ValueError("available references require a cohort")
        if self.reference.status != "available" and self.cohort is not None:
            raise ValueError("unavailable references cannot carry a cohort")
        if any(
            r.status in {"measured", "estimated"} and r.cohort != self.cohort
            for r in self.axes.values()
        ):
            raise ValueError("Axis comparison references must match the report.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "cohort": self.cohort.to_dict() if self.cohort is not None else None,
            "reference": self.reference.to_dict(),
            "axes": {axis: self.axes[axis].to_dict() for axis in AXIS_ORDER},
            "accepted_photos": self.accepted_photos,
            "instrument_fingerprint": self.instrument_fingerprint,
            "photo_fingerprint": self.photo_fingerprint,
        }


class ProgressiveAnalysisService:
    """Cache by evidence and instrument while allowing each axis to advance alone."""

    def __init__(
        self,
        store: SessionStore,
        *,
        adapter: MeasurementAdapter | None = None,
        resolver: CohortResolverLike | None = None,
    ) -> None:
        self.store = store
        self.adapter = adapter or SetMeasurementAdapter()
        self.resolver = resolver
        self._lock = RLock()

    def analyze(self, session_id: str, *, force: bool = False) -> AnalysisReport:
        with self._lock:
            return self._analyze(session_id, force=force)

    def _analyze(self, session_id: str, *, force: bool) -> AnalysisReport:
        session = self.store.load(session_id)
        catalog = PhotoCatalog(self.store, session_id)
        readiness = catalog.readiness()
        if not readiness.ready:
            raise AnalysisNotReady(readiness.message)

        entries = catalog.entries()
        cohort, reference = self._resolve_reference(session.demographics)
        if cohort is not None and not isinstance(cohort, CohortUse):
            raise AnalysisStateError("cohort resolver did not return a CohortUse")
        instrument = self.adapter.fingerprint()
        if not isinstance(instrument, str) or not instrument.strip():
            raise AnalysisStateError("measurement adapter fingerprint is invalid")
        photo_fingerprint = self._photo_fingerprint(entries)
        previous = self._previous_state(session.state)
        configure = getattr(self.adapter, "set_forehead_overrides", None)
        overrides = {}
        if callable(configure):
            from .frontal_routes import frontal_payloads

            for review in frontal_payloads(session, self.store):
                status = review.get("forehead_visibility")
                if status in {"forehead_hidden", "uncertain"} or (
                    status in {"visible", "hairline_hidden"}
                    and not review.get("visibility_only")
                ):
                    overrides[review["evidence_photo_id"]] = status
            configure(overrides)
        visibility_fingerprint = hashlib.sha256(
            json.dumps(overrides, sort_keys=True).encode()
        ).hexdigest()
        if (
            previous is not None
            and previous.get("visibility_fingerprint") != visibility_fingerprint
        ):
            force = True

        if (
            not force
            and previous is not None
            and previous.get("instrument_fingerprint") == instrument
            and previous.get("photo_fingerprint") == photo_fingerprint
            and (
                previous.get("cohort") == cohort.to_dict()
                if cohort is not None
                else previous.get("cohort") is None
            )
            and previous.get("reference") == reference.to_dict()
        ):
            report = self._report_from_state(session_id, previous)
            report = self._with_profile_controls(report, session, entries)
            session.state[ANALYSIS_STATE_KEY]["axes"] = report.to_dict()["axes"]
            self.store.save(session)
            return report

        photos = tuple(
            PhotoInput(
                entry=entry,
                path=self.store.path_for(session_id, "uploads", entry.stored_name),
            )
            for entry in entries
        )
        routed = self.adapter.route(photos, cohort)
        routed_inputs = self._validate_routes(routed, photos)

        previous_axes: dict[str, AxisResult] = {}
        previous_evidence: dict[str, tuple[str, ...]] = {}
        reusable_previous = (
            not force
            and previous is not None
            and previous.get("instrument_fingerprint") == instrument
            and (
                previous.get("cohort") == cohort.to_dict()
                if cohort is not None
                else previous.get("cohort") is None
            )
            and previous.get("reference") == reference.to_dict()
        )
        if reusable_previous:
            previous_report = self._report_from_state(session_id, previous)
            previous_axes = previous_report.axes
            raw_evidence = previous.get("eligible_hashes", {})
            if not isinstance(raw_evidence, dict):
                raise AnalysisStateError("persisted eligible evidence is invalid")
            previous_evidence = {
                axis: tuple(raw_evidence.get(axis, ())) for axis in AXIS_ORDER
            }

        axes: dict[str, AxisResult] = {}
        eligible_hashes: dict[str, tuple[str, ...]] = {}
        for axis in AXIS_ORDER:
            axis_photos = routed_inputs[axis]
            hashes = tuple(photo.entry.sha256 for photo in axis_photos)
            eligible_hashes[axis] = hashes
            if axis in previous_axes and previous_evidence.get(axis) == hashes:
                refresh = getattr(self.adapter, "refresh_capture_diagnostics", None)
                axes[axis] = (
                    refresh(previous_axes[axis])
                    if callable(refresh)
                    else previous_axes[axis]
                )
                continue

            result = self.adapter.measure_axis(axis, axis_photos, cohort)
            self._validate_axis_result(axis, cohort, result)
            axes[axis] = result

        report = AnalysisReport(
            session_id=session_id,
            cohort=cohort,
            axes=axes,
            accepted_photos=len(entries),
            instrument_fingerprint=instrument,
            photo_fingerprint=photo_fingerprint,
            reference=reference,
        )
        report = self._with_profile_controls(report, session, entries)
        session.state[ANALYSIS_STATE_KEY] = {
            "version": ANALYSIS_STATE_VERSION,
            **report.to_dict(),
            "visibility_fingerprint": visibility_fingerprint,
            "eligible_hashes": {
                axis: list(eligible_hashes[axis]) for axis in AXIS_ORDER
            },
        }
        self.store.save(session)
        return report

    def _with_profile_controls(self, report, session, entries):
        """Resolve review/removal intent without altering the cached axis arithmetic."""
        from .profile_routes import _binding
        from .frontal_routes import frontal_payloads

        controls = {}
        records = session.state.get("profile_reviews", {})
        for entry in entries:
            raw = records.get(entry.photo_id)
            if not isinstance(raw, dict) or raw.get("binding") != _binding(
                session, self.store, entry
            ):
                continue
            if raw.get("reviewed"):
                controls[entry.photo_id] = "reviewed"
            elif raw.get("automatic_suppressed") or raw.get("supersedes_legacy"):
                controls[entry.photo_id] = "suppressed"
            elif raw.get("restored_hints"):
                controls[entry.photo_id] = "restored_hints"
        legacy = session.state.get("profile_review", {})
        legacy_id = legacy.get("photo_id")
        if legacy_id in {entry.photo_id for entry in entries} and not records.get(
            legacy_id, {}
        ).get("supersedes_legacy"):
            controls[legacy_id] = "legacy_review"
        for raw in frontal_payloads(session, self.store):
            controls[raw["evidence_photo_id"]] = "frontal_review"
        harmony = report.axes["harmony"]
        harmony = replace(
            harmony, details={**harmony.details, "profile_controls": controls}
        )
        return replace(report, axes={**report.axes, "harmony": harmony})

    def _resolve_reference(
        self, demographics
    ) -> tuple[CohortUse | None, ReferenceAvailability]:
        if demographics is None:
            return None, ReferenceAvailability.not_requested()
        if self.resolver is None:
            self.resolver = self._default_resolver()
        try:
            cohort = self.resolver.resolve(demographics)
        except NoCompatibleCohort as exc:
            requested_label = (
                f"{demographics.sex}, {', '.join(demographics.ethnicities)}, "
                f"age {demographics.age}"
            )
            return None, ReferenceAvailability.unavailable(
                reason=str(exc),
                requested_label=requested_label,
            )
        return cohort, ReferenceAvailability.available(cohort)

    def load_report(self, session_id: str) -> AnalysisReport | None:
        session = self.store.load(session_id)
        previous = self._previous_state(session.state)
        if previous is None:
            return None
        return self._report_from_state(session_id, previous)

    @staticmethod
    def _default_resolver():
        import bank_reference as bank

        from .cohorts import CohortResolver

        return CohortResolver(
            bank.cohort_inventory(),
            floor=int(bank.MIN_COHORT_N),
            allow_virtual_pools=False,
        )

    @staticmethod
    def _photo_fingerprint(entries: tuple[PhotoEntry, ...]) -> str:
        digest = hashlib.sha256()
        for value in sorted(entry.sha256 for entry in entries):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _previous_state(state: dict[str, Any]) -> dict[str, Any] | None:
        payload = state.get(ANALYSIS_STATE_KEY)
        if payload is None:
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != ANALYSIS_STATE_VERSION
        ):
            raise AnalysisStateError("persisted analysis state is invalid")
        return payload

    @staticmethod
    def _validate_routes(
        routed: dict[str, tuple[str, ...]],
        photos: tuple[PhotoInput, ...],
    ) -> dict[str, tuple[PhotoInput, ...]]:
        if not isinstance(routed, dict) or set(routed) != set(AXIS_ORDER):
            raise AnalysisStateError("measurement router must return exactly four axes")
        by_id = {photo.entry.photo_id: photo for photo in photos}
        normalized: dict[str, tuple[PhotoInput, ...]] = {}
        for axis in AXIS_ORDER:
            raw_ids = routed[axis]
            if isinstance(raw_ids, str):
                raise AnalysisStateError(
                    f"{axis} route must be a sequence of photo ids"
                )
            photo_ids = tuple(raw_ids)
            if len(photo_ids) != len(set(photo_ids)):
                raise AnalysisStateError(f"{axis} route contains duplicate photo ids")
            if any(photo_id not in by_id for photo_id in photo_ids):
                raise AnalysisStateError(f"{axis} route contains an unknown photo id")
            normalized[axis] = tuple(by_id[photo_id] for photo_id in photo_ids)
        return normalized

    @staticmethod
    def _validate_axis_result(
        axis: str,
        cohort: CohortUse,
        result: AxisResult,
    ) -> None:
        if not isinstance(result, AxisResult):
            raise AnalysisStateError(f"{axis} evaluator did not return an AxisResult")
        if result.axis != axis:
            raise AnalysisStateError(f"{axis} evaluator returned the wrong axis name")
        if result.cohort != cohort:
            raise AnalysisStateError(f"{axis} evaluator returned the wrong cohort")

    @staticmethod
    def _report_from_state(session_id: str, payload: dict[str, Any]) -> AnalysisReport:
        raw_axes = payload.get("axes")
        if not isinstance(raw_axes, dict) or set(raw_axes) != set(AXIS_ORDER):
            raise AnalysisStateError("persisted report axes are invalid")
        try:
            raw_cohort = payload.get("cohort")
            cohort = CohortUse.from_dict(raw_cohort) if raw_cohort is not None else None
            reference = ReferenceAvailability.from_dict(
                payload.get("reference")
                if "reference" in payload
                else (
                    ReferenceAvailability.available(cohort).to_dict()
                    if cohort is not None
                    else ReferenceAvailability.not_requested().to_dict()
                )
            )
            axes = {axis: AxisResult.from_dict(raw_axes[axis]) for axis in AXIS_ORDER}
            return AnalysisReport(
                session_id=session_id,
                cohort=cohort,
                axes=axes,
                accepted_photos=int(payload.get("accepted_photos")),
                instrument_fingerprint=str(payload.get("instrument_fingerprint")),
                photo_fingerprint=str(payload.get("photo_fingerprint")),
                reference=reference,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise AnalysisStateError("persisted analysis report is invalid") from exc


class SetMeasurementAdapter:
    """Bridge the persistent service to the existing local MediaPipe evaluators."""

    _UNRANKED_COHORT = CohortUse(
        requested_id="reference_unavailable",
        used_id="reference_unavailable",
        fallback_level="unavailable",
        sample_size=0,
        requested_label="No comparison reference",
        used_label="No comparison reference",
    )

    def __init__(self) -> None:
        self._contexts: dict[tuple[str, str, tuple[str, ...]], Any] = {}
        self._primary_routes: dict[
            tuple[str, str, tuple[str, ...]], tuple[str, ...]
        ] = {}
        self._photo_facts: tuple[Any, ...] = ()
        self._automatic_forehead: list[dict[str, Any]] = []
        self._forehead_overrides: dict[str, str] = {}
        self._profile_proposals: list[dict[str, Any]] = []

    def set_forehead_overrides(self, overrides: dict[str, str]) -> None:
        if any(
            not isinstance(photo, str)
            or status
            not in {"visible", "hairline_hidden", "forehead_hidden", "uncertain"}
            for photo, status in overrides.items()
        ):
            raise ValueError("Unsupported forehead visibility correction.")
        self._forehead_overrides = dict(overrides)

    def _clear_forehead_ids(self) -> set[str]:
        from .automatic_forehead import coverage_exclusions
        from types import SimpleNamespace

        evidence = SimpleNamespace(
            details={
                "automatic_forehead": self._automatic_forehead,
                "forehead_overrides": self._forehead_overrides,
            }
        )
        excluded = coverage_exclusions({"harmony": evidence})
        return {
            row["photo_id"]
            for row in self._automatic_forehead
            if row["photo_id"] not in excluded
        }

    def routing_diagnostics(self) -> dict[str, Any]:
        """Aggregate the last route's actual gates without exporting photo IDs."""
        from collections import Counter

        return {
            axis: {
                "submitted_photos": len(self._photo_facts),
                "detected_photos": sum(fact.detected for fact in self._photo_facts),
                "eligible_photos": sum(
                    fact.eligible_for(axis) for fact in self._photo_facts
                ),
                "failure_counts": dict(
                    Counter(
                        reason
                        for fact in self._photo_facts
                        for reason in fact.failures_for(axis)
                    )
                ),
            }
            for axis in AXIS_ORDER
        }

    def refresh_capture_diagnostics(self, result: AxisResult) -> AxisResult:
        """Refresh selection explanations even when the numerical result is cached."""
        return replace(
            result,
            details={
                **result.details,
                **(
                    {"automatic_forehead": getattr(self, "_automatic_forehead", [])}
                    if result.axis == "harmony"
                    else {}
                ),
                **(
                    {"forehead_overrides": getattr(self, "_forehead_overrides", {})}
                    if result.axis == "harmony"
                    else {}
                ),
                **(
                    {"profile_proposals": self._profile_proposals}
                    if result.axis == "harmony"
                    else {}
                ),
                **(
                    {"appearance_sources": getattr(self, "_appearance_sources", [])}
                    if result.axis == "harmony"
                    else {}
                ),
                "capture_diagnostics": {
                    **self.routing_diagnostics()[result.axis],
                    "selected_axis_photos": result.photo_count,
                },
            },
        )

    def fingerprint(self) -> str:
        root = Path(__file__).resolve().parent.parent
        artifacts = (
            root / "local_app" / "analysis.py",
            root / "local_app" / "contracts.py",
            root / "local_app" / "cohorts.py",
            root / "local_app" / "estimates.py",
            root / "local_app" / "regional_evidence.py",
            root / "local_app" / "appearance.py",
            root / "local_app" / "anatomical_observations.py",
            root / "local_app" / "automatic_observations.py",
            root / "local_app" / "observations.py",
            root / "local_app" / "reading_registry.py",
            root / "local_app" / "profile_proposals.py",
            root / "local_app" / "profile_detection.py",
            root / "local_app" / "expanded_measurements.py",
            root / "local_app" / "automatic_forehead.py",
            root / "forehead_context.bin",
            root / "macos_app" / "config.py",
            root / "fa_common.py",
            root / "face_landmarker.task",
            root / "local_app" / "guidance.py",
            root / "capture_check.py",
            root / "averageness_band.py",
            root / "local_app" / "product_report.py",
            root / "local_app" / "illustrations.py",
            root / "bank_derive.py",
            root / "public_reference.py",
            root / "bank" / "public_reference.json",
            root / "facial_catalogue.py",
            root / "assess_set.py",
            root / "axis_stability.json",
            root / "averageness_set.json",
            root / "symmetry_yaw_gate.json",
            root / "scut_harmony.json",
            root / "bank" / "harmony_model.json",
            root / "bank" / "provenance.json",
            root / "bank" / "dimorphism_calibration.json",
            root / "bank" / "derived.npz",
        )
        digest = hashlib.sha256()
        for path in artifacts:
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
            if not path.exists():
                digest.update(b"missing\0")
                continue
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()

    def route(
        self,
        photos: tuple[PhotoInput, ...],
        cohort: CohortUse | None,
    ) -> dict[str, tuple[str, ...]]:
        import assess_set
        from .guidance import PhotoFact

        by_name = {photo.entry.stored_name: photo for photo in photos}
        self._source_photos = photos
        self._contexts.clear()
        self._primary_routes.clear()
        kept, rejected = assess_set.gate_photos(
            [str(photo.path) for photo in photos], retain_estimates=True
        )
        facts: list[PhotoFact] = []
        normalized_kept: list[dict] = []
        for record in kept:
            photo = by_name[record["photo"]]
            raw = record["fact"]
            facts.append(
                PhotoFact(
                    photo_id=photo.entry.photo_id,
                    yaw=raw.get("yaw"),
                    pitch=raw.get("pitch"),
                    roll=raw.get("roll"),
                    ipd_px=raw.get("ipd_px"),
                    neutral=bool(raw.get("neutral")),
                    detected=bool(raw.get("detected")),
                )
            )
            normalized_kept.append({**record, "photo": photo.entry.photo_id})
        for record in rejected:
            photo = by_name[record["photo"]]
            raw = record["fact"]
            facts.append(
                PhotoFact(
                    photo_id=photo.entry.photo_id,
                    yaw=raw.get("yaw"),
                    pitch=raw.get("pitch"),
                    roll=raw.get("roll"),
                    ipd_px=raw.get("ipd_px"),
                    neutral=bool(raw.get("neutral")),
                    detected=bool(raw.get("detected")),
                )
            )
            if record.get("landmarks") is not None:
                normalized_kept.append({**record, "photo": photo.entry.photo_id})

        values = [
            assess_set.BD.metrics_for(record["landmarks"]) for record in normalized_kept
        ]
        self._photo_facts = tuple(facts)
        from .appearance import observe_appearance

        self._appearance_sources = observe_appearance(photos, tuple(normalized_kept))
        from .automatic_forehead import observe_photos

        self._automatic_forehead = observe_photos(photos, normalized_kept)
        from .profile_proposals import propose_profile
        from .profile_detection import profile_detector_inputs

        model = Path(__file__).resolve().parent.parent / "face_landmarker.task"
        model_hash = (
            hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else None
        )
        detected = {r["photo"]: r for r in normalized_kept}
        self._profile_proposals = []
        for number, photo in enumerate(photos, 1):
            raw = detected.get(photo.entry.photo_id, {})
            try:
                source_hash = hashlib.sha256(photo.path.read_bytes()).hexdigest()
            except OSError:
                source_hash = None
            points, pose, normalization = profile_detector_inputs(
                photo.path, raw, photo.entry.width, photo.entry.height
            )
            proposal = propose_profile(
                points if source_hash else None,
                width=photo.entry.width,
                height=photo.entry.height,
                photo_id=photo.entry.photo_id,
                photo_number=number,
                yaw=pose["yaw"],
                pitch=pose["pitch"],
                roll=pose["roll"],
                model_sha256=model_hash,
            )
            self._profile_proposals.append(
                {
                    **proposal,
                    "source_sha256": source_hash,
                    "inference_orientation": normalization,
                }
            )
        context_cohort = cohort or self._UNRANKED_COHORT
        context = assess_set.AssessmentContext(
            cohort=context_cohort,
            kept=tuple(normalized_kept),
            values=tuple(values),
            photo_facts=tuple(facts),
        )
        routes: dict[str, tuple[str, ...]] = {}
        for axis in AXIS_ORDER:
            photo_ids = tuple(context.evidence_ids(axis))
            if not photo_ids:
                # Low detail prevents a measured result, but does not justify
                # discarding neutral, pose-compatible repeat observations.
                available = {record["photo"] for record in normalized_kept}
                photo_ids = tuple(
                    fact.photo_id
                    for fact in facts
                    if fact.photo_id in available
                    and fact.failures_for(axis) == ("too_small",)
                )
            if not photo_ids and normalized_kept:
                # Prefer the least compromised real observation; never fill missing
                # anatomy with the reference mean or average it into cleaner photos.
                available = {record["photo"] for record in normalized_kept}
                candidates = [fact for fact in facts if fact.photo_id in available]
                best = min(
                    candidates,
                    key=lambda fact: (
                        len(fact.failures_for(axis)),
                        abs(fact.yaw or 0),
                        abs(fact.pitch or 0),
                        fact.photo_id,
                    ),
                )
                photo_ids = (best.photo_id,)
            primary_ids = photo_ids
            if axis == "harmony":
                from .regional_evidence import metric_evidence

                regional_ids = tuple(
                    p["photo"]
                    for p, _ in metric_evidence(context, primary_ids, "eye_spacing")
                )
                photo_ids = tuple(dict.fromkeys((*primary_ids, *regional_ids)))
            routes[axis] = photo_ids
            key = (axis, context_cohort.used_id, photo_ids)
            self._contexts[key] = context
            self._primary_routes[key] = primary_ids
        return routes

    def measure_axis(
        self,
        axis: str,
        photos: tuple[PhotoInput, ...],
        cohort: CohortUse | None,
    ) -> AxisResult:
        import assess_set

        photo_ids = tuple(photo.entry.photo_id for photo in photos)
        context_cohort = cohort or self._UNRANKED_COHORT
        key = (axis, context_cohort.used_id, photo_ids)
        context = self._contexts.get(key)
        if context is None:
            raise RuntimeError("measurement route must run before an axis evaluator")
        primary_ids = self._primary_routes.get(key, photo_ids)
        evaluators = {
            "harmony": assess_set.assess_harmony_axis,
            "dimorphism": assess_set.assess_dimorphism_axis,
            "symmetry": assess_set.assess_symmetry_axis,
            "averageness": assess_set.assess_averageness_axis,
        }
        try:
            from .estimates import estimate_axis, regional_measurements
            from .illustrations import observed_diagram

            if cohort is None:
                if axis in {"harmony", "symmetry"} and not photo_ids:
                    result = replace(evaluators[axis](context), cohort=None)
                elif axis in {"harmony", "symmetry"}:
                    result = replace(
                        estimate_axis(
                            axis, context, primary_ids, reference_available=False
                        ),
                        cohort=None,
                    )
                else:
                    return AxisResult.unavailable(
                        axis=axis,
                        summary=(
                            f"{axis.capitalize()} requires a comparison reference. "
                            "The available geometry is still shown in the report."
                        ),
                        reason_code="reference_unavailable",
                        photo_count=len(photos),
                        details={"evidence_photo_ids": list(photo_ids)},
                    )
            else:
                try:
                    result = evaluators[axis](context)
                except assess_set.AxisDataError:
                    if axis not in {"harmony", "symmetry"} or not photo_ids:
                        raise
                    # A missing calibration must not erase a directly observed ratio.
                    result = estimate_axis(axis, context, primary_ids)
            if result.status == "needs_photo" and photo_ids:
                result = replace(
                    estimate_axis(axis, context, primary_ids),
                    cohort=cohort,
                )
            if axis == "harmony" and result.status in {"measured", "estimated"}:
                from .anatomical_observations import observed_iris_rows

                result = replace(
                    result,
                    details={
                        **result.details,
                        "regional_measurements": (
                            regional_measurements(
                                context,
                                primary_ids,
                                forehead_clear_ids=self._clear_forehead_ids(),
                                reference_available=cohort is not None,
                            )
                            + observed_iris_rows(
                                context,
                                primary_ids,
                                {photo.entry.photo_id: photo.path for photo in photos},
                                forehead_clear_ids=self._clear_forehead_ids(),
                            )
                        ),
                        "illustration": observed_diagram(context, primary_ids),
                    },
                )
            if axis == "harmony":
                from .automatic_observations import automatic_observations
                from .expanded_measurements import expanded_frontal_observations

                records = automatic_observations(
                    result.details.get("regional_measurements", []),
                    context,
                    getattr(self, "_source_photos", photos),
                    forehead_statuses={
                        r["photo_id"]: r["status"]
                        for r in getattr(self, "_automatic_forehead", [])
                    },
                    forehead_overrides=getattr(self, "_forehead_overrides", {}),
                )
                model = Path(__file__).resolve().parent.parent / "face_landmarker.task"
                expanded = expanded_frontal_observations(
                    context,
                    getattr(self, "_source_photos", photos),
                    model_sha256=(
                        hashlib.sha256(model.read_bytes()).hexdigest()
                        if model.is_file()
                        else None
                    ),
                )
                result = replace(
                    result,
                    details={
                        **result.details,
                        "observations": [r.model_dump(mode="json") for r in records],
                        "expanded_observations": [
                            r.model_dump(mode="json") for r in expanded
                        ],
                    },
                )
            return self.refresh_capture_diagnostics(result)
        except assess_set.AxisDataError as exc:
            return self.refresh_capture_diagnostics(
                AxisResult.processing_error(
                    axis=axis,
                    error_code=exc.code,
                    error_message=exc.public_message,
                    retryable=exc.retryable,
                    cohort=cohort,
                    photo_count=len(photos),
                    details={"evidence_photo_ids": list(photo_ids)},
                )
            )
