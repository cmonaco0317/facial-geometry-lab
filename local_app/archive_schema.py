"""Strict, photo-free numerical snapshots and conservative comparisons."""

from __future__ import annotations
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .product_report import public_regions
from .score_audit import audit, contract
from .observations import Observation, public_observations
from .current_observations import archive_observations
from .product_report import current_composite, current_score_axes
from .profile_lineage import ProposalOrigin
from .history_context import SessionContext

Text = Annotated[str, Field(max_length=512)]
Short = Annotated[str, Field(max_length=120)]
Number = Annotated[float, Field(allow_inf_nan=False)]


class Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, frozen=True
    )


class Reading(Strict):
    id: Short
    label: Text
    method: Short
    contract: Short = ""
    unit: Short
    value: Number
    side: Literal["", "left", "right"] = ""
    photo: int = Field(default=0, ge=0, le=10000)


class Component(Strict):
    id: Literal["harmony", "symmetry", "averageness"]
    value: Number = Field(ge=0, le=100)
    effective_weight: Number = Field(ge=0, le=1)
    status: Literal["measured", "estimated"]


class Idea(Strict):
    id: Short
    title: Text
    action: Text


class EnteredDemographics(Strict):
    age: int = Field(ge=18, le=150)
    sex: Literal["male", "female", "another", "intersex", "prefer_not_to_say"]
    ethnicities: list[Short] = Field(min_length=1, max_length=20)


class Snapshot(Strict):
    version: Literal[1, 2] = 2
    demographics: EnteredDemographics | None = None
    adult_confirmed: bool = False
    created_at: Short
    profile: Short = Field(min_length=1)
    captured_at: Short = ""
    protocol: Literal["unspecified", "casual", "controlled", "mixed"] = "unspecified"
    camera: Short = ""
    formula: Short
    formula_contract: Short
    reference: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    cohort: Short
    fallback: Short
    instrument: Short
    input_digest: Short
    photo_count: int = Field(ge=0, le=10000)
    routes: list[Short] = Field(max_length=4)
    harmony_ids: list[Short] = Field(max_length=100)
    score: Number | None = Field(default=None, ge=0, le=100)
    components: list[Component] = Field(max_length=3)
    readings: list[Reading] = Field(max_length=2000)
    observations: tuple[Observation, ...] = Field(default=(), max_length=2000)
    ideas: list[Idea] = Field(default_factory=list, max_length=30)
    preferences: dict[Short, Short] = Field(default_factory=dict, max_length=20)

    @field_validator("created_at", "captured_at")
    @classmethod
    def date(cls, value):
        if value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Dates require a timezone.")
        return value

    @model_validator(mode="after")
    def unique(self):
        from .consultation import preferences

        preferences(self.preferences)
        public_observations(self.observations)
        if self.demographics is None and not self.adult_confirmed:
            raise ValueError("Adult confirmation is required without demographics.")
        if len({r.id for r in self.components}) != len(self.components):
            raise ValueError("Duplicate components.")
        if len(set(self.harmony_ids)) != len(self.harmony_ids):
            raise ValueError("Duplicate proportions.")
        if len(set(self.routes)) != len(self.routes):
            raise ValueError("Duplicate routes.")
        keys = [
            (r.id, r.method, r.contract, r.unit, r.side, r.photo) for r in self.readings
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate readings.")
        if self.formula == "geometry-composite-v1":
            from .product_report import SCORE_WEIGHTS

            coverage = round(sum(SCORE_WEIGHTS[c.id] for c in self.components), 2)
            for component in self.components:
                if (
                    abs(
                        component.effective_weight
                        - SCORE_WEIGHTS[component.id] / coverage
                    )
                    > 1e-12
                ):
                    raise ValueError(
                        "Component weights do not match the declared formula."
                    )
            expected = (
                round(
                    sum(c.value * SCORE_WEIGHTS[c.id] for c in self.components)
                    / coverage,
                    1,
                )
                if coverage
                else None
            )
            if self.score != expected:
                raise ValueError(
                    "Headline does not reconcile with its component inputs."
                )
        return self


class Annotation(Strict):
    kind: Literal["frontal", "profile"]
    method: Short
    contract: Short
    revision: int = Field(ge=1)
    points: dict[Short, list[Number] | None] = Field(max_length=40)
    facing: Literal["", "left", "right"] = ""
    level_degrees: Number = Field(default=0.0, ge=-45, le=45)
    proposal_ancestry: ProposalOrigin | None = None
    forehead_visibility: Literal[
        "unassessed", "visible", "hairline_hidden", "forehead_hidden", "uncertain"
    ] = "unassessed"

    @model_validator(mode="after")
    def validate_anchors(self):
        from .landmark_review import validate_points as frontal
        from .profile_v2 import validate_points as profile

        (frontal if self.kind == "frontal" else profile)(self.points)
        return self


class Evidence(Strict):
    kind: Literal["source-photo"] = "source-photo"
    ordinal: int = Field(ge=1, le=10000)
    jpeg_b64: str = Field(max_length=8_000_000)
    annotations: list[Annotation] = Field(default_factory=list, max_length=2)

    @field_validator("jpeg_b64")
    @classmethod
    def image(cls, value):
        from .photos import PhotoCatalog, MAX_PIXELS
        import cv2
        import numpy as np

        try:
            raw = base64.b64decode(value, validate=True)
        except ValueError as error:
            raise ValueError("Invalid photo encoding.") from error
        if not 1 <= len(raw) <= 5_000_000 or PhotoCatalog._sniff_format(raw) != "jpeg":
            raise ValueError("Evidence requires a bounded normalized JPEG.")
        dims = PhotoCatalog._declared_dimensions(raw, "jpeg")
        if not dims or dims[0] * dims[1] > MAX_PIXELS:
            raise ValueError("Photo dimensions exceed the limit.")
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or (image.shape[1], image.shape[0]) != dims:
            raise ValueError("Evidence cannot be decoded.")
        return value


class Record(Strict):
    version: Literal[1, 2] = 1
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    name: Short = Field(min_length=1)
    archived_at: Short
    imported_at: Short = ""
    snapshot: Snapshot
    snapshot_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence: list[Evidence] = Field(default_factory=list, max_length=30)
    context: SessionContext | None = None

    @model_validator(mode="after")
    def integrity(self):
        if self.context is not None and self.version != 2:
            raise ValueError("Saved report context requires record version 2.")
        if (
            self.context is not None
            and self.context.workflow is not None
            and self.context.saved_report is None
        ):
            raise ValueError(
                "Editable history requires its explicitly retained full report."
            )
        if self.context is not None and self.context.sources:
            sources = {s.photo_number: s for s in self.context.sources}
            if set(sources) != set(range(1, self.snapshot.photo_count + 1)):
                raise ValueError(
                    "Source context does not cover the original input slots."
                )
            from .photos import PhotoCatalog

            for item in self.evidence:
                raw = base64.b64decode(item.jpeg_b64, validate=True)
                source = sources.get(item.ordinal)
                if (
                    source is None
                    or hashlib.sha256(raw).hexdigest() != source.source_sha256
                    or PhotoCatalog._declared_dimensions(raw, "jpeg")
                    != (source.width, source.height)
                ):
                    raise ValueError(
                        "Saved photo does not match its archived source context."
                    )
        if self.context is not None and self.context.saved_report is not None:
            from .report_questions import _load

            payload = _load(self.context.saved_report)
            demographic = self.snapshot.demographics
            expected_demographics = {
                "entered_by_user": demographic is not None,
                "adult_confirmed": self.snapshot.adult_confirmed,
                "comparison_requested": demographic is not None,
                **(demographic.model_dump() if demographic else {}),
            }
            component_fields = ("id", "value", "effective_weight", "status")
            components = [
                {k: r.get(k) for k in component_fields}
                for r in payload.get("composite", {}).get("components", [])
            ]

            def signature(rows):
                normalized = []
                for row in rows:
                    value = {
                        k: v
                        for k, v in row.items()
                        if k not in {"geometry_retained", "score_eligible"}
                    }
                    value["sources"] = [
                        {k: v for k, v in s.items() if k != "points"}
                        for s in row["sources"]
                    ]
                    normalized.append(
                        json.dumps(
                            value,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
                return sorted(normalized)

            if (
                str(payload.get("format_version")) != self.context.report_version
                or payload.get("instrument_version") != self.snapshot.instrument
                or payload.get("accepted_photo_count") != self.snapshot.photo_count
                or payload.get("composite", {}).get("value") != self.snapshot.score
                or payload.get("composite", {}).get("version") != self.snapshot.formula
                or payload.get("score_audit", {}).get("contract")
                != self.snapshot.formula_contract
                or components != [r.model_dump() for r in self.snapshot.components]
                or payload.get("demographics") != expected_demographics
                or payload.get("consultation", {}).get("preferences")
                != self.snapshot.preferences
                or signature(payload["observations"])
                != signature([r.model_dump() for r in self.snapshot.observations])
            ):
                raise ValueError(
                    "Saved full report does not match its numerical snapshot."
                )
        hashes = {snapshot_hash(self.snapshot)}
        if self.snapshot.version == 1 and not self.snapshot.observations:
            # v1 existed before either observations or optional references.
            # Preserve both historical serializations without altering their values.
            hashes.add(
                hashlib.sha256(
                    self.snapshot.model_dump_json(exclude={"observations"}).encode()
                ).hexdigest()
            )
            if self.snapshot.demographics is not None:
                hashes.add(
                    hashlib.sha256(
                        self.snapshot.model_dump_json(
                            exclude={"observations", "adult_confirmed"}
                        ).encode()
                    ).hexdigest()
                )
        if self.snapshot_hash not in hashes:
            raise ValueError("Snapshot integrity mismatch.")
        if len({e.ordinal for e in self.evidence}) != len(self.evidence):
            raise ValueError("Duplicate evidence.")
        return self


def now():
    return datetime.now(timezone.utc).isoformat()


def snapshot_hash(snapshot):
    return hashlib.sha256(snapshot.model_dump_json().encode()).hexdigest()


def reference_fingerprint():
    root = Path(__file__).resolve().parent.parent / "bank"
    paths = [root / "derived.npz", root / "provenance.json"] + sorted(
        root.glob("*/measurements.npz")
    )
    digest = hashlib.sha256()
    try:
        for path in paths:
            digest.update(str(path.relative_to(root)).encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
    except OSError:
        return None
    return digest.hexdigest()


def freeze_report(
    report,
    demographics,
    *,
    profile,
    protocol="unspecified",
    camera="",
    captured_at="",
    reference=None,
    frontal=None,
    profiles=None,
    settings=None,
    ideas=None,
    adult_confirmed=False,
):
    from .consultation import consultation_payload, preferences

    score = current_composite(report.axes, frontal, profiles)
    audited = audit(current_score_axes(report.axes, frontal, profiles))
    rows = [
        Reading(
            id=r["id"],
            label=r["label"],
            method=r["method"],
            unit=r["unit"],
            value=float(r["value"]),
        )
        for r in public_regions(report.axes, frontal=frontal)
    ]
    profiles = [profiles] if isinstance(profiles, dict) else profiles or []
    for raw in list(frontal or []) + list(profiles):
        for r in raw["measurements"]:
            rows.append(
                Reading(
                    id=r["id"],
                    label=r["label"],
                    method=raw["method"],
                    contract=raw.get("contract", ""),
                    unit=r["unit"],
                    value=float(r["value"]),
                    side=raw.get("facing", ""),
                    photo=raw.get("photo_number", 0),
                )
            )
    for axis, result in report.axes.items():
        if (
            result.status in ("measured", "estimated")
            and result.headline_available
            and result.value is not None
        ):
            rows.append(
                Reading(
                    id=axis,
                    label=axis.title(),
                    method="automatic-axis-v1",
                    unit=result.unit or "",
                    value=float(result.value),
                )
            )
    consultation = consultation_payload(
        public_regions(report.axes, frontal=frontal),
        settings,
        frontal=frontal,
        profile=profiles,
        ideas=ideas,
    )
    recommendations = [
        Idea(id=r["id"], title=r.get("title", ""), action=r.get("action", ""))
        for r in consultation.get("suggestions", [])
    ]
    return Snapshot(
        observations=archive_observations(report.axes, frontal, profiles),
        demographics=(
            EnteredDemographics(**demographics.to_dict())
            if demographics is not None
            else None
        ),
        adult_confirmed=adult_confirmed if demographics is None else True,
        created_at=now(),
        profile=profile.strip(),
        protocol=protocol,
        camera=camera,
        captured_at=captured_at,
        formula=score["version"],
        formula_contract=contract(),
        reference=reference,
        cohort=report.cohort.used_id if report.cohort is not None else "",
        fallback=(
            report.cohort.fallback_level
            if report.cohort is not None
            else report.reference.status
        ),
        instrument=report.instrument_fingerprint,
        input_digest=hashlib.sha256(report.photo_fingerprint.encode()).hexdigest(),
        photo_count=report.accepted_photos,
        routes=sorted(
            f"{k}:{v.status}:{v.photo_count}" for k, v in report.axes.items()
        ),
        harmony_ids=audited["harmony_ratio_ids"],
        score=score["value"],
        components=[
            Component(
                **{k: r[k] for k in ("id", "value", "effective_weight", "status")}
            )
            for r in score["components"]
        ],
        readings=rows,
        ideas=recommendations,
        preferences=preferences(settings or {}),
    )


def compare(a: Snapshot, b: Snapshot):
    from math import isfinite

    fields = (
        "profile",
        "formula",
        "formula_contract",
        "reference",
        "cohort",
        "fallback",
        "instrument",
        "photo_count",
        "routes",
        "harmony_ids",
        "protocol",
        "camera",
    )
    differences = [k for k in fields if getattr(a, k) != getattr(b, k)]
    if [(c.id, c.effective_weight, c.status) for c in a.components] != [
        (c.id, c.effective_weight, c.status) for c in b.components
    ]:
        differences.append("components")

    def keys(r):
        return (r.id, r.method, r.contract, r.unit, r.side, r.photo)

    before = {keys(r): r for r in a.readings}
    after = {keys(r): r for r in b.readings}
    if (
        a.formula != "geometry-composite-v1"
        or b.formula != "geometry-composite-v1"
        or a.formula_contract != contract()
        or b.formula_contract != contract()
    ):
        differences.append("unverified_formula_contract")
    if set(before) != set(after):
        differences.append("measurement_set")
    same = a.input_digest == b.input_digest
    unknown = (
        a.reference is None
        or b.reference is None
        or not a.camera
        or not b.camera
        or a.protocol == "unspecified"
        or b.protocol == "unspecified"
        or a.score is None
        or b.score is None
    )
    changed = any(
        k in differences
        for k in (
            "formula",
            "formula_contract",
            "unverified_formula_contract",
            "reference",
            "cohort",
            "fallback",
            "instrument",
        )
    )
    status = (
        "insufficient_evidence"
        if unknown
        else (
            "method_changed"
            if changed
            else "partly_comparable" if differences else "compatible"
        )
    )
    deltas = [
        dict(
            id=before[k].id,
            label=before[k].label,
            method=before[k].method,
            unit=before[k].unit,
            side=before[k].side,
            photo=before[k].photo,
            before=before[k].value,
            after=after[k].value,
            delta=(
                (after[k].value - before[k].value)
                if isfinite(after[k].value - before[k].value)
                else None
            ),
        )
        for k in sorted(before.keys() & after.keys())
        if (not changed or not before[k].method.startswith("automatic"))
    ]
    warnings = [
        "Difference observed; significance not established. Original headlines are preserved."
    ]
    if same:
        warnings.append("Same input set: these are not independent retakes.")
    if not a.captured_at or not b.captured_at:
        warnings.append("Capture date missing; no chronological progress claim.")
    elif datetime.fromisoformat(
        a.captured_at.replace("Z", "+00:00")
    ) >= datetime.fromisoformat(b.captured_at.replace("Z", "+00:00")):
        warnings.append(
            "Capture dates are equal or reversed; no chronological progress claim."
        )
    return dict(
        status=status,
        differences=differences,
        score_delta=round(b.score - a.score, 1) if status == "compatible" else None,
        headlines=[a.score, b.score],
        same_inputs=same,
        readings=deltas,
        observed_range=(
            [min(a.score, b.score), max(a.score, b.score)]
            if status == "compatible" and not same
            else None
        ),
        warnings=warnings,
    )
