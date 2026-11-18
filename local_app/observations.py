"""Versioned measurement provenance; legacy instruments remain separate.

This contract records evidence and eligibility, not anatomical accuracy. Builders
must calculate values with their declared instrument; loading a record alone does
not verify its arithmetic or promote an estimation method.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Token = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,160}$")]
Text = Annotated[str, Field(min_length=1, max_length=1000)]
Unit = Annotated[str, Field(min_length=1, max_length=40)]
Number = Annotated[float, Field(allow_inf_nan=False)]
View = Literal[
    "frontal", "frontal_neutral", "profile_left", "profile_right", "smile", "unknown"
]


class Frozen(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class Point(Frozen):
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_]{1,64}$")]
    x: Number = Field(ge=0, le=1)
    y: Number = Field(ge=0, le=1)


class SourceEvidence(Frozen):
    photo_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,96}$")]
    photo_number: int = Field(ge=1, le=10000)
    view: View
    coordinate_space: Literal["normalized_original_image_xy"] = (
        "normalized_original_image_xy"
    )
    width: int = Field(ge=1, le=100000)
    height: int = Field(ge=1, le=100000)
    revision: int = Field(ge=1)
    review_revision: int = Field(default=0, ge=0)
    points: tuple[Point, ...] = Field(max_length=478)
    visibility: Literal["visible", "occluded", "uncertain", "unassessed"] = "unassessed"
    forehead_visibility: (
        Literal[
            "unassessed", "visible", "hairline_hidden", "forehead_hidden", "uncertain"
        ]
        | None
    ) = None
    automatic_forehead_status: (
        Literal["forehead_clear", "hair_overlap", "uncertain", "unavailable", "no_face"]
        | None
    ) = None
    capture_eligible: bool = False

    @model_validator(mode="after")
    def unique_points(self) -> SourceEvidence:
        if len({p.name for p in self.points}) != len(self.points):
            raise ValueError("Source point names must be unique.")
        return self


class Evaluation(Frozen):
    evaluation_id: Token
    basis: Literal["synthetic", "ai_annotations", "human_annotations"]
    definition_key: Token
    definition_version: Token
    method: Token
    unit: Unit
    capture_condition: Text
    sample_count: int = Field(ge=1)
    error_metric: Literal[
        "median_absolute_error", "mean_absolute_error", "p95_absolute_error"
    ]
    error: Number = Field(ge=0)


class ReferenceMatch(Frozen):
    definition_key: Token
    definition_version: Token
    method: Token
    unit: Unit
    reference_id: Token
    percentile: Number = Field(ge=0, le=100)
    sample_size: int | None = Field(default=None, ge=1)


class Observation(Frozen):
    version: Literal[1] = 1
    geometry_retained: bool = True
    definition_key: Token
    definition_version: Token
    label: Text
    definition: Text
    formula: Text
    method: Token
    unit: Unit
    status: Literal["automatic", "estimated", "proxy", "user_reviewed", "unavailable"]
    value: Number | None
    revision: int = Field(ge=1)
    sources: tuple[SourceEvidence, ...] = Field(default=(), max_length=10000)
    display_decimals: int = Field(default=4, ge=0, le=6)
    proxy_for: Token | None = None
    withheld_reason: Text | None = None
    evaluation: Evaluation | None = None
    reference: ReferenceMatch | None = None
    score_eligible: bool = False
    score_exclusion_reason: Text | None = (
        "Not admitted to a compatible score instrument."
    )

    @model_validator(mode="after")
    def consistency(self) -> Observation:
        if not self.geometry_retained and (
            self.score_eligible or any(s.points for s in self.sources)
        ):
            raise ValueError(
                "Report-only archives omit geometry and cannot supply score inputs."
            )
        if len({s.photo_id for s in self.sources}) != len(self.sources) or len(
            {s.photo_number for s in self.sources}
        ) != len(self.sources):
            raise ValueError(
                "Source photo identifiers and ordinals must be unique within a reading."
            )
        if self.status == "unavailable":
            if (
                self.value is not None
                or self.reference is not None
                or self.evaluation is not None
                or self.score_eligible
                or not self.withheld_reason
                or not self.withheld_reason.strip()
            ):
                raise ValueError(
                    "Unavailable observations need a reason and no numeric result, reference or score."
                )
            return self
        if (
            self.value is None
            or not self.sources
            or (
                self.geometry_retained
                and any(not source.points for source in self.sources)
            )
            or self.withheld_reason is not None
        ):
            raise ValueError(
                "Available observations need finite values and sources, without a withheld reason."
            )
        if self.status == "estimated":
            if self.evaluation is None or (
                self.evaluation.definition_key,
                self.evaluation.definition_version,
                self.evaluation.method,
                self.evaluation.unit,
            ) != (self.definition_key, self.definition_version, self.method, self.unit):
                raise ValueError(
                    "An estimate needs evaluation of the same definition and units."
                )
        elif self.evaluation is not None:
            raise ValueError(
                "Estimation evaluation belongs only to an estimated observation."
            )
        if self.status == "proxy":
            if self.proxy_for is None or self.proxy_for == self.definition_key:
                raise ValueError("A proxy must identify a different target definition.")
        elif self.proxy_for is not None:
            raise ValueError("Only a proxy may identify a substituted target.")
        if self.status == "user_reviewed" and not any(
            s.review_revision for s in self.sources
        ):
            raise ValueError(
                "A user-reviewed observation needs an annotation revision."
            )
        if self.reference is not None and (
            self.reference.definition_key,
            self.reference.definition_version,
            self.reference.method,
            self.reference.unit,
        ) != (self.definition_key, self.definition_version, self.method, self.unit):
            raise ValueError(
                "Reference distribution must match the actual definition, method and units."
            )
        if self.score_eligible and (
            self.reference is None
            or not all(
                s.capture_eligible
                and s.visibility == "visible"
                and not s.review_revision
                for s in self.sources
            )
            or self.status in {"estimated", "user_reviewed"}
            or self.score_exclusion_reason is not None
        ):
            raise ValueError(
                "Score eligibility requires compatible automatic/reference/capture inputs."
            )
        if not self.score_eligible and (
            not self.score_exclusion_reason or not self.score_exclusion_reason.strip()
        ):
            raise ValueError("A score exclusion needs an explanation.")
        return self

    def fingerprint(self) -> str:
        record = type(self).model_validate(self)
        payload = json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_public_dict(self) -> dict[str, Any]:
        """Stable within-report ordinals replace private session photo identifiers."""
        record = type(self).model_validate(self)
        result = record.model_dump(mode="json")
        for source in result["sources"]:
            source["photo_id"] = f"photo-{source['photo_number']}"
        return result


def public_observations(records: tuple[Observation, ...]) -> list[dict[str, Any]]:
    """Prevent alias collisions or stale image frames across an entire report."""
    frames: dict[str, tuple[int, int, int, int]] = {}
    views: dict[str, str] = {}
    identities: dict[int, str] = {}
    validated = tuple(Observation.model_validate(record) for record in records)
    for record in validated:
        for source in record.sources:
            frame = (
                source.photo_number,
                source.width,
                source.height,
                source.revision,
            )
            if source.photo_id in frames and frames[source.photo_id] != frame:
                raise ValueError(
                    "A source photo has conflicting ordinal, frame or image revision."
                )
            if source.view != "unknown":
                if source.photo_id in views and views[source.photo_id] != source.view:
                    raise ValueError("A source photo has conflicting declared views.")
                views[source.photo_id] = source.view
            if (
                source.photo_number in identities
                and identities[source.photo_number] != source.photo_id
            ):
                raise ValueError(
                    "Distinct source photos cannot share a public ordinal."
                )
            frames[source.photo_id] = frame
            identities[source.photo_number] = source.photo_id
    return [record.to_public_dict() for record in validated]
