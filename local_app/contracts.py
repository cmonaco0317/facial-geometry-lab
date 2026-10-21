"""Typed payloads shared by analysis, persistence, rendering, and export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal

AxisName = Literal["harmony", "dimorphism", "symmetry", "averageness"]
AxisStatus = Literal["measured", "estimated", "needs_photo", "processing_error"]

AXES = frozenset({"harmony", "dimorphism", "symmetry", "averageness"})
STATUSES = frozenset({"measured", "estimated", "needs_photo", "processing_error"})
SEXES = frozenset({"female", "male", "another", "intersex", "prefer_not_to_say"})

# symmetry_population.symmetry_468 divides by the 33/263 outer-canthus span,
# not pupil/iris spacing. Share the label across strict and estimated paths.
SYMMETRY_UNIT = "% outer-eye span asymmetry"


def _normalized_token(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    normalized = "_".join(value.strip().lower().replace("-", " ").split())
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"payload contains a non-JSON value: {type(value).__name__}")


@dataclass(frozen=True)
class Demographics:
    """User-provided comparison fields; nothing here is inferred from a face."""

    age: int
    sex: str
    ethnicities: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.age, bool) or not isinstance(self.age, int):
            raise ValueError("age must be a whole number")
        if self.age < 18:
            raise ValueError("analysis is available only to adults 18 and older")

        sex = _normalized_token(self.sex, "sex")
        if sex not in SEXES:
            raise ValueError(f"unsupported sex selection: {self.sex}")

        if isinstance(self.ethnicities, str):
            raise ValueError("ethnicities must be a sequence of selections")
        normalized: list[str] = []
        for raw in self.ethnicities:
            token = _normalized_token(raw, "ethnicity")
            if token not in normalized:
                normalized.append(token)
        if not normalized:
            raise ValueError("at least one ethnicity selection is required")

        object.__setattr__(self, "sex", sex)
        object.__setattr__(self, "ethnicities", tuple(normalized))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class CohortUse:
    """The comparison group requested by the user and the group actually read."""

    requested_id: str
    used_id: str
    fallback_level: str
    sample_size: int
    requested_label: str = ""
    used_label: str = ""
    member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("requested_id", "used_id", "fallback_level"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
        if isinstance(self.sample_size, bool) or not isinstance(self.sample_size, int):
            raise ValueError("sample_size must be a whole number")
        if self.sample_size < 0:
            raise ValueError("sample_size cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe(asdict(self))
        payload["exact"] = self.fallback_level == "exact"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CohortUse":
        if not isinstance(payload, dict):
            raise ValueError("cohort payload must be an object")
        return cls(
            requested_id=payload.get("requested_id"),
            used_id=payload.get("used_id"),
            fallback_level=payload.get("fallback_level"),
            sample_size=payload.get("sample_size"),
            requested_label=payload.get("requested_label", ""),
            used_label=payload.get("used_label", ""),
            member_ids=tuple(payload.get("member_ids", ())),
        )


@dataclass(frozen=True)
class PhotoRequest:
    """One concrete capture instruction that can unlock an axis."""

    view: str
    expression: str
    lighting: str
    framing: str
    reason: str

    def __post_init__(self) -> None:
        for field_name in ("view", "expression", "lighting", "framing", "reason"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"photo request {field_name} cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhotoRequest":
        if not isinstance(payload, dict):
            raise ValueError("photo request payload must be an object")
        return cls(
            view=payload.get("view"),
            expression=payload.get("expression"),
            lighting=payload.get("lighting"),
            framing=payload.get("framing"),
            reason=payload.get("reason"),
        )


@dataclass(frozen=True)
class AxisResult:
    """The only result shape accepted by the progressive product path."""

    axis: AxisName | str
    status: AxisStatus | str
    summary: str
    headline_available: bool = True
    value: float | None = None
    unit: str | None = None
    percentile: float | None = None
    interval: tuple[float, float] | None = None
    cohort: CohortUse | None = None
    photo_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    photo_request: PhotoRequest | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"unknown analysis axis: {self.axis}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown axis status: {self.status}")
        if not isinstance(self.summary, str):
            raise ValueError("summary must be text")
        if not isinstance(self.headline_available, bool):
            raise ValueError("headline_available must be true or false")
        if isinstance(self.photo_count, bool) or not isinstance(self.photo_count, int):
            raise ValueError("photo_count must be a whole number")
        if self.photo_count < 0:
            raise ValueError("photo_count cannot be negative")
        _json_safe(self.details)

        if self.status == "measured":
            self._validate_measured()
        elif self.status == "estimated":
            self._validate_estimated()
        elif self.status == "needs_photo":
            self._validate_needs_photo()
        else:
            self._validate_processing_error()

    def _validate_estimated(self) -> None:
        """Estimates retain observed values without borrowing calibrated intervals."""
        if not self.summary.strip() or self.cohort is None or self.photo_count < 1:
            raise ValueError("estimates require a summary, cohort and observed photo")
        if self.interval is not None:
            raise ValueError("estimates cannot claim a calibrated uncertainty interval")
        if self.photo_request is not None or self.error_code or self.error_message:
            raise ValueError("estimates cannot carry request or error fields")
        if self.headline_available:
            _finite_number(self.value, "estimate")
            if not isinstance(self.unit, str) or not self.unit.strip():
                raise ValueError("estimates require a unit")
            self._validate_optional_percentile(self.percentile)
        else:
            if any(v is not None for v in (self.value, self.unit, self.percentile)):
                raise ValueError("profile estimates cannot carry scalar fields")
            profile = self.details.get("profile")
            if not isinstance(profile, dict) or not profile:
                raise ValueError("profile estimates require observed components")
            for row in profile.values():
                if not isinstance(row, dict):
                    raise ValueError("estimated profile rows must be objects")
                _finite_number(row.get("value"), "estimated component")
                self._validate_optional_percentile(row.get("percentile"))
                if row.get("interval") is not None:
                    raise ValueError("estimated components cannot claim calibrated intervals")

    @staticmethod
    def _validate_optional_percentile(value: object) -> None:
        if value is not None and not 0 <= _finite_number(value, "percentile") <= 100:
            raise ValueError("percentile must be between 0 and 100")

    def _validate_measured(self) -> None:
        if not self.summary.strip():
            raise ValueError("measured results require a summary")
        if self.headline_available:
            self._validate_headline()
        else:
            self._validate_profile()
        if self.cohort is None:
            raise ValueError("measured results require a cohort")
        if self.photo_request is not None or self.error_code or self.error_message:
            raise ValueError("measured results cannot carry request or error fields")

    def _validate_headline(self) -> None:
        _finite_number(self.value, "value")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("measured results require a unit")
        percentile = _finite_number(self.percentile, "percentile")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be between 0 and 100")
        if self.interval is None or len(self.interval) != 2:
            raise ValueError("measured results require a two-value interval")
        low = _finite_number(self.interval[0], "interval lower bound")
        high = _finite_number(self.interval[1], "interval upper bound")
        if not (0.0 <= low <= percentile <= high <= 100.0):
            raise ValueError("interval must be ordered around the percentile within 0 to 100")

    def _validate_profile(self) -> None:
        if any(item is not None for item in (self.value, self.unit, self.percentile, self.interval)):
            raise ValueError("profile measurements cannot invent a scalar headline")
        profile = self.details.get("profile")
        if not isinstance(profile, dict) or not profile:
            raise ValueError("profile measurements require a nonempty profile")
        for name, row in profile.items():
            if not isinstance(row, dict):
                raise ValueError(f"profile row {name} must be an object")
            _finite_number(row.get("value"), f"profile row {name} value")
            percentile = _finite_number(
                row.get("percentile"),
                f"profile row {name} percentile",
            )
            interval = row.get("interval")
            if not 0.0 <= percentile <= 100.0:
                raise ValueError(f"profile row {name} percentile must be between 0 and 100")
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(f"profile row {name} requires a two-value interval")
            low = _finite_number(interval[0], f"profile row {name} interval lower bound")
            high = _finite_number(interval[1], f"profile row {name} interval upper bound")
            if not (0.0 <= low <= percentile <= high <= 100.0):
                raise ValueError(
                    f"profile row {name} interval must surround its percentile"
                )

    def _validate_needs_photo(self) -> None:
        if self.photo_request is None:
            raise ValueError("needs_photo results require a photo request")
        if any(item is not None for item in (self.value, self.unit, self.percentile, self.interval)):
            raise ValueError("needs_photo results cannot fabricate a measurement")
        if self.error_code or self.error_message:
            raise ValueError("needs_photo results cannot carry processing errors")

    def _validate_processing_error(self) -> None:
        if not self.error_code or not self.error_message:
            raise ValueError("processing_error results require an error code and safe error message")
        if self.photo_request is not None:
            raise ValueError("processing errors cannot masquerade as photo requests")
        if any(item is not None for item in (self.value, self.unit, self.percentile, self.interval)):
            raise ValueError("processing errors cannot carry a measurement")

    @classmethod
    def measured(
        cls,
        *,
        axis: AxisName | str,
        summary: str,
        value: float,
        unit: str,
        percentile: float,
        interval: tuple[float, float],
        cohort: CohortUse,
        photo_count: int,
        details: dict[str, Any] | None = None,
    ) -> "AxisResult":
        return cls(
            axis=axis,
            status="measured",
            summary=summary,
            headline_available=True,
            value=value,
            unit=unit,
            percentile=percentile,
            interval=interval,
            cohort=cohort,
            photo_count=photo_count,
            details={} if details is None else dict(details),
        )

    @classmethod
    def measured_profile(
        cls,
        *,
        axis: AxisName | str,
        summary: str,
        cohort: CohortUse,
        photo_count: int,
        details: dict[str, Any],
    ) -> "AxisResult":
        return cls(
            axis=axis,
            status="measured",
            summary=summary,
            headline_available=False,
            cohort=cohort,
            photo_count=photo_count,
            details=dict(details),
        )

    @classmethod
    def needs_photo(
        cls,
        *,
        axis: AxisName | str,
        request: PhotoRequest,
        photo_count: int,
        details: dict[str, Any] | None = None,
        cohort: CohortUse | None = None,
    ) -> "AxisResult":
        return cls(
            axis=axis,
            status="needs_photo",
            summary=request.reason,
            headline_available=False,
            cohort=cohort,
            photo_count=photo_count,
            details={} if details is None else dict(details),
            photo_request=request,
        )

    @classmethod
    def processing_error(
        cls,
        *,
        axis: AxisName | str,
        error_code: str,
        error_message: str,
        retryable: bool,
        photo_count: int = 0,
        details: dict[str, Any] | None = None,
        cohort: CohortUse | None = None,
    ) -> "AxisResult":
        return cls(
            axis=axis,
            status="processing_error",
            summary=error_message,
            headline_available=False,
            cohort=cohort,
            photo_count=photo_count,
            details={} if details is None else dict(details),
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.cohort is not None:
            payload["cohort"] = self.cohort.to_dict()
        if self.photo_request is not None:
            payload["photo_request"] = self.photo_request.to_dict()
        return _json_safe(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AxisResult":
        if not isinstance(payload, dict):
            raise ValueError("axis result payload must be an object")
        raw_interval = payload.get("interval")
        interval = tuple(raw_interval) if raw_interval is not None else None
        raw_cohort = payload.get("cohort")
        cohort = CohortUse.from_dict(raw_cohort) if raw_cohort is not None else None
        raw_request = payload.get("photo_request")
        request = PhotoRequest.from_dict(raw_request) if raw_request is not None else None
        details = payload.get("details", {})
        if not isinstance(details, dict):
            raise ValueError("axis result details must be an object")
        status = payload.get("status")
        return cls(
            axis=payload.get("axis"),
            status=status,
            summary=payload.get("summary"),
            headline_available=payload.get(
                "headline_available",
                status == "measured",
            ),
            value=payload.get("value"),
            unit=payload.get("unit"),
            percentile=payload.get("percentile"),
            interval=interval,
            cohort=cohort,
            photo_count=payload.get("photo_count", 0),
            details=dict(details),
            photo_request=request,
            error_code=payload.get("error_code"),
            error_message=payload.get("error_message"),
            retryable=payload.get("retryable", False),
        )
