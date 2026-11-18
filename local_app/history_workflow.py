"""Bounded editable journals and original-source slots for opt-in archives."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .observations import Frozen, View

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
PhotoId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,96}$")]
Number = Annotated[float, Field(allow_inf_nan=False)]
StateJSON = Annotated[str, Field(max_length=2_000_000)]


class SourceContext(Frozen):
    photo_id: PhotoId
    photo_number: int = Field(ge=1, le=10000)
    source_sha256: Digest
    width: int = Field(ge=1, le=100000)
    height: int = Field(ge=1, le=100000)
    views: tuple[View, ...] = Field(default=("unknown",), min_length=1, max_length=6)
    pose: dict[Literal["yaw", "pitch", "roll"], Number | None] = Field(
        default_factory=lambda: dict(yaw=None, pitch=None, roll=None)
    )
    pose_basis: Literal["detector_estimate_uncalibrated", "unavailable"] = "unavailable"
    role: Literal["unknown", "neutral", "smile", "profile"] = "unknown"
    role_basis: Literal[
        "user_recorded_unverified", "detector_candidate_unverified", "unavailable"
    ] = "unavailable"

    @model_validator(mode="after")
    def pose_fields(self) -> SourceContext:
        if set(self.pose) != {"yaw", "pitch", "roll"} or len(set(self.views)) != len(
            self.views
        ):
            raise ValueError("Invalid archived source context.")
        if any(value is not None for value in self.pose.values()) != (
            self.pose_basis == "detector_estimate_uncalibrated"
        ):
            raise ValueError("Pose values must identify their actual estimation basis.")
        return self


class ArchiveOrigin(Frozen):
    archive_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    appearance_revision: int = Field(ge=0, le=500)
    source_map: dict[str, int] = Field(max_length=10000)
    sources: tuple[SourceContext, ...] = Field(max_length=10000)

    @model_validator(mode="after")
    def slots(self) -> ArchiveOrigin:
        slots = {s.photo_number for s in self.sources}
        if (
            len(slots) != len(self.sources)
            or len({s.photo_id for s in self.sources}) != len(self.sources)
            or len(set(self.source_map.values())) != len(self.source_map)
            or any(
                not re.fullmatch(r"[1-9][0-9]{0,3}|10000", key)
                or not 1 <= value <= 10000
                or int(key) not in slots
                for key, value in self.source_map.items()
            )
        ):
            raise ValueError("Invalid restoration source map.")
        return self


def _digest(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("Invalid correction source hash.")


def _object(value: object, fields: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Unsupported editable archive fields.")
    return value


def _photo_setting(value: object) -> None:
    from .appearance import ROLES

    if value is None:
        return
    row = _object(value, {"source_sha256", "role"})
    _digest(row["source_sha256"])
    if row["role"] not in ROLES:
        raise ValueError("Invalid saved photo role.")


def _override(family: str, value: object) -> None:
    from .appearance import review_appearance

    if value is None:
        return
    row = _object(
        value,
        {
            "source_sha256",
            "box",
            "role",
            "visibility",
            "lighting",
            "glasses",
            "makeup",
            "facial_hair",
            "note",
            "points",
        },
    )
    _digest(row["source_sha256"])
    raw = [
        dict(photo_id="validation", photo_number=1, source_sha256=row["source_sha256"])
    ]
    result = review_appearance(
        raw,
        {},
        photo_number=1,
        family=family,
        revision=0,
        action="save",
        **{k: v for k, v in row.items() if k != "source_sha256"},
    )
    if result["overrides"]["validation:" + family] != row:
        raise ValueError("Saved appearance correction is not canonical.")


def _appearance(raw: object) -> None:
    from .appearance import FAMILIES

    if (
        not isinstance(raw, dict)
        or not {"revision", "overrides", "history"} <= set(raw)
        or set(raw) - {"revision", "overrides", "history", "photo_settings"}
    ):
        raise ValueError("Invalid appearance journal.")
    events = raw["history"]
    if (
        type(raw["revision"]) is not int
        or not isinstance(events, list)
        or len(events) > 500
        or raw["revision"] != len(events)
    ):
        raise ValueError("Invalid appearance revision history.")
    for key, row in _bounded_map(raw["overrides"]).items():
        photo, family = _split_key(key)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", photo):
            raise ValueError("Invalid saved photo key.")
        _override(family, row)
    for photo, setting in _bounded_map(raw.get("photo_settings", {})).items():
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", photo):
            raise ValueError("Invalid saved photo key.")
        _photo_setting(setting)
    for n, event in enumerate(events, 1):
        e = _object(
            event,
            {
                "revision",
                "photo_number",
                "family",
                "action",
                "source_sha256",
                "previous",
                "current",
                "previous_photo",
                "current_photo",
            },
        )
        if (
            type(e["revision"]) is not int
            or e["revision"] != n
            or type(e["photo_number"]) is not int
            or not 1 <= e["photo_number"] <= 10000
            or e["family"] not in FAMILIES
            or e["action"] not in {"save", "reset"}
        ):
            raise ValueError("Invalid appearance history event.")
        _digest(e["source_sha256"])
        for field in ("previous", "current"):
            _override(e["family"], e[field])
        for field in ("previous_photo", "current_photo"):
            _photo_setting(e[field])
        if (e["current"] is None) != (e["action"] == "reset"):
            raise ValueError(
                "Appearance action does not match its retained correction."
            )


def _bounded_map(value: object) -> dict:
    if not isinstance(value, dict) or len(value) > 500:
        raise ValueError("Editable journal exceeds its limit.")
    return value


def _split_key(key: str) -> tuple[str, str]:
    from .appearance import FAMILIES

    if not isinstance(key, str) or ":" not in key:
        raise ValueError("Invalid saved region key.")
    photo, family = key.rsplit(":", 1)
    if family not in FAMILIES:
        raise ValueError("Invalid saved appearance family.")
    return photo, family


def _review_entry(value: object) -> None:
    from .consultation_review import update_review

    if value is None:
        return
    e = _object(
        value,
        {
            "version",
            "reviewer_name",
            "reviewed_on",
            "scope",
            "note",
            "binding",
            "user_attested",
        },
    )
    _digest(e["binding"])
    result = update_review(
        {},
        expected=0,
        submitted=e["binding"],
        current=e["binding"],
        action="save",
        reviewer_name=e["reviewer_name"],
        reviewed_on=e["reviewed_on"],
        scope=e["scope"],
        note=e["note"],
        confirmed=e["user_attested"],
    )
    if result["entry"] != e:
        raise ValueError("Invalid saved reviewer attribution.")


def _reviewer(raw: object) -> None:
    row = _object(raw, {"revision", "entry", "history"})
    events = row["history"]
    if (
        type(row["revision"]) is not int
        or not isinstance(events, list)
        or len(events) > 100
        or row["revision"] != len(events)
    ):
        raise ValueError("Invalid reviewer revision history.")
    previous = None
    for n, event in enumerate(events, 1):
        e = _object(event, {"revision", "action", "previous", "current"})
        if (
            type(e["revision"]) is not int
            or e["revision"] != n
            or e["action"] not in {"save", "reset"}
            or e["previous"] != previous
        ):
            raise ValueError("Broken reviewer history lineage.")
        _review_entry(e["previous"])
        _review_entry(e["current"])
        if (e["current"] is None) != (e["action"] == "reset"):
            raise ValueError("Invalid reviewer action.")
        previous = e["current"]
    if row["entry"] != previous:
        raise ValueError("Reviewer entry does not match its history.")


def _decode(encoded: str) -> dict:
    if len(encoded.encode()) > 2_000_000:
        raise ValueError("Editable journal exceeds its byte limit.")

    def reject_constant(value: str) -> None:
        raise ValueError("Nonfinite saved journal.")

    raw = json.loads(encoded, parse_constant=reject_constant)
    if not isinstance(raw, dict):
        raise ValueError("Editable journal must be an object.")
    return raw


class ArchivedWorkflow(Frozen):
    version: Literal[1] = 1
    appearance_json: StateJSON | None = None
    reviewer_json: StateJSON | None = None
    idea_choices_json: StateJSON | None = None
    origins: tuple[ArchiveOrigin, ...] = Field(default=(), max_length=1000)

    @model_validator(mode="after")
    def validate_journals(self) -> ArchivedWorkflow:
        from .idea_choices import normalize

        try:
            if self.appearance_json is not None:
                from .history_journal import validate_appearance_journal

                raw = _decode(self.appearance_json)
                _appearance(raw)
                validate_appearance_journal(raw, self.origins)
            elif any(origin.appearance_revision for origin in self.origins):
                raise ValueError("Source epochs require their appearance journal.")
            if self.reviewer_json is not None:
                _reviewer(_decode(self.reviewer_json))
            if self.idea_choices_json is not None:
                normalize(_decode(self.idea_choices_json))
        except (
            KeyError,
            TypeError,
            AttributeError,
            ArithmeticError,
            RecursionError,
        ) as error:
            raise ValueError("Malformed editable archive journal.") from error
        return self

    @classmethod
    def from_state(cls, state: dict) -> ArchivedWorkflow:
        if not isinstance(state, dict):
            raise ValueError("Editable state must be an object.")
        fields = {}
        for source, target in (
            ("appearance_review", "appearance_json"),
            ("consultation_reviewer", "reviewer_json"),
            ("idea_choices", "idea_choices_json"),
        ):
            if state.get(source) is not None:
                fields[target] = json.dumps(
                    state[source],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
        fields["origins"] = tuple(
            ArchiveOrigin.model_validate_json(json.dumps(origin))
            for origin in state.get("appearance_history_origins", [])
        )
        return cls(**fields)

    def validate_source_bindings(self, sources: tuple[SourceContext, ...]) -> None:
        if self.appearance_json is None:
            return
        raw = _decode(self.appearance_json)
        from .history_journal import validate_appearance_journal

        validate_appearance_journal(raw, self.origins, sources)
        by_id = {s.photo_id: s for s in sources}
        rows = [(_split_key(key)[0], value) for key, value in raw["overrides"].items()]
        rows += list(raw.get("photo_settings", {}).items())
        for photo, row in rows:
            if photo not in by_id or row["source_sha256"] != by_id[photo].source_sha256:
                raise ValueError(
                    "Appearance correction does not match its original source slot."
                )
        for photo, setting in raw.get("photo_settings", {}).items():
            if (
                by_id[photo].role != setting["role"]
                or by_id[photo].role_basis != "user_recorded_unverified"
            ):
                raise ValueError(
                    "Expression label differs from its editable source setting."
                )


def restore_workflow(
    workflow: ArchivedWorkflow,
    sources: tuple[SourceContext, ...],
    mapping: dict[int, SourceContext],
    *,
    archive_id: str | None = None,
) -> dict:
    """Activate only selected byte/frame-matched corrections; retain original journals."""
    bindings = {}
    for source in sources:
        current = mapping.get(source.photo_number)
        if current is None:
            continue
        if (source.source_sha256, source.width, source.height) != (
            current.source_sha256,
            current.width,
            current.height,
        ):
            raise ValueError(
                "Restored source bytes or frame do not match archived context."
            )
        bindings[source.photo_id] = current.photo_id
    state = {}
    if workflow.appearance_json is not None:
        raw = _decode(workflow.appearance_json)
        raw["overrides"] = {
            bindings[photo] + ":" + family: value
            for key, value in raw["overrides"].items()
            for photo, family in [_split_key(key)]
            if photo in bindings
        }
        if "photo_settings" in raw:
            raw["photo_settings"] = {
                bindings[key]: value
                for key, value in raw["photo_settings"].items()
                if key in bindings
            }
        state["appearance_review"] = raw
    if workflow.reviewer_json is not None:
        state["consultation_reviewer"] = _decode(workflow.reviewer_json)
    if workflow.idea_choices_json is not None:
        state["idea_choices"] = _decode(workflow.idea_choices_json)
    origins = list(workflow.origins)
    if archive_id is not None:
        origins.append(
            ArchiveOrigin(
                archive_id=archive_id,
                appearance_revision=state.get("appearance_review", {}).get(
                    "revision", 0
                ),
                source_map={str(old): new.photo_number for old, new in mapping.items()},
                sources=sources,
            )
        )
    state["appearance_history_origins"] = [
        origin.model_dump(mode="json") for origin in origins
    ]
    return deepcopy(state)
