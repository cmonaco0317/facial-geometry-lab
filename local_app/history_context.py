"""Optional versioned history context, separate from legacy snapshot checksums."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .observations import Frozen, Observation, public_observations
from .report_questions import _load
from .history_workflow import ArchivedWorkflow, SourceContext


class CaptureContext(Frozen):
    """User-recorded conditions, never calibrated pose or capture verification."""

    pose: Literal["unknown", "straight_on", "true_profile", "turned", "mixed"] = (
        "unknown"
    )
    expression: Literal["unknown", "neutral", "smile", "other", "mixed"] = "unknown"
    lighting: Literal["unknown", "even", "uneven", "mixed"] = "unknown"
    distance: Annotated[str, Field(max_length=120)] = ""
    basis: Literal["user_recorded_unverified"] = "user_recorded_unverified"


class SessionContext(Frozen):
    version: Literal[1] = 1
    report_version: Annotated[str, Field(min_length=1, max_length=120)]
    capture: CaptureContext = Field(default_factory=CaptureContext)
    saved_report: dict[str, str] | None = None
    sources: tuple[SourceContext, ...] = Field(default=(), max_length=10000)
    workflow: ArchivedWorkflow | None = None

    @model_validator(mode="after")
    def validate_saved_report(self) -> SessionContext:
        try:
            self._validate_saved_report()
        except (
            KeyError,
            TypeError,
            AttributeError,
            ArithmeticError,
            RecursionError,
        ) as error:
            raise ValueError(
                "The saved report context is malformed or unsupported."
            ) from error
        return self

    def _validate_saved_report(self) -> None:
        if len({s.photo_id for s in self.sources}) != len(self.sources) or len(
            {s.photo_number for s in self.sources}
        ) != len(self.sources):
            raise ValueError("Archived source slots must be unique.")
        if self.workflow is not None:
            self.workflow.validate_source_bindings(self.sources)
        if self.saved_report is None:
            return
        if set(self.saved_report) != {"revision", "report_json"}:
            raise ValueError("Unsupported saved report context fields.")
        payload = _load(self.saved_report)
        for field in (
            "appearance",
            "consultation",
            "composite",
            "score_audit",
            "demographics",
            "axes",
            "comparison",
        ):
            if not isinstance(payload.get(field), dict):
                raise ValueError(
                    "Saved report metadata must contain supported objects."
                )
        if self.workflow is not None:
            from .consultation_review import public_review
            from .history_appearance import validate_frozen_appearance

            validate_frozen_appearance(
                self.workflow, self.sources, payload["appearance"]
            )
            if self.workflow.reviewer_json is not None:
                state = json.loads(self.workflow.reviewer_json)
                if (
                    public_review(
                        state, payload["consultation"]["reviewer_context_binding"]
                    )
                    != payload["consultation"]["reviewer_record"]
                ):
                    raise ValueError("Reviewer journal differs from its frozen report.")
        public_observations(
            tuple(
                Observation.model_validate_json(json.dumps(row))
                for row in payload["observations"]
            )
        )
        if "report_questions" in payload:
            from .question_session import public_questions

            if not isinstance(payload["report_questions"], dict):
                raise ValueError("Saved question context must be an object.")
            public_questions(payload["report_questions"])
        if "photo_experiment" in payload:
            from .photo_experiments import public_experiment

            public_experiment(payload["photo_experiment"])
        from .archive_rendering import render_saved_report

        render_saved_report(payload)

    def without_images(self) -> SessionContext:
        """A full report can retain crops and simulations even without originals."""
        return self.model_copy(update={"saved_report": None, "workflow": None})
