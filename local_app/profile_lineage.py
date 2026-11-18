"""Bounded proposal ancestry, separate from every frozen numerical instrument."""

import json
from typing import Annotated, Literal

from pydantic import Field

from .observations import Frozen, Token

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class ProposalOrigin(Frozen):
    method: Token
    contract: Digest
    proposal_id: Digest
    model_sha256: Digest
    changed_anchors: tuple[
        Literal[
            "nasion", "tip", "subnasale", "upper_lip", "lower_lip", "pogonion", "menton"
        ],
        ...,
    ] = Field(default=(), max_length=7)

    def description(self) -> str:
        changed = ", ".join(self.changed_anchors) or "none"
        return f"Initial positions came from {self.method}; proposal {self.proposal_id}, contract {self.contract}, model {self.model_sha256}. Anchors edited after proposal: {changed}. A confirmed save is not independent placement or anatomical accuracy evidence."


def proposal_origin(raw) -> ProposalOrigin | None:
    return ProposalOrigin.model_validate_json(json.dumps(raw)) if raw else None
