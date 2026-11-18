"""Cross-check editable corrections against their frozen visible representation."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from .appearance import FAMILIES, VERSION

if TYPE_CHECKING:
    from .history_workflow import ArchivedWorkflow, SourceContext


def validate_frozen_appearance(
    workflow: ArchivedWorkflow, sources: tuple[SourceContext, ...], payload: dict
) -> None:
    if workflow.appearance_json is None:
        return
    raw = json.loads(workflow.appearance_json)
    if payload.get("version") != VERSION or payload.get("revision") != raw["revision"]:
        raise ValueError(
            "Appearance journal version or revision differs from its frozen report."
        )
    families = {family["id"]: family for family in payload["families"]}
    if set(families) != set(FAMILIES) or len(payload["families"]) != len(FAMILIES):
        raise ValueError("Frozen appearance families are incompatible.")
    by_id = {s.photo_id: s for s in sources}
    for key, override in raw["overrides"].items():
        photo, family = key.rsplit(":", 1)
        source = by_id[photo]
        rows = [
            row
            for row in families[family]["findings"]
            if row["photo_number"] == source.photo_number
        ]
        visible = override["visibility"] not in {"obscured", "not_visible", "uncertain"}
        if family in {"hair", "ears", "neck"}:
            visible = visible and override["visibility"] == "visible"
        if family == "smile":
            visible = visible and source.role == "smile"
        if not visible:
            if rows:
                raise ValueError(
                    "A hidden correction has a conflicting visible frozen finding."
                )
            continue
        if len(rows) != 1:
            raise ValueError("Visible correction is missing from its frozen report.")
        row = rows[0]
        expected_method = (
            "user-reviewed-image-region-v1"
            if override["visibility"] == "visible"
            else "user-adjusted-region-proposal-v1"
        )
        if row["method"] != expected_method or row["source_dimensions"] != [
            source.width,
            source.height,
        ]:
            raise ValueError(
                "Frozen region method or image frame differs from its editable correction."
            )
        for field in ("source_sha256", "box", "visibility", "note"):
            if row[field] != override[field]:
                raise ValueError("Editable correction differs from its frozen finding.")
        if row["role"] != source.role:
            raise ValueError("Frozen expression label differs from its source context.")
        conditions = (
            f'Lighting: {override["lighting"]}; '
            + ", ".join(
                f'{field.replace("_"," ")}: {override[field]}'
                for field in ("glasses", "makeup", "facial_hair")
            )
            + "."
        )
        if re.findall(r"Lighting: [^.]*\.", row["uncertainty"]) != [conditions]:
            raise ValueError(
                "Editable lighting or covering labels differ from the frozen finding."
            )
        adjusted = override["points"] is not None
        if row["geometry_method"] != (
            "user-adjusted-image-points-v1" if adjusted else "detector-image-points-v1"
        ):
            raise ValueError(
                "Editable endpoint method differs from its frozen finding."
            )
        if adjusted:
            points = {
                "p"
                + name: (
                    [value[0] * source.width, value[1] * source.height]
                    if value is not None
                    else None
                )
                for name, value in override["points"].items()
            }
            if row["points"] != points:
                raise ValueError(
                    "Editable endpoints differ from their frozen image coordinates."
                )
