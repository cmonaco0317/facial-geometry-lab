"""Replay appearance journal transitions across explicit source-slot epochs."""

from __future__ import annotations

from collections import Counter
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .history_workflow import ArchiveOrigin, SourceContext


def _counter(items) -> Counter:
    return Counter(
        json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for item in items
    )


def _palette(sources) -> dict:
    return {s.photo_number: s for s in sources} if sources is not None else {}


def validate_appearance_journal(
    raw: dict,
    origins: tuple[ArchiveOrigin, ...],
    sources: tuple[SourceContext, ...] | None = None,
) -> None:
    events = raw["history"]
    cutoffs = [o.appearance_revision for o in origins]
    if cutoffs != sorted(cutoffs) or any(cutoff > len(events) for cutoff in cutoffs):
        raise ValueError("Invalid appearance source epoch boundaries.")
    overrides, settings = {}, {}
    epoch_index = 0
    for revision in range(len(events) + 1):
        while (
            epoch_index < len(origins)
            and origins[epoch_index].appearance_revision == revision
        ):
            epoch = origins[epoch_index]
            old = _palette(epoch.sources)
            following = (
                origins[epoch_index + 1].sources
                if epoch_index + 1 < len(origins)
                else sources
            )
            new = _palette(following)
            for original, restored in epoch.source_map.items():
                before = old[int(original)]
                after = new.get(restored)
                if following is not None and (
                    after is None
                    or (before.source_sha256, before.width, before.height)
                    != (after.source_sha256, after.width, after.height)
                ):
                    raise ValueError(
                        "Appearance source epochs do not retain the same bytes and frame."
                    )
            overrides = {
                (epoch.source_map[str(n)], digest, family): value
                for (n, digest, family), value in overrides.items()
                if str(n) in epoch.source_map
            }
            settings = {
                (epoch.source_map[str(n)], digest): value
                for (n, digest), value in settings.items()
                if str(n) in epoch.source_map
            }
            epoch_index += 1
        if revision == len(events):
            break
        event = events[revision]
        number, digest, family = (
            event["photo_number"],
            event["source_sha256"],
            event["family"],
        )
        palette = (
            origins[epoch_index].sources if epoch_index < len(origins) else sources
        )
        if palette is not None:
            source = _palette(palette).get(number)
            if source is None or source.source_sha256 != digest:
                raise ValueError(
                    "Appearance history event does not match its source epoch."
                )
        key = (number, digest, family)
        photo = (number, digest)
        if event["previous"] != overrides.get(key) or event[
            "previous_photo"
        ] != settings.get(photo):
            raise ValueError("Broken appearance correction lineage.")
        expected_photo = (
            {"source_sha256": digest, "role": event["current"]["role"]}
            if event["action"] == "save"
            else event["previous_photo"]
        )
        if event["current_photo"] != expected_photo:
            raise ValueError(
                "Appearance action does not preserve its photo-setting transition."
            )
        if any(
            value is not None and value["source_sha256"] != digest
            for value in (
                event["previous"],
                event["current"],
                event["previous_photo"],
                event["current_photo"],
            )
        ):
            raise ValueError("Appearance event contains a conflicting source hash.")
        if event["current"] is None:
            overrides.pop(key, None)
        else:
            overrides[key] = event["current"]
        if event["current_photo"] is None:
            settings.pop(photo, None)
        else:
            settings[photo] = event["current_photo"]
    actual = [
        (row["source_sha256"], key.rsplit(":", 1)[1], row)
        for key, row in raw["overrides"].items()
    ]
    expected = [(digest, family, row) for (_, digest, family), row in overrides.items()]
    if _counter(actual) != _counter(expected) or _counter(
        raw.get("photo_settings", {}).values()
    ) != _counter(settings.values()):
        raise ValueError("Appearance active state does not match its journal.")
    if sources is not None:
        by_id = {s.photo_id: s for s in sources}
        actual_overrides = {}
        actual_settings = {}
        for key, row in raw["overrides"].items():
            photo, family = key.rsplit(":", 1)
            source = by_id.get(photo)
            if source is None:
                raise ValueError("Appearance photo binding is unavailable.")
            actual_overrides[(source.photo_number, source.source_sha256, family)] = row
        for photo, row in raw.get("photo_settings", {}).items():
            source = by_id.get(photo)
            if source is None:
                raise ValueError("Appearance photo binding is unavailable.")
            actual_settings[(source.photo_number, source.source_sha256)] = row
        if actual_overrides != overrides or actual_settings != settings:
            raise ValueError(
                "Appearance active source slots disagree with their journal."
            )
