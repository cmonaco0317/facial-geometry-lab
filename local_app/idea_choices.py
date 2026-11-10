"""Bounded per-session idea bookmarks; no external persistence or user prose."""

from __future__ import annotations

from .recommendations import RULES


def normalize(raw):
    if raw is None:
        return dict(revision=1, saved=[], dismissed=[])
    ids = {r.id for r in RULES}
    if (
        not isinstance(raw, dict)
        or set(raw) != {"revision", "saved", "dismissed"}
        or type(raw["revision"]) is not int
        or raw["revision"] < 1
    ):
        raise ValueError("Invalid idea choices.")
    for field in ("saved", "dismissed"):
        values = raw[field]
        if (
            not isinstance(values, list)
            or len(values) > len(ids)
            or any(not isinstance(v, str) or v not in ids for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("Invalid idea choices.")
    if set(raw["saved"]) & set(raw["dismissed"]):
        raise ValueError("An idea cannot be saved and dismissed at once.")
    return dict(
        revision=raw["revision"],
        saved=sorted(raw["saved"]),
        dismissed=sorted(raw["dismissed"]),
    )


def update(raw, action, idea, *, expected):
    state = normalize(raw)
    if type(expected) is not int or expected != state["revision"]:
        raise RuntimeError(
            "Idea choices changed. Reload the report before changing them."
        )
    if (
        action not in ("save", "unsave", "dismiss", "restore")
        or not isinstance(idea, str)
        or idea not in {r.id for r in RULES}
    ):
        raise ValueError("Choose a supported idea action.")
    if action == "save":
        state["saved"] = sorted(set(state["saved"]) | {idea})
        state["dismissed"] = [s for s in state["dismissed"] if s != idea]
    elif action == "dismiss":
        state["dismissed"] = sorted(set(state["dismissed"]) | {idea})
        state["saved"] = [s for s in state["saved"] if s != idea]
    else:
        field = "saved" if action == "unsave" else "dismissed"
        state[field] = [s for s in state[field] if s != idea]
    state["revision"] += 1
    return state
