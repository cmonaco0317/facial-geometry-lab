#!/usr/bin/env python3
"""Records when a fallback fired, because every fallback here fails FLATTERINGLY.

THE FINDING THIS EXISTS FOR — 2026-09-04 self-audit
-----------------------------------------------------
Six handlers in shipping stages swallow an exception and return a fabricated
value. Each is individually documented and individually defensible. Read
together they share one direction:

    analyze_catalogue._subject_scalar  -> 1.0          no widening of any margin
    analyze_advanced.race_evidence     -> (None, None) no race gate, so no refusal
    analyze_extended._phrase_gated     -> False        "no tiers means no refusals"

⛔ EVERY ONE OF THEM FAILS TOWARD MORE CONFIDENCE. `_subject_scalar` exists ONLY
to widen error bars — it is clamped at 1.0 so it can never narrow one — so the
value it returns when it breaks is the narrowest report the code can produce. The
race gate can only ever REFUSE, so losing it can only ever ASSERT. A tier lookup
that fails says "no new refusals" in its own comment.

So a broken dependency does not make this instrument cautious. It makes it
confident, and says nothing. That is the exact failure this project was built to
refuse, arriving through the back door of its own error handling.

⭐ WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT DO. It does not change any
fallback's VALUE — doing that would alter every number the report prints, and the
fallbacks are correct as behaviour. It makes them AUDIBLE: each records that it
fired, with where and why, and the record is written where a guard and a reader
can both find it. A degradation that is disclosed is a limitation; one that is
silent is a lie of omission.

⚠️ NOT A CRASH HANDLER. Recording is best-effort and never raises — a failure to
record a failure must not take down a run that would otherwise have produced a
report with a stated limitation.
"""

from __future__ import annotations

import json
import os
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "degradations.json")
_LOCK = threading.Lock()
_EVENTS: list[dict] = []


def record(where: str, what: str, effect: str = "unknown") -> None:
    """Note that a fallback fired. Never raises.

    `effect` says which DIRECTION the fallback moved the report, because that is
    the part a reader needs and the part that is easy to leave out:
    "more confident" | "more cautious" | "neutral" | "unknown".
    """
    try:
        with _LOCK:
            _EVENTS.append({"where": where, "what": str(what)[:400],
                            "effect": effect})
            _flush()
    except Exception:  # noqa: BLE001 - see the docstring: never take down a run
        pass


def _flush() -> None:
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({
            "_what": "fallbacks that fired during this run, and which way each "
                     "moved the report",
            "_why": "every swallowing handler in a shipping stage returns a value "
                    "that makes the report MORE confident, so a broken dependency "
                    "quietly produces a more assertive document rather than a "
                    "more careful one",
            "n": len(_EVENTS),
            "any_more_confident": any(e["effect"] == "more confident"
                                      for e in _EVENTS),
            "events": _EVENTS,
        }, fh, indent=2)
    os.replace(tmp, PATH)


def reset() -> None:
    """Start a run clean. Called by the pipeline, not by stages."""
    with _LOCK:
        _EVENTS.clear()
        _flush()


def events() -> list[dict]:
    with _LOCK:
        return list(_EVENTS)


def load() -> dict:
    if not os.path.exists(PATH):
        return {"n": 0, "events": [], "any_more_confident": False}
    try:
        with open(PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {"n": 0, "events": [], "any_more_confident": False}
