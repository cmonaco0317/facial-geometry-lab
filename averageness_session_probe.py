#!/usr/bin/env python3
"""Does the averageness percentile survive being photographed on another day?

THE QUESTION THIS DECIDES
-------------------------
Averageness is one of only two axes this report calls ROBUST, and since
2026-08-04 the SYMMETRIC COMPONENT is its primary figure -- promoted on
measurement, because across a reference-bank pose sweep it moved 3.4 percentile
points where the full-shape figure moved 32.4.

Both of those numbers answer questions about the REFERENCE BANK and about HEAD
TURN. Neither answers the question a paying user actually asks, which is:

    if I photograph myself again next week, do I get the same number?

That has never been measured. Every reproducibility figure this project holds
for averageness comes from ONE sitting -- the 26-second burst -- so it cannot
separate "this instrument is stable" from "these ten photographs were taken 26
seconds apart".

WHY THIS IS NOT THE POSE QUESTION AGAIN
---------------------------------------
Only photographs INSIDE the shipped pose gate are used (|yaw| <= 5 deg, the
refuse threshold from reference_values `pose_gate.averageness_yaw_deg`). If the
reading moves anyway, head turn is not the explanation and the gate is not the
fix. That is the whole design: hold the thing already known to matter constant,
and see what is left.

THE DECOMPOSITION THAT MAKES IT RIGOROUS
----------------------------------------
The available in-gate frontals fall into three sittings, and one of those
sittings contributed several frames. So the spread splits:

  WITHIN-SESSION  -- frames minutes apart, same light, same distance, same day.
                     This is the detector-and-expression floor.
  BETWEEN-SESSION -- different day, light, distance, camera framing.
                     This is what a returning user experiences.

A large between-session spread sitting on top of a small within-session one is
the finding. If both are large, the instrument is simply noisy and the burst
already said so. If both are small, averageness ships as-is.

⛔ WHAT WOULD MAKE THIS PROBE WRONG
-----------------------------------
It must score the subject through the SHIPPED code path -- `make_landmarker` +
`landmarks_for`, exactly as `analyze_averageness2.load_subject` does, and the
same `analyze_set` the report calls. It must NOT read `landmarks_frontal.csv`,
which on the set path is the multi-view fit: scoring a fitted subject against a
raw single-photo bank is the MULTIVIEW-WIRING 5.1 hazard, measured and confirmed
on 2026-08-11 (claim `averageness-must-not-read-the-fit`). A probe that quietly
"tidied" its subject loading into consistency with the rest of the pipeline
would measure the hazard instead of the question.

THE BAR IT IS JUDGED AGAINST
----------------------------
1. **The CI the report already prints.** That interval is a bootstrap over the
   REFERENCE POOL -- sampling error of the 243-face bank. If the same face
   photographed on another day lands outside it, the printed interval is not
   wrong arithmetic, it is answering a narrower question than the reader will
   take it to answer.
2. **The 25%-of-scale working rule** from the 2026-08-11 proposal brief: a
   quantity whose measurement error exceeds ~25% of the scale it is judged on is
   REFUSED, bar or no bar. Averageness is a percentile, so the scale is 100
   points and the cutoff is 25 points.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import analyze_averageness2 as A
import intake

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "averageness_session_probe.json")

# The shipped refuse threshold for averageness, read rather than retyped.
try:
    import reference_values as RV

    GATE_REFUSE_DEG = float(RV.require("pose_gate.averageness_yaw_deg").value[1])
except Exception:  # pragma: no cover - reference_values is always present
    GATE_REFUSE_DEG = 5.0

# Below intake's own floor a frame is down-weighted as "small in frame". Such a
# photograph is a resolution result, not a session result, so it is excluded and
# the exclusion is reported rather than silently applied.
MIN_IPD_PX = intake.MIN_IPD_PX

# Percentile scale, and the fraction of it an error may occupy (proposal brief
# trap #1: a margin of error is not a licence to print anything).
SCALE_PTS = 100.0
MAX_ERROR_FRAC = 0.25

CANDIDATES = [
    # Private development note omitted.
    # them. Two more sittings for free -- but they are NOT equivalent to the
    # staged frames and the difference is reported rather than smoothed over:
    # their IPDs are 180/186 px against 308-675 for the deliberate shots, so if
    # the spread grows when they are added, resolution is a live explanation
    # alongside session. The probe prints per-photograph IPD for exactly this.
    # ⛔ The rest of the camera roll was REJECTED, and why is worth keeping: the
    # Photos library's 14 detected faces are from 2024-08 to 2025-06, one to two
    # years old. Every reproducibility figure in this project rests on "one face
    # that cannot have changed"; across two years it can, so those photographs
    # would mix real facial change into a number reported as measurement error --
    # OVERSTATING the defect, which is the unflattering direction and therefore
    # the one that is easy to accept without checking.
    ("captures/_extra_frontals/IMG_1048.png", "2026-05-25"),
    ("captures/_extra_frontals/IMG_1265.png", "2026-06-18"),
    ("_two_photo_inputs/frontal.png", "2026-06-16"),
    ("captures/2026-08-01_cross-session-pair/IMG_1870.jpg", "2026-08-01"),
    ("captures/2026-08-01_pose-burst/IMG_1886.png", "2026-08-01"),
    ("captures/2026-08-01_pose-burst/IMG_1887.png", "2026-08-01"),
    ("captures/2026-08-12_second-set/IMG_2052.png", "2026-08-12_a"),
    # THIRD SITTING, shot 2026-08-12 17:55 — about 2.5 hours after the second set
    # and in a different room. ⚠️ Labelled as its own sitting but the SAME DAY,
    # which is a distinction this probe could not previously draw: every other
    # pair of sittings here is also a different day, so "sitting" and "day" were
    # confounded. These three frames separate them. Two of them (IMG_2061/2062)
    # are at essentially identical distance (330 and 329 px IPD) with the phone
    # put down and re-framed between, which is the constant-distance re-framing
    # pair the within-sitting term needs.
    ("captures/2026-08-12_third-set/IMG_2061.png", "2026-08-12_b"),
    ("captures/2026-08-12_third-set/IMG_2062.png", "2026-08-12_b"),
    ("captures/2026-08-12_third-set/IMG_2063.png", "2026-08-12_b"),
    # ⭐ THE CONTROLLED RESOLUTION FRAME, shot 18:19 — twenty minutes after the
    # other three, same room, same light, deliberately from further back. IPD
    # 136px against their 330/329/380. Labelled as the SAME sitting on purpose:
    # that is what makes it a control rather than another confounded data point.
    # The 180/186px camera-roll photographs that suggested this effect were each
    # their own sitting, so resolution and session could not be separated there.
    # Here session, room, light and day are all held fixed and distance is
    # essentially the only thing that changed.
    ("captures/_extra_frontals/IMG_2083.png", "2026-08-12_b"),
    ("captures/_reference_sheet/frontal_good.jpg", "2026-08-02"),
]


def score_photo(path: str, cfgs: dict, perm) -> dict | None:
    """Score one photograph exactly as load_subject + analyze_set would."""
    lmk = A.make_landmarker()
    try:
        L, _ = A.landmarks_for(lmk, path)
    finally:
        lmk.close()
    if L is None or L.shape != (468, 2):
        return None
    return {name: A.analyze_set(L, cfg, perm) for name, cfg in cfgs.items()}


def spread(vals: list[float]) -> float:
    return float(max(vals) - min(vals)) if len(vals) > 1 else 0.0


def main() -> int:
    print("=" * 78)
    print("AVERAGENESS ACROSS SESSIONS — does the number survive another day?")
    print("=" * 78)

    # ---- select in-gate photographs ---------------------------------------
    kept, dropped = [], []
    for rel, session in CANDIDATES:
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            dropped.append((rel, session, "missing"))
            continue
        r = intake.assess_photo(p)
        yaw, ipd = r.get("yaw"), r.get("ipd_px")
        if yaw is None:
            dropped.append((rel, session, "no face"))
        elif abs(yaw) > GATE_REFUSE_DEG:
            dropped.append(
                (rel, session, f"yaw {yaw:+.1f} past the {GATE_REFUSE_DEG:g}° gate")
            )
        elif ipd is not None and ipd < MIN_IPD_PX:
            dropped.append(
                (
                    rel,
                    session,
                    f"IPD {ipd:.0f}px under intake's {MIN_IPD_PX:.0f}px floor",
                )
            )
        else:
            kept.append(
                {"path": p, "rel": rel, "session": session, "yaw": yaw, "ipd": ipd}
            )

    print(f"\n  in-gate photographs used : {len(kept)}")
    for k in kept:
        print(
            f"    {k['session']}  {os.path.basename(k['rel']):22s} yaw {k['yaw']:+5.1f}  IPD {k['ipd']:.0f}px"
        )
    if dropped:
        print("  excluded, with reason:")
        for rel, session, why in dropped:
            print(f"    {session}  {os.path.basename(rel):22s} {why}")

    sessions = sorted({k["session"] for k in kept})
    if len(kept) < 2 or len(sessions) < 2:
        print("\n!! not enough in-gate photographs across sessions to measure this")
        return 1

    # ---- score every one through the shipped path -------------------------
    perm = A.load_mirror_perm()
    cache = A.build_cache()
    cfgs = {name: cache[name] for name in A.SETS}
    cfgs["white_all"] = np.concatenate(
        [cache["white_10_19"], cache["white_20_29"]], axis=0
    )

    for k in kept:
        k["scores"] = score_photo(k["path"], cfgs, perm)
        if k["scores"] is None:
            print(f"!! detection failed on {k['rel']}")
            return 1

    # ---- report per reference set -----------------------------------------
    results: dict = {}
    print()
    print("=" * 78)
    print("READINGS — one unchanging face, every photograph inside the pose gate")
    print("=" * 78)

    for setname in list(cfgs.keys()):
        sym = [k["scores"][setname]["symmetric_typicality_pct"] for k in kept]
        symz = [k["scores"][setname]["symmetric_z"] for k in kept]
        full = [k["scores"][setname]["percentile"] for k in kept]
        ci = kept[0]["scores"][setname]["symmetric_ci"]
        ci_full = kept[0]["scores"][setname]["ci"]

        within = {}
        for s in sessions:
            v = [
                k["scores"][setname]["symmetric_typicality_pct"]
                for k in kept
                if k["session"] == s
            ]
            if len(v) > 1:
                within[s] = spread(v)
        within_max = max(within.values()) if within else None

        # Between-session spread on session MEANS, so a sitting that happened to
        # contribute more frames does not dominate the figure.
        sess_means = [
            float(
                np.mean(
                    [
                        k["scores"][setname]["symmetric_typicality_pct"]
                        for k in kept
                        if k["session"] == s
                    ]
                )
            )
            for s in sessions
        ]
        between = spread(sess_means)

        sym_spread, full_spread = spread(sym), spread(full)
        covered = min(sym) >= ci[0] and max(sym) <= ci[1]

        results[setname] = {
            "symmetric_readings": sym,
            "symmetric_z_readings": symz,
            "full_shape_readings": full,
            "symmetric_spread_pp": round(sym_spread, 1),
            "full_shape_spread_pp": round(full_spread, 1),
            "within_session_spread_pp": (
                None if within_max is None else round(within_max, 1)
            ),
            "between_session_spread_pp": round(between, 1),
            "printed_symmetric_ci": ci,
            "printed_full_ci": ci_full,
            "all_readings_inside_printed_ci": bool(covered),
            "passes_25pct_of_scale_rule": bool(
                sym_spread <= MAX_ERROR_FRAC * SCALE_PTS
            ),
        }

        print(f"\n  {setname}")
        print(f"    symmetric (PRIMARY) : {[round(v,1) for v in sym]}")
        print(
            f"      spread            : {sym_spread:.1f} pp"
            f"   (within-session {('%.1f' % within_max) if within_max is not None else 'n/a'} pp,"
            f" between-session {between:.1f} pp)"
        )
        print(
            f"      printed CI        : {ci}   -> all readings inside it: {'YES' if covered else 'NO'}"
        )
        print(
            f"      <= 25% of scale   : {'YES' if sym_spread <= 25 else 'NO'}  ({sym_spread:.1f} of 100 points)"
        )
        print(
            f"    full shape (SECONDARY): {[round(v,1) for v in full]}   spread {full_spread:.1f} pp"
        )

    # ---- verdict -----------------------------------------------------------
    prim = [results[s] for s in cfgs]
    worst = max(r["symmetric_spread_pp"] for r in prim)
    worst_between = max(r["between_session_spread_pp"] for r in prim)
    any_within = [
        r["within_session_spread_pp"]
        for r in prim
        if r["within_session_spread_pp"] is not None
    ]
    worst_within = max(any_within) if any_within else None
    all_covered = all(r["all_readings_inside_printed_ci"] for r in prim)
    passes_rule = all(r["passes_25pct_of_scale_rule"] for r in prim)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(
        f"  worst symmetric spread across in-gate photographs : {worst:.1f} percentile points"
    )
    if worst_within is not None:
        print(
            f"    of which within a single sitting                : {worst_within:.1f} pp"
        )
    print(
        f"    across sittings (session means)                 : {worst_between:.1f} pp"
    )
    print(
        f"  every reading inside the CI the report prints      : {'YES' if all_covered else 'NO'}"
    )
    print(
        f"  passes the 25%-of-scale rule                       : {'YES' if passes_rule else 'NO'}"
    )
    print()
    if not all_covered:
        print(
            "  The printed interval is a bootstrap over the REFERENCE POOL. It does not"
        )
        print(
            "  contain the variation the same face produces on another day, so a reader"
        )
        print("  who takes it as 'how much this number could move' is being misled.")
    if not passes_rule:
        print(
            "  Past the working rule from the proposal brief: an error this large cannot"
        )
        print("  support the verdict printed beside it. A bar does not rescue it.")

    # ---- resolution is confounded with session, so say so ------------------
    # Added 2026-08-12 after two camera-roll photographs were pooled in and the
    # headline spread jumped 27.8 -> 48.1 points. They are also the only two
    # below 200px IPD, and they returned the two HIGHEST readings of the seven --
    # every low-resolution photograph above every high-resolution one. Each is
    # its own sitting, so resolution and session cannot be separated in this
    # pool, and the pooled figure must NOT be quoted as cross-session variation.
    # ⛔ The comparable-resolution figure is not a "cleaned" result either: the
    # 200px line was drawn AFTER seeing which photographs it separates, which is
    # the same move as widening a matching tolerance to admit a pair that helps.
    # Both figures are printed, neither is called the answer, and the split is
    # reported as a hypothesis needing its own test.
    print()
    print("=" * 78)
    print("⚠️ RESOLUTION IS CONFOUNDED WITH SESSION IN THIS POOL")
    print("=" * 78)
    ipds = [k["ipd"] for k in kept]
    ref_for_split = "white_all" if "white_all" in results else list(results)[0]
    reads = results[ref_for_split]["symmetric_readings"]
    lo_r = [v for k, v in zip(kept, reads) if k["ipd"] < 200]
    hi_r = [v for k, v in zip(kept, reads) if k["ipd"] >= 200]
    print(f"  IPD range across the pool: {min(ipds):.0f}-{max(ipds):.0f}px")
    if lo_r and hi_r:
        print(f"  below 200px IPD ({len(lo_r)}): {[round(v,1) for v in lo_r]}")
        print(f"  at or above     ({len(hi_r)}): {[round(v,1) for v in hi_r]}")
        ordered = min(lo_r) > max(hi_r)
        print(
            f"  every low-resolution reading above every high-resolution one: {'YES' if ordered else 'no'}"
        )
        print(
            f"  spread among comparable-resolution photographs only: {spread(hi_r):.1f} pp"
        )
        print(
            f"  spread pooled                                       : {spread(reads):.1f} pp"
        )
        if ordered:
            print()
            print(
                "  ⛔ DO NOT quote the pooled figure as cross-session variation. Each"
            )
            print(
                "  low-resolution photograph is also its own sitting, so the two cannot"
            )
            print(
                "  be separated here. And note the DIRECTION: lower resolution reads MORE"
            )
            print("  average, which is the flattering direction, and is NOT the noise")
            print(
                "  mechanism 5.1 describes — noise inflates a distance and would read LESS"
            )
            print(
                "  average. A detector falling back toward its own prior when there is less"
            )
            print(
                "  detail to fit would do this. That is a hypothesis, not a result: n=2."
            )

    payload = {
        "_what": "averageness reproducibility across capture sessions, inside the pose gate",
        "_resolution_confound": {
            "ipd_min_px": round(min(ipds), 1),
            "ipd_max_px": round(max(ipds), 1),
            "below_200px_readings": [round(v, 1) for v in lo_r],
            "at_or_above_200px_readings": [round(v, 1) for v in hi_r],
            "every_low_res_above_every_high_res": bool(
                lo_r and hi_r and min(lo_r) > max(hi_r)
            ),
            "spread_comparable_resolution_only_pp": (
                round(spread(hi_r), 1) if hi_r else None
            ),
            "spread_pooled_pp": round(spread(reads), 1),
            "_note": (
                "Resolution and session are perfectly confounded in this pool -- each "
                "sub-200px photograph is its own sitting -- so the pooled spread must not be "
                "quoted as cross-session variation. The shipped reference value stays the "
                "comparable-resolution figure."
            ),
        },
        "_method": (
            "Every in-gate frontal photograph on disk is scored through the SHIPPED path "
            "(make_landmarker + landmarks_for, as load_subject does; then analyze_set). "
            "landmarks_frontal.csv is never read -- on the set path it is the multi-view fit, "
            "and scoring a fitted subject against a raw single-photo bank is the 5.1 hazard."
        ),
        "_gate_deg": GATE_REFUSE_DEG,
        "photographs": [
            {
                "file": os.path.basename(k["rel"]),
                "session": k["session"],
                "yaw": k["yaw"],
                "ipd_px": k["ipd"],
            }
            for k in kept
        ],
        "excluded": [
            {"file": os.path.basename(r), "session": s, "why": w} for r, s, w in dropped
        ],
        "n_sessions": len(sessions),
        "by_reference_set": results,
        "worst_symmetric_spread_pp": round(worst, 1),
        "worst_within_session_spread_pp": (
            None if worst_within is None else round(worst_within, 1)
        ),
        "worst_between_session_spread_pp": round(worst_between, 1),
        "all_readings_inside_printed_ci": bool(all_covered),
        "passes_25pct_of_scale_rule": bool(passes_rule),
        "_limit": (
            "n=1 subject and a handful of photographs across three sittings. This establishes "
            "that the reading MOVES and roughly how far; it does not calibrate a distribution, "
            "and a shipped error bar must not be fitted to it."
        ),
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {os.path.basename(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
