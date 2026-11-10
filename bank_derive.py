#!/usr/bin/env python3
"""Derived metrics for every bank cohort — a CACHE, never a source of truth.

`bank/<cohort>/measurements.npz` holds 468x2 landmarks and nothing else, because
storing derived metrics in the bank would freeze today's definitions into it and
make tomorrow's definition change a refetch.  This module computes every metric
the report ranks -- the catalogue ratios and the dimorphism features -- from
those landmarks, and writes `bank/derived.npz`.

⛔ IT IS SAFE TO DELETE `bank/derived.npz` AT ANY TIME.  That is the test of
whether something is a cache.  If a consumer breaks in a way rebuilding cannot
fix, the thing was not a cache and the design is wrong.

⭐ THE CACHE CARRIES THE INSTRUMENT DIGEST OF THE BANK IT WAS DERIVED FROM.
A derived file computed from an older bank is exactly the "one label over two
quantities" defect, one level down.

    ./venv/bin/python bank_derive.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dimorphism_bank_probe as DBP  # noqa: E402
import facial_catalogue as FC  # noqa: E402
import harmony_axes as HA  # noqa: E402
import symmetry_population as SP  # noqa: E402

BANK = os.path.join(HERE, "bank")
OUT = os.path.join(BANK, "derived.npz")


# ⛔ THE TWO FAMILIES ARE NAMESPACED BECAUSE THEY COLLIDE.
# `lower_third` is a CATALOGUE ratio (lower third as a % of face height) AND a
# DIMORPHISM feature (lower-face height / eye spacing). They are different
# quantities. Written into one dict under one key, the second silently
# overwrote the first -- which is the one-label-over-two-quantities defect this
# project has now hit five times, and it produced a nonsense 107%-of-scale
# movement figure by dividing one family's value by the other family's scale.
# The prefix is derived from the family, not from a list of known collisions,
# so a future collision cannot reintroduce this.
DIM_PREFIX = "dim__"


HARM_PREFIX = "harm__"


def metric_ids() -> tuple[list[str], list[str]]:
    ratios = [c[0] for c in FC.CATALOGUE if c[3] == "ratio"]
    feats = ([DIM_PREFIX + f[0] for f in DBP.FEATURES]
             + [HARM_PREFIX + a for a in HA.AXES] + ["asymmetry"])
    return ratios, feats


def metrics_for(p: np.ndarray) -> dict[str, float]:
    """⭐ THE ONE DEFINITION, used for a BANK face and a SUBJECT face alike.

    `bank-and-subject-are-measured-by-the-same-instrument` is a claim about code
    identity, and it is only true if the two sides call the SAME function. This
    is that function: every probe that measures a stranger, a subject or a bank
    face imports it from here rather than assembling its own dict, because two
    assembled dicts are two vocabularies and they drift.
    """
    out: dict[str, float] = {}
    p = np.asarray(p, dtype=np.float64)
    m = FC.measure_all(p)
    for cid, _l, _fn, unit, *_ in FC.CATALOGUE:
        if unit == "ratio" and m.get(cid) is not None and np.isfinite(m[cid]):
            out[cid] = float(m[cid])
    try:
        d = DBP.measures(p)
        for fid, _l, fn, _hi in DBP.FEATURES:
            v = float(fn(d))
            if np.isfinite(v):
                out[DIM_PREFIX + fid] = v
    except Exception:  # noqa: BLE001
        pass
    for k, v in HA.magnitudes(p).items():
        if np.isfinite(v):
            out[HARM_PREFIX + k] = float(v)
    a = SP.asym_of(p)
    if a is not None and np.isfinite(a):
        out["asymmetry"] = float(a)
    return out


def derive_cohort(L: np.ndarray) -> dict[str, np.ndarray]:
    ratios, feats = metric_ids()
    acc = {k: np.full(L.shape[0], np.nan) for k in ratios + feats}
    for i in range(L.shape[0]):
        for k, v in metrics_for(L[i]).items():
            if k in acc:
                acc[k][i] = v
    return acc


def main() -> int:
    prov = json.load(open(os.path.join(BANK, "provenance.json")))
    # ⛔ THE PROVENANCE IS THE COHORT LIST, NOT THE DIRECTORY.
    # Globbing bank/*/ measured whatever happened to be on disk, which on
    # 2026-09-07 included 36 cohorts left behind by an earlier partial build --
    # 98 measured where the build emitted 62, and the dimorphism layer downstream
    # reported 49 cohorts instead of 30 without anything going red. One build
    # says what it produced; that record is what gets read.
    out, meta, t0 = {}, {}, time.time()
    for cohort in sorted(prov["cohorts"]):
        cdir = os.path.join(BANK, cohort, "measurements.npz")
        if not os.path.exists(cdir):
            raise SystemExit(f"bank/{cohort}/measurements.npz is missing but "
                             f"provenance declares it — rebuild with build_bank.py")
        L = np.load(cdir)["landmarks"]
        acc = derive_cohort(L)
        for k, v in acc.items():
            out[f"{cohort}|{k}"] = v.astype(np.float32)
        meta[cohort] = {k: int(np.isfinite(v).sum()) for k, v in acc.items()}
        print(f"  {cohort:34s} n={L.shape[0]:5d}  "
              f"{time.time()-t0:6.1f}s", flush=True)
    np.savez_compressed(OUT, **out)
    with open(os.path.join(BANK, "derived_meta.json"), "w") as f:
        json.dump({
            "_what": "a CACHE derived from bank/*/measurements.npz. Safe to delete.",
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "from_bank_instrument_digest": prov["instrument"]["digest"],
            "n_cohorts": len(meta),
            "finite_counts": meta,
        }, f, indent=2)
    print(f"\nwrote {OUT}  ({len(out)} arrays, {len(meta)} cohorts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
