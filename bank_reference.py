#!/usr/bin/env python3
"""The cohort reference, READ FROM THE BUILT BANK — a drop-in for `race_reference`.

WHAT CHANGES FOR A CONSUMER
---------------------------
Nothing but the import.  `COHORTS` and `DIMORPHISM_BY_RACE` carry the same shape
`race_reference.py` carried, so every call site keeps its own error handling and
its own fallbacks.  What changes is underneath:

    race_reference.py     42 cohorts,   ~5,000 faces, 22 cohorts under the floor,
                          frozen into a 1.7 MB hand-uneditable .py
    bank/                 84 declared FairFace adult cells through 70+, plus
                          explicitly named widened cohorts where needed, rebuilt
                          by one command and counted by identity clusters

⭐ AND IT IS A LOADER, NOT A FROZEN FILE.  §5.3: re-measuring after an instrument
change has to be a compute job rather than a rebuild of source code, or "floor
first" is unaffordable and the bank silently ages instead.

⛔ THE INSTRUMENT GUARD IS A HARD FAILURE, NOT A WARNING.  If the instrument that
would measure a subject is not the one that measured the bank, `COHORTS` raises.
It does not quietly serve the old numbers.  A percentile computed across two
instruments is the one-label-two-quantities defect this project has hit four
times, and it is invisible in the output when it happens.

⚠️ LABEL HONESTY IS CARRIED IN THE DATA, NOT LEFT TO THE CALLER.  Every cohort
record ships a `label` reading "faces labelled X by FairFace annotators".
FairFace's ethnicity labels are annotator judgements about photographs -- not
self-report, not biology -- and any surface naming a cohort must say so.
"""

from __future__ import annotations

import json
import os
import sys
from types import MappingProxyType

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BANK = os.path.join(HERE, "bank")

_STATE: dict = {}


def _read_json(filename: str):
    with open(os.path.join(BANK, filename), encoding="utf-8") as stream:
        return json.load(stream)


def _load():
    if _STATE:
        return _STATE
    import bank_instrument as BI
    import public_reference
    prov = _read_json("provenance.json")
    public_reference.metadata()
    ok, message = BI.check_against(prov["instrument"])
    if not ok:
        raise RuntimeError(f"PUBLIC BANK INSTRUMENT MISMATCH — {message}")
    _STATE.update(prov=prov, dim=_read_json("dimorphism.json"),
                  cal=_read_json("dimorphism_calibration.json"), instrument_ok=message)
    return _STATE


def _split(cohort: str) -> tuple[str, str, str]:
    """(sex, race, age band) from a cohort id.

    ⚠️ A WIDENED COHORT IS `<sex>_<race>_<lo1>_<lo2>_to_<hi1>_<hi2>`, and naive
    rsplit on "_" made its race come out as "east_asian_40_49_to". A race label
    that carries an age band inside it would then be printed as a race.
    """
    sex, rest = cohort.split("_", 1)
    sexw = "male" if sex == "m" else "female"
    if "_to_" in rest:
        left, _, right = rest.partition("_to_")
        race = left.rsplit("_", 2)[0]
        lo = "-".join(left.rsplit("_", 2)[1:])
        return sexw, race, f"{lo} to {right.replace('_', '-')}"
    parts = rest.rsplit("_", 2)
    return sexw, parts[0], "-".join(parts[1:])


def _cohort_label(sex: str, race: str, band: str) -> str:
    return (f"faces labelled {race.replace('_', ' ').title()}, "
            f"{'men' if sex == 'male' else 'women'} aged {band}, "
            f"by FairFace annotators")


def _label_for_meta(cohort_id: str, meta: dict, sex: str, race: str, band: str) -> str:
    sources = tuple(meta.get("sources", ("fairface",)))
    if sources == ("fairface",):
        return _cohort_label(sex, race, band)
    label = meta.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return f"faces in the disclosed {cohort_id} reference cohort"


def _source_disclosure(source_ids: tuple[str, ...]) -> str:
    registry = _read_json("source_registry.json").get("sources", {})
    disclosures = []
    for source_id in source_ids:
        source = registry.get(source_id, {})
        label_kind = source.get("label_kind")
        if isinstance(label_kind, str) and label_kind:
            disclosures.append(label_kind)
    return "; ".join(sorted(set(disclosures)))


def _source_names(source_ids: tuple[str, ...]) -> str:
    registry = _read_json("source_registry.json").get("sources", {})
    names = []
    for source_id in source_ids:
        source = registry.get(source_id, {})
        display_name = source.get("display_name", source_id)
        if isinstance(display_name, str) and display_name.strip():
            names.append(display_name.strip())
    return "; ".join(sorted(set(names)))


def cohort_inventory():
    """Return cohort metadata without exposing mutable loader state or arrays."""
    prov = _load()["prov"]
    inventory = {}
    for cohort_id, meta in prov["cohorts"].items():
        sex, race, band = _split(cohort_id)
        band_id = band.replace(" to ", "_to_").replace("-", "_")
        clusters = int(meta.get("identity_cluster_count", meta["n"]))
        source_ids = tuple(meta.get("sources", ("fairface",)))
        inventory[cohort_id] = MappingProxyType({
            "n": clusters,
            "identity_cluster_count": clusters,
            "source_rows": int(meta.get("source_rows", meta["n"])),
            "sex": sex,
            "ethnicity": race,
            "age_band": band_id,
            "label": _label_for_meta(cohort_id, meta, sex, race, band),
            "merged": bool(meta["merged"]),
            "meets_floor": bool(meta["meets_floor"]),
            "members": tuple(meta.get("members", (cohort_id,))),
            "sources": source_ids,
            "source_buckets": tuple(meta.get("source_buckets", ())),
            "source": _source_names(source_ids),
            "source_label_kind": _source_disclosure(source_ids),
        })
    return MappingProxyType(inventory)


class _Cohorts(dict):
    """Lazy: a cohort's rows are computed from the bank the first time it is asked
    for.  Eager loading would spend seconds of every report build on 56 cohorts
    when a subject belongs to exactly one."""

    def __init__(self):
        super().__init__()
        s = _load()
        self._prov = s["prov"]
        self._names = sorted(self._prov["cohorts"])
        for n in self._names:
            super().__setitem__(n, None)

    def __getitem__(self, k):
        v = super().__getitem__(k)
        if v is not None:
            return v
        meta = self._prov["cohorts"][k]
        sex, race, band = _split(k)
        rows = {}
        import public_reference
        for metric in public_reference.metric_names(k):
            a = public_reference.pool(k, metric)
            rows[metric] = {"n": int(a.size), "median": round(float(np.median(a)), 6),
                "p5": round(float(np.percentile(a, 5)), 6),
                "p95": round(float(np.percentile(a, 95)), 6), "values": a}
        # ⚠️ Built here from the parts, not copied from provenance: a cohort
        # name must never be printed as "people of ethnicity X". FairFace's
        # labels are annotator judgements about photographs -- not self-report,
        # not biology -- and the disclosure travels WITH the record so a caller
        # cannot print the cohort without it.
        label = _label_for_meta(k, meta, sex, race, band)
        cluster_count = int(meta.get("identity_cluster_count", meta["n"]))
        rec = {"n": cluster_count, "identity_cluster_count": cluster_count,
               "source_rows": int(meta.get("source_rows", meta["n"])),
               "sex": sex, "race": race, "age_band": band,
               "label": label, "merged": meta["merged"],
               "meets_floor": meta["meets_floor"],
               "bootstrap_interval_pp_est": meta["bootstrap_interval_pp_est"],
               "rows": rows}
        super().__setitem__(k, rec)
        return rec

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default

    # ⛔ LAZY MUST NOT MEAN "BROKEN FOR EVERY ACCESS EXCEPT []".
    # The first version stored None as a placeholder and inherited dict.values(),
    # so `{r["race"] for r in COHORTS.values()}` handed a caller a set of Nones
    # instead of raising -- silently wrong rather than loudly absent, which is the
    # worse of the two. Every read path materialises.
    def values(self):
        return [self[k] for k in self._names]

    def items(self):
        return [(k, self[k]) for k in self._names]

    def __iter__(self):
        return iter(self._names)

    def keys(self):
        return list(self._names)


def __getattr__(name):
    """Module-level lazy attributes, so importing this costs nothing."""
    if name == "COHORTS":
        v = _Cohorts()
        globals()["COHORTS"] = v
        return v
    if name == "DIMORPHISM_BY_RACE":
        v = _load()["dim"]["cohorts"]
        globals()["DIMORPHISM_BY_RACE"] = v
        return v
    if name == "CALIBRATION":
        v = _load()["cal"]["cohorts"]
        globals()["CALIBRATION"] = v
        return v
    if name in ("MIN_COHORT_N", "N_RACE_FACES", "BOOTSTRAP_SAMPLES", "N_COHORTS"):
        # Kept at the same names `race_reference` used so every consumer stays a
        # one-line import change. The VALUES are the built bank's, not the old
        # frozen file's -- MIN_COHORT_N is now the manifest's own 150-face floor
        # rather than the 40 the smaller bank could afford.
        prov = _load()["prov"]
        vals = {"MIN_COHORT_N": _read_json("manifest.json")["depth"]["floor"],
                "N_RACE_FACES": prov["n_faces_unique"],
                "BOOTSTRAP_SAMPLES": 2000,
                "N_COHORTS": prov["n_cohorts_emitted"]}
        globals()[name] = vals[name]
        return vals[name]
    if name == "PROVENANCE":
        v = _load()["prov"]
        globals()["PROVENANCE"] = v
        return v
    raise AttributeError(name)


if __name__ == "__main__":
    s = _load()
    print("instrument:", s["instrument_ok"])
    print("cohorts   :", len(s["prov"]["cohorts"]),
          " faces:", s["prov"]["n_faces_unique"])
    C = _Cohorts()
    k = "m_white_20_29"
    rec = C[k]
    print(f"\n{k}: n={rec['n']}  {rec['label']}")
    print(f"  floor met: {rec['meets_floor']}   sampling interval "
          f"~{rec['bootstrap_interval_pp_est']}pp   rows: {len(rec['rows'])}")
    for m in ("fwhr", "alar_ratio", "jaw_width"):
        if m in rec["rows"]:
            r = rec["rows"][m]
            print(f"  {m:12s} n={r['n']:5d} median {r['median']:.4f} "
                  f"p5 {r['p5']:.4f} p95 {r['p95']:.4f}")
    import bank_dimorphism  # noqa: F401
    d = _load()["dim"]["cohorts"]["white_20_29"]
    print("\ndimorphism white_20_29:",
          {k2: v2["verdict"] for k2, v2 in d.items()})
