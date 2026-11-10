#!/usr/bin/env python3
"""Sagittal measurements, and the 3D reference bank that finally lets them be normed.

THE BLOCKER THIS REMOVES, AND WHY IT WAS NEVER STRUCTURAL
----------------------------------------------------------
`face_angles.py` recorded a hard limit beside the refused convexity metric:

    "the 243-face reference bank is 2D LANDMARKS ONLY, so it has no z and can
     supply NO population norm for any sagittal quantity. Frontal-plane ratios
     can be normed against it; profile angles cannot be, by construction. Qoves
     ships normed profile measurements; this instrument cannot match that WITH
     THIS BANK."

The last three words were the whole of it. `mp468_embedding.py` (2026-08-12) put
all 468 MediaPipe points onto any FLAME-topology mesh, so the D3DFACS scans yield
the same points the fit produces — **in 3D**. A sagittal reference bank became
buildable out of material already on disk.

TWO POPULATIONS, AND THE SMALL ONE AUDITS THE LARGE ONE
--------------------------------------------------------
  * **REAL** — 40 meshes from 10 scanned humans. Honest, and far too few for a
    percentile: ranks over 10 subjects move in 10-point steps.
  * **SYNTHETIC** — identities sampled from FLAME's own prior, which is learned
    from real 4D head scans. Large enough to rank against.

⛔ THE SYNTHETIC BANK IS NEVER TRUSTED ON ITS OWN. For each metric the two
distributions are compared, and a metric is only normed against the synthetic
population when the real scans corroborate it — mean within 1 SD, spread within
a factor of ~2. A modelled population that nobody checked against real geometry
is a norm made of assumptions.

⚠️ WHAT A PERCENTILE HERE MEANS, AND DOES NOT. It is a rank against FLAME's
modelled identity space, corroborated by 10 real people. It is NOT a rank against
a demographically matched human population, and it carries none of the ethnic
matching the frontal bank at least attempts. Say so wherever it prints.

⛔ SCALE-FREE ONLY. Every metric here is an angle or a ratio. Absolute
millimetres stay gated behind the +-6% iris anchor, and a magnitude scored
against a bank is the SS5.1 hazard this project has been bitten by repeatedly.
"""

from __future__ import annotations

import os

import numpy as np

import fa_common as _FC
import face_angles as _FA

HERE = os.path.dirname(os.path.abspath(__file__))

PRONASALE = 4  # MediaPipe nose tip
NASION = 168

# The project's standing rule: an error bar wider than a quarter of the scale a
# reading is judged on cannot support the words printed beside it.
SHIP_MAX_FRAC_OF_SCALE = 0.25
SCALE_SD_RANGE = 4.0  # the bank's 4-SD range is the judged scale, stated not implied

# Agreement gates between the real and synthetic populations.
MAX_MEAN_SHIFT_SD = 1.0
SD_RATIO_BAND = (0.4, 2.5)


def _sagittal_angle(a, b, c) -> float:
    """Angle at b in the (y, z) plane — the plane a profile is actually read in.

    Measured in 2D on purpose. The construct these are compared against comes
    from profile photographs and cephalometry, both of which are sagittal
    projections; computing in full 3D would answer a different question and then
    be compared against the wrong literature.
    """
    u = np.array([a[1] - b[1], a[2] - b[2]])
    v = np.array([c[1] - b[1], c[2] - b[2]])
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(u @ v / max(n, 1e-12), -1.0, 1.0))))


def sagittal_metrics(shape3d: np.ndarray) -> dict:
    """Every scale-free sagittal measurement, on a 468-point 3D shape.

    ⛔ `mentolabial` was a candidate and is DROPPED, not refused-with-a-number:
    it needs labrale inferius, the sulcus and pogonion to be three distinct
    points, and the corrected derivation lets pogonion COINCIDE with the sulcus
    when the chin's most anterior point sits at the band edge. A metric whose
    inputs can collapse is not a metric with a wide error bar; it is undefined on
    some faces, and printing it with a bracket would dress that up as precision.
    """
    p = _FA.profile_points(shape3d)
    gl, pg = p["glabella"], p["pogonion"]
    if gl is None or pg is None:
        return {}
    sn = shape3d[_FC.SUBNASALE]
    pn = shape3d[PRONASALE]
    na = shape3d[NASION]
    ls = shape3d[_FC.LABRALE_SUP]
    li = shape3d[_FC.LABRALE_INF]
    width = float(np.linalg.norm(shape3d[_FC.ZYGION_L] - shape3d[_FC.ZYGION_R]))

    # Ricketts' aesthetic line: pronasale -> pogonion. Lip offset from it,
    # as a percentage of face width so the reading is scale-free.
    d = np.array([pg[1] - pn[1], pg[2] - pn[2]])
    d = d / max(np.linalg.norm(d), 1e-12)

    def offset(q) -> float:
        v = np.array([q[1] - pn[1], q[2] - pn[2]])
        return float((v - (v @ d) * d)[1] / max(width, 1e-12) * 100.0)

    return {
        "facial_convexity": _sagittal_angle(gl, sn, pg),
        "nasofrontal_angle": _sagittal_angle(gl, na, pn),
        "nasomental_angle": _sagittal_angle(na, pn, pg),
        "nasal_projection_ratio": float(
            abs(pn[2] - sn[2]) / max(abs(na[1] - sn[1]), 1e-12)
        ),
        "chin_projection_ratio": float((pg[2] - sn[2]) / max(width, 1e-12)),
        "eline_upper_lip": offset(ls),
        "eline_lower_lip": offset(li),
        "midface_depth_ratio": float(
            (pn[2] - shape3d[_FC.ZYGION_L][2]) / max(width, 1e-12)
        ),
    }


def build_banks(n_synthetic: int = 300, seed: int = 7, scan_dir: str | None = None):
    """(real, synthetic) metric distributions, each a dict of metric -> array."""
    import flame_fit as FF
    import mp468_embedding as EMB
    import p8_validate as P8

    flame = FF.load_flame()
    tri, bary = EMB.load()

    real: dict[str, list[float]] = {}
    subjects: list[str] = []
    scan_dir = scan_dir or os.path.join(HERE, "_scans", "d3dfacs")
    if os.path.isdir(scan_dir):
        for name, verts in P8.real_identities(scan_dir, flame):
            subjects.append(name.split("/")[0])
            for k, v in sagittal_metrics(
                EMB.landmarks_from_mesh(verts, tri, bary)
            ).items():
                real.setdefault(k, []).append(v)

    rng = np.random.default_rng(seed)
    synth: dict[str, list[float]] = {}
    for _ in range(n_synthetic):
        shape = EMB.landmarks_from_mesh(P8.sample_identity(flame, rng, 1.0), tri, bary)
        for k, v in sagittal_metrics(shape).items():
            synth.setdefault(k, []).append(v)

    clean = lambda d: {  # noqa: E731
        k: np.asarray([x for x in v if np.isfinite(x)], float) for k, v in d.items()
    }
    return clean(real), clean(synth), sorted(set(subjects))


def agreement(real: np.ndarray, synth: np.ndarray) -> dict:
    """Does the small real population corroborate the large synthetic one?"""
    if real.size == 0 or synth.size == 0:
        return {"agrees": False, "why": "a population is empty"}
    shift = float(abs(real.mean() - synth.mean()) / max(synth.std(), 1e-12))
    ratio = float(real.std() / max(synth.std(), 1e-12))
    ok = shift < MAX_MEAN_SHIFT_SD and SD_RATIO_BAND[0] < ratio < SD_RATIO_BAND[1]
    return {
        "agrees": bool(ok),
        "mean_shift_sd": round(shift, 3),
        "sd_ratio": round(ratio, 3),
        "real_mean": float(real.mean()),
        "real_sd": float(real.std()),
        "synth_mean": float(synth.mean()),
        "synth_sd": float(synth.std()),
        "why": (
            "the 10 real scanned subjects corroborate the modelled population"
            if ok
            else f"real and modelled populations disagree (mean {shift:.2f} SD "
            f"apart, spread ratio {ratio:.2f})"
        ),
    }


def ships(
    metric_swing: float, synth: np.ndarray, agrees: bool
) -> tuple[bool, float, str]:
    """Does a metric earn a place? Reproducibility AND a corroborated norm.

    ⛔ Both, not either. A metric that reproduces but has no trustworthy norm is a
    number with nothing to compare it to; a metric with a good norm that does not
    reproduce is the defect this project keeps refusing. `facial_convexity` is
    the standing example of the second: its norm is now available and it still
    swings across its own cutoff.
    """
    scale = SCALE_SD_RANGE * float(synth.std()) if synth.size else 0.0
    frac = metric_swing / max(scale, 1e-12)
    if not agrees:
        return False, frac, "no corroborated norm to rank against"
    if frac >= SHIP_MAX_FRAC_OF_SCALE:
        return (
            False,
            frac,
            (f"moves {frac:.0%} of the judged scale on one unchanging face"),
        )
    return True, frac, "reproduces, and its norm is corroborated by real scans"


def percentile(value: float, synth: np.ndarray) -> float | None:
    if synth.size == 0 or not np.isfinite(value):
        return None
    return float(100.0 * np.mean(synth < value))


# ---------------------------------------------------------------------------
# THE CACHED BANK — so a report can be built without the licensed scans
# ---------------------------------------------------------------------------
# Building the bank needs FLAME and the D3DFACS registrations, and the scans are
# non-distributable. What the report needs is far less: the distribution of each
# metric, which is NUMBERS, and the numbers are publishable where the geometry is
# not. Caching them keeps the licensed material out of the shipping path
# entirely rather than relying on the pipeline never touching it.
#
# ⛔ THE CORROBORATION FLAG TRAVELS WITH THE NUMBERS. A synthetic distribution
# nobody checked against real geometry is a norm made of assumptions, so
# `agrees` is cached alongside and a metric whose real and modelled populations
# disagree must not be ranked. Caching the distribution without the flag would
# turn a two-part test into a one-part one at the point of use.
BANK_CACHE = os.path.join(HERE, "sagittal_bank.npz")


def build_cache(n_synthetic: int = 300, seed: int = 7, path: str = BANK_CACHE) -> dict:
    """Compute the sagittal bank once and cache the numbers it produces."""
    real, synth, subjects = build_banks(n_synthetic=n_synthetic, seed=seed)
    if not synth:
        raise RuntimeError("no synthetic population built — is FLAME present?")
    payload: dict[str, np.ndarray] = {}
    for k, v in synth.items():
        agr = agreement(real.get(k, np.array([])), v)
        payload[f"synth::{k}"] = v
        payload[f"agrees::{k}"] = np.array([1.0 if agr["agrees"] else 0.0])
        payload[f"realstat::{k}"] = np.array(
            [
                float(real.get(k, np.array([])).size),
                agr.get("real_mean", float("nan")),
                agr.get("real_sd", float("nan")),
                agr.get("mean_shift_sd", float("nan")),
                agr.get("sd_ratio", float("nan")),
            ]
        )
    payload["n_real_subjects"] = np.array([float(len(subjects))])
    np.savez_compressed(path, **payload)
    return {"metrics": sorted(synth), "n_real_subjects": len(subjects)}


def load_cache(path: str = BANK_CACHE) -> dict | None:
    """The cached bank, or None if it was never built. Never a silent fallback."""
    if not os.path.exists(path):
        return None
    d = np.load(path)
    out: dict = {"metrics": {}, "n_real_subjects": int(d["n_real_subjects"][0])}
    for key in d.files:
        if not key.startswith("synth::"):
            continue
        k = key.split("::", 1)[1]
        stat = d[f"realstat::{k}"]
        out["metrics"][k] = {
            "synth": d[key],
            "agrees": bool(d[f"agrees::{k}"][0]),
            "n_real": int(stat[0]),
            "real_mean": float(stat[1]),
            "real_sd": float(stat[2]),
            "mean_shift_sd": float(stat[3]),
            "sd_ratio": float(stat[4]),
        }
    return out


def cached_percentile(name: str, value: float, cache: dict | None) -> float | None:
    """Rank against the cached bank — None unless the real scans corroborate it."""
    if cache is None:
        return None
    row = cache["metrics"].get(name)
    if row is None or not row["agrees"]:
        return None
    return percentile(value, row["synth"])


if __name__ == "__main__":
    import sys

    info = build_cache()
    print(
        f"wrote {BANK_CACHE}: {len(info['metrics'])} metrics, corroborated "
        f"against {info['n_real_subjects']} real scanned subjects"
    )
    sys.exit(0)
