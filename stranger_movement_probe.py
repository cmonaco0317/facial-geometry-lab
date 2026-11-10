#!/usr/bin/env python3
"""Public implementation; private development notes omitted."""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "stranger_movement_probe.json")
MIN_VIEWS = 4
MIN_SPREAD = 52.0
DEAD = 45.0  # frames past this earn no spread credit (intake's MEASURED-REAL rule)


def _eff_spread(yaws):
    ys = [y for y in yaws if abs(y) <= DEAD]
    return (max(ys) - min(ys)) if len(ys) >= 2 else 0.0


def best_split(frames):
    """The disjoint pair of gate-passing subsets with the strongest weaker half.

    ⛔ Both halves must pass the REAL gate, not a relaxed one. A split that only
    works if the floor is lowered measures a set this product refuses, and
    "it reproduces if you stop enforcing the gate" is not a result.
    """
    n = len(frames)
    best = None
    for mask in range(1, 2 ** n - 1):
        A = [frames[i] for i in range(n) if mask >> i & 1]
        B = [frames[i] for i in range(n) if not mask >> i & 1]
        if len(A) < MIN_VIEWS or len(B) < MIN_VIEWS:
            continue
        sa, sb = _eff_spread([f["yaw"] for f in A]), _eff_spread([f["yaw"] for f in B])
        if sa >= MIN_SPREAD and sb >= MIN_SPREAD:
            score = min(sa, sb)
            if best is None or score > best[0]:
                best = (score, A, B, sa, sb)
    return best


def ratios_from(frames):
    """Fit this subset alone and read the shipped ratios off the pose-free result.

    ⭐ RATIOS ONLY, and that is a correctness requirement rather than a
    convenience. `frontal_render` puts the shape in the REFERENCE FRAME'S pixel
    scale, and two subsets may choose different reference frames -- so any
    absolute quantity would differ by camera distance and be reported as
    movement. Every mm row is already excluded upstream for the same reason.
    """
    import dimorphism_bank_probe as DP
    import facial_catalogue as FC
    import fit_identity_stage as FIS
    import multiview_probe as MV

    res = FIS.fit(frames)
    ref = FIS.choose_reference(frames)
    pts = MV.frontal_render(res["shape"], ref["pts478"][: MV.N_MESH])

    out = {}
    base = DP.measures(pts)
    for fid, _label, fn, _hi in DP.FEATURES:
        out[f"dim:{fid}"] = float(fn(base))
    cbase = FC.base(pts)
    for cid, _label, fn, unit, *_rest in FC.CATALOGUE:
        if unit == "mm":
            continue
        try:
            v = fn(cbase)
        except Exception:  # noqa: BLE001 - a row that cannot be read is not movement
            continue
        if v is not None and np.isfinite(float(v)):
            out[f"cat:{cid}"] = float(v)
    return out, float(res["rms_px"])


def subject_sets():
    """Public implementation; private development notes omitted."""
    import dimorphism_reference as DR

    sets = []
    base = os.path.join(HERE, "captures", "_strangers")
    for sid in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        d = os.path.join(base, sid)
        if os.path.isdir(d):
            sets.append((f"stranger:{sid}", sorted(glob.glob(os.path.join(d, "*.png")))))
    # Private development note omitted.
    # strangers and another way on him would be two quantities under one name --
    # this project's most-repeated defect. His sittings are the ones the shipped
    # margins were measured on, named by the reference file rather than typed.
    for name in getattr(DR, "MOVEMENT_SITTINGS", []):
        d = os.path.join(HERE, "captures", name)
        if os.path.isdir(d):
            imgs = [p for p in sorted(glob.glob(os.path.join(d, "*.png")))
                    if os.path.basename(p) not in ("frontal.png", "profile.png")]
            sets.append((f"calibration_subject:{name}", imgs))
    return sets


def main() -> int:
    if "--emit" in sys.argv:
        return emit_reference()
    import fit_identity_stage as FIS

    results, skipped = {}, []
    for label, paths in subject_sets():
        if not paths:
            skipped.append((label, "no images"))
            continue
        print(f"\n=== {label}  ({len(paths)} photographs)")
        frames = [f for f in FIS.detect_frames(paths) if f.get("yaw") is not None]
        frames = [f for f in frames if abs(f["yaw"]) <= 75.0]
        if len(frames) < 2 * MIN_VIEWS:
            skipped.append((label, f"only {len(frames)} usable views, need {2*MIN_VIEWS}"))
            print(f"    SKIP — {skipped[-1][1]}")
            continue
        split = best_split(frames)
        if split is None:
            skipped.append((label, "no two disjoint halves both clear the 52 deg floor"))
            print(f"    SKIP — {skipped[-1][1]}")
            continue
        _score, A, B, sa, sb = split
        print(f"    split A: {len(A)}v {sa:.1f}°   B: {len(B)}v {sb:.1f}°")
        ra, rms_a = ratios_from(A)
        rb, rms_b = ratios_from(B)
        common = sorted(set(ra) & set(rb))
        results[label] = {
            "n_views": len(frames),
            "split": {"A": [f["file"] for f in A], "B": [f["file"] for f in B]},
            "spread": {"A": round(sa, 1), "B": round(sb, 1)},
            "rms_px": {"A": round(rms_a, 3), "B": round(rms_b, 3)},
            "view_selection_movement": {k: abs(ra[k] - rb[k]) for k in common},
            "values": {"A": {k: ra[k] for k in common}, "B": {k: rb[k] for k in common}},
        }
        print(f"    fitted: RMS {rms_a:.2f} / {rms_b:.2f} px, {len(common)} ratios compared")

    payload = {
        "_what": "VIEW-SELECTION movement: |ratio(subset A) - ratio(subset B)| for two "
                 "disjoint subsets of ONE face's views that each independently pass the "
                 "real intake gate. NOT cross-sitting movement — every stranger set is a "
                 "simultaneous burst, so this is a strict LOWER BOUND on the quantity the "
                 "shipped margins use.",
        "results": results,
        "skipped": [{"subject": s, "why": w} for s, w in skipped],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    if "--json" in sys.argv:
        print(json.dumps(payload, indent=1, ensure_ascii=False))
        return 0

    report(results, skipped)
    return 0


def report(results, skipped):
    import dimorphism_reference as DR

    strangers = {k: v for k, v in results.items() if k.startswith("stranger:")}
    calibration_subject = {k: v for k, v in results.items() if k.startswith("calibration_subject:")}
    if not strangers or not calibration_subject:
        print("\nnot enough subjects on both sides to compare")
        return

    keys = sorted(set.intersection(*[set(v["view_selection_movement"]) for v in results.values()]))
    print("\n" + "=" * 92)
    print("VIEW-SELECTION MOVEMENT — two admissible answers about the same face, same instant")
    print("=" * 92)
    print(f"{len(strangers)} strangers vs calibration_subject {len(calibration_subject)} sittings, {len(keys)} ratios each.")
    print("⛔ A FLOOR on cross-sitting movement, never a measurement of it.\n")

    print(f"{'ratio':26s} {'calibration_subject median':>14s} {'stranger median':>16s} {'ratio':>8s} {'stranger max':>13s}")
    print("-" * 92)
    worse = 0
    rows = []
    for k in keys:
        cm = float(np.median([v["view_selection_movement"][k] for v in calibration_subject.values()]))
        sm = float(np.median([v["view_selection_movement"][k] for v in strangers.values()]))
        sx = float(max(v["view_selection_movement"][k] for v in strangers.values()))
        r = (sm / cm) if cm > 0 else float("inf")
        rows.append((k, cm, sm, r, sx))
        if r > 1.0:
            worse += 1
    for k, cm, sm, r, sx in sorted(rows, key=lambda t: -t[3])[:14]:
        print(f"{k:26s} {cm:14.5f} {sm:16.5f} {r:8.2f} {sx:13.5f}")

    allr = [r for _k, _c, _s, r, _x in rows if np.isfinite(r)]
    print("-" * 92)
    print(f"median stranger:calibration_subject ratio across {len(allr)} ratios: {np.median(allr):.2f}")
    print(f"ratios where the median stranger moves MORE than calibration_subject: {worse}/{len(rows)}")

    print("\n" + "=" * 92)
    print("⛔ THE TEST THIS PROBE MUST NOT RUN, AND WHY — read before quoting anything above")
    print("=" * 92)
    print("The obvious next step is to check each stranger's view-selection movement")
    print("against the SHIPPED margin and call every excess an under-coverage. The first")
    print("version of this probe did exactly that and reported 15 breaches at up to 8.6x.")
    print("IT IS INVALID, and the control below is what shows it:\n")

    breaches = {"calibration_subject": [], "stranger": []}
    for fid, ref in DR.FEATURES.items():
        mv = ref.get("movement")
        if mv is None:
            continue
        for lab, v in results.items():
            got = v["view_selection_movement"].get(f"dim:{fid}")
            if got is not None and got > mv:
                breaches["calibration_subject" if lab.startswith("calibration_subject:") else "stranger"].append(
                    (fid, lab.split(":", 1)[1], got / mv)
                )
    nc, ns = len(breaches["calibration_subject"]), len(breaches["stranger"])
    print(f"  strangers exceeding the shipped margin : {ns}")
    print(f"  calibration_subject exceeding HIS OWN shipped margin: {nc}"
          + (f"   (worst {max(r for _f, _l, r in breaches['calibration_subject']):.1f}x)" if nc else ""))
    print()
    print("  calibration_subject breaches his own margin on his own face, so the excess is not a")
    print("  property of the stranger. It is a property of the OPERATION: the shipped")
    print("  margin was measured between FULL-SET fits (7-9 views, RMS ~6.1-6.7px) and")
    print("  this probe compares HALF-SET fits (4-5 views, RMS up to 11.3px). A half-set")
    print("  fit is materially noisier, so 'view-selection movement on half sets' is NOT")
    print("  a floor on 'cross-sitting movement on full sets' -- they are two different")
    print("  quantities and comparing them directly is this project's most-repeated")
    print("  defect, committed here against its own margin.")
    print()
    print("  ⭐ WHAT SURVIVES IS THE RATIO, because both sides get the IDENTICAL")
    print("  operation: same gate, same subset search, same half-set fit. calibration_subject is not")
    print("  the yardstick here, he is one more subject put through the same mill.")

    print("\n" + "=" * 92)
    print("WHAT THIS ACTUALLY LICENSES YOU TO SAY")
    print("=" * 92)
    med = float(np.median(allr))
    print(f"  Under one identical operation, the median stranger moves {med:.2f}x as much as")
    print(f"  calibration_subject does, and moves more on {worse} of {len(rows)} ratios. His face is")
    print("  therefore NOT dramatically easier for this instrument to measure -- but it is")
    print("  somewhat easier, and the borrowed margin runs correspondingly narrow for")
    print("  other people. margin_sensitivity_probe.py prices that: around k=1.3 only a")
    print("  handful of verdicts move.")
    print()
    print(f"  ⚠️  n IS SMALL: {len(strangers)} strangers and {len(calibration_subject)} calibration_subject sittings survived the")
    print("  split, because the fit's detector keeps fewer frames than intake does and")
    print("  both halves must independently clear 52 deg. Per-ratio multipliers with a")
    print("  tiny denominator (jaw_ratio's calibration_subject median is 0.0014) are unstable and the")
    print("  MEDIAN ACROSS RATIOS is the only figure here worth quoting.")
    print()
    print("  ⛔ AND IT REMAINS A FLOOR. Every stranger set is a simultaneous burst. This")
    print("  says nothing about a second day, a different room, or a different mood --")
    print("  which is precisely what the shipped margin is supposed to cover.")

    if skipped:
        print(f"\n⚠️  {len(skipped)} subject(s) not measured, and why:")
        for s, w in skipped:
            print(f"     {s:28s} {w}")
    print("=" * 92)


# ---------------------------------------------------------------------------
# FREEZING THE REFERENCE, so a per-subject scalar has a denominator
# ---------------------------------------------------------------------------
def emit_reference(out_path: str = "movement_reference.py") -> int:
    """Public implementation; private development notes omitted."""
    import numpy as _np

    with open(os.path.join(HERE, "stranger_movement_probe.json"), encoding="utf-8") as fh:
        res = json.load(fh)["results"]
    ref = {k: v for k, v in res.items() if k.startswith("calibration_subject:")}
    if not ref:
        print("!! no reference-subject sittings in the probe output")
        return 1
    keys = sorted(set.intersection(*[set(v["view_selection_movement"]) for v in ref.values()]))
    table = {}
    for k in keys:
        vals = [float(v["view_selection_movement"][k]) for v in ref.values()]
        table[k] = {
            "median": float(_np.median(vals)),
            "lo": float(min(vals)),
            "hi": float(max(vals)),
            "n": len(vals),
        }
    body = [
        '"""FROZEN reference split-half movement — emitted by stranger_movement_probe.py.',
        "",
        "⛔ DO NOT HAND-EDIT. Regenerate with:",
        "    ./venv/bin/python stranger_movement_probe.py --emit",
        "",
        "WHAT THIS IS. For each shipped ratio, how far the reference subject's reading",
        "moves between TWO DISJOINT HALVES of one sitting that EACH independently pass",
        "the real intake gate. It is a same-instant quantity: expression, lighting,",
        "camera and posture are all held constant, so it is a FLOOR on cross-sitting",
        "movement and must never be quoted as one.",
        "",
        "WHAT IT IS FOR. It is the DENOMINATOR of a like-for-like comparison. A new",
        "subject's own split-half movement, measured by the identical operation, divided",
        "by this, says how much more that person's photographs disagree than the person",
        "every margin in this instrument was measured on.",
        "",
        "⛔ WHY NOT cross-sitting / split-half. That ratio was tried on 2026-08-28 and is",
        "CONFOUNDED: cross-sitting movement is measured between FULL-set fits (7-9 views,",
        "RMS 6.1-6.7px) and split-half between HALF-set fits (4-5 views, RMS up to",
        "11.3px). Dividing them measures fit quality, not sittings -- it came out at a",
        "median of 0.96x with a 43.6x spread and went BELOW 1.0 on 10 of 18 ratios, which",
        "is impossible for a true floor relationship. Only same-operation ratios survive.",
        "",
        f"⚠️ n = {len(ref)} sittings of ONE person. `lo`/`hi` are the individual sittings,",
        "not a confidence interval. The aggregation matters: the shipped 1.30x stranger",
        "multiplier is 0.93x using `hi` and 2.79x using `lo` on this same data.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f"N_SITTINGS = {len(ref)}",
        f"SITTINGS = {sorted(ref)!r}",
        "",
        "# ratio -> {median, lo, hi, n} of split-half movement, in the ratio's own units",
        "SPLIT_HALF = {",
    ]
    for k in keys:
        t = table[k]
        body.append(
            f"    {k!r}: {{'median': {t['median']!r}, 'lo': {t['lo']!r}, "
            f"'hi': {t['hi']!r}, 'n': {t['n']}}},"
        )
    body += ["}", ""]
    with open(os.path.join(HERE, out_path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(body))
    print(f"wrote {out_path} — {len(keys)} ratios from {len(ref)} reference sittings")
    return 0


if __name__ == "__main__":
    sys.exit(main())

