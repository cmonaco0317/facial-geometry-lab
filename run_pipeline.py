#!/usr/bin/env python
"""End-to-end facial harmony / morphometric analysis pipeline.

TWO PATHS. Both produce facial_analysis_report.html (+ .pdf) and every
intermediate figure and JSON.

  TWO-PHOTO (default, unchanged)
      Drop `frontal.png` and `profile.png` (near-frontal + true lateral) here:
          ./venv/bin/python run_pipeline.py
      This is the BASELINE the re-scoped goal is stated against, so it stays.

  SET  --set <dir>
      A directory of 4+ casual photographs of one person at different angles,
      plus one `profile.*` for the silhouette:
          ./venv/bin/python run_pipeline.py --set captures/my-set
      `intake.assess_set` gates the SET and halts with specific advice if the
      angles do not span enough to identify pose; then one 3D identity is fitted
      across every accepted frame and projected head-on. The thirteen geometry
      stages downstream consume the result unchanged, because it arrives in the
      same 478-row landmark contract they already read.

      What it buys: 4.7x the reproducibility of the best single photograph the
      pose gate allows. What it does not: millimetres, skin, or a silhouette.

      HEIC IS HANDLED FOR YOU. Anything straight off an iPhone is converted to
      PNG next to the original before the set is read (see heic_convert.py).

  CHECK  --check <dir>
      Report on a folder of photographs and STOP. No measurement, no report, no
      file in the project root touched:
          ./venv/bin/python run_pipeline.py --check captures/my-set
      Says whether the set can support a fit, which photo is weak and why, and
      what is missing -- exit 0 if it is ready, 2 if it is not.

      The set path OVERWRITES frontal.png and profile.png with its own reference
      frame and lateral, because every appearance stage reads those names. The
      pair in place beforehand is copied to `_two_photo_inputs/` the first time,
      and `--restore-two-photo` puts them back, so the baseline stays runnable.

Reuse for a NEW subject:
  1. Put their photographs in a folder and run `--check` on it, then `--set`.
     (Two-photo path: replace frontal.png / profile.png by hand instead.)
  2. Set who they are in `subject.json` -- `reference_set` chooses which bank of
     real faces their typicality is compared against, and there are eight (four
     male, four female), listed in `subject.example.json`. Getting it wrong does
     not widen the answer, it answers a different question.
  3. Re-run this script.

Stages are ordered by dependency; each is a standalone script you can also run alone.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import heic_convert  # noqa: E402
import orientation  # noqa: E402

PY = (
    sys.executable
    if "venv" in sys.executable
    else os.path.join(HERE, "venv", "bin", "python")
)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

STAGES = [
    # ⭐ FIRST, because three later stages read its scalar. It measures how much
    # THIS subject's own photographs disagree with each other and widens every
    # movement margin by that factor -- so the brackets stop being one person's
    # applied to everyone. Cheap on the two-photo and single-frame paths: with no
    # set to split it short-circuits to 1.0 and the shipped margin is unchanged.
    ("This subject's own movement", "subject_movement.py"),
    ("Frontal landmarks + canon metrics", "analyze_frontal.py"),
    ("Profile cephalometrics (silhouette)", "analyze_profile.py"),
    ("Symmetry / dimorphism / template-averageness", "analyze_advanced.py"),
    ("Pose-gated matched averageness", "analyze_averageness2.py"),
    ("Measured precision (augmentation)", "ensemble_uncertainty.py"),
    ("Social-perception reading", "social_perception.py"),
    ("Extended morphometry + styling", "analyze_extended.py"),
    ("Skin & surface (cosmetic-dermatology layer)", "analyze_skin.py"),
    ("Self-improvement ROI", "self_improvement.py"),
    ("Anatomical reference map", "analyze_anatomy.py"),
    ("Facial adiposity (condition pillar)", "analyze_adiposity.py"),
    ("Nasal + cranial detail (coverage gap)", "analyze_regions.py"),
    ("Regional coverage audit", "region_coverage.py"),
    ("Measurement catalogue", "analyze_catalogue.py"),
    ("Facial harmony (relational balance)", "analyze_harmony.py"),
    ("Plain-English visuals (gauge + report card)", "plain_visuals.py"),
    ("Canon dashboard", "dashboard.py"),
    ("Front page — what this found", "headline.py"),
    ("Build consolidated HTML report", "build_report.py"),
]


def run(script, *args):
    t = time.time()
    r = subprocess.run(
        [PY, os.path.join(HERE, script), *args],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    return r.returncode, time.time() - t, (r.stderr or "")[-400:]


# ⭐ MODULE LEVEL SO THE GUARD CAN READ IT. This was local to
# `_reset_identity_for_set`, so `selfcheck_subject_isolation` had no choice but to
# keep a SECOND copy -- and a sixth identity field would have been cleared by the
# code while the guard went on checking five, silently. Identity carry-over is the
# defect that was live for the life of this project; its guard must not be able to
# drift from the thing it guards.
IDENTITY_KEYS = ("reference_set", "age", "race", "pronoun", "is_validation_subject")

TWO_PHOTO_KEEP = os.path.join(HERE, "_two_photo_inputs")


def frontal_is_set_derived():
    """Is the frontal.png on disk a set run's chosen reference frame?

    Answered from the bytes, not from a flag: hash frontal.png and compare it to
    the file the last fit named as its reference. A stale metrics file or a
    hand-edited marker cannot fool this.
    """
    try:
        import hashlib

        with open(os.path.join(HERE, "fit_diagnostics.json")) as fh:
            meta = json.load(fh)["_meta"]
        ref = os.path.join(meta["set_dir"], meta["reference_photo"])
        if not (
            os.path.exists(ref) and os.path.exists(os.path.join(HERE, "frontal.png"))
        ):
            return False

        def h(p):
            d = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    d.update(chunk)
            return d.hexdigest()

        return h(ref) == h(os.path.join(HERE, "frontal.png"))
    except (OSError, ValueError, KeyError):
        return False


def preserve_two_photo_inputs():
    """Copy the two-photo inputs aside before the set path overwrites them.

    THIS EXISTS BECAUSE IT ALREADY WENT WRONG. The set path chooses a reference
    frame and writes it over `frontal.png`, and copies the set's lateral over
    `profile.png`. The first `--set` run therefore destroyed the very pair the
    two-photo baseline runs on -- and those files are gitignored, because they are
    photographs of a person, so there was no copy to restore from.

    The goal is stated as "beat the two-photo report", and §6 of the wiring plan
    requires that baseline to still run. An input the set path silently eats is
    not a baseline. Preserved once, never overwritten again, restored with
    `--restore-two-photo`.
    """
    if frontal_is_set_derived():
        # The pair on disk is already a set run's output. Preserving it here would
        # file a set frame under "the two-photo baseline" and make the mistake
        # permanent and invisible.
        return []
    made = []
    os.makedirs(TWO_PHOTO_KEEP, exist_ok=True)
    for name in ("frontal.png", "profile.png"):
        src, dst = os.path.join(HERE, name), os.path.join(TWO_PHOTO_KEEP, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
            made.append(name)
    if made:
        print(
            f"     preserved the two-photo inputs -> _two_photo_inputs/ ({', '.join(made)})"
        )
    return made


def restore_two_photo_inputs():
    missing = [
        n
        for n in ("frontal.png", "profile.png")
        if not os.path.exists(os.path.join(TWO_PHOTO_KEEP, n))
    ]
    if missing:
        print(f"!! nothing preserved to restore: {', '.join(missing)}")
        print("   _two_photo_inputs/ is written the first time --set runs. If the")
        print("   originals predate that, re-export them from the source photographs.")
        return 1
    for n in ("frontal.png", "profile.png"):
        shutil.copyfile(os.path.join(TWO_PHOTO_KEEP, n), os.path.join(HERE, n))
    print("restored frontal.png and profile.png from _two_photo_inputs/")
    print("run `./venv/bin/python run_pipeline.py` to rebuild the two-photo baseline")
    return 0


# The face-size floor for a SINGLE frame. Same number intake enforces for the
# set, and for the same measured reason: at 96px IPD an axis changes side
# against the same face at 383px, and at 191px it does not.
MIN_IPD_FOR_SINGLE = 120.0


def _restore_default_identity():
    """Bring the baseline subject's profile back for a no-set run.

    ⛔ Only ever RESTORES. It cannot invent a cohort and it cannot overwrite one
    that is already active -- if somebody has deliberately configured the profile
    for this run, that choice wins.
    """
    import json as _json

    cfg = os.path.join(HERE, "subject.json")
    if not os.path.exists(cfg):
        return None
    try:
        with open(cfg, encoding="utf-8") as fh:
            data = _json.load(fh)
    except (OSError, ValueError):
        return None
    if data.get("reference_set") or data.get("age"):
        return None  # already configured; do not touch it
    archive = dict(data.get("archived_profiles") or {})
    prof = archive.pop("<unassigned>", None)
    if not prof:
        return None
    data.update(prof)
    data["archived_profiles"] = archive
    data.pop("subject_for", None)
    data.pop("identity_cleared_note", None)
    with open(cfg, "w", encoding="utf-8") as fh:
        _json.dump(data, fh, indent=2)
        fh.write("\n")
    print(f"   baseline subject profile restored ({prof.get('reference_set')})")
    return prof


def _set_photos(set_dir):
    exts = ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG")
    out = []
    for e in exts:
        out += glob.glob(os.path.join(set_dir, f"*.{e}"))
    return sorted(set(out))


def _pick_best_frontal(photos):
    """The single frame most worth measuring when the set cannot support a fit.

    Squarest to camera first, because pose is the error term this project has
    measured most and the one a single frame cannot correct. Ties go to the
    larger face. Frames the at-rest gate refused are never eligible -- an
    expression would be measured as anatomy.
    """
    ok = [
        p
        for p in photos
        if p.get("status") != "not-at-rest"
        and p.get("yaw") is not None
        and (p.get("ipd_px") or 0) >= MIN_IPD_FOR_SINGLE
    ]
    if not ok:
        return None
    ok.sort(key=lambda p: (abs(p["yaw"]) + abs(p.get("pitch") or 0.0), -(p.get("ipd_px") or 0)))
    return ok[0]


def _reset_identity_for_set(set_dir):
    """Clear an identity that belongs to a DIFFERENT subject. Returns a note or None.

    `subject.json` gains `subject_for`: the set this profile was deliberately
    configured against. A run whose set does not match is a new person, so the
    cohort, age, race and pronoun are dropped rather than inherited.

    ⛔ It never INVENTS a cohort -- there is no ground truth for that. It only
    removes one that provably belongs to somebody else.
    """
    import json as _json

    cfg = os.path.join(HERE, "subject.json")
    if not os.path.exists(cfg):
        return None
    try:
        with open(cfg, encoding="utf-8") as fh:
            data = _json.load(fh)
    except (OSError, ValueError):
        return None
    # ⭐ FOUR STATES, and the middle one is what keeps `--cohort` usable.
    #   absent            -> unknown provenance, treat as somebody else's. CLEAR.
    #   "<explicit>"      -> a human JUST set it and has not run a set yet.
    #                        Accept it once and BIND it to this set.
    #   == this set       -> already bound here. Keep.
    #   == a different set-> provably another subject's. CLEAR.
    # An explicit choice therefore survives exactly one set run and then belongs
    # to that set, so it cannot silently follow a third person around.
    tag = os.path.basename(os.path.normpath(set_dir))
    owner = data.get("subject_for")
    if owner == tag:
        return None
    if owner == "<explicit>":
        data["subject_for"] = tag
        with open(cfg, "w", encoding="utf-8") as fh:
            _json.dump(data, fh, indent=2)
            fh.write("\n")
        return None
    identity = {k: data.get(k) for k in IDENTITY_KEYS}
    has_identity = any(v for v in identity.values())

    # ⭐ ARCHIVE, DO NOT DESTROY. The first version deleted the identity outright,
    # so running one other person's set silently threw away a configuration a
    # human had entered, with no way back but retyping it. Clearing is right;
    # losing it is not. The outgoing identity is filed under the owner it
    # belonged to, and a later run of THAT set gets it back — so two people can
    # alternate without re-entering anything, and neither can ever be scored
    # against the other's cohort.
    #
    # ⛔ The bookkeeping runs even when there is nothing to archive. An earlier
    # version returned early on an already-unconfigured profile, which skipped
    # the re-binding AND the restore: `subject_for` then kept pointing at a
    # previous set, and a returning subject never got their profile back.
    archive = dict(data.get("archived_profiles") or {})
    prev_owner = data.get("subject_for") or "<unassigned>"
    if has_identity and prev_owner != tag:
        archive[prev_owner] = {k: v for k, v in identity.items() if v is not None}

    for k in IDENTITY_KEYS:
        data.pop(k, None)
    restored = archive.pop(tag, None)
    if restored:
        data.update(restored)

    data["archived_profiles"] = archive
    data["subject_for"] = tag
    if restored:
        data.pop("identity_cleared_note", None)
    elif has_identity:
        data["identity_cleared_note"] = (
            "Cleared automatically: the profile on disk was configured for a "
            f"different subject ({identity}). It was archived, not discarded, and "
            "is restored if that subject's set is run again."
        )
    with open(cfg, "w", encoding="utf-8") as fh:
        _json.dump(data, fh, indent=2)
        fh.write("\n")
    if restored:
        return None  # their own profile is back; nothing was taken away
    return identity if has_identity else None
    data["identity_cleared_note"] = (
        "Cleared automatically: the profile on disk was configured for a "
        f"different subject ({identity}). It was not carried over."
    )
    with open(cfg, "w", encoding="utf-8") as fh:
        _json.dump(data, fh, indent=2)
        fh.write("\n")
    return identity


def flag_dir(argv, flag):
    """`<flag> <dir>` -> the directory, or None if the flag is absent."""
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise SystemExit(f"{flag} needs a directory of photographs")
    return os.path.abspath(argv[i + 1])


def set_mode_dir(argv):
    """`--set <dir>` -> the directory, or None for the two-photo path."""
    return flag_dir(argv, "--set")


def check_only(set_dir):
    """Report on a folder of photographs and stop.

    THE POINT IS THAT IT WRITES NOTHING IN THE PROJECT. Everything else in this
    file measures: it overwrites frontal.png, twelve JSON files and the report,
    so "let me just see if my photos are any good" was previously indistinguishable
    from "replace the current subject". A person deciding whether their photos are
    worth submitting must be able to ask without consequences.

    It does convert HEIC in the PHOTO folder, which is the one deliberate side
    effect, and says so -- the conversion is what makes the answer true, since the
    fit reads PNG/JPEG and would otherwise be judging a set that is missing every
    iPhone photo in it.
    """
    import intake

    if not os.path.isdir(set_dir):
        print(f"!! --check: not a directory: {set_dir}")
        return 1
    print(f"==> checking {set_dir}")
    heic_convert.convert_dir(set_dir)
    # RUNS IN --check TOO, not only in --set, so this reports on the same folder
    # a real run would measure. A sideways photograph passes intake looking
    # "fine" (measured 2026-08-19), so a --check that skipped this would tell a
    # person their set was READY and then --set would measure a different thing.
    orientation.normalize_dir(set_dir)

    photos = heic_convert.usable_photos(set_dir)
    if not photos:
        print("\n!! no photographs found (png / jpg / jpeg, or HEIC to convert).")
        return 2
    result = intake.assess_set(photos)
    intake.render(result)

    profile = heic_convert.find_profile(set_dir)
    print()
    if profile:
        print(f"  lateral      : {os.path.basename(profile)} — present")
    else:
        print(
            "  lateral      : MISSING. The fit recovers shape from landmarks and no\n"
            "                 landmark set contains a silhouette, so one true side-on\n"
            "                 shot is still needed. Name it profile.jpg (or .png)\n"
            "                 and put it in this folder."
        )
    ready = bool(result["ready"]) and profile is not None
    print()
    print(f"  VERDICT      : {'READY' if ready else 'NOT READY'}")
    if ready:
        print(
            "  Nothing was measured and nothing in the project was touched. To build\n"
            f"  the report:  ./venv/bin/python run_pipeline.py --set {set_dir}"
        )
    else:
        print("  Fix the above and run --check again. Nothing was written.")
    return 0 if ready else 2


def run_set_frontend(set_dir):
    """Fit one identity across the set, in place of detecting on one photograph.

    Streams rather than capturing: intake's refusal advice is the useful output
    when a set is not ready, and swallowing it would leave the user with an exit
    code and no idea which photograph to retake.
    """
    print("==> multi-view front end", flush=True)
    r = subprocess.run(
        [PY, os.path.join(HERE, "fit_identity_stage.py"), set_dir], cwd=HERE
    )
    return r.returncode



def age_span(names):
    """Lowest and highest age the named banks cover, READ FROM THE NAMES.

    Every bank name ends `_<lo>_<hi>`. Reading the span off the names rather
    than restating it is what stops this page from claiming an age range the
    banks stopped having.
    """
    lo, hi = [], []
    open_ended = False
    for n in names:
        if n == "70_plus" or n.endswith("_70_plus") or n.endswith("70-plus"):
            lo.append(70)
            open_ended = True
        m = re.search(r"(\d+)_(\d+)$", n)
        if m:
            lo.append(int(m.group(1)))
            hi.append(int(m.group(2)))
    if not lo:
        return None, None
    return min(lo), "70+" if open_ended else max(hi)


def cohort_limits(bank_names=None, matched=None):
    """The limits paragraph, DERIVED from the cohorts that actually exist.

    ⛔ THIS WAS A HAND-WRITTEN STRING AND IT WENT FALSE. Until 2026-08-25 it
    read "There is no bank outside 10-29 and none for any other single
    ancestry, so a subject outside those is compared to nobody rather than to
    the wrong population." That was true the day Step 6 wrote it and false from
    2026-08-23, when 42 race x sex x age cohorts covering ages 20-49 landed. A
    35-year-old friend running `--cohort list` was told, on an authoritative
    screen, that the software could not place them -- while the software placed
    them correctly. Nobody had to break anything: the fact moved and its second
    copy did not.

    Same fault the shoot guide was built to avoid, in the same surface. So this
    reads the banks instead, and `selfcheck_shoot_guide.py` mutates the inputs
    and fails if the paragraph does not follow.

    Both arguments are injectable so the guard can exercise this against
    fabricated cohort sets rather than against whatever happens to be on disk.
    """
    import averageness_band as AB
    import bank_reference as RR

    banks = sorted(bank_names if bank_names is not None else AB.COHORT_LABEL)
    cohorts = RR.COHORTS if matched is None else matched

    b_lo, b_hi = age_span(banks)
    bands = sorted({c["age_band"] for c in cohorts.values()})
    races = sorted({c["race"] for c in cohorts.values()})
    sexes = sorted({c["sex"] for c in cohorts.values()})
    m_lo, m_hi = age_span([b.replace("-", "_") for b in bands])
    faces = sum(c["n"] for c in cohorts.values())

    lines = []
    lines.append(
        f"SHAPE-REFERENCE BANKS (above) span ages {b_lo}-{b_hi}. `broad` means ALL "
        f"RACES pooled; the others are the white subset of the same source."
    )
    if cohorts:
        lines.append(
            f"MATCHED COHORTS are separate and wider: {len(cohorts)} of them, "
            f"{faces:,} faces, ages {m_lo}-{m_hi}, {len(races)} race groups "
            f"({', '.join(r.replace('_', ' ') for r in races)}), "
            f"{' and '.join(sexes)}. They decide the percentiles, the dimorphism "
            f"bands and typicality, and you set one with age + race rather than "
            f"by name:"
        )
        lines.append(
            "    ./venv/bin/python intake.py <photos> --age 35 --race Black --save-profile"
        )
        lines.append(
            f"A subject outside {m_lo}-{m_hi} has no matched cohort and is compared to "
            f"nobody rather than to the wrong population."
        )
    else:
        lines.append(
            "There are NO matched cohorts on this machine, so every subject is "
            "compared to nobody rather than to the wrong population."
        )
    return "\n".join(lines)


def list_cohorts():
    """Print the banks a subject can be compared against, and their limits."""
    import averageness_band as AB
    import numpy as np

    cache = os.path.join(HERE, "ref_landmarks2.npz")
    counts = {}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        counts = {k: len(z[k]) for k in z.files if not k.endswith(("_raw", "_race"))}
    names = sorted(counts or AB.COHORT_LABEL)
    print("COMPARISON COHORTS\n" + "=" * 60)
    for name in names:
        n = counts.get(name)
        print(f"  {name:22s} {AB.label_for(name):32s}" + (f"  n={n}" if n else ""))
    print()
    print(cohort_limits(bank_names=names))
    print()
    print("Set one with:   ./venv/bin/python run_pipeline.py --cohort <name>")
    print("Clear it with:  ./venv/bin/python run_pipeline.py --cohort none")
    return 0


def set_cohort(name, cfg=None):
    """Write the chosen cohort into subject.json, preserving everything else.

    `cfg` overrides the target path so a guard can exercise this against a
    throwaway file instead of the real subject's profile -- a test that has to
    back up and restore live configuration is one crash away from destroying it.

    ⭐ THIS EXISTS BECAUSE HAND-EDITING subject.json WAS THE ONLY WAY, and the
    field decides BOTH the comparison population AND which advice is allowed to
    print. Getting it wrong answers a different question at full confidence --
    the route by which a female subject was once told to grow a beard. A flag
    that validates the name against the banks that actually exist cannot make
    that particular mistake.
    """
    import json as _json

    import averageness_band as AB
    import numpy as np

    cfg = cfg or os.path.join(HERE, "subject.json")
    data = {}
    if os.path.exists(cfg):
        try:
            with open(cfg, encoding="utf-8") as fh:
                data = _json.load(fh)
        except (OSError, ValueError):
            print("!! subject.json is unreadable; refusing to overwrite it")
            return 1

    if name.lower() in ("none", "clear", "unset"):
        data.pop("reference_set", None)
        with open(cfg, "w", encoding="utf-8") as fh:
            _json.dump(data, fh, indent=2)
            fh.write("\n")
        print("cohort cleared — the report will make no cohort claim about this subject")
        return 0

    cache = os.path.join(HERE, "ref_landmarks2.npz")
    valid = set()
    if os.path.exists(cache):
        valid = {
            k for k in np.load(cache, allow_pickle=False).files
            if not k.endswith(("_raw", "_race"))
        }
    valid |= set(AB.POOLED_NAME.values())
    if name not in valid:
        # ⛔ REFUSE RATHER THAN ACCEPT A NAME THAT WILL SILENTLY FALL BACK.
        # subject.load() resolves an unknown bank to the DEFAULT, so a typo would
        # otherwise be indistinguishable from a deliberate choice.
        print(f"!! not a bank this build has: {name!r}")
        print("   run  --cohort list  to see them")
        return 1

    data["reference_set"] = name
    # ⭐ MARK IT AS DELIBERATE. Without this, `--cohort <name>` followed by
    # `--set <dir>` would have the set run clear the choice as somebody else's
    # leftovers -- the documented flow undoing itself. "<explicit>" means a human
    # just chose this and no set has claimed it yet; the next set run binds it.
    data["subject_for"] = "<explicit>"
    data.pop("identity_cleared_note", None)
    with open(cfg, "w", encoding="utf-8") as fh:
        _json.dump(data, fh, indent=2)
        fh.write("\n")
    print(f"cohort set to {name}  ({AB.label_for(name)})")
    print("⚠️ this also decides which advice may print. Re-run the pipeline to apply it.")
    return 0

# ⭐ ONE SOURCE FOR WHAT THIS ACCEPTS. `usage()` renders it and `main()`
# validates against it, so a flag cannot be documented without being accepted or
# accepted without being documented. `takes_value` is what makes the following
# word an argument rather than a flag to be checked.
KNOWN_FLAGS = {
    "--help": (False, "print this and stop"),
    "-h": (False, "print this and stop"),
    "--set": (True, "<dir> — a folder of 4+ casual photographs of one person"),
    "--check": (True, "<dir> — report on a folder and STOP; writes nothing"),
    "--cohort": (True, "<name|list|none> — choose the shape-reference bank"),
    "--how-to-shoot": (False, "print the capture sheet, derived from the gate"),
    "--shoot-page": (False, "write how_to_shoot.html — the same sheet, with the angles drawn"),
    "--restore-two-photo": (False, "put the saved frontal/profile pair back"),
    "--keep-going": (False, "do not halt on a failed stage (DEBUGGING ONLY)"),
    "--force": (False, "run past the capture gate (research; report still refuses)"),
}


def usage():
    """What this accepts, rendered from the table that validates it."""
    out = [
        "usage: run_pipeline.py [flags]",
        "",
        "  no flags          the two-photo baseline: reads frontal.png + profile.png",
        "",
    ]
    for flag, (_takes, why) in KNOWN_FLAGS.items():
        if flag == "-h":
            continue
        out.append(f"  {flag:<20s}{why}")
    out += [
        "",
        "A full run is 16 stages and about 40 seconds, and it OVERWRITES the report.",
        "Use --check first if you only want to know whether a folder is usable.",
    ]
    return "\n".join(out)


def unknown_flags(argv):
    """Flags this does not accept — WITHOUT eating the arguments of ones it does.

    ⛔ BEFORE THIS EXISTED, EVERY UNRECOGNISED FLAG RAN THE WHOLE PIPELINE.
    Argument handling here is a series of `in sys.argv` tests, so a typo, a flag
    from a sibling script, or `--help` itself matched nothing, fell through, and
    started a 40-second measuring run that overwrote the report. The friend most
    likely to type `--help` is the one least able to tell that from working.
    """
    bad, skip = [], False
    for arg in argv[1:]:
        if skip:
            skip = False
            continue
        if not arg.startswith("-"):
            continue
        if arg in KNOWN_FLAGS:
            skip = KNOWN_FLAGS[arg][0]
            continue
        bad.append(arg)
    return bad


def main():
    if any(a in ("--help", "-h") for a in sys.argv[1:]):
        print(usage())
        return 0

    bad = unknown_flags(sys.argv)
    if bad:
        print(f"!! not a flag this understands: {', '.join(bad)}")
        print()
        print(usage())
        return 2

    if "--restore-two-photo" in sys.argv:
        return restore_two_photo_inputs()

    # ⭐ THE THINGS A PERSON NEEDS BEFORE THEY HAVE A REPORT, added for
    # FINISH-LINE Step 6. Both run first and exit: neither touches a photograph,
    # neither writes an artifact, and `--cohort` writes only subject.json.
    if "--how-to-shoot" in sys.argv:
        import shoot_guide

        print(shoot_guide.guide_text())
        return 0

    # ⭐ THE SAME SHEET WITH THE ANGLES DRAWN. The binding requirement is a head
    # angle, and six of the ten Step 3 strangers were refused for having no frame
    # in the lateral band -- the most common single reason a set fails. Nobody
    # estimates 62 degrees from a sentence, so the page draws it. Generated from
    # the same thresholds, never hand-edited.
    if "--shoot-page" in sys.argv:
        import shoot_page

        return shoot_page.main()

    if "--cohort" in sys.argv:
        i = sys.argv.index("--cohort")
        if i + 1 >= len(sys.argv):
            print("!! --cohort needs a name, or 'list'")
            return 1
        arg = sys.argv[i + 1]
        return list_cohorts() if arg == "list" else set_cohort(arg)

    check_dir = flag_dir(sys.argv, "--check")
    if check_dir is not None:
        return check_only(check_dir)

    set_dir = set_mode_dir(sys.argv)

    # ---- THE SET FRONT END -------------------------------------------------
    # Runs BEFORE the input check below, because it is what produces frontal.png:
    # the set path chooses its own reference photograph from the set rather than
    # being handed one. Everything after this point is identical in both paths --
    # that is the design, and it is why thirteen stages needed no changes.
    #
    # The two-photo path is untouched and still the default. It is the baseline
    # the goal is stated against ("more accurate than the two-photo report"), so
    # deleting it would destroy the comparison the whole re-scope rests on.
    no_lateral = False
    single_frame = None
    # ⭐ THE TWO-PHOTO PATH IS THE DEFAULT SUBJECT, so it restores that subject's
    # identity. Found 2026-08-27: a LEDGER CLAIM that exercises `--set` archived
    # the baseline profile, and because that profile had never been bound to a
    # set it was filed under "<unassigned>" and nothing ever asked for it back.
    # The validation subject's own report then printed "(DEFAULT — no cohort was
    # configured)" about a subject whose cohort has been configured all along.
    #
    # Running with no --set means "the baseline subject", so that is exactly when
    # the unassigned profile should come home. Symmetric with the set path: each
    # entry point restores the identity that belongs to it.
    if set_dir is None:
        _restore_default_identity()
    if set_dir is not None:
        if not os.path.isdir(set_dir):
            print(f"!! --set: not a directory: {set_dir}")
            return 1
        # RUNS BEFORE THE PROFILE LOOKUP BELOW, because `profile.HEIC` is a
        # perfectly reasonable thing for a person to hand this and the lookup
        # only knows about PNG and JPEG. Converting first means the same photo
        # qualifies whichever format the phone chose to save it in.
        heic_convert.convert_dir(set_dir)
        # BEFORE THE LATERAL LOOKUP AND BEFORE THE FIT, for the same reason HEIC
        # conversion is: a photograph the pipeline cannot read correctly must be
        # made readable before anything decides what the set contains. A set whose
        # EXIF orientation was stripped in transit still passes intake as READY
        # and still produces a full report -- off a face lying on its side, with
        # a 6% scale error and a symmetry bracket that does not overlap the truth.
        # Measured 2026-08-19; see orientation.py.
        orientation.normalize_dir(set_dir)
        # A true lateral is NOT something the set replaces -- the fit recovers
        # shape from landmarks, and no landmark set contains a silhouette. It must
        # come from the set directory rather than whatever profile.png happens to
        # be lying in the project root, or a new subject's report would silently
        # inherit the previous subject's profile.
        prof = [
            p
            for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG")
            for p in glob.glob(os.path.join(set_dir, f"profile.{ext}"))
        ]
        # ⭐ A MISSING LATERAL NO LONGER KILLS THE RUN. It used to `return 1`
        # before a single stage executed -- but only FOUR of the nineteen stages
        # touch the profile, so somebody with seven good frontal frames and no
        # side-on shot was getting NOTHING when the honest answer is every
        # frontal measurement plus an explicit refusal on the silhouette rows.
        # All-or-nothing was the defect, not the missing photograph.
        # ⛔ The silhouette is still NOT recoverable from the set: the fit reads
        # landmarks and no landmark set contains an outline. So the rows are
        # REFUSED, never estimated, and analyze_profile.py --absent writes that
        # refusal out in full rather than leaving the previous run's file on disk.
        # ⛔⛔ IDENTITY DOES NOT CARRY OVER BETWEEN SUBJECTS. Found 2026-08-26,
        # by walking straight into it: a second person's photographs were run
        # Private development note omitted.
        # white_20_29, race White, is_validation_subject true. The report built
        # cleanly, all nineteen stages green, and every population figure in it
        # (percentiles, dimorphism bands, typicality, cohort-gated advice) was
        # computed against a population assigned by leftover configuration
        # rather than by the person in the photographs. Structurally complete
        # and substantively wrong, which is the worst combination this project
        # can produce.
        #
        # ⭐ THE PROJECT ALREADY KNEW. `step3_strangers.py` deliberately runs
        # every stranger with the cohort LEFT UNCONFIGURED and says why: there
        # is no ground truth for it and inventing one would be wrong. The
        # RESEARCH harness was protected; the path a friend would actually use
        # was not, and the docstring listed "set who they are" as a manual step
        # it never enforced.
        #
        # So a set that is not the one the profile was written for CLEARS the
        # identity rather than inheriting it. Clearing is safe by construction:
        # the report already renders "(DEFAULT — no cohort was configured)" and
        # `cohort_sex()` returns None, so it makes no claim about the person. A
        # narrower report is recoverable; a confidently mis-populated one is not.
        subject_reset_note = _reset_identity_for_set(set_dir)
        no_lateral = not prof
        if no_lateral:
            print(f"!! --set: no profile.* in {set_dir}")
            print(
                "   Continuing WITHOUT the silhouette. Every frontal measurement still\n"
                "   runs; the four profile rows (nasolabial, convexity, chin projection,\n"
                "   cervicomental) will be refused rather than estimated.\n"
                "   To get them: add one shot turned 50-75 degrees — both eyes still\n"
                "   visible — named profile.jpg in the set folder, and re-run.\n"
                "   `capture_check.py --scan <dir>` triages candidates."
            )
        # ⭐⭐ THIRD MODE: WHEN THE SET CANNOT SUPPORT A FIT, MEASURE ONE FRAME.
        # Measured 2026-08-27 across eight stock-portrait subjects: SEVEN halted
        # at intake, and the blocker was almost always ANGULAR SPREAD (3-30 deg
        # against a 30 deg hard floor), not crashes and not expression. Portrait
        # photography clusters near-frontal, so demanding 52 degrees of turn
        # means most real photo sets produce nothing at all.
        #
        # ⛔ THE FLOOR IS NOT LOWERED. Below 30 degrees a multi-view fit genuinely
        # does not work -- 6.5-11% shape error, almost nothing usable -- so the
        # FIT is still refused. What changes is that refusing the fit stops
        # meaning refusing the report: the pipeline falls back to measuring the
        # best single frontal frame, which is exactly what the two-photo baseline
        # has always done and is the most validated path in this project.
        #
        # The person gets every frontal measurement with SINGLE-PHOTO error bars
        # instead of fit error bars, and the report says which it used. That is a
        # different instrument, not a quieter version of the same one.
        single_frame = None
        if not no_lateral or True:
            import intake as _IN

            _paths = _set_photos(set_dir)
            _v = _IN.assess_set(_paths) if _paths else None
            if _v and not _v["ready"]:
                _cov, _cal = _v["coverage"], _v["calibration"]
                if _cov["n_usable"] < _cal["min_views"] or _cov[
                    "effective_spread_deg"
                ] < 30.0:
                    single_frame = _pick_best_frontal(_v["photos"])

        if subject_reset_note:
            print()
            print("!! SUBJECT IDENTITY CLEARED — the profile on disk belonged to")
            print(f"   somebody else: {subject_reset_note}")
            print("   Nothing about that person is carried into this report. Every")
            print("   population comparison will say '(DEFAULT — no cohort was")
            print("   configured)' rather than rank this face against the wrong group.")
            print("   To compare them against their own cohort, set it explicitly:")
            print("       ./venv/bin/python intake.py <photos> --age 35 --race Black --save-profile")
            print("       ./venv/bin/python run_pipeline.py --cohort <name>")
            print()
        preserve_two_photo_inputs()
        if single_frame is not None:
            print()
            print("!! SET CANNOT SUPPORT A MULTI-VIEW FIT — measuring ONE frame instead.")
            print(f"   Chosen: {single_frame['file']}  (yaw {single_frame['yaw']:+.1f}, "
                  f"IPD {single_frame['ipd_px']:.0f}px)")
            print("   The 3D fit is refused, not the report: below 30 degrees of spread")
            print("   the fit's own shape error runs 6.5-11%. Every frontal measurement")
            print("   still runs, on SINGLE-PHOTO error bars rather than fit error bars,")
            print("   and the report says which it used.")
            print()
            shutil.copyfile(single_frame["path"], os.path.join(HERE, "frontal.png"))
            with open(os.path.join(HERE, "capture_mode.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "mode": "single-frame",
                        "reason": "set below the 30 deg hard floor for a multi-view fit, "
                        "or fewer usable views than the fit needs",
                        "frame": single_frame["file"],
                        "yaw": single_frame["yaw"],
                        "ipd_px": single_frame["ipd_px"],
                        "what_is_lost": (
                            "pose correction. A single frame cannot separate head turn from "
                            "face shape, so every reading carries this project's measured "
                            "single-photo fragility rather than the multi-view fit's."
                        ),
                        "what_is_unaffected": (
                            "which measurements are produced. The two-photo baseline this "
                            "project is validated against measures exactly one frontal frame."
                        ),
                    },
                    fh,
                    indent=1,
                )
                fh.write("\n")
        else:
            if os.path.exists(os.path.join(HERE, "capture_mode.json")):
                os.remove(os.path.join(HERE, "capture_mode.json"))
            if run_set_frontend(set_dir) != 0:
                return 1
        if not no_lateral:
            shutil.copyfile(prof[0], os.path.join(HERE, "profile.png"))
            print(f"     profile <- {os.path.basename(prof[0])}")

    required = ["frontal.png", "face_landmarker.task", "selfie_segmenter.tflite",
                "canonical_face_model.obj"]
    if not (set_dir is not None and no_lateral):
        required.insert(1, "profile.png")
    for f in required:
        if not os.path.exists(os.path.join(HERE, f)):
            print(f"!! missing required input/asset: {f}")
            if f.endswith((".task", ".tflite", ".obj")):
                print(
                    "   (download the MediaPipe assets — see "
                    "docs/TECHNICAL_OVERVIEW.md, Setup)"
                )
            return 1

    # The two-photo path WARNS and does not rotate. See orientation.warn_if_not_upright:
    # Private development note omitted.
    # rather than something a run does to him on the way past. The set path, which
    # is the one a friend's photographs arrive through, does correct.
    if set_dir is None:
        orientation.warn_if_not_upright(
            [os.path.join(HERE, "frontal.png"), os.path.join(HERE, "profile.png")]
        )

    # matched-averageness reference: rebuild if neither cache nor images exist
    if not os.path.exists(os.path.join(HERE, "ref_landmarks2.npz")):
        has_imgs = any(
            os.path.exists(os.path.join(HERE, "reference_faces2", s))
            and os.listdir(os.path.join(HERE, "reference_faces2", s))
            for s in ("white_10_19", "white_20_29", "broad_10_19")
        )
        if not has_imgs:
            print(
                "==> no averageness reference cache; fetching FairFace sample (~5 min)…"
            )
            rc, dt, err = run("fetch_reference2.py")
            print(f"    fetch_reference2.py  rc={rc}  {dt:.0f}s")

    keep_going = "--keep-going" in sys.argv

    # ---- CAPTURE GATE ------------------------------------------------------
    # Refuse to build a report the photographs cannot support. This runs FIRST and
    # halts, because the alternative -- measure anyway, caveat afterwards -- is the
    # failure mode this whole project exists to avoid. Measured 2026-08-01: a frame
    # 12.9 deg off axis, which passes this project's own reference-population filter
    # and looks like an ordinary photo, moves symmetry 5.2x and averageness 58
    # percentile points. A caveat under a number that wrong is not honesty.
    #
    # --force exists for research use on deliberately bad captures (that is how the
    # tolerances were measured in the first place). It is not a way to get a report
    # out of an unusable photo -- the per-metric gates downstream still refuse.
    #
    # IN THE SET PATH THE FRONTAL GATE IS INTAKE'S, AND IT ALREADY RAN. Re-gating
    # the chosen reference frame here would refuse exactly what the set path
    # exists to accept: no single casual photograph has to be ideal, because the
    # geometry comes from all of them. The silhouette keeps its gate -- nothing in
    # the set substitutes for a lateral.
    gate_args = ["--profile-only"] if set_dir is not None else []
    if "--force" not in sys.argv:
        print(
            "==> capture check"
            + (" (profile only — intake gated the set)" if gate_args else "")
        )
        rc, dt, err = run("capture_check.py", *gate_args)
        if rc != 0:
            subprocess.run(
                [PY, os.path.join(HERE, "capture_check.py"), *gate_args], cwd=HERE
            )
            print()
            print(
                "!! HALTED before measuring anything: the capture cannot support a report."
            )
            print("   Nothing was written. Retake as described above and run again.")
            print("   (--force measures anyway; the per-metric gates will still refuse")
            print("    individually, and the report will say so.)")
            return 1
        print(f"[OK ] {'Capture quality gate':46s} ({dt:5.1f}s)  capture_check.py")

    print(f"Python: {PY}\n")
    ok = True
    for name, script in STAGES:
        # Stage 1 is the only one that behaves differently: --from-fit makes it
        # MEASURE the fitted landmark set instead of overwriting it with a fresh
        # single-photo detection. It still detects on frontal.png, because the
        # appearance stages, the px->mm iris anchor and every overlay need the
        # photograph's own coordinates. See fa_common.GEOMETRY_CSV / PHOTO_CSV.
        extra = (
            ["--from-fit"]
            if (set_dir and single_frame is None and script == "analyze_frontal.py")
            else []
        )
        # ⛔ WRITE THE REFUSAL, DO NOT SKIP THE STAGE. Skipping would leave the
        # PREVIOUS run's metrics_profile.json on disk while the pipeline reported
        # success -- and on a second subject that publishes the first person's
        # profile. --absent emits every key with a null value and a stated reason,
        # so the three downstream consumers keep their contract.
        if no_lateral and script == "analyze_profile.py":
            extra = ["--absent", f"no profile.* in {os.path.basename(set_dir or '')}"]
        rc, dt, err = run(script, *extra)
        status = "OK " if rc == 0 else "FAIL"
        print(f"[{status}] {name:46s} ({dt:5.1f}s)  {script}")
        if rc != 0:
            ok = False
            print("        stderr:", err.replace("\n", " ")[-300:])
            if not keep_going:
                # HALT. Every stage reads its inputs from files that persist in
                # this directory, so continuing past a failure lets downstream
                # stages silently consume the PREVIOUS run's artifacts and emit
                # a report that looks successful but is built on stale data.
                # Refusing to continue is the honest behaviour; --keep-going is
                # available for debugging only.
                print(
                    "\n!! HALTED at a failed stage.\n"
                    "   Downstream stages read persisted files, so continuing would\n"
                    "   rebuild the report from the PREVIOUS run's data and look fine.\n"
                    "   Fix the error above and re-run (or pass --keep-going to override)."
                )
                return 1
    print()

    html = os.path.join(HERE, "facial_analysis_report.html")
    if os.path.exists(html) and os.path.exists(CHROME):
        pdf = os.path.join(HERE, "facial_analysis_report.pdf")
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--virtual-time-budget=20000",
                f"--print-to-pdf={pdf}",
                f"file://{html}",
            ],
            capture_output=True,
        )
        print(f"PDF: {'OK' if os.path.exists(pdf) else 'FAILED'}  {pdf}")

    print("\nDONE." if ok else "\nDONE WITH ERRORS (see above).")
    print(f"Report: {html}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
