"""Pre-flight capture QC — decide whether these photographs can support a report.

    ./venv/bin/python capture_check.py            # human-readable verdict
    ./venv/bin/python capture_check.py --json     # machine-readable
    ./venv/bin/python capture_check.py --frontal a.png --profile b.png

Exit status is 0 when the capture is usable and 1 when it is not, so the pipeline
can gate on it.

WHY THIS EXISTS. Until 2026-08-01 this pipeline measured whatever it was given and
reported confident numbers regardless. Then the tolerances were measured, and they
are far tighter than anyone assumed:

  * bilateral symmetry and shape-averageness hold to about **2 degrees of head yaw**.
    At 12.9 degrees -- a frame that PASSES this project's own reference-population
    filter -- symmetry read 5.2x high and averageness 58 percentile points low.
  * the skin layer's lesion count is resolution-bound: 20 at IPD 676 px against 6 on
    identical skin at IPD ~300, and that gap is an information limit, not a
    calibration one.
  * the acne grade moved a whole band, across its own dermatologist-referral
    threshold, driven by illumination rather than pose.

So the honest design is to refuse BEFORE measuring, and to say exactly what to change.
A report built on a photograph that cannot support it is worse than no report: it is
confident, plausible, and wrong, which is precisely the failure this category is
criticised for.

HONESTY ABOUT THE THRESHOLDS THEMSELVES. Each check declares its basis:

  MEASURED       — derived from this project's own measurements, with the n stated.
  AUTHOR-CHOSEN  — a judgement call, informed by observed values but not validated.

They are not equivalent and the output never renders them identically. The pose and
resolution gates are measured. Sharpness, exposure and expression are author-chosen.

ORDERING MATTERS, and it is not cosmetic. Expression is scored from FaceLandmarker
blendshapes, and those are pose-contaminated: at high yaw the far eye is occluded and
the model reports it as a squint (measured: eyeSquint 0.88 on a 58-degree frame versus
0.15-0.33 across relaxed near-frontal frames). So the expression check is only
meaningful once pose has passed, and it is skipped, not guessed, when pose fails.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reference_values as rv  # noqa: E402

MEASURED = "MEASURED"
AUTHOR = "AUTHOR-CHOSEN"

# --- expression thresholds -------------------------------------------------
# AUTHOR-CHOSEN, informed by blendshapes across 11 real frames of one subject
# (10-frame burst + baseline). Relaxed near-frontal frames sat at: eyeSquint
# 0.15-0.33, eyeBlink 0.03-0.15, mouthSmile <=0.003, jawOpen <=0.004, browDown
# <=0.025. Cutoffs are set clear of that range so an ordinary neutral face passes.
# mouthPucker and cheekPuff are deliberately EXCLUDED: pucker read 0.10-0.59 on
# faces that were doing nothing at all, so it carries no signal about posing, and
# cheekPuff was identically zero on every frame.
EXPRESSION = [
    ("eyeSquintLeft", 0.45, "squinting (left eye)"),
    ("eyeSquintRight", 0.45, "squinting (right eye)"),
    ("eyeBlinkLeft", 0.40, "left eye part-closed"),
    ("eyeBlinkRight", 0.40, "right eye part-closed"),
    ("mouthSmileLeft", 0.20, "smiling"),
    ("mouthSmileRight", 0.20, "smiling"),
    ("jawOpen", 0.15, "mouth open"),
    ("browDownLeft", 0.25, "brow pulled down / frowning"),
    ("browDownRight", 0.25, "brow pulled down / frowning"),
]

# MEASURED band for a usable profile turn. Lower bound: below this it is a turned
# frontal, not a profile. Upper bound: the far eye disappears and detection fails --
# known good at 67 deg (the June profile behind every shipped cephalometric), still
# detecting at 69.5, failing at ~90 on two separate true laterals. The exact death
# point between 70 and 90 is UNMEASURED, so 75 is a conservative stop, not a finding.
PROFILE_YAW_BAND = (50.0, 75.0)

SHARPNESS_MIN = 60.0  # AUTHOR-CHOSEN: variance of Laplacian on a 512px face crop
CLIP_MAX_PCT = 2.0  # AUTHOR-CHOSEN: % of face pixels crushed to 0 or blown to 255
LUMA_BAND = (60.0, 205.0)  # AUTHOR-CHOSEN: median face luminance, 0-255


def read_image(path):
    """cv2.imread, plus transparent HEIC support.

    iPhones shoot HEIC by default and OpenCV cannot decode it -- so every photo
    straight off the camera roll read as "unreadable" until this existed. Converts
    via macOS `sips` into a temp PNG, which is the same route the project already
    documents for preparing frontal/profile inputs. Lossless enough for measurement:
    sips decodes the HEIC and re-encodes PNG without resampling.
    """
    img = cv2.imread(path)
    if img is not None:
        return img
    if not path.lower().endswith((".heic", ".heif")):
        return None
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        dest = os.path.join(td, "conv.png")
        r = subprocess.run(
            ["sips", "-s", "format", "png", path, "--out", dest],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or not os.path.exists(dest):
            return None
        return cv2.imread(dest)


def _check(cid, title, status, detail, basis, fix=None, value=None, threshold=None):
    return {
        "id": cid,
        "title": title,
        "status": status,  # pass | warn | fail | skipped
        "detail": detail,
        "basis": basis,
        "fix": fix,
        "value": value,
        "threshold": threshold,
    }


def detect(path):
    """Landmarks + head pose + blendshapes in one pass."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(
            model_asset_path=os.path.join(HERE, "face_landmarker.task")
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    img = read_image(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            )
        )
    if not res.face_landmarks:
        return {"img": img, "wh": (w, h), "found": False}
    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=float)
    out = {
        "img": img,
        "wh": (w, h),
        "found": True,
        "pts": pts,
        "blend": {},
        "pose": None,
    }
    if res.face_blendshapes:
        out["blend"] = {c.category_name: c.score for c in res.face_blendshapes[0]}
    if res.facial_transformation_matrixes:
        R = np.array(res.facial_transformation_matrixes[0])[:3, :3]
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        out["pose"] = (
            float(np.degrees(np.arctan2(R[2, 1], R[2, 2]))),
            float(np.degrees(np.arctan2(-R[2, 0], sy))),
            float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        )
    return out


def face_crop(img, pts, pad=0.12):
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    h, w = img.shape[:2]
    return img[
        max(0, int(y0 - py)) : min(h, int(y1 + py)),
        max(0, int(x0 - px)) : min(w, int(x1 + px)),
    ]


def check_frontal(path):
    checks = []
    d = detect(path)
    if d is None:
        return [
            _check(
                "read",
                "Photograph readable",
                "fail",
                f"Could not open {os.path.basename(path)}.",
                MEASURED,
                "Supply a readable JPEG or PNG.",
            )
        ], None
    if not d["found"]:
        return [
            _check(
                "face",
                "Face detected",
                "fail",
                "No face was found in the frontal photograph.",
                MEASURED,
                "Retake facing the camera, with the whole head in frame and "
                "nothing covering the face.",
            )
        ], d
    checks.append(
        _check("face", "Face detected", "pass", "A single face was located.", MEASURED)
    )

    pts = d["pts"]
    ipd = float(np.linalg.norm(pts[468] - pts[473])) if len(pts) > 473 else float("nan")
    calib = rv.require("capture_gate.skin_calibration_ipd_px").value

    # --- resolution (MEASURED) --------------------------------------------
    if ipd >= 0.9 * calib:
        checks.append(
            _check(
                "resolution",
                "Face resolution",
                "pass",
                f"{ipd:.0f} px between the pupils, against a {calib:.0f} px "
                "calibration. Skin counts are comparable.",
                MEASURED,
                value=round(ipd, 1),
                threshold=0.9 * calib,
            )
        )
    else:
        checks.append(
            _check(
                "resolution",
                "Face resolution",
                "fail",
                f"{ipd:.0f} px between the pupils, against a {calib:.0f} px "
                "calibration. Measured: the lesion count reads about 3.3x "
                "lower at this scale on identical skin, and that is an "
                "information limit — the detail is not in the file.",
                MEASURED,
                "Fill more of the frame with the face: hold the camera "
                "closer, or crop less. Portrait orientation, head and "
                "shoulders only.",
                value=round(ipd, 1),
                threshold=round(0.9 * calib, 1),
            )
        )

    # --- pose (MEASURED) ---------------------------------------------------
    pose_ok = False
    if d["pose"] is None:
        checks.append(
            _check(
                "pose",
                "Head pose",
                "fail",
                "Head pose could not be estimated.",
                MEASURED,
                "Retake facing the camera squarely.",
            )
        )
    else:
        pitch, yaw, roll = d["pose"]
        sym_max = rv.require("pose_gate.symmetry_yaw_deg").value[0]
        if abs(yaw) <= sym_max:
            pose_ok = True
            checks.append(
                _check(
                    "pose",
                    "Head straight to camera",
                    "pass",
                    f"yaw {yaw:+.1f}°, pitch {pitch:+.1f}°, roll {roll:+.1f}°. "
                    f"Within the measured {sym_max:g}° yaw tolerance, so "
                    "symmetry and shape-averageness are reportable.",
                    MEASURED,
                    value=round(yaw, 2),
                    threshold=sym_max,
                )
            )
        else:
            side = "right" if yaw > 0 else "left"
            checks.append(
                _check(
                    "pose",
                    "Head straight to camera",
                    "fail",
                    f"yaw {yaw:+.1f}° (pitch {pitch:+.1f}°, roll {roll:+.1f}°). "
                    f"The measured tolerance is {sym_max:g}°. Measured on a "
                    "face that could not have changed, this much turn moves "
                    "symmetry by several times and averageness by tens of "
                    "percentile points.",
                    MEASURED,
                    f"You are turned to your {side}. Face the lens squarely "
                    "— look straight down the barrel, not at the screen, and "
                    "keep your chin level.",
                    value=round(yaw, 2),
                    threshold=sym_max,
                )
            )
        if abs(pitch) > 12.0:
            checks.append(
                _check(
                    "pitch",
                    "Chin level",
                    "warn",
                    f"pitch {pitch:+.1f}°. Tilting the head changes the "
                    "vertical thirds.",
                    AUTHOR,
                    "Chin down" if pitch > 0 else "Chin up",
                    value=round(pitch, 2),
                    threshold=12.0,
                )
            )
        if abs(roll) > 8.0:
            checks.append(
                _check(
                    "roll",
                    "Head not tilted sideways",
                    "warn",
                    f"roll {roll:+.1f}°. Corrected for, but it widens the "
                    "error bars.",
                    AUTHOR,
                    "Level your head sideways.",
                    value=round(roll, 2),
                    threshold=8.0,
                )
            )

    # --- expression (AUTHOR-CHOSEN, and only meaningful once pose passes) --
    if not pose_ok:
        checks.append(
            _check(
                "expression",
                "Relaxed, neutral expression",
                "skipped",
                "Not assessed: expression is scored from blendshapes, and "
                "those are pose-contaminated — at yaw the far eye is "
                "occluded and reads as a squint (measured 0.88 on a 58° "
                "frame against 0.15–0.33 relaxed). Fix the pose first.",
                AUTHOR,
            )
        )
    else:
        bad = [
            (lbl, d["blend"].get(k, 0.0), thr)
            for k, thr, lbl in EXPRESSION
            if d["blend"].get(k, 0.0) > thr
        ]
        if bad:
            worst = ", ".join(sorted({b[0] for b in bad}))
            checks.append(
                _check(
                    "expression",
                    "Relaxed, neutral expression",
                    "fail",
                    f"Detected: {worst}.",
                    AUTHOR,
                    "Relax your face completely. Eyes open normally — not "
                    "wide, not narrowed. Mouth closed, lips together but "
                    "not pressed. No smile, no frown, no raised or lowered "
                    "brows. Look at the lens and let your face go slack.",
                    value={b[0]: round(b[1], 3) for b in bad},
                )
            )
        else:
            checks.append(
                _check(
                    "expression",
                    "Relaxed, neutral expression",
                    "pass",
                    "No squinting, smiling, jaw opening or brow movement "
                    "above threshold.",
                    AUTHOR,
                )
            )

    # --- sharpness (AUTHOR-CHOSEN) ----------------------------------------
    crop = face_crop(d["img"], pts)
    if crop.size:
        g = cv2.cvtColor(cv2.resize(crop, (512, 512)), cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        if sharp >= SHARPNESS_MIN:
            checks.append(
                _check(
                    "sharpness",
                    "In focus",
                    "pass",
                    f"focus score {sharp:.0f}.",
                    AUTHOR,
                    value=round(sharp, 1),
                    threshold=SHARPNESS_MIN,
                )
            )
        else:
            checks.append(
                _check(
                    "sharpness",
                    "In focus",
                    "fail",
                    f"focus score {sharp:.0f}, below {SHARPNESS_MIN:.0f}. "
                    "The face is soft or motion-blurred.",
                    AUTHOR,
                    "Steady the camera or use a timer, make sure the focus "
                    "locks on the face, and use more light so the shutter "
                    "is faster.",
                    value=round(sharp, 1),
                    threshold=SHARPNESS_MIN,
                )
            )

        # --- exposure (AUTHOR-CHOSEN) -------------------------------------
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clip = 100.0 * float(((gray <= 2) | (gray >= 253)).mean())
        luma = float(np.median(gray))
        if clip <= CLIP_MAX_PCT and LUMA_BAND[0] <= luma <= LUMA_BAND[1]:
            checks.append(
                _check(
                    "exposure",
                    "Even, adequate light",
                    "pass",
                    f"median brightness {luma:.0f}, {clip:.1f}% clipped.",
                    AUTHOR,
                    value={"luma": round(luma, 1), "clipped_pct": round(clip, 2)},
                )
            )
        else:
            why = []
            if clip > CLIP_MAX_PCT:
                why.append(f"{clip:.1f}% of the face is pure black or pure white")
            if luma < LUMA_BAND[0]:
                why.append(f"the face is underexposed (brightness {luma:.0f})")
            if luma > LUMA_BAND[1]:
                why.append(f"the face is overexposed (brightness {luma:.0f})")
            checks.append(
                _check(
                    "exposure",
                    "Even, adequate light",
                    "fail",
                    "; ".join(why) + ". This matters more than it sounds: "
                    "the acne-severity grade was measured moving a whole "
                    "band across its own referral threshold on unchanging "
                    "skin, driven by illumination rather than pose.",
                    AUTHOR,
                    "Face a window or a soft, even light source. Avoid "
                    "direct overhead light, harsh shadows on one side, and "
                    "flash. Nothing bright directly behind you.",
                    value={"luma": round(luma, 1), "clipped_pct": round(clip, 2)},
                )
            )
    return checks, d


def check_profile(path):
    """The profile has a narrower usable range than most people expect."""
    import profile_silhouette as PS

    checks = []
    img = read_image(path)
    if img is None:
        return [
            _check(
                "profile_read",
                "Profile photograph readable",
                "fail",
                f"Could not open {os.path.basename(path)}.",
                MEASURED,
                "Supply a readable JPEG or PNG.",
            )
        ]
    model = os.path.join(HERE, "face_landmarker.task")
    fl = None
    try:
        fl = PS.detect_facelandmarker(img, model)
    except Exception:
        fl = None
    if fl is None:
        checks.append(
            _check(
                "profile_anchor",
                "Profile usable",
                "fail",
                "No anchor face was found in the profile photograph. This "
                "is what a true 90° side-on shot does — the landmarker "
                "needs a partially visible far eye.",
                MEASURED,
                "Turn about three-quarters, not fully side-on: from the "
                "camera's view the far eyebrow and the far edge of the far "
                "eye should still just be visible. This is the single most "
                "common profile mistake, because the shot most people think "
                "is correct is outside the usable range.",
            )
        )
        return checks
    checks.append(
        _check(
            "profile_anchor",
            "Profile usable",
            "pass",
            "Anchor landmarks found — the turn is inside the usable range.",
            MEASURED,
        )
    )
    mask = PS.segment_person(img)
    facing, conf, method = PS.detect_facing(mask, img, model)
    if conf < 0.25:
        checks.append(
            _check(
                "profile_facing",
                "Profile facing direction",
                "warn",
                f"Facing is only {conf:.2f} confident ({method}).",
                MEASURED,
                "Make sure the head is clearly in profile against a plain, "
                "contrasting background.",
                value=round(conf, 3),
            )
        )
    else:
        checks.append(
            _check(
                "profile_facing",
                "Profile facing direction",
                "pass",
                f"Facing {facing} ({conf:.2f} confidence). Mirrored "
                "automatically if needed.",
                MEASURED,
                value=round(conf, 3),
            )
        )
    return checks


def verdict(checks):
    fails = [c for c in checks if c["status"] == "fail"]
    warns = [c for c in checks if c["status"] == "warn"]
    return ("RETAKE" if fails else "READY"), fails, warns


def render(checks, frontal, profile):
    ok, fails, warns = verdict(checks)
    W = 78
    print("=" * W)
    print("CAPTURE CHECK — can these photographs support an accurate report?")
    print("=" * W)
    print(f"  frontal: {os.path.basename(frontal)}")
    print(f"  profile: {os.path.basename(profile)}")
    print()
    mark = {
        "pass": "  ok  ",
        " warn ": " warn ",
        "warn": " warn ",
        "fail": " FAIL ",
        "skipped": " --   ",
    }
    for c in checks:
        print(f"[{mark.get(c['status'], '      ')}] {c['title']:<34} ({c['basis']})")
        print(f"          {c['detail']}")
        if c["fix"]:
            print(f"          -> {c['fix']}")
    print()
    print("=" * W)
    if ok == "READY":
        print("VERDICT: READY — these photographs can support the full report.")
        if warns:
            print(
                f"         {len(warns)} advisory item(s) above; the report will widen"
            )
            print("         its error bars rather than refuse.")
    else:
        print("VERDICT: RETAKE NEEDED — the report was NOT generated.")
        print()
        print(
            "  This is not a failure of your face or your camera. It is this pipeline"
        )
        print("  declining to publish numbers a photograph cannot support. Measured on")
        print(
            "  one unchanging face, an off-axis shot moved its own headline figures by"
        )
        print("  more than twenty times, and a report built on that would have been")
        print("  confident, plausible and wrong.")
        print()
        print("  Fix these, in order:")
        for i, c in enumerate(fails, 1):
            print(f"    {i}. {c['title']} — {c['fix'] or c['detail']}")
        print()
        print("  The shot that works: portrait orientation, head and shoulders filling")
        print("  the frame, camera at eye level about an arm's length away, facing a")
        print("  window or other soft even light, looking straight into the lens, face")
        print("  completely relaxed — no smile, no squint, mouth closed. Then the same")
        print("  again turned about three-quarters for the profile.")
        # ⭐ MEASURED, NOT ADVICE. "Mouth closed" above is not enough, and the
        # difference is not cosmetic: between two sittings the reference
        # subject's lips went from barely pursed to noticeably pursed, and the
        # shape-typicality figure moved 23.6-30.2 percentile points on a face
        # that had not changed. mouthPucker correlates -0.92 with the dominant
        # shape axis. The gate refuses the photograph; this is the sentence that
        # stops it being taken in the first place.
        print()
        print("  And let your mouth go slack — lips together but NOT pressed,")
        print("  pushed forward or held. Pursing them is easy to do without")
        print("  noticing and it moves the shape measurements more than a")
        print("  different day, a different room or a different camera does.")
    print("=" * W)
    return ok


def scan_profiles(folder):
    """Triage a folder of candidate PROFILE shots. Which ones are usable?

    Exists because the usable range is narrower than anyone expects and is not
    obvious by eye: a true 90-degree side-on shot -- the one most people believe is
    the correct profile photograph -- fails outright, because the landmarker needs a
    partially visible far eye. Guessing costs a re-shoot; this costs a few seconds.
    """
    import glob

    import profile_silhouette as PS

    model = os.path.join(HERE, "face_landmarker.task")
    paths = sorted(
        p
        for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG", "heic", "HEIC")
        for p in glob.glob(os.path.join(folder, f"*.{ext}"))
    )
    if not paths:
        print(f"no images found in {folder}")
        return 1
    print("=" * 78)
    print(f"PROFILE TRIAGE — {len(paths)} candidate(s) in {folder}")
    print("=" * 78)
    print(f"{'file':<26}{'usable':>9}{'facing':>9}{'yaw':>7}{'face h':>9}   note")
    usable = []
    for path in paths:
        name = os.path.basename(path)[:25]
        img = read_image(path)
        if img is None:
            print(f"{name:<26}{'--':>9}{'--':>9}{'--':>7}{'--':>9}   unreadable")
            continue
        try:
            fl = PS.detect_facelandmarker(img, model)
        except Exception:
            fl = None
        if fl is None:
            # These two causes CANNOT be separated here and the message says so
            # rather than guessing. Both the anchor detector and the full landmarker
            # are FaceLandmarker, so a true 90-degree lateral of a real face fails
            # identically to an image with no face in it. An earlier version guessed
            # "too far side-on" and told a user to turn their head less when the file
            # was a photograph of a sculpture.
            print(
                f"{name:<26}{'NO':>9}{'--':>9}{'--':>7}{'--':>9}   "
                "no landmarks — either turned too far side-on, or not a face photo"
            )
            continue
        # How far is the head actually turned? detect_facing answers WHICH WAY, not
        # HOW FAR -- it returns a flat 0.95 whenever the nose is off centre, so a
        # near-frontal frame scored "usable profile" on this function's first run.
        # Measured band: the June profile that produced every shipped cephalometric
        # sits at yaw 67.0 deg; detection still works at 69.5; both true laterals
        # (~90) fail outright. Below ~50 it is not a profile, it is a turned frontal.
        dd = detect(path)
        yaw = abs(dd["pose"][1]) if (dd and dd.get("pose")) else None
        if yaw is not None and yaw < PROFILE_YAW_BAND[0]:
            print(
                f"{name:<26}{'NO':>9}{'--':>9}{yaw:>7.0f}{'--':>9}   "
                f"only {yaw:.0f}deg turned — that is a frontal, not a profile"
            )
            continue
        if yaw is not None and yaw > PROFILE_YAW_BAND[1]:
            print(
                f"{name:<26}{'NO':>9}{'--':>9}{yaw:>7.0f}{'--':>9}   "
                f"{yaw:.0f}deg is too far round — turn back toward the camera"
            )
            continue
        mask = PS.segment_person(img)
        facing, conf, _m = PS.detect_facing(mask, img, model)
        ys, xs = PS.anterior_curve(mask, "right" if facing == "right" else "left")
        fh = float(ys.max() - ys.min()) if len(ys) else 0.0
        frac = 100.0 * fh / img.shape[0]
        # Low facing confidence means the silhouette is near-symmetric, i.e. the head
        # is not actually turned. Caught on this function's own first run: a FRONTAL
        # photo scored "usable" at 0.03 confidence. Anchor landmarks alone are not
        # evidence of a profile -- a face-on shot has them in abundance.
        if conf < 0.25:
            print(
                f"{name:<26}{'NO':>9}{facing:>9}{yaw or 0:>7.0f}{fh:>9.0f}   "
                "silhouette near-symmetric — not a clear profile"
            )
            continue
        note = "usable" if frac > 15 else "usable, but subject is small in frame"
        usable.append(path)
        print(
            f"{name:<26}{'yes':>9}{facing:>9}{(yaw or 0):>7.0f}{fh:>9.0f}   {note} "
            f"({frac:.0f}% of frame)"
        )
    print()
    print(f"{len(usable)} of {len(paths)} usable as profile input.")
    if len(usable) >= 4:
        print("Enough for a within-session repeatability measurement (needs >= 4).")
    else:
        print("NOT enough yet — a repeatability floor needs at least 4 usable frames")
        print("from one sitting. Turn less far to the side and re-shoot.")
    return 0 if usable else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scan",
        metavar="DIR",
        help="triage a folder of candidate PROFILE shots and stop",
    )
    ap.add_argument("--frontal", default=os.path.join(HERE, "frontal.png"))
    ap.add_argument("--profile", default=os.path.join(HERE, "profile.png"))
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--out", default=os.path.join(HERE, "capture_qc.json"))
    ap.add_argument(
        "--profile-only",
        action="store_true",
        help=(
            "gate the PROFILE only. For the set path, where intake.assess_set has "
            "already judged the frontal photographs as a SET -- which is the whole "
            "point: no single frame has to be ideal, so re-gating one casual frame "
            "against a deliberate-capture standard would refuse exactly the input "
            "the set path exists to accept. The silhouette gets no such reprieve, "
            "because nothing in the set replaces a true lateral."
        ),
    )
    a = ap.parse_args()

    if a.scan:
        return scan_profiles(a.scan)

    checks = [] if a.profile_only else check_frontal(a.frontal)[0]
    checks += check_profile(a.profile)
    ok, fails, warns = verdict(checks)

    payload = {
        "verdict": ok,
        "frontal": a.frontal,
        "profile": a.profile,
        "n_fail": len(fails),
        "n_warn": len(warns),
        "checks": checks,
        "_basis_note": (
            "MEASURED thresholds come from this project's own 2026-08-01 measurements "
            "(one subject, n=5 near-frontal frames of a 26-second burst). AUTHOR-CHOSEN "
            "thresholds are judgement calls informed by observed values across 11 real "
            "frames but not validated. They are not equivalent and are never rendered "
            "identically."
        ),
    }
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)

    if a.json:
        print(json.dumps(payload, indent=2))
    else:
        render(checks, a.frontal, a.profile)
        print(f"\nwrote {os.path.basename(a.out)}")
    return 0 if ok == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
