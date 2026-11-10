"""Frontal facial-harmony analysis.

Computes facial thirds, fifths, golden-ratio (phi) checks, bilateral symmetry,
and eye metrics from MediaPipe FaceMesh landmarks; writes metrics_frontal.json,
landmarks_frontal.csv, and frontal_annotated.png.
"""

import json
import math
import os
import sys
import numpy as np
import cv2

import fa_common as F
import fa_refine
import reference_values as rv

HERE = os.path.dirname(os.path.abspath(__file__))

IMG = os.path.join(HERE, "frontal.png")
OUT_JSON = os.path.join(HERE, "metrics_frontal.json")
OUT_CSV = os.path.join(HERE, F.GEOMETRY_CSV)
OUT_PHOTO_CSV = os.path.join(HERE, F.PHOTO_CSV)
OUT_IMG = os.path.join(HERE, "frontal_annotated.png")
OUT_SCALE = os.path.join(HERE, "scale_mm.json")


def compute(pts):
    P = pts
    m = {}

    # ---- Facial thirds (horizontal bands -> vertical extents) --------------
    brow_center = (P[F.BROW_L] + P[F.BROW_R]) / 2.0
    y_tri, y_brow, y_sub, y_men = (
        P[F.TRICHION][1],
        brow_center[1],
        P[F.SUBNASALE][1],
        P[F.MENTON][1],
    )
    upper = abs(y_brow - y_tri)
    middle = abs(y_sub - y_brow)
    lower = abs(y_men - y_sub)
    total_h = upper + middle + lower
    thirds = []
    for name, val in [
        ("Upper (trichion->brow)", upper),
        ("Middle (brow->subnasale)", middle),
        ("Lower (subnasale->menton)", lower),
    ]:
        pct = 100.0 * val / total_h
        dev = pct - rv.require("thirds.equal_share").value
        thirds.append(
            {
                "zone": name,
                "pixels": round(val, 1),
                "pct": round(pct, 2),
                "dev_pp": round(dev, 2),
                "grade": F.grade(abs(dev), *rv.require("grade_band.thirds").value),
            }
        )
    m["thirds"] = {
        "items": thirds,
        "total_height_px": round(total_h, 1),
        "brow_center": [round(brow_center[0], 1), round(brow_center[1], 1)],
    }

    # ---- Facial fifths (vertical bands -> horizontal extents) --------------
    xs = {
        k: P[k][0]
        for k in [
            F.ZYGION_L,
            F.L_CANTHUS_LAT,
            F.L_CANTHUS_MED,
            F.R_CANTHUS_MED,
            F.R_CANTHUS_LAT,
            F.ZYGION_R,
        ]
    }
    face_w = abs(xs[F.ZYGION_R] - xs[F.ZYGION_L])
    segs = [
        ("Left outer (234->33)", abs(xs[F.L_CANTHUS_LAT] - xs[F.ZYGION_L])),
        ("Left eye (33->133)", abs(xs[F.L_CANTHUS_MED] - xs[F.L_CANTHUS_LAT])),
        ("Intercanthal (133->362)", abs(xs[F.R_CANTHUS_MED] - xs[F.L_CANTHUS_MED])),
        ("Right eye (362->263)", abs(xs[F.R_CANTHUS_LAT] - xs[F.R_CANTHUS_MED])),
        ("Right outer (263->454)", abs(xs[F.ZYGION_R] - xs[F.R_CANTHUS_LAT])),
    ]
    fifths = []
    for name, val in segs:
        pct = 100.0 * val / face_w
        dev = pct - rv.require("fifths.equal_share").value
        fifths.append(
            {
                "zone": name,
                "pixels": round(val, 1),
                "pct": round(pct, 2),
                "dev_pp": round(dev, 2),
                "grade": F.grade(abs(dev), *rv.require("grade_band.fifths").value),
            }
        )
    m["fifths"] = {"items": fifths, "face_width_px": round(face_w, 1)}

    # ---- Phi / golden-ratio checks -----------------------------------------
    face_height = abs(P[F.MENTON][1] - P[F.TRICHION][1])
    eye_w_L = F.hdist(P, F.L_CANTHUS_LAT, F.L_CANTHUS_MED)
    eye_w_R = F.hdist(P, F.R_CANTHUS_MED, F.R_CANTHUS_LAT)
    avg_eye_w = (eye_w_L + eye_w_R) / 2.0
    nose_w = F.hdist(P, F.ALA_L, F.ALA_R)
    mouth_w = F.hdist(P, F.MOUTH_L, F.MOUTH_R)
    upper_lip_seg = F.vdist(P, F.SUBNASALE, F.LABRALE_SUP)  # nose base -> upper lip
    chin_seg = F.vdist(P, F.LABRALE_INF, F.MENTON)  # lower lip -> chin

    def phi_row(name, actual, ideal, note=None):
        """One golden-ratio row.

        `ideal=None` means NO documented canon exists for this ratio as computed.
        In that case the value is reported descriptively with no deviation and no
        grade, rather than scored against a benchmark that does not apply — the
        same refuse-rather-than-fabricate rule used for the nasion-based angles
        (analyze_extended.py) and the cervicomental angle (analyze_profile.py).
        """
        row = {"ratio": name, "actual": round(actual, 3), "ideal": ideal}
        if ideal is None:
            row["dev_pct"] = None
            row["grade"] = "n/a"
        else:
            dev_rel = 100.0 * (actual - ideal) / ideal
            row["dev_pct"] = round(dev_rel, 1)
            row["grade"] = F.grade(abs(dev_rel), *rv.require("grade_band.phi").value)
        if note:
            row["note"] = note
        return row

    m["phi"] = [
        phi_row(
            "Face height / width",
            face_height / face_w,
            rv.require("phi.face_height_over_width").value,
        ),
        phi_row(
            "Nose width / avg eye width",
            nose_w / avg_eye_w,
            rv.require("phi.nose_width_over_eye_width").value,
        ),
        phi_row(
            "Mouth width / nose width",
            mouth_w / nose_w,
            rv.require("phi.mouth_width_over_nose_width").value,
        ),
        # NO canon applies here as computed. These segments are the philtrum
        # (subnasale->labrale superius) and the lower-lip->menton span; the lip
        # vermilion between them belongs to neither, so this is NOT the classical
        # lower-third division. That division runs subnasale->stomion->menton and
        # its convention is a neoclassical 1:2 split (ratio ~0.5) — an artistic
        # canon, not a golden-ratio claim. Grading this proxy against 1.618 was
        # guaranteed to read as a large deviation on every human face, since the
        # philtrum is always far shorter than the lip-to-chin span. It was scored
        # red here and then explained away in a report footnote; reporting it
        # without a grade is the honest form.
        phi_row(
            "Lower face (philtrum / lip-chin)",
            upper_lip_seg / chin_seg,
            None,
            note=(
                "Descriptive only — no canon applies. These segments are not the "
                "classical lower-third division (which runs subnasale->stomion->menton "
                "and follows a neoclassical 1:2 convention, not phi), so there is no "
                "documented ideal to compare against."
            ),
        ),
    ]
    m["phi_inputs"] = {
        "face_height_px": round(face_height, 1),
        "face_width_px": round(face_w, 1),
        "eye_w_L": round(eye_w_L, 1),
        "eye_w_R": round(eye_w_R, 1),
        "nose_w": round(nose_w, 1),
        "mouth_w": round(mouth_w, 1),
        "upper_lip_seg": round(upper_lip_seg, 1),
        "chin_seg": round(chin_seg, 1),
    }

    # ---- Bilateral symmetry -------------------------------------------------
    midline_x = float(np.mean([P[i][0] for i in F.MIDLINE_IDS]))
    sym = []
    for li, ri, name in F.SYMMETRY_PAIRS:
        dl = abs(P[li][0] - midline_x)
        dr = abs(P[ri][0] - midline_x)
        mean_d = (dl + dr) / 2.0
        asym = 100.0 * abs(dl - dr) / mean_d if mean_d > 0 else 0.0
        vtilt = float(P[li][1] - P[ri][1])  # vertical offset (px), +=left lower
        sym.append(
            {
                "pair": name,
                "left_px": round(dl, 1),
                "right_px": round(dr, 1),
                "asym_pct": round(asym, 2),
                "vert_offset_px": round(vtilt, 1),
                "grade": F.grade(
                    asym, *rv.require("grade_band.frontal_symmetry_pairs").value
                ),
            }
        )
    m["symmetry"] = {
        "items": sym,
        "midline_x": round(midline_x, 1),
        # NO AGGREGATE IS PUBLISHED HERE — deliberately. Retired 2026-08-01 after
        # measurement, not after inspection. Across 5 near-frontal frames of a
        # CONSTANT face (26-second burst) the old avg_asym_pct ran 2.77%–64.69%:
        # noise, not an error bar. A 2x2 decomposition then showed WHY the obvious
        # repair does not work — swapping the fixed midline for a fitted axis moved
        # the CV 74.8 -> 73.1, and swapping per-pair scale for IPD scale moved it
        # 74.8 -> 74.1. The dominant term is head YAW, which no version corrects,
        # and which inflates even analyze_advanced's 42-pair figure (CV 71.1 on the
        # same frames). So this field was a strictly worse duplicate of a number
        # that is itself pose-conditional. The 42-pair version in
        # analyze_advanced.symmetry() is the one the report reads, and it now
        # carries an explicit pose_condition. Per-pair items are kept because the
        # report shows them as a labelled panel, not as an aggregate claim.
        "aggregate_withheld": (
            "See analyze_advanced.symmetry() -> mean_asym_pct_ipd, which fits a "
            "symmetry axis and carries a measured pose_condition. Publishing a "
            "second, noisier symmetry scalar here invited it to be read as a "
            "measurement."
        ),
    }

    # ---- Eye analysis -------------------------------------------------------
    def canthal_tilt(med, lat):
        dy = P[med][1] - P[lat][1]  # +ve if lateral is higher (smaller y)
        dx = abs(P[lat][0] - P[med][0])
        return math.degrees(math.atan2(dy, dx))

    ear_L = F.vdist(P, F.L_EYE_TOP, F.L_EYE_BOT) / eye_w_L
    ear_R = F.vdist(P, F.R_EYE_TOP, F.R_EYE_BOT) / eye_w_R
    tilt_L = canthal_tilt(F.L_CANTHUS_MED, F.L_CANTHUS_LAT)
    tilt_R = canthal_tilt(F.R_CANTHUS_MED, F.R_CANTHUS_LAT)
    ipd = F.dist(P, F.L_IRIS, F.R_IRIS)
    m["eyes"] = {
        "ear_left": round(ear_L, 3),
        "ear_right": round(ear_R, 3),
        "canthal_tilt_left_deg": round(tilt_L, 2),
        "canthal_tilt_right_deg": round(tilt_R, 2),
        "ipd_px": round(ipd, 1),
        "ipd_pct_face_width": round(100.0 * ipd / face_w, 2),
        "eye_width_L": round(eye_w_L, 1),
        "eye_width_R": round(eye_w_R, 1),
    }
    m["midline_x"] = round(midline_x, 1)
    return m


def annotate(img, pts, m):
    P = pts
    h, w = img.shape[:2]
    out = img.copy()
    x_left = float(min(P[:, 0]))
    x_right = float(max(P[:, 0]))
    y_top = float(min(P[:, 1]))
    y_bot = float(max(P[:, 1]))

    # All 478 landmark dots (red)
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), 2, F.RED, -1, cv2.LINE_AA)

    # Thirds: green dashed horizontal lines
    brow_center = m["thirds"]["brow_center"]
    for y in [P[F.TRICHION][1], brow_center[1], P[F.SUBNASALE][1], P[F.MENTON][1]]:
        F.dashed_hline(out, y, x_left - 40, x_right + 40, F.GREEN, 3)
    F.label(
        out,
        "THIRDS",
        (int(x_right) + 50, int(P[F.TRICHION][1]) + 10),
        F.GREEN,
        1.1,
        3,
        bg=(0, 0, 0),
    )

    # Fifths: orange dotted vertical lines
    for k in [
        F.ZYGION_L,
        F.L_CANTHUS_LAT,
        F.L_CANTHUS_MED,
        F.R_CANTHUS_MED,
        F.R_CANTHUS_LAT,
        F.ZYGION_R,
    ]:
        F.dotted_vline(out, P[k][0], y_top - 40, y_bot + 40, F.ORANGE, 4)
    F.label(
        out,
        "FIFTHS",
        (int(x_left) - 30, int(y_top) - 50),
        F.ORANGE,
        1.1,
        3,
        bg=(0, 0, 0),
    )

    # Midline: blue solid vertical
    mx = int(m["midline_x"])
    cv2.line(out, (mx, int(y_top) - 40), (mx, int(y_bot) + 40), F.BLUE, 3, cv2.LINE_AA)
    F.label(out, "midline", (mx + 8, int(y_bot) + 70), F.BLUE, 1.0, 2, bg=(0, 0, 0))

    # Canthal tilt LINES per eye (cyan), extended. The angle LABELS are drawn
    # further down, after the landmark indices — see the note there.
    tilt_labels = []
    for med, lat, tilt, side in [
        (F.L_CANTHUS_MED, F.L_CANTHUS_LAT, m["eyes"]["canthal_tilt_left_deg"], "L"),
        (F.R_CANTHUS_MED, F.R_CANTHUS_LAT, m["eyes"]["canthal_tilt_right_deg"], "R"),
    ]:
        a = P[med].astype(int)
        b = P[lat].astype(int)
        d = (b - a).astype(float)
        d = d / (np.linalg.norm(d) + 1e-9)
        p1 = (a - d * 70).astype(int)
        p2 = (b + d * 70).astype(int)
        cv2.line(out, tuple(p1), tuple(p2), F.CYAN, 3, cv2.LINE_AA)
        # Anchor the label OUTSIDE the lateral canthus, past the end of the drawn
        # line, instead of at the eye midpoint. The midpoint sits directly on the
        # iris-centre landmarks (468 / 473), whose index labels were overprinting
        # the angle: the report rendered "L tilt 468.9deg" and "R tilt +473deg"
        # where the true values are +3.9 and +1.6. A reader could take a landmark
        # ID for part of a measurement.
        text = f"{side} tilt {tilt:+.1f}deg"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        lx = int(p2[0] - tw - 12) if d[0] < 0 else int(p2[0] + 12)
        lx = max(8, min(lx, out.shape[1] - tw - 8))  # keep inside the frame
        tilt_labels.append((text, (lx, int(p2[1] - 14))))

    # Key landmark labels (white on dark)
    for idx, txt in F.KEY_LANDMARKS.items():
        x, y = int(P[idx][0]), int(P[idx][1])
        cv2.circle(out, (x, y), 5, F.WHITE, -1, cv2.LINE_AA)
        F.label(out, txt, (x + 7, y - 7), F.WHITE, 0.6, 1, bg=(30, 30, 30))

    # Measured values go on LAST so a debug landmark index can never occlude a
    # number a reader is meant to trust.
    for text, org in tilt_labels:
        F.label(out, text, org, F.CYAN, 0.9, 2, bg=(0, 0, 0))

    return out


def head_pose(path):
    """Measure head pose once, here, so downstream stages don't have to guess it.

    Bilateral symmetry turned out to be almost entirely a yaw reading (measured
    2026-08-01: 1.18%–27.29% of IPD across 26 seconds of a constant face), so the
    pose has to reach analyze_advanced, and no stage before this one had access to
    it. Emitting it here — the first stage, which already has the photo open — is
    the shortest path and also starts the capture-conditions block Track F wants.

    `pose_detect` is imported lazily: compute() must stay usable with no model asset
    present, which is what lets selfcheck.py run on a clean clone.
    """
    try:
        import pose_detect as FZ

        _P, M, _wh, _img = FZ.detect_3d(path)
        pitch, yaw, roll = FZ.euler(M[:3, :3])
    except Exception as e:  # detection genuinely fails on true laterals
        return {
            "yaw_deg": None,
            "pitch_deg": None,
            "roll_deg": None,
            "error": f"{type(e).__name__}: {e}",
            "_method": "FaceLandmarker facial transformation matrix -> Euler angles",
        }
    return {
        "yaw_deg": round(float(yaw), 2),
        "pitch_deg": round(float(pitch), 2),
        "roll_deg": round(float(roll), 2),
        "error": None,
        "_method": (
            "FaceLandmarker facial transformation matrix -> Euler angles. A rigid "
            "fit of the canonical 3D model, so it reaches pose by a different route "
            "than the landmark pair residuals it is used to qualify — which is why "
            "it, and not analyze_advanced's own yaw_balance_pct, is the gate. "
            "yaw_balance_pct is computed from the same residuals as the asymmetry "
            "and would be partly circular."
        ),
    }


def write_landmark_csv(path, pts):
    with open(path, "w") as f:
        f.write("index,x,y\n")
        for i, (x, y) in enumerate(pts):
            f.write(f"{i},{x:.2f},{y:.2f}\n")


def main():
    from_fit = "--from-fit" in sys.argv

    # Refined landmarks: multi-view fusion + sub-pixel edge refinement + spline
    # de-jitter of the FACE_OVAL. HONEST CAVEAT: fusion reduces *detector jitter
    # on this single photo* (a lower bound on uncertainty) -- it is NOT
    # test-retest stability (different day/lighting/pose/expression).
    pts, dispersion, rmeta = fa_refine.refined_landmarks(IMG)
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    if pts is None:
        print("FRONTAL: no face detected")
        return
    print(
        f"FRONTAL: detected {len(pts)} refined landmarks on {w}x{h} image "
        f"({rmeta['n_variants']} variants fused)"
    )

    # THE PHOTOGRAPH'S OWN LANDMARKS, always written. Anything that draws on
    # frontal.png or samples pixels out of it reads this file, in both paths.
    write_landmark_csv(OUT_PHOTO_CSV, pts)

    # THE GEOMETRY FILE. In the two-photo path it is the same detection, so the
    # two files are byte-identical and the split costs nothing. Under --from-fit
    # it was already written by fit_identity_stage.py from a shape fitted across
    # the whole set, and overwriting it here would throw that away and silently
    # return the pipeline to single-photo geometry.
    if from_fit:
        print(f"FRONTAL: --from-fit — measuring {F.GEOMETRY_CSV} from the set fit")
    else:
        write_landmark_csv(OUT_CSV, pts)

    # Emit the iris-derived px->mm scale (no reference object needed).
    scale = rmeta["scale_mm"]
    ipd_px = F.dist(pts, F.L_IRIS, F.R_IRIS)
    ipd_mm = round(ipd_px / scale["px_per_mm"], 1)
    with open(OUT_SCALE, "w") as f:
        json.dump(
            {
                "px_per_mm": scale["px_per_mm"],
                "iris_px": scale["iris_px"],
                "err_pct": scale["err_pct"],
                "ipd_mm": ipd_mm,
                "_method": (
                    "iris horizontal diameter = 11.7mm (SD ~0.5mm); "
                    "near-frontal only; +/-~4-5%"
                ),
            },
            f,
            indent=2,
        )
    print(f"FRONTAL: scale {scale['px_per_mm']} px/mm  ->  IPD {ipd_mm} mm")

    # WHAT GETS MEASURED. Under --from-fit the geometry comes from the set fit,
    # which is pose-free by construction: the fitter solves each frame's rotation
    # explicitly and `frontal_render` projects the recovered shape head-on. So the
    # yaw the downstream gates read is 0 -- not because a photograph was taken at
    # 0 deg, but because pose was removed rather than avoided, and `pose.source`
    # says which of those it is so no reader can mistake one for the other.
    geom = F.load_landmark_csv("geometry", HERE) if from_fit else pts
    m = compute(geom)
    if from_fit:
        m["pose"] = {
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "source": "multi-view fit — pose solved per frame and removed",
            "photo_pose": head_pose(IMG),
        }
    else:
        m["pose"] = head_pose(IMG)
    m["_meta"] = {
        "image": IMG,
        "width": w,
        "height": h,
        "n_landmarks": int(len(geom)),
        "geometry_source": "multi-view fit" if from_fit else "this photograph",
        # Stamped so downstream stages can prove they are reading landmarks derived
        # from the photo currently on disk. See fa_common.assert_fresh.
        "source_sha256": F.image_fingerprint(IMG),
    }
    if from_fit:
        # Carry the SET fingerprint too, so a stage can prove the geometry it is
        # about to measure came from the photographs sitting in the set directory
        # now. See fa_common.assert_fresh_set.
        try:
            with open(os.path.join(HERE, "fit_diagnostics.json")) as fh:
                m["_meta"]["set_sha256"] = json.load(fh)["_meta"]["set_sha256"]
        except (OSError, ValueError, KeyError):
            raise SystemExit(
                "--from-fit but fit_diagnostics.json is missing or unstamped. "
                "Run fit_identity_stage.py first; measuring the geometry file "
                "without knowing which photographs produced it is how a report "
                "gets built on the previous subject."
            )
    with open(OUT_JSON, "w") as f:
        json.dump(m, f, indent=2)
    p = m["pose"]
    if p.get("source"):
        print(f"FRONTAL: pose {p['source']} — yaw 0 by construction")
    elif p["yaw_deg"] is None:
        print(
            f"FRONTAL: head pose UNAVAILABLE ({p['error']}) — pose-conditional "
            "metrics downstream will decline to grade themselves"
        )
    else:
        print(
            f"FRONTAL: head pose  yaw {p['yaw_deg']:+.1f}  pitch {p['pitch_deg']:+.1f}"
            f"  roll {p['roll_deg']:+.1f} deg"
        )

    # The annotated image is ink on the photograph, so it is drawn from `pts` --
    # the detection -- never from the fitted geometry. See fa_common.PHOTO_CSV.
    out = annotate(img, pts, m)
    cv2.imwrite(OUT_IMG, out)
    print(f"Wrote {OUT_JSON}, {OUT_PHOTO_CSV}, {OUT_IMG}")

    # Quick console sanity print
    print("\n-- thirds --")
    for t in m["thirds"]["items"]:
        print(
            f"   {t['zone']:<28} {t['pct']:6.2f}%  dev {t['dev_pp']:+.2f}pp  [{t['grade']}]"
        )
    print("-- fifths --")
    for t in m["fifths"]["items"]:
        print(
            f"   {t['zone']:<26} {t['pct']:6.2f}%  dev {t['dev_pp']:+.2f}pp  [{t['grade']}]"
        )
    print("-- phi --")
    for t in m["phi"]:
        if t["ideal"] is None:
            print(
                f"   {t['ratio']:<42} {t['actual']:6.3f} (no canon applies — descriptive only)"
            )
        else:
            print(
                f"   {t['ratio']:<42} {t['actual']:6.3f} (ideal {t['ideal']}) dev {t['dev_pct']:+.1f}% [{t['grade']}]"
            )
    print("-- symmetry --")
    for t in m["symmetry"]["items"]:
        print(
            f"   {t['pair']:<20} L{t['left_px']:7.1f} R{t['right_px']:7.1f}  asym {t['asym_pct']:5.2f}%  [{t['grade']}]"
        )
    print(
        "   (no aggregate — see analyze_advanced mean_asym_pct_ipd and its pose gate)"
    )
    print("-- eyes --")
    e = m["eyes"]
    print(
        f"   EAR L {e['ear_left']}  R {e['ear_right']}   tilt L {e['canthal_tilt_left_deg']}  R {e['canthal_tilt_right_deg']}"
    )
    print(f"   IPD {e['ipd_px']}px = {e['ipd_pct_face_width']}% of face width")


if __name__ == "__main__":
    main()
