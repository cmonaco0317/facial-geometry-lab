#!/usr/bin/env python3
"""Public implementation; private development notes omitted."""

from __future__ import annotations

import html
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import shoot_guide as SG  # noqa: E402

OUT = os.path.join(HERE, "how_to_shoot.html")


def shot_list(t: dict) -> list[dict]:
    """The concrete shots to take, DERIVED from the gate's own numbers.

    The gate wants three things at once and a person can only hold one: enough
    total spread, enough frames, and one frame inside the lateral band. This
    turns them into a numbered list where following it satisfies all three with
    margin, so nobody has to do the arithmetic that produced it.
    """
    half = t["min_spread_deg"] / 2.0  # each way from straight on
    lateral = (t["profile_yaw_lo"] + t["profile_yaw_hi"]) / 2.0  # mid-band, safest
    mid = math.ceil(half)  # round AWAY from the floor, never toward it
    # ⭐ SIGN CONVENTION, because getting it wrong is worse than omitting it.
    # These are TOP-DOWN diagrams with the camera at the bottom, so the subject
    # faces the bottom of the frame. Someone facing you-from-above who turns to
    # their own LEFT swings their nose to the RIGHT of the picture. Positive yaw
    # here means "nose moves right in the drawing", which is therefore the
    # subject's own left.
    shots = [
        {
            "n": 1,
            "yaw": 0.0,
            "name": "Straight on",
            "note": "Look right down the lens. This is the reference frame.",
            "file": None,
        },
        {
            "n": 2,
            "yaw": float(mid),
            "name": f"Turn left {mid:.0f}°",
            "note": "Head turns, camera stays put. Eyes still on the lens.",
            "file": None,
        },
        {
            "n": 3,
            "yaw": float(-mid),
            "name": f"Turn right {mid:.0f}°",
            "note": "The mirror of shot 2. This pair is what creates the spread.",
            "file": None,
        },
        {
            "n": 4,
            "yaw": lateral,
            "name": f"Three-quarter, about {lateral:.0f}°",
            "note": (
                "THE ONE MOST SETS ARE MISSING. Either side — whichever is easier. "
                "Both eyes must still be visible: past "
                f"{t['dead_abs_yaw']:.0f}° the far eye disappears and the frame is dropped."
            ),
            "file": "profile",
        },
    ]
    return shots


def returns_table() -> list[dict]:
    """How much each extra photograph actually buys, from the calibration sweep.

    ⭐ THIS EXISTS BECAUSE "FOUR" IS A FLOOR AND READS LIKE A RECOMMENDATION.
    `intake_calibration.json` measured 400 trials against known synthetic truth:
    going 2 -> 4 photographs more than halves the shape error (12.3% -> 5.4%),
    and 4 -> 8 only takes it to 3.8%. Spread does more than count at every point
    (correlation with error -0.52 against -0.45), which is why the page spends
    its emphasis on turning rather than on shooting more.
    """
    import json

    import intake as IN

    # ⚠️ READ THE RAW ARTEFACT, not `load_calibration()`. That normaliser exposes
    # only what the GATE needs (`min_views`, `min_spread_deg`, `by_spread_band`)
    # and drops `by_n_views` — so going through it returned an empty table and
    # rendered a table with headings and no rows, which looks like a styling bug
    # rather than missing data. Same file, same command, one hop earlier.
    with open(IN.CALIBRATION, encoding="utf-8") as fh:
        by_n = (json.load(fh).get("derived") or {}).get("by_n_views") or {}
    return [
        {
            "n": int(k),
            "err": float(by_n[k]["median_error_pct"]),
            "pass": float(by_n[k]["pass_rate"]),
        }
        for k in sorted(by_n, key=int)
    ]


def head_svg(yaw_deg: float, size: int = 132) -> str:
    """A top-down head at `yaw_deg`, drawn from the angle rather than picked from a set.

    Top-down is the only view where a yaw angle is legible as an angle. A person
    looking at a front-on photograph of a turned head cannot tell 35 degrees from
    55, which is precisely the confusion that costs them the report.
    """
    w, h = size, size + 22
    cx, cy = w / 2.0, h * 0.40
    r = w * 0.195
    a = math.radians(yaw_deg)
    cam_y = h - 13

    def at(theta: float, k: float) -> tuple[float, float]:
        """A point k*r from the head centre, theta off the camera direction.

        ⭐ +y IS TOWARD THE CAMERA. The first version of this had `cy - cos(a)`,
        which drew every subject facing AWAY from the lens — the diagram said
        the opposite of the instruction beside it, and looked plausible enough
        to ship.
        """
        return cx + math.sin(theta) * r * k, cy + math.cos(theta) * r * k

    nx, ny = at(a, 1.52)  # nose tip
    lx, ly = at(a - 0.46, 0.98)  # nose base, one side
    rx, ry = at(a + 0.46, 0.98)  # nose base, other side
    e1x, e1y = at(a - 1.5708, 1.0)  # ears, so rotation reads at 0°
    e2x, e2y = at(a + 1.5708, 1.0)

    # The arc sits BELOW the head and ABOVE the camera, and the degree is pinned
    # to the top of the frame rather than riding the arc — on the first attempt
    # it rode the arc and landed on top of the camera at 26°.
    arc = ""
    if abs(yaw_deg) >= 1:
        ar = r * 1.98
        ex, ey = cx + math.sin(a) * ar, cy + math.cos(a) * ar
        sweep = 0 if yaw_deg > 0 else 1
        arc = (
            f'<path d="M {cx:.1f} {cy + ar:.1f} A {ar:.1f} {ar:.1f} 0 0 {sweep} '
            f'{ex:.1f} {ey:.1f}" class="arc"/>'
            f'<text x="{cx:.1f}" y="15" class="deg">{abs(yaw_deg):.0f}°</text>'
        )

    return f"""<svg viewBox="0 0 {w} {h}" role="img" aria-label="seen from above: head turned {abs(yaw_deg):.0f} degrees from the camera">
  <line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cam_y - 11}" class="axis"/>
  {arc}
  <circle cx="{e1x:.1f}" cy="{e1y:.1f}" r="{r * 0.2:.1f}" class="ear"/>
  <circle cx="{e2x:.1f}" cy="{e2y:.1f}" r="{r * 0.2:.1f}" class="ear"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" class="head"/>
  <polygon points="{lx:.1f},{ly:.1f} {nx:.1f},{ny:.1f} {rx:.1f},{ry:.1f}" class="nose"/>
  <g class="cam">
    <rect x="{cx - 12:.1f}" y="{cam_y - 7:.1f}" width="24" height="15" rx="3"/>
    <circle cx="{cx}" cy="{cam_y - 0.5:.1f}" r="4.2" class="lens"/>
  </g>
</svg>"""


def page_html(t: dict | None = None) -> str:
    """The whole page. `t` is injectable so the guard can prove it follows."""
    t = t or SG.thresholds()
    shots = shot_list(t)
    spread = max(s["yaw"] for s in shots) - min(s["yaw"] for s in shots)

    cards = "\n".join(f"""      <li class="shot{' key' if s['file'] else ''}">
        <div class="dia">{head_svg(s['yaw'])}</div>
        <div class="body">
          <span class="num">Photo {s['n']}</span>
          <h3>{html.escape(s['name'])}</h3>
          <p>{html.escape(s['note'])}</p>
          {'<p class="file">Name this one <code>profile.jpg</code> — the software looks for it by name.</p>' if s['file'] else ''}
        </div>
      </li>""" for s in shots)

    floor_n = int(t["min_views"])
    rows = "\n".join(
        '      <tr{cls}><td class="n">{n}</td>'
        '<td class="n">{err:.1f}%</td>'
        '<td class="n">{pas:.0f}%</td></tr>'.format(
            cls=' class="floor"' if r["n"] == floor_n else "",
            n=r["n"],
            err=r["err"],
            pas=r["pass"] * 100,
        )
        for r in returns_table()
    )

    return f"""<title>How To Shoot Your Set</title>
<style>
  :root {{
    --bg:#fbfaf8; --panel:#ffffff; --ink:#1b1a19; --dim:#5f5b56; --line:#e3ded7;
    --accent:#8a5a2b; --key:#b4741f; --keybg:#fdf6ea; --warn:#8c2f1e; --warnbg:#fdf0ed;
    --head:#e8e2d9; --nose:#8a5a2b; --cam:#c9c2b8;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#141311; --panel:#1c1b19; --ink:#f0ece6; --dim:#a49d94; --line:#302d29;
      --accent:#d9a066; --key:#e0a75f; --keybg:#241c11; --warn:#e8907c; --warnbg:#2a1713;
      --head:#33302b; --nose:#d9a066; --cam:#4a4540;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#141311; --panel:#1c1b19; --ink:#f0ece6; --dim:#a49d94; --line:#302d29;
    --accent:#d9a066; --key:#e0a75f; --keybg:#241c11; --warn:#e8907c; --warnbg:#2a1713;
    --head:#33302b; --nose:#d9a066; --cam:#4a4540;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font:16px/1.62 ui-serif, Georgia, "Iowan Old Style", serif;
    -webkit-font-smoothing:antialiased;
  }}
  .wrap {{ max-width:760px; margin:0 auto; padding:56px 22px 90px; }}
  h1 {{
    font-size:clamp(30px,5.4vw,44px); line-height:1.14; margin:0 0 14px;
    letter-spacing:-.021em; font-weight:600;
  }}
  .lede {{ font-size:19px; color:var(--dim); margin:0 0 10px; }}
  .why {{
    margin:26px 0 0; padding:16px 18px; background:var(--panel);
    border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:3px;
  }}
  .why p {{ margin:0; font-size:15.5px; }}
  h2 {{
    font-size:13px; text-transform:uppercase; letter-spacing:.13em; font-weight:700;
    color:var(--accent); margin:52px 0 18px;
    font-family:ui-sans-serif,-apple-system,system-ui,sans-serif;
  }}
  ol.shots {{ list-style:none; padding:0; margin:0; display:grid; gap:14px; }}
  .shot {{
    display:flex; gap:18px; align-items:flex-start; background:var(--panel);
    border:1px solid var(--line); border-radius:5px; padding:16px 18px;
  }}
  .shot.key {{ border-color:var(--key); background:var(--keybg); }}
  .dia {{ flex:0 0 132px; }}
  .dia svg {{ width:132px; height:154px; display:block; }}
  .head {{ fill:var(--head); stroke:var(--line); stroke-width:1.5; }}
  .ear {{ fill:var(--head); stroke:var(--line); stroke-width:1.2; }}
  .nose {{ fill:var(--nose); }}
  .axis {{ stroke:var(--line); stroke-width:1.5; stroke-dasharray:3 4; }}
  .arc {{ fill:none; stroke:var(--accent); stroke-width:1.6; }}
  .deg {{
    fill:var(--accent); font:600 12px ui-sans-serif,system-ui,sans-serif;
    text-anchor:middle;
  }}
  .cam rect {{ fill:var(--cam); }}
  .cam .lens {{ fill:var(--bg); stroke:var(--cam); stroke-width:1.5; }}
  .num {{
    font-family:ui-sans-serif,system-ui,sans-serif; font-size:11px; font-weight:700;
    letter-spacing:.11em; text-transform:uppercase; color:var(--dim);
  }}
  .shot h3 {{ margin:3px 0 6px; font-size:20px; font-weight:600; letter-spacing:-.01em; }}
  .shot p {{ margin:0; font-size:15px; color:var(--dim); }}
  .shot .file {{ margin-top:8px; color:var(--key); font-size:14.5px; }}
  code {{
    font:13.5px/1.5 ui-monospace, "SF Mono", Menlo, monospace;
    background:var(--bg); border:1px solid var(--line); border-radius:3px; padding:1px 5px;
  }}
  ul.rules {{ margin:0; padding:0; list-style:none; display:grid; gap:11px; }}
  ul.rules li {{ padding-left:24px; position:relative; font-size:15.5px; }}
  ul.rules li::before {{
    content:"·"; position:absolute; left:8px; top:-2px;
    color:var(--accent); font-size:22px; font-weight:700;
  }}
  ul.rules b {{ font-weight:600; color:var(--ink); }}
  .warn {{
    margin-top:16px; padding:14px 18px; background:var(--warnbg);
    border:1px solid var(--warn); border-radius:4px; font-size:15px; color:var(--ink);
  }}
  .warn b {{ color:var(--warn); }}
  pre {{
    background:var(--panel); border:1px solid var(--line); border-radius:4px;
    padding:14px 16px; overflow-x:auto;
    font:13.5px/1.6 ui-monospace,"SF Mono",Menlo,monospace;
  }}
  .cols {{ display:grid; gap:14px; grid-template-columns:1fr 1fr; }}
  @media (max-width:620px) {{
    .cols {{ grid-template-columns:1fr; }}
    .shot {{ flex-direction:column; align-items:center; text-align:center; }}
  }}
  .col {{
    background:var(--panel); border:1px solid var(--line);
    border-radius:5px; padding:16px 18px;
  }}
  table.ret {{
    width:100%; border-collapse:collapse; background:var(--panel);
    border:1px solid var(--line); border-radius:5px; overflow:hidden;
    font-size:15px;
  }}
  table.ret th {{
    text-align:left; padding:10px 14px; font-size:11.5px; letter-spacing:.09em;
    text-transform:uppercase; color:var(--dim); font-weight:700;
    font-family:ui-sans-serif,system-ui,sans-serif; border-bottom:1px solid var(--line);
  }}
  table.ret td {{ padding:9px 14px; border-bottom:1px solid var(--line); }}
  table.ret tr:last-child td {{ border-bottom:none; }}
  table.ret tr.floor td {{ background:var(--keybg); color:var(--ink); font-weight:600; }}
  table.ret td.n {{ font-variant-numeric:tabular-nums; }}
  .col h4 {{ margin:0 0 9px; font-size:16px; font-weight:600; }}
  .col p {{ margin:0; font-size:14.5px; color:var(--dim); }}
  footer {{
    margin-top:56px; padding-top:20px; border-top:1px solid var(--line);
    font-size:13.5px; color:var(--dim);
  }}
</style>

<div class="wrap">
  <h1>The photographs this needs</h1>
  <p class="lede">
    Six to eight pictures, about three minutes, any room with even light.
    Take them on purpose — photographs you already have almost never work, and the
    reason is on this page.
  </p>

  <div class="why">
    <p>
      The software builds one 3D estimate of your face from several angles, which is
      what lets it ignore the room, the lens and the lighting. That only works if the
      pictures <b>disagree about the angle</b>. Ten near-identical selfies carry no more
      information than one. It is the turning that does the work, not the number of files.
    </p>
  </div>

  <h2>The four that must be in there</h2>
  <ol class="shots">
{cards}
  </ol>

  <div class="warn">
    <b>Photo 4 is the one that decides whether you get a report at all.</b>
    Of ten test subjects whose photographs came from a professional camera rig, six were
    turned away for having nothing in this range — it is by far the most common reason a
    set is refused. Turned far enough that one cheek is hidden, near enough that both eyes
    still show.
  </div>

  <h2>Then take a few more — here is exactly how much it buys</h2>
  <p style="font-size:15.5px;color:var(--dim);margin:0 0 14px">
    Four is the <b>floor</b>, not the target. Add two to four more at angles
    <b>in between</b> the ones above — halfway to each turn, a little higher, a little
    lower. Duplicates of an angle you already have add almost nothing; new angles
    keep paying. Measured over 400 trials against faces whose true shape was known:
  </p>
  <table class="ret">
    <thead><tr><th>Photographs</th><th>Typical shape error</th><th>Sets that came out usable</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <p style="font-size:15px;color:var(--dim);margin:14px 0 0">
    Two photographs are not nearly enough. Four is roughly where it becomes usable.
    Past eight the curve is flat, so there is no prize for thirty.
    <b>And turning matters more than counting:</b> across the same trials the spread
    of angles predicts the error better than the number of files does. A set
    covering {spread:.0f}° beats a bigger set that stays near the front.
  </p>

  <h2>While you shoot</h2>
  <ul class="rules">
    <li><b>Turn your head, not the camera.</b> Keep the phone still and rotate yourself.</li>
    <li><b>Relaxed, closed mouth.</b> No smile, no pout, no clenched jaw — mouth posture
        measurably moves one of the readings, so there is a gate on it.</li>
    <li><b>Eyes open, on the lens</b> in every shot, including the turned ones.</li>
    <li><b>Nothing across your face.</b> No glasses, no hat, no hair over the brow.</li>
    <li><b>Even, indirect light.</b> No hard shadow down one side, no strong colour cast —
        a dim green-lit room once made a fair-skinned subject read as dark-skinned, and
        the skin section had to refuse rather than report it.</li>
    <li><b>Fill the frame</b>, head and shoulders. Your eyes need to be at least
        {t['min_ipd_px']:.0f} pixels apart or the photo is dropped as too small.</li>
    <li><b>Send the originals.</b> Do not crop, filter, screenshot or re-save — a re-save
        can strip the orientation tag while leaving the picture identical, and the face is
        then measured sideways.</li>
  </ul>

  <h2>Check them before you put the set away</h2>
  <p style="font-size:15.5px;color:var(--dim);margin:0 0 12px">
    This reads the folder and writes nothing. It tells you which photo is weak and which
    way to turn, so you can re-shoot while you are still standing there.
  </p>
  <pre>./venv/bin/python run_pipeline.py --check /path/to/your/photos</pre>

  <h2>What comes back, and what does not</h2>
  <div class="cols">
    <div class="col">
      <h4>Holds across rooms and cameras</h4>
      <p>Proportions, symmetry, sex-typicality and the graded measurement catalogue.
         That is what building the 3D estimate is for.</p>
    </div>
    <div class="col">
      <h4>Describes the photograph, not you</h4>
      <p>Skin, colour and texture are read from a single picture and do not transfer.
         They are reported as properties of that photograph.</p>
    </div>
    <div class="col">
      <h4>Compared with people like you</h4>
      <p>{t['n_cohorts']} groups by race, sex and age band, covering ages
         {t['cohort_age_lo']}–{t['cohort_age_hi']}. Outside that range it compares you with
         nobody rather than with the wrong population.</p>
    </div>
    <div class="col">
      <h4>No score, ever</h4>
      <p>There is no rating and no single number, by design and in three separate places
         in the code. Anything it cannot measure well enough, it refuses and says why.</p>
    </div>
  </div>

  <footer>
    Every number on this page — {t['min_views']:.0f} photographs,
    {t['min_spread_deg']:.0f}° of spread, the
    {t['profile_yaw_lo']:.0f}–{t['profile_yaw_hi']:.0f}° turn,
    {t['min_ipd_px']:.0f} pixels between the eyes — is read from the code that enforces it
    when this page is generated, so it cannot drift from what the software actually checks.
    The set drawn above spans {spread:.0f}°, against a floor of {t['min_spread_deg']:.0f}°.
    Nothing is uploaded and nothing is collected; the page never sees a photograph.
  </footer>
</div>"""


def main() -> int:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(page_html())
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
