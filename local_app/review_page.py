"""Private partial profile editor; only derived geometry enters saved reports."""

from __future__ import annotations

import base64
import html

from .profile_review import ANCHORS, render_profile
from .profile_v2 import METHOD, ROWS, measure


def render_review(session, entries, selected, image_bytes, record, *, draft=None):
    from .render import _document

    esc = html.escape
    base = f"/sessions/{session.session_id}"
    value = draft or record
    points = value["points"]
    facing = value["facing"]
    level = value["level_degrees"]
    options = "".join(
        f'<option value="{esc(e.photo_id, quote=True)}"{" selected" if e.photo_id == selected.photo_id else ""}>Photo {i + 1} · {e.width} × {e.height}</option>'
        for i, e in enumerate(entries)
    )
    controls = ""
    for group, keys in (
        ("Nose", ("nasion", "tip", "subnasale")),
        ("Lips", ("upper_lip", "lower_lip")),
        ("Chin", ("pogonion", "menton")),
    ):
        controls += f'<section class="profile-anchor-group"><h2>{group}</h2>'
        for key in keys:
            pair = points[key]
            missing = pair is None
            controls += f'<fieldset class="landmark-row"><legend>{esc(ANCHORS[key])}</legend><label><input type="radio" name="active_anchor" value="{key}"{" checked" if key == "nasion" else ""}> Place this point</label><label><input type="checkbox" name="missing_{key}" value="yes" data-missing="{key}"{" checked" if missing else ""}> Not visible / unavailable</label>'
            for i, axis in enumerate(("x", "y")):
                coord = "" if missing else format(pair[i] * 100, ".6f")
                controls += f'<label>{axis.upper()} %<input type="number" min="0" max="100" step="any" name="{key}_{axis}" value="{coord}" data-anchor="{key}" data-coordinate="{axis}"></label>'
            controls += "</fieldset>"
        controls += "</section>"
    hidden = "".join(
        f'<input type="hidden" name="{key}" value="{esc(str(v), quote=True)}">'
        for key, v in dict(
            csrf_token=session.csrf_token,
            photo=selected.photo_id,
            method=METHOD,
            revision=record["revision"],
            binding=record["binding"],
        ).items()
    )
    clear_hidden = hidden
    metrics = '<option value="all">All measured lines</option>' + "".join(
        f'<option value="{r["id"]}">{esc(r["label"])}</option>' for r in ROWS
    )
    facing_options = '<option value="">Choose direction</option>' + "".join(
        f'<option value="{v}"{" selected" if facing == v else ""}>Nose points {v} in this image</option>'
        for v in ("left", "right")
    )
    preview = ""
    if facing:
        payload = measure(points, selected.width, selected.height, facing=facing)
        payload["revision"] = record["revision"]
        preview = render_profile(payload)
    warning = (
        '<p class="draft-note">Draft preview — the saved report has not changed.</p>'
        if draft
        else ""
    )
    if record.get("migrated"):
        warning += "<p>Legacy v1 point positions loaded. Confirm the view and review them to save a separate v2 revision; no new points were fabricated.</p>"
    body = f'''<main class="review-shell"><p class="local-mark">Private session · manual profile v2</p><h1>Review your profile landmarks</h1>
<p>Choose a true lateral photo with a level head, relaxed lips and a visible nose/chin contour. Facing describes the nose direction on this image, not anatomical left/right. A frontal image cannot establish profile readings.</p><p><a href="{base}/results">Back to report</a></p>
<form method="get" action="{base}/profile" class="photo-selector"><label>Photograph<select name="photo">{options}</select></label><button>Load photograph</button></form>{warning}
<form method="post" action="{base}/profile" id="profile-review-form" data-draft="{"true" if draft else "false"}">{hidden}
<div class="frontal-toolbar"><label>Facing<select name="facing" required>{facing_options}</select></label><label>Zoom<select id="profile-zoom"><option value="1">100%</option><option value="1.5">150%</option><option value="2">200%</option><option value="3">300%</option></select></label><label>Visual leveling, degrees<input type="number" name="level_degrees" id="profile-level" min="-45" max="45" step="0.5" value="{level}"></label><label>Measurement lines<select name="metric" id="profile-metric">{metrics}</select></label><button type="button" id="profile-undo">Undo point edit</button><button type="button" id="profile-reset">Mark all unavailable</button></div>
<p>The leveling aid rotates only the display. Coordinates map back to the original normalized image; line-to-line readings do not need a vertical reference and gain no 3D accuracy from leveling.</p>
<div class="review-grid"><div><div class="profile-viewport"><img id="profile-source" src="data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}" alt="Selected photograph for manual profile review"><canvas id="profile-canvas" data-width="{selected.width}" data-height="{selected.height}" tabindex="0" aria-label="Profile landmarks. Select a point, then click, drag or use arrow keys; numeric fields are also available."></canvas></div><p id="profile-status" role="status">Select a point, then place it. Missing points exclude only dependent readings. Use arrows for one image pixel, Shift and arrows for ten.</p></div><div>{controls}</div></div>
<label class="review-confirm"><input type="checkbox" name="lateral_confirmed" value="yes" required> This is a lateral photograph with a visible profile contour.</label><label class="review-confirm"><input type="checkbox" name="confirmed" value="yes" required> I reviewed the visible points and marked obscured points unavailable.</label><label class="review-confirm"><input type="checkbox" name="acknowledged" value="yes"> I checked and accept any displayed facing, point-order or detail warnings.</label><button name="action" value="preview" formnovalidate>Preview readings</button><button name="action" value="save">Save profile measurements</button></form>
{preview}<form method="post" action="{base}/profile-clear">{clear_hidden}<button>Remove this photo’s profile review</button></form><p>Each photo has its own revision. Opposite-facing views are not averaged or used as anatomical asymmetry evidence. These readings do not change the frontal score. Ending the session deletes them.</p></main><script src="/static/review.js" defer></script>'''
    return _document(
        title="Review profile landmarks", body=body, page_class="review-page"
    )
