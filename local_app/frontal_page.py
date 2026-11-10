"""Accessible photo-bound frontal review; numerical forms work without JavaScript."""

from __future__ import annotations

import base64
import html
from typing import Any

from .landmark_review import (
    ANCHORS, FOREHEAD_VISIBILITY, calculate, render_reviewed, geometry_warnings,
    forehead_points,
)
from .measurement_registry import REGISTRY_VERSION, measurement_contract


def render_frontal(
    session: Any,
    entries: tuple,
    selected: Any,
    image_bytes: bytes,
    record: dict,
    *,
    draft: dict | None = None,
    draft_visibility: str | None = None,
) -> str:
    from .render import _document

    esc = html.escape
    base = f"/sessions/{session.session_id}"
    visibility = draft_visibility if draft_visibility is not None else record.get("forehead_visibility", "unassessed")
    automatic_status = record.get("automatic_forehead_status")
    points = forehead_points(draft if draft is not None else record["points"], visibility,
                             automatic_status)
    from .automatic_forehead import EXPLANATIONS

    automatic_note = EXPLANATIONS.get(automatic_status, "Run analysis for an automatic hair and forehead check.")
    visibility_options = "".join(
        f'<option value="{key}"{" selected" if key == visibility else ""}>{esc(label)}</option>'
        for key, label in FOREHEAD_VISIBILITY.items()
    )
    options = "".join(
        f'<option value="{esc(e.photo_id, quote=True)}"{" selected" if e.photo_id == selected.photo_id else ""}>Photo {i + 1} · {e.width} × {e.height}</option>'
        for i, e in enumerate(entries)
    )
    groups = {}
    for i, (key, (label, _index, region)) in enumerate(ANCHORS.items(), 1):
        pair = points[key]
        original = record["automatic"][key]
        controls = f'<fieldset class="landmark-row" data-region="{region}"><legend>{i}. {esc(label)}</legend><label><input type="radio" name="active_anchor" value="{key}"{" checked" if i == 1 else ""}> Place point</label>'
        for j, axis in enumerate(("x", "y")):
            value = "" if pair is None else format(pair[j] * 100, ".8f")
            auto = "" if original is None else format(original[j] * 100, ".8f")
            controls += f'<label>{axis.upper()} %<input type="number" name="{key}_{axis}" value="{value}" min="0" max="100" step="any" data-anchor="{key}" data-coordinate="{axis}" data-original="{auto}"></label>'
        controls += f'<label><input type="checkbox" name="missing_{key}" value="yes" data-missing="{key}"{" checked" if pair is None else ""}> Unavailable / not visible</label><button type="button" data-reset="{key}">Reset point</button></fieldset>'
        groups.setdefault(region, []).append(controls)
    controls = "".join(
        f'<section class="anchor-region" data-region="{region}"><h2>{esc(region.title())}</h2>{"".join(items)}</section>'
        for region, items in groups.items()
    )
    warnings = geometry_warnings(points)
    warning_text = "".join(f"<li>{esc(w)}</li>" for w in warnings)
    rows = calculate(points, selected.width, selected.height)
    payload = dict(
        method=REGISTRY_VERSION,
        contract=measurement_contract(),
        revision=record["revision"],
        forehead_visibility=visibility,
        **({"automatic_forehead_status": automatic_status} if automatic_status is not None else {}),
        measurements=rows,
        diagram=dict(points=points, width=selected.width, height=selected.height),
    )
    original_rows = {
        r["id"]: r
        for r in calculate(record["automatic"], selected.width, selected.height)
    }
    changes = "".join(
        f"<tr><th>{esc(r['label'])}</th><td>{original_rows[r['id']]['value'] if r['id'] in original_rows else 'Unavailable'}</td><td>{r['value']} {esc(r['unit'])}</td></tr>"
        for r in rows
    )
    data = base64.b64encode(image_bytes).decode("ascii")
    selected_index = next(
        i for i, e in enumerate(entries, 1) if e.photo_id == selected.photo_id
    )
    body = f"""<main class="review-shell"><p class="local-mark">Private session · frontal review</p><h1>Review frontal landmarks and hairline</h1>
<p>Use a frontal photograph. Check points against visible outlines, not an imagined ideal. A/B name the two labelled sides on the original photograph; visual mirroring does not change stored coordinates. Mark hidden points unavailable. The mesh forehead top and actual hairline are separate.</p>
<p><a href="{base}/results">Back to report</a> · Reviewing photo {selected_index}. Measurements below use this single photograph.</p>
<form method="get" action="{base}/frontal"><label>Photograph<select name="photo">{options}</select></label><button>Load photograph</button></form>
<form method="post" action="{base}/frontal" id="frontal-review-form"><input type="hidden" name="csrf_token" value="{esc(session.csrf_token, quote=True)}"><input type="hidden" name="photo" value="{esc(selected.photo_id, quote=True)}"><input type="hidden" name="revision" value="{record["revision"]}"><input type="hidden" name="binding" value="{record["binding"]}">
<p><strong>Automatic finding:</strong> {esc(automatic_note)}</p><label>Forehead and hairline visibility in this photo<select name="forehead_visibility" id="forehead-visibility" data-automatic-status="{esc(automatic_status or '', quote=True)}">{visibility_options}</select></label><p>The automatic assessment is used by default. You can correct it here if needed. Hair covering the forehead is not the hairline. Hidden points are excluded; add a clear frontal photo with hair lifted away for true hairline proportions, then explicitly confirm a visible central hairline before placing that point.</p>
<div class="frontal-toolbar"><label>Region<select id="frontal-region"><option value="all">All regions</option><option value="forehead">Forehead and hairline</option><option value="eyes">Eyes and brows</option><option value="nose">Nose</option><option value="lips">Lips</option><option value="jaw">Cheeks, jaw and chin</option></select></label><label>Zoom<select id="frontal-zoom"><option value="1">100%</option><option value="1.5">150%</option><option value="2">200%</option><option value="3">300%</option></select></label><button type="button" id="frontal-mirror" aria-pressed="false">Mirror view</button><button type="button" id="frontal-original" aria-pressed="false">Show automatic points</button><button type="button" id="frontal-undo">Undo</button><button type="button" id="frontal-reset">Reset all to automatic</button></div>
<div class="review-grid"><div class="frontal-image-panel"><div class="frontal-viewport"><img id="frontal-source" src="data:image/jpeg;base64,{data}" alt="Selected photograph for frontal review"><canvas id="frontal-canvas" tabindex="0" aria-label="Place selected landmark; arrow keys adjust it. Numeric inputs provide an alternative."></canvas></div><p id="frontal-status" role="status">Choose a point, then click to place it. Scroll the zoomed view to pan. Coordinates are percentages of the original image.</p></div><div>{controls}</div></div>
<ul role="status">{warning_text}</ul><label class="review-confirm"><input type="checkbox" name="acknowledged" value="yes"> If placement warnings appear, I checked them and intentionally accept these positions.</label><label class="review-confirm"><input type="checkbox" name="confirmed" value="yes" required> I reviewed available points against the photograph.</label><button type="submit" name="action" value="preview" formnovalidate>Calculate draft readings</button><button type="submit" name="action" value="save">Save reviewed measurements</button></form>
<form method="post" action="{base}/frontal-clear"><input type="hidden" name="csrf_token" value="{esc(session.csrf_token, quote=True)}"><input type="hidden" name="photo" value="{esc(selected.photo_id, quote=True)}"><input type="hidden" name="revision" value="{record["revision"]}"><input type="hidden" name="binding" value="{record["binding"]}"><button>Remove saved frontal review for this photo</button></form>
<p>{"Draft preview — not saved. Check the readings, confirm the points and save." if draft is not None else "Saved revision " + str(record["revision"]) if record["reviewed"] else "Automatic starting points — no frontal review saved yet."}</p>
<div class="table-wrap"><table><thead><tr><th>Measurement</th><th>Automatic</th><th>Current points</th></tr></thead><tbody>{changes}</tbody></table></div>{render_reviewed(payload)}</main><script src="/static/frontal.js" defer></script>"""
    return _document(
        title="Review frontal landmarks", body=body, page_class="review-page"
    )
