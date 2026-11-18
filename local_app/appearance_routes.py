"""Owned-session appearance inspection with source-bound, stale-safe corrections."""

from __future__ import annotations

import html
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .appearance import (
    COVERINGS,
    FAMILIES,
    LIGHTING,
    LANDMARKS,
    NOTES,
    ROLES,
    VISIBILITY,
    appearance_payload,
    render_appearance,
    review_appearance,
)


def _sources(handler: Any, session: Any) -> tuple:
    report = handler.server.analyze_session(session.session_id)
    current = handler.server.store.load(session.session_id)
    return report.axes["harmony"].details.get("appearance_sources", []), current


def _select(name: str, values: tuple, selected: str, label: str) -> str:
    esc = html.escape
    options = "".join(
        f'<option value="{v}"{" selected" if v == selected else ""}>{esc(v.replace("_", " "))}</option>'
        for v in values
    )
    return f'<label>{esc(label)}<select name="{name}">{options}</select></label>'


def get_appearance(handler: Any, session: Any) -> None:
    from .render import _document

    try:
        query = parse_qs(
            urlsplit(handler.path).query, keep_blank_values=True, max_num_fields=2
        )
        if set(query) - {"photo", "family"} or any(len(v) != 1 for v in query.values()):
            raise ValueError("Choose one photograph and appearance region.")
        raw, current = _sources(handler, session)
        if not raw:
            handler._error(
                409,
                "No appearance source is available. Add a photograph and retry analysis.",
            )
            return
        number = int(query.get("photo", [str(raw[0]["photo_number"])])[0])
        family = query.get("family", ["smile"])[0]
        photo = next((p for p in raw if p["photo_number"] == number), None)
        if photo is None or family not in FAMILIES:
            raise ValueError("Choose a current photograph and appearance region.")
        review = current.state.get("appearance_review", {})
        row = review.get("overrides", {}).get(f'{photo["photo_id"]}:{family}', {})
        if row.get("source_sha256") != photo["source_sha256"]:
            row = {}
        box = row.get("box") or photo["regions"][family] or [0.1, 0.1, 0.9, 0.9]
        photo_settings = review.get("photo_settings", {}).get(photo["photo_id"], {})
        if photo_settings.get("source_sha256") != photo["source_sha256"]:
            photo_settings = {}
        endpoints = row.get("points") or photo["points"].get(family, {})
        esc = html.escape
        base = f"/sessions/{session.session_id}"
        options = "".join(
            f'<option value="{p["photo_number"]}"{" selected" if p["photo_number"] == number else ""}>Photo {p["photo_number"]}</option>'
            for p in raw
        )
        regions = "".join(
            f'<option value="{key}"{" selected" if key == family else ""}>{esc(title)}</option>'
            for key, title in FAMILIES.items()
        )
        controls = _select(
            "role",
            ROLES,
            photo_settings.get("role", photo["role"]),
            "Expression/view for every region in this photo",
        )
        controls += _select(
            "visibility",
            VISIBILITY,
            row.get("visibility", "unassessed"),
            "Can you see this region in the crop?",
        )
        controls += _select(
            "lighting",
            LIGHTING,
            row.get("lighting", "unknown"),
            "Lighting in this region",
        )
        for name in ("glasses", "makeup", "facial_hair"):
            controls += _select(
                name,
                COVERINGS,
                row.get(name, "unknown"),
                name.replace("_", " ").capitalize(),
            )
        coords = "".join(
            f'<label>{label}<input type="number" name="{key}" value="{value*100:.3f}" min="0" max="100" step="any" required></label>'
            for key, label, value in zip(
                ("x0", "y0", "x1", "y1"),
                ("Left %", "Top %", "Right %", "Bottom %"),
                box,
            )
        )
        x0, y0, x1, y1 = box
        point_fields = ""
        marks = ""
        for key, pair in endpoints.items():
            for axis, index in (("x", 0), ("y", 1)):
                value = format(pair[index] * 100, ".12g") if pair is not None else ""
                point_fields += f'<label>p{key} {axis} %<input name="p{key}_{axis}" type="number" value="{value}" min="0" max="100" step="any"></label>'
            if pair is not None:
                marks += f'<circle cx="{pair[0]*100}" cy="{pair[1]*100}" r=".55" fill="#ffc876"/><text x="{pair[0]*100+1}" y="{pair[1]*100}" font-size="2.2" fill="#ffc876">p{key}</text>'
        if point_fields:
            point_fields = (
                "<fieldset><legend>Appearance ratio endpoints</legend><p>Move endpoints using photo coordinates, or leave both coordinates blank when hidden. These edits affect appearance ratios only.</p>"
                + point_fields
                + "</fieldset>"
            )
        diagram = f'<svg class="appearance-photo" viewBox="0 0 100 100" width="480" role="img" aria-label="Selected photo and proposed region rectangle"><image href="data:image/jpeg;base64,{photo["jpeg_b64"]}" width="100" height="100" preserveAspectRatio="none"/><rect x="{x0*100}" y="{y0*100}" width="{(x1-x0)*100}" height="{(y1-y0)*100}" fill="none" stroke="#34f29f" stroke-width=".6"/>{marks}</svg>'
        # Preserve the original aspect ratio on the SVG itself, including tall photos.
        diagram = diagram.replace(
            'viewBox="0 0 100 100" width="480"',
            f'viewBox="0 0 100 100" width="480" height="{max(1,round(480*photo["height"]/photo["width"]))}" preserveAspectRatio="none"',
        )
        body = f'<main class="results-shell"><a href="{base}/results">Back to report</a><h1>Inspect appearance evidence</h1><p>These are image regions and mesh proposals. Inspect each crop; correct its rectangle or mark it obscured/not visible. Hair, ears and neck start without automatic crops. A note records your observation, not professional review.</p>'
        body += f'<form method="get" action="{base}/appearance"><label>Photo<select name="photo">{options}</select></label><label>Region<select name="family">{regions}</select></label><button>Show region</button></form><h2>{esc(FAMILIES[family])}</h2><p>{esc(NOTES[family])}</p>{diagram}'
        body += f'<form method="post" action="{base}/appearance"><input type="hidden" name="csrf_token" value="{esc(session.csrf_token, quote=True)}"><input type="hidden" name="photo_number" value="{number}"><input type="hidden" name="family" value="{family}"><input type="hidden" name="revision" value="{review.get("revision",0)}">{coords}{controls}{point_fields}<label>Your appearance note (optional)<textarea name="note" maxlength="240">{esc(row.get("note", ""))}</textarea></label><button name="action" value="save">Save correction</button><button name="action" value="reset">Reset crop and endpoints (keep photo role)</button></form>'
        body += (
            '<p>Editing a crop changes its pixel description. The endpoints above change appearance ratios. The independent frontal editor remains available: <a href="'
            + base
            + '/frontal">review frontal geometry</a>. The original photo is preserved. Exported HTML/JSON now include the visible crops; share them only as you intend.</p>'
        )
        body += render_appearance(appearance_payload(raw, review)) + "</main>"
        page = _document(
            title="Inspect appearance evidence", body=body, page_class="results-page"
        )
        handler._send(200, page.encode(), "text/html; charset=utf-8")
    except (ValueError, LookupError) as error:
        handler._error(422, str(error))
    except (OSError, RuntimeError):
        handler._error(
            500,
            "Appearance evidence could not be read. Your session is still available.",
        )


def update_appearance(handler: Any, session: Any, content_length: int) -> None:
    if content_length > 16_000:
        handler._error(413, "This form is too large.")
        return
    fields = handler._simple_form(content_length)
    if fields is None or not handler._valid_csrf(
        session, handler._one(fields, "csrf_token")
    ):
        return
    allowed = {
        "csrf_token",
        "photo_number",
        "family",
        "revision",
        "action",
        "role",
        "visibility",
        "lighting",
        "glasses",
        "makeup",
        "facial_hair",
        "note",
        "x0",
        "y0",
        "x1",
        "y1",
    }
    try:
        family = handler._one(fields, "family")
        point_keys = {
            f"p{i}_{axis}" for i in LANDMARKS.get(family, ()) for axis in ("x", "y")
        }
        if set(fields) != allowed | point_keys or any(
            len(v) != 1 for v in fields.values()
        ):
            raise ValueError("Unsupported, missing or repeated appearance fields.")
        raw, current = _sources(handler, session)
        data = {k: v[0] for k, v in fields.items()}
        number = int(data.pop("photo_number"))
        family = data.pop("family")
        revision = int(data.pop("revision"))
        data.pop("csrf_token")
        box = [float(data.pop(k)) / 100 for k in ("x0", "y0", "x1", "y1")]
        points = {}
        for i in LANDMARKS.get(family, ()):
            x, y = data.pop(f"p{i}_x"), data.pop(f"p{i}_y")
            if bool(x) != bool(y):
                raise ValueError(
                    "Enter both endpoint coordinates, or leave both blank."
                )
            points[str(i)] = [float(x) / 100, float(y) / 100] if x else None
        current.state["appearance_review"] = review_appearance(
            raw,
            current.state.get("appearance_review", {}),
            photo_number=number,
            family=family,
            revision=revision,
            box=box,
            points=points if point_keys else None,
            **data,
        )
        handler.server.store.save(current)
        handler._redirect(
            f"/sessions/{session.session_id}/appearance?photo={number}&family={family}"
        )
    except ValueError as error:
        handler._error(409 if "stale" in str(error) else 422, str(error))
    except (OSError, RuntimeError):
        handler._error(
            500,
            "The appearance correction could not be saved. Your session is still available.",
        )
