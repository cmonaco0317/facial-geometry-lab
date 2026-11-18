"""Server-rendered placement controls for illustrative styling and geometry."""

from __future__ import annotations

import html

from .landmark_review import ANCHORS, forehead_points
from .photo_experiments import render_experiment


def _geometry_controls(
    base: str, photo: str, baseline: dict, selected: str | None, saved: dict | None
) -> tuple[str, bool]:
    esc = html.escape
    points = forehead_points(
        baseline["points"],
        baseline.get("forehead_visibility", "unassessed"),
        baseline.get("automatic_forehead_status"),
    )
    available = [key for key, p in points.items() if p is not None]
    if not available:
        return (
            "<p>No current source points are available for a geometry experiment. Choose a clearer photo or inspect the source landmarks first.</p>",
            False,
        )
    anchor = selected or (
        saved["parameters"]["anchor"]
        if saved and saved["kind"] == "geometry"
        else "eye_inner_a" if "eye_inner_a" in available else available[0]
    )
    if anchor not in available:
        raise ValueError("That source point is unavailable. Choose an available input.")
    params = (
        saved["parameters"]
        if saved
        and saved["kind"] == "geometry"
        and saved["parameters"]["anchor"] == anchor
        else dict(x=points[anchor][0], y=points[anchor][1], radius=0.2)
    )
    options = "".join(
        f'<option value="{key}" {"selected" if key==anchor else ""}>{esc(ANCHORS[key][0])}</option>'
        for key in available
    )
    return (
        f'<form method="get" action="{base}/experiments"><input type="hidden" name="photo" value="{photo}"><input type="hidden" name="mode" value="geometry"><label>Source input<select name="anchor">{options}</select></label><button>Choose input</button></form><p>Current source input: {esc(ANCHORS[anchor][0])}. Click a nearby target in the original photo above, or enter target coordinates below. Gold marks the original input and green the simulated input. One bounded local warp is saved at a time. No anatomy is generated for missing inputs.</p><input type="hidden" name="kind" value="geometry" form="experiment-save"><input type="hidden" name="anchor" value="{anchor}" form="experiment-save"><label>Target horizontal position (fraction of image width)<input id="geometry-target-x" name="x" type="number" min="0" max="1" step="any" value="{params["x"]}" form="experiment-save"></label><label>Target vertical position (fraction of image height)<input id="geometry-target-y" name="y" type="number" min="0" max="1" step="any" value="{params["y"]}" form="experiment-save"></label><label>Influence radius (fraction of shorter image dimension)<input name="radius" type="number" min=".08" max=".4" step=".01" value="{params["radius"]}" form="experiment-save"></label><p id="geometry-target-status" role="status"></p><script src="/static/experiments.js" defer></script>',
        True,
    )


def render_page(
    session,
    entries,
    entry,
    number: int,
    public: dict,
    basis: str,
    *,
    mode: str = "styling",
    baseline: dict | None = None,
    anchor: str | None = None,
) -> str:
    from .render import _document

    esc = html.escape
    base = f"/sessions/{session.session_id}"
    options = "".join(
        f'<option value="{e.photo_id}" {"selected" if e.photo_id==entry.photo_id else ""}>Photo {i}</option>'
        for i, e in enumerate(entries, 1)
    )
    modes = "".join(
        f'<option value="{key}" {"selected" if mode==key else ""}>{label}</option>'
        for key, label in (
            ("styling", "Hair or eyewear overlay"),
            ("geometry", "Geometry point experiment"),
        )
    )
    hidden = f'<input type="hidden" name="csrf_token" value="{esc(session.csrf_token,quote=True)}"><input type="hidden" name="revision" value="{public["revision"]}"><input type="hidden" name="binding" value="{basis}"><input type="hidden" name="photo" value="{entry.photo_id}">'
    saved = (
        public["entry"]
        if public["entry"]
        and public["entry"]["photo_number"] == number
        and public["binding_status"] == "current"
        else None
    )
    body = f'<main class="results-shell"><a href="{base}/results">Back to report</a><h1>Reversible photo experiments</h1><p>Try a geometric styling aid or a small local point warp. These are illustrations, not realistic predictions or fit prescriptions. Measured values and source photo bytes stay unchanged. The saved illustration and its original display are included in report exports; Undo removes the current illustration.</p><form method="get" action="{base}/experiments"><label>Source photo<select name="photo">{options}</select></label><label>Experiment type<select name="mode">{modes}</select></label><button>Choose photo and experiment</button></form><figure><img id="experiment-original" width="420" class="experiment-image" src="{base}/experiment-original?photo={entry.photo_id}" alt="Original selected photo"><figcaption>Photo {number}, original normalized stored bytes.</figcaption></figure>'
    if mode == "geometry":
        geometry_controls, can_save = _geometry_controls(
            base, entry.photo_id, baseline, anchor, saved
        )
        body += geometry_controls
        controls = ""
    else:
        can_save = True
        params = (
            saved["parameters"]
            if saved and saved["kind"] in {"hair", "eyewear"}
            else dict(x=0.5, y=0.3, scale=0.4)
        )
        style = saved["kind"] if saved else "eyewear"
        kinds = "".join(
            f'<option value="{kind}" {"selected" if style==kind else ""}>{label}</option>'
            for kind, label in (
                ("eyewear", "Eyewear outline"),
                ("hair", "Hair framing shape"),
            )
        )
        controls = (
            f'<label>Styling experiment<select name="kind">{kinds}</select></label>'
        )
        controls += "".join(
            f'<label>{label}<input type="range" name="{key}" min="{low}" max="{high}" step="0.001" value="{params[key]}"></label>'
            for key, label, low, high in (
                ("x", "Horizontal position", 0, 1),
                ("y", "Vertical position", 0, 1),
                ("scale", "Illustration width", 0.1, 0.8),
            )
        )
    body += f'<form id="experiment-save" class="preferences-form" method="post" action="{base}/experiments">{hidden}{controls}<button name="action" value="save" {"" if can_save else "disabled"}>Save illustration</button><button name="action" value="reset" formnovalidate>Undo and restore original</button></form>'
    if public["entry"]:
        body += f'<p><a href="{base}/experiment.svg">Download labeled simulation SVG</a></p>'
    body += render_experiment(public) + "</main>"
    return _document(title="Photo experiments", body=body, page_class="results-page")
