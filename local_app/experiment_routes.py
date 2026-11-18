"""Private photo illustration forms, downloads and exact original-byte recovery."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from .photos import PhotoCatalog
from .photo_experiments import (
    public_experiment,
    report_basis,
    styling_experiment,
    update_experiment,
)
from .report_question_routes import _context
from .landmark_review import detected_points, observation_binding, validate_points
from .frontal_routes import _instrument, _automatic_status

STATE_KEY = "photo_experiment"


def _baseline(handler: Any, session: Any, entry: Any, source: bytes) -> dict:
    """Prepare input points without creating or changing a measured annotation."""
    binding = observation_binding(
        entry, _instrument() + hashlib.sha256(source).hexdigest()
    )
    current = session.state.get("frontal_reviews", {}).get(entry.photo_id)
    if current and current.get("binding") == binding:
        result = deepcopy(current)
    else:
        path = handler.server.store.path_for(
            session.session_id, "uploads", entry.stored_name
        )
        points = validate_points(detected_points(path))
        if path.read_bytes() != source:
            raise ValueError("The source photo changed; reload the experiment.")
        result = dict(
            binding=binding,
            revision=1,
            points=points,
            reviewed=False,
            forehead_visibility="unassessed",
        )
    result["automatic_forehead_status"] = _automatic_status(session, entry.photo_id)
    return result


def _selection(handler: Any, session: Any, photo: str | None) -> tuple:
    entries = PhotoCatalog(handler.server.store, session.session_id).entries()
    selected = (
        next((e for e in entries if e.photo_id == photo), None)
        if photo is not None
        else next(iter(entries), None)
    )
    if selected is None:
        raise LookupError("Photograph not found in this session.")
    number = next(
        i for i, e in enumerate(entries, 1) if e.photo_id == selected.photo_id
    )
    source = handler.server.store.path_for(
        session.session_id, "uploads", selected.stored_name
    ).read_bytes()
    return entries, selected, number, source


def get_experiments(handler: Any, session: Any, resource: str) -> None:
    try:
        if resource == "experiment.svg":
            state = public_experiment(
                handler.server.store.load(session.session_id).state.get(STATE_KEY)
            )
            if state["entry"] is None:
                handler._error(404, "No illustration was saved.")
                return
            handler._send(
                200,
                state["entry"]["simulation_svg"].encode(),
                "image/svg+xml; charset=utf-8",
                extra_headers=[
                    (
                        "Content-Disposition",
                        'attachment; filename="facial-geometry-illustrative-simulation.svg"',
                    )
                ],
            )
            return
        query = parse_qs(
            urlsplit(handler.path).query, keep_blank_values=True, max_num_fields=3
        )
        if set(query) - {"photo", "mode", "anchor"} or any(
            len(v) != 1 for v in query.values()
        ):
            raise ValueError("Choose one source photograph.")
        entries, entry, number, source = _selection(
            handler, session, query.get("photo", [None])[0]
        )
        if resource == "experiment-original":
            if set(query) - {"photo"}:
                raise ValueError("Choose one source photograph.")
            handler._send(
                200,
                source,
                "image/jpeg",
                extra_headers=[
                    (
                        "Content-Disposition",
                        'inline; filename="original-normalized-photo.jpg"',
                    )
                ],
            )
            return
        payload, current = _context(handler, session, include_experiment=False)
        basis = report_basis(payload)
        state = current.state.get(STATE_KEY, {})
        public = public_experiment(state, basis)
        mode = query.get("mode", ["styling"])[0]
        if mode not in {"styling", "geometry"}:
            raise ValueError("Choose styling or geometry.")
        baseline = (
            _baseline(handler, current, entry, source) if mode == "geometry" else None
        )
        from .experiment_page import render_page

        page = render_page(
            current,
            entries,
            entry,
            number,
            public,
            basis,
            mode=mode,
            baseline=baseline,
            anchor=query.get("anchor", [None])[0],
        )
        handler._send(200, page.encode(), "text/html; charset=utf-8")
    except LookupError as error:
        handler._error(404, str(error))
    except ValueError as error:
        handler._error(422, str(error))
    except (OSError, RuntimeError):
        handler._error(
            500,
            "The illustration could not be opened. Your session is still available.",
        )


def save_experiment(handler: Any, session: Any, content_length: int) -> None:
    if content_length > 12000:
        handler._error(413, "This experiment form is too large.")
        return
    fields = handler._simple_form(content_length)
    if fields is None or not handler._valid_csrf(
        session, handler._one(fields, "csrf_token")
    ):
        return
    try:
        required = {"csrf_token", "revision", "action"}
        allowed = required | {
            "binding",
            "photo",
            "kind",
            "x",
            "y",
            "scale",
            "anchor",
            "radius",
        }
        if (
            not required <= set(fields)
            or set(fields) - allowed
            or any(len(v) != 1 for v in fields.values())
        ):
            raise ValueError("Unsupported, missing or repeated experiment fields.")
        values = {k: v[0] for k, v in fields.items()}
        current = handler.server.store.load(session.session_id)
        state = current.state.get(STATE_KEY, {})
        expected = int(values["revision"])
        update_experiment(state, expected=expected, entry=None)
        if values["action"] == "reset":
            candidate = None
        elif values["action"] == "save":
            if not {"binding", "photo", "kind", "x", "y"} <= set(values):
                raise ValueError("Choose the source and illustration parameters.")
            payload, current = _context(handler, session, include_experiment=False)
            basis = report_basis(payload)
            if basis != values["binding"]:
                raise ValueError(
                    "This form describes an earlier report; reload it first."
                )
            _, entry, number, source = _selection(handler, current, values["photo"])
            if values["kind"] == "geometry":
                if not {"anchor", "radius"} <= set(values):
                    raise ValueError("Choose the source input and influence radius.")
                from .geometry_experiments import geometry_experiment

                candidate = geometry_experiment(
                    source,
                    _baseline(handler, current, entry, source),
                    photo_number=number,
                    basis=basis,
                    anchor=values["anchor"],
                    x=float(values["x"]),
                    y=float(values["y"]),
                    radius=float(values["radius"]),
                )
            else:
                if "scale" not in values:
                    raise ValueError("Choose an illustration width.")
                candidate = styling_experiment(
                    source,
                    photo_number=number,
                    basis=basis,
                    style=values["kind"],
                    x=float(values["x"]),
                    y=float(values["y"]),
                    scale=float(values["scale"]),
                )
        else:
            raise ValueError("Choose save or undo.")
        current.state[STATE_KEY] = update_experiment(
            current.state.get(STATE_KEY, {}), expected=expected, entry=candidate
        )
        handler.server.store.save(current)
        query = {"mode": "geometry" if values.get("kind") == "geometry" else "styling"}
        if any(
            e.photo_id == values.get("photo")
            for e in PhotoCatalog(handler.server.store, session.session_id).entries()
        ):
            query["photo"] = values["photo"]
        handler._redirect(
            f"/sessions/{session.session_id}/experiments?{urlencode(query)}#photo-experiment"
        )
    except LookupError as error:
        handler._error(404, str(error))
    except ValueError as error:
        handler._error(
            409 if "stale" in str(error) or "earlier report" in str(error) else 422,
            str(error),
        )
    except (OSError, RuntimeError):
        handler._error(
            500, "The illustration could not be saved. Your session is still available."
        )
