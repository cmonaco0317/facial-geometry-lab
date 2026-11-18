"""Photo-bound, revisioned partial profile reviews; v1 records remain readable."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from urllib.parse import parse_qs, urlsplit

from .photos import PhotoCatalog
from .profile_review import ANCHORS, measure_profile
from .profile_v2 import METHOD, contract, measure, validate_points, warnings
from .review_page import render_review
from .profile_lineage import proposal_origin
from .profile_proposals import MESH_PROFILE_METHOD, profile_proxy_contract

STATE_KEY = "profile_reviews"


def _selection(session, store, photo):
    entries = PhotoCatalog(store, session.session_id).entries()
    selected = (
        next((e for e in entries if e.photo_id == photo), None)
        if photo is not None
        else next(iter(entries), None)
    )
    if selected is None:
        raise LookupError("Photograph not found in this session.")
    return entries, selected


def _binding(session, store, entry):
    path = store.path_for(session.session_id, "uploads", entry.stored_name)
    value = [
        entry.photo_id,
        entry.sha256,
        entry.width,
        entry.height,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        contract(),
    ]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _record(session, store, entry):
    binding = _binding(session, store, entry)
    prior = session.state.get(STATE_KEY, {}).get(entry.photo_id)
    proposals = (
        session.state.get("analysis_report", {})
        .get("axes", {})
        .get("harmony", {})
        .get("details", {})
        .get("profile_proposals", [])
    )
    source_hash = hashlib.sha256(
        store.path_for(session.session_id, "uploads", entry.stored_name).read_bytes()
    ).hexdigest()
    proposal = next(
        (
            p
            for p in proposals
            if p.get("photo_id") == entry.photo_id
            and p.get("status") == "candidate"
            and p.get("source_sha256") == source_hash
            and p.get("width") == entry.width
            and p.get("height") == entry.height
            and p.get("method") == MESH_PROFILE_METHOD
            and p.get("contract") == profile_proxy_contract(p.get("model_sha256"))
        ),
        None,
    )
    if prior and prior.get("binding") == binding:
        if (
            proposal
            and not any(
                prior.get(k)
                for k in (
                    "reviewed",
                    "migrated",
                    "automatic_suppressed",
                    "supersedes_legacy",
                    "restored_hints",
                )
            )
            and (prior.get("automatic_proposal") or {}).get("proposal_id")
            != proposal["proposal_id"]
        ):
            prior = {
                **prior,
                "points": deepcopy(proposal["points"]),
                "facing": proposal["facing"],
                "automatic_proposal": proposal,
                "revision": prior["revision"] + 1,
            }
            session.state[STATE_KEY][entry.photo_id] = prior
            store.save(session)
        return prior
    points = {k: None for k in ANCHORS}
    migrated = False
    legacy = session.state.get("profile_review", {})
    if prior is None and legacy.get("photo_id") == entry.photo_id:
        try:
            points = validate_points(legacy["points"])
            migrated = True
        except (ValueError, KeyError):
            pass
    value = dict(
        method=METHOD,
        points=points,
        revision=prior["revision"] + 1 if prior else 1,
        binding=binding,
        reviewed=False,
        facing="",
        level_degrees=0.0,
        migrated=migrated,
        supersedes_legacy=bool(
            prior and (prior.get("supersedes_legacy") or prior.get("reviewed"))
        ),
        automatic_suppressed=False,
        automatic_proposal=proposal if not migrated else None,
    )
    if proposal and not migrated:
        value.update(points=deepcopy(proposal["points"]), facing=proposal["facing"])
    session.state.setdefault(STATE_KEY, {})[entry.photo_id] = value
    store.save(session)
    return value


def _display(handler, session, entries, entry, record, *, draft=None):
    import cv2

    path = handler.server.store.path_for(
        session.session_id, "uploads", entry.stored_name
    )
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("This photograph could not be opened.")
    factor = min(1, 1200 / max(image.shape[:2]))
    if factor < 1:
        image = cv2.resize(
            image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("This photograph could not be displayed.")
    page = render_review(
        session, entries, entry, encoded.tobytes(), record, draft=draft
    )
    handler._send(200, page.encode(), "text/html; charset=utf-8")


def get_profile(handler, session):
    try:
        query = parse_qs(
            urlsplit(handler.path).query, keep_blank_values=True, max_num_fields=2
        )
        if set(query) - {"photo", "draft"} or any(len(v) != 1 for v in query.values()):
            raise ValueError("Choose one photograph.")
        entries, entry = _selection(
            session, handler.server.store, query.get("photo", [None])[0]
        )
        handler.server.analyze_session(session.session_id)
        session = handler.server.store.load(session.session_id)
        if "draft" in query and query["draft"] != ["yes"]:
            raise ValueError("Unsupported preview selection.")
        record = _record(session, handler.server.store, entry)
        draft = None
        if query.get("draft") == ["yes"]:
            pending = session.state.get("profile_drafts", {}).get(entry.photo_id)
            if (
                pending
                and pending.get("binding") == record["binding"]
                and pending.get("base_revision") == record["revision"]
            ):
                draft = pending
        _display(handler, session, entries, entry, record, draft=draft)
    except LookupError as error:
        handler._error(404, str(error))
    except ValueError as error:
        handler._error(422, str(error))
    except (OSError, RuntimeError):
        handler._error(
            500,
            "The profile review could not be prepared. Your session is still available.",
        )


def update_profile(handler, session, resource, fields):
    try:
        allowed = {"csrf_token", "photo", "method", "revision", "binding"}
        if resource == "profile":
            allowed |= (
                {
                    "action",
                    "facing",
                    "confirmed",
                    "lateral_confirmed",
                    "acknowledged",
                    "active_anchor",
                    "level_degrees",
                    "metric",
                }
                | {k + "_" + a for k in ANCHORS for a in ("x", "y")}
                | {"missing_" + k for k in ANCHORS}
            )
        if (
            set(fields) - allowed
            or any(len(v) != 1 for v in fields.values())
            or fields.get("method") != [METHOD]
        ):
            raise ValueError("Unsupported or repeated profile fields.")
        current = handler.server.store.load(session.session_id)
        entries, entry = _selection(
            current, handler.server.store, handler._one(fields, "photo") or ""
        )
        record = _record(current, handler.server.store, entry)
        revision = int(handler._one(fields, "revision") or "")
        if (
            revision != record["revision"]
            or handler._one(fields, "binding") != record["binding"]
        ):
            handler._error(
                409, "The profile or photograph changed. Reload before saving."
            )
            return
        if resource == "profile-clear":
            current.state[STATE_KEY][entry.photo_id] = {
                **record,
                "points": {k: None for k in ANCHORS},
                "reviewed": False,
                "revision": revision + 1,
                "migrated": False,
                "supersedes_legacy": True,
                "automatic_suppressed": True,
                "automatic_proposal": None,
                "proposal_ancestry": None,
            }
            if (
                current.state.get("profile_review", {}).get("photo_id")
                == entry.photo_id
            ):
                current.state.pop("profile_review", None)
        elif handler._one(fields, "action") == "propose":
            if (
                current.state.get("profile_review", {}).get("photo_id")
                == entry.photo_id
            ):
                current.state.pop("profile_review", None)
            current.state[STATE_KEY][entry.photo_id] = {
                **record,
                "points": {k: None for k in ANCHORS},
                "reviewed": False,
                "automatic_suppressed": False,
                "supersedes_legacy": False,
                "migrated": False,
                "restored_hints": False,
                "automatic_proposal": None,
                "proposal_ancestry": None,
                "revision": revision + 1,
            }
            handler.server.store.save(current)
            handler._redirect(
                f"/sessions/{session.session_id}/profile?photo={entry.photo_id}"
            )
            return
        else:
            action = handler._one(fields, "action")
            if action not in ("save", "preview"):
                raise ValueError("Choose preview or save.")
            if (
                "active_anchor" in fields
                and handler._one(fields, "active_anchor") not in ANCHORS
            ):
                raise ValueError("Unsupported selected point.")
            points = {}
            for key in ANCHORS:
                if "missing_" + key in fields:
                    if fields["missing_" + key] != ["yes"]:
                        raise ValueError("Invalid visibility selection.")
                    points[key] = None
                else:
                    points[key] = [
                        float(handler._one(fields, key + "_" + axis) or "") / 100
                        for axis in ("x", "y")
                    ]
            points = validate_points(points)
            facing = handler._one(fields, "facing")
            level = float(handler._one(fields, "level_degrees") or "0")
            if not math.isfinite(level) or not -45 <= level <= 45:
                raise ValueError("Visual leveling must be between -45 and 45 degrees.")
            if "metric" in fields:
                from .profile_v2 import ROWS

                if handler._one(fields, "metric") not in {"all"} | {
                    r["id"] for r in ROWS
                }:
                    raise ValueError("Unsupported displayed measurement.")
            reading = measure(points, entry.width, entry.height, facing=facing)
            draft = dict(points=points, facing=facing, level_degrees=level)
            if action == "preview":
                current.state.setdefault("profile_drafts", {})[entry.photo_id] = {
                    **draft,
                    "binding": record["binding"],
                    "base_revision": revision,
                }
                handler.server.store.save(current)
                handler._redirect(
                    f"/sessions/{session.session_id}/profile?photo={entry.photo_id}&draft=yes"
                )
                return
            if fields.get("confirmed") != ["yes"] or fields.get(
                "lateral_confirmed"
            ) != ["yes"]:
                raise ValueError(
                    "Confirm a lateral photograph and review the visible points before saving."
                )
            if not reading["measurements"]:
                raise ValueError(
                    "Place enough visible points for at least one reading."
                )
            from .frontal_routes import frontal_payloads

            if any(
                p.get("evidence_photo_id") == entry.photo_id
                for p in frontal_payloads(current, handler.server.store)
            ):
                raise ValueError(
                    "This photo has a frontal review. Use a separate side photo, or remove its frontal review before saving a profile review."
                )
            if warnings(
                points, entry.width, entry.height, facing=facing
            ) and fields.get("acknowledged") != ["yes"]:
                raise ValueError(
                    "Check the profile warnings and explicitly acknowledge unusual geometry before saving."
                )
            current.state[STATE_KEY][entry.photo_id] = {
                **deepcopy(record),
                **draft,
                "reviewed": True,
                "revision": revision + 1,
                "migrated": False,
                "supersedes_legacy": True,
                "automatic_suppressed": False,
            }
            automatic = record.get("automatic_proposal")
            previous_origin = proposal_origin(record.get("proposal_ancestry"))
            if automatic or previous_origin:

                def quantized(pair):
                    return (
                        tuple(round(v, 8) for v in pair) if pair is not None else None
                    )

                origin = proposal_origin(
                    dict(
                        **(
                            previous_origin.model_dump(
                                mode="json", exclude={"changed_anchors"}
                            )
                            if previous_origin
                            else {
                                key: automatic[key]
                                for key in (
                                    "method",
                                    "contract",
                                    "proposal_id",
                                    "model_sha256",
                                )
                            }
                        ),
                        changed_anchors=sorted(
                            set(
                                previous_origin.changed_anchors
                                if previous_origin
                                else ()
                            )
                            | {
                                k
                                for k in ANCHORS
                                if quantized(points[k])
                                != quantized(record["points"][k])
                            }
                        ),
                    )
                )
                current.state[STATE_KEY][entry.photo_id]["proposal_ancestry"] = (
                    origin.model_dump(mode="json")
                )
        current.state.get("profile_drafts", {}).pop(entry.photo_id, None)
        handler.server.store.save(current)
        handler._redirect(f"/sessions/{session.session_id}/results")
    except LookupError as error:
        handler._error(404, str(error))
    except (ValueError, TypeError) as error:
        handler._error(422, str(error))
    except (OSError, RuntimeError):
        handler._error(
            500, "The profile could not be updated. Your session is still available."
        )


def profile_payloads(session, store):
    result = []
    reviewed = set()
    entries = PhotoCatalog(store, session.session_id).entries()
    for index, entry in enumerate(entries, 1):
        raw = session.state.get(STATE_KEY, {}).get(entry.photo_id)
        if not raw or not raw.get("reviewed"):
            continue
        try:
            if raw.get("binding") != _binding(session, store, entry):
                continue
            value = measure(
                raw["points"], entry.width, entry.height, facing=raw["facing"]
            )
            result.append(
                {
                    **value,
                    "revision": raw["revision"],
                    "photo_number": index,
                    "evidence_photo_id": entry.photo_id,
                    "proposal_ancestry": (
                        proposal_origin(raw.get("proposal_ancestry")).model_dump(
                            mode="json"
                        )
                        if raw.get("proposal_ancestry")
                        else None
                    ),
                }
            )
            reviewed.add(entry.photo_id)
        except (ValueError, OSError, KeyError):
            continue
    legacy = session.state.get("profile_review")
    legacy_record = (
        session.state.get(STATE_KEY, {}).get(legacy.get("photo_id"), {})
        if legacy
        else {}
    )
    superseded = legacy_record.get("reviewed") or legacy_record.get("supersedes_legacy")
    if legacy and not superseded and legacy.get("photo_id") not in reviewed:
        entry = next((e for e in entries if e.photo_id == legacy.get("photo_id")), None)
        if entry:
            try:
                result.append(
                    measure_profile(legacy.get("points"), entry.width, entry.height)
                )
            except ValueError:
                pass
    if len(result) == 1 and result[0]["method"] == "manual-profile-v1":
        return result[0]
    return result or None
