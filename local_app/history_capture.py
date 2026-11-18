"""Capture metadata records estimates and user labels without validating anatomy."""

from __future__ import annotations

import hashlib

from .history_workflow import SourceContext
from .photos import PhotoCatalog


def source_contexts(session, store, report, snapshot) -> tuple[SourceContext, ...]:
    raw = {
        r["photo_id"]: r
        for r in report.axes["harmony"].details.get("appearance_sources", [])
    }
    settings = session.state.get("appearance_review", {}).get("photo_settings", {})
    views = {}
    for row in snapshot.observations:
        for source in row.sources:
            views.setdefault(source.photo_number, set()).add(source.view)
    result = []
    for number, entry in enumerate(
        PhotoCatalog(store, session.session_id).entries(), 1
    ):
        source = store.path_for(
            session.session_id, "uploads", entry.stored_name
        ).read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        detected = raw.get(entry.photo_id, {})
        if detected.get("source_sha256") != digest or (
            detected.get("width"),
            detected.get("height"),
        ) != (entry.width, entry.height):
            detected = {}
        entered = settings.get(entry.photo_id, {})
        if entered.get("source_sha256") != digest:
            entered = {}
        role = entered.get("role", detected.get("role", "unknown"))
        pose = detected.get("pose", dict(yaw=None, pitch=None, roll=None))
        result.append(
            SourceContext(
                photo_id=entry.photo_id,
                photo_number=number,
                source_sha256=digest,
                width=entry.width,
                height=entry.height,
                views=tuple(sorted(views.get(number, {"unknown"}))),
                pose=pose,
                pose_basis=(
                    "detector_estimate_uncalibrated"
                    if any(value is not None for value in pose.values())
                    else "unavailable"
                ),
                role=role,
                role_basis=(
                    "user_recorded_unverified"
                    if entered
                    else (
                        "detector_candidate_unverified"
                        if role != "unknown"
                        else "unavailable"
                    )
                ),
            )
        )
    return tuple(result)
