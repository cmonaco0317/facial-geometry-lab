"""One current observation collection shared by live, exported and saved reports."""

from __future__ import annotations

import html
import json
from types import SimpleNamespace

from .automatic_observations import automatic_observations
from .observations import Observation, public_observations
from .review_observations import reviewed_observations
from .reading_registry import VERSION, current_reading_registry
from .profile_proposals import profile_proxy_contract
from .expanded_measurements import (
    expanded_frontal_observations,
    expanded_profile_observations,
    exclude_reviewed_frontal_sources,
)


def current_observations(axes, frontal=None, profiles=None) -> tuple[Observation, ...]:
    harmony = axes.get("harmony")
    raw = harmony.details.get("observations", []) if harmony else []
    automatic = tuple(Observation.model_validate_json(json.dumps(r)) for r in raw)
    if any(not r.geometry_retained for r in automatic):
        raise ValueError(
            "Archived numerical records must be remeasured before live use."
        )
    if not automatic:
        automatic = automatic_observations(
            [], SimpleNamespace(kept=(), photo_facts=()), ()
        )
    from .automatic_forehead import coverage_exclusions, FOREHEAD_METRICS

    covered = coverage_exclusions(axes, frontal)
    refreshed = []
    for record in automatic:
        if record.definition_key.split(":", 1)[1] in FOREHEAD_METRICS and any(
            s.photo_id in covered for s in record.sources
        ):
            record = Observation.model_validate(
                {
                    **record.model_dump(),
                    "status": "unavailable",
                    "value": None,
                    "proxy_for": None,
                    "reference": None,
                    "evaluation": None,
                    "score_eligible": False,
                    "withheld_reason": "Source forehead visibility excludes this reading. Reanalyze to select another compatible source.",
                }
            )
        refreshed.append(record)
    automatic = tuple(refreshed)
    profiles = [profiles] if isinstance(profiles, dict) else profiles or []
    controls = dict(harmony.details.get("profile_controls", {})) if harmony else {}
    for raw in list(frontal or ()) + profiles:
        if raw.get("evidence_photo_id"):
            controls[raw["evidence_photo_id"]] = "reviewed"
    proposals = harmony.details.get("profile_proposals", []) if harmony else []
    proxy_records = tuple(
        Observation.model_validate_json(json.dumps(record))
        for proposal in proposals
        if proposal.get("status") == "candidate"
        and proposal.get("photo_id") not in controls
        for record in proposal.get("observations", [])
    )
    if any(
        r.method != "mesh-profile-proxy-v1"
        or not r.geometry_retained
        or r.status not in {"proxy", "unavailable"}
        or r.score_eligible
        or r.reference is not None
        for r in proxy_records
    ):
        raise ValueError("Invalid automatic profile proxy record.")
    # v1 profile snapshots retain their original table; never relabel them v2.
    profiles = [r for r in profiles if r.get("method") != "manual-profile-v1"]
    reviewed = reviewed_observations(tuple(frontal or ()), tuple(profiles))
    expanded_raw = harmony.details.get("expanded_observations", []) if harmony else []
    expanded_frontal = tuple(
        Observation.model_validate_json(json.dumps(r)) for r in expanded_raw
    ) or expanded_frontal_observations(SimpleNamespace(kept=(), photo_facts=()), ())
    if any(
        not r.geometry_retained
        or r.method != "expanded-frontal-mesh-v1"
        or r.status not in {"proxy", "unavailable"}
        or r.score_eligible
        or r.reference is not None
        for r in expanded_frontal
    ):
        raise ValueError("Invalid expanded frontal mesh record.")
    reviewed_ids = {
        raw["evidence_photo_id"]
        for raw in list(frontal or ()) + profiles
        if raw.get("evidence_photo_id")
    }
    # The frontal editor has a different anchor set. Do not present unchanged mesh
    # constructions as if they incorporated a manual correction to that photo.
    expanded_frontal = exclude_reviewed_frontal_sources(expanded_frontal, reviewed_ids)
    expanded_profiles = expanded_profile_observations(reviewed + proxy_records)
    keys = {r.definition_key for r in reviewed + proxy_records}
    absent = tuple(
        Observation(
            definition_key=d["key"],
            definition_version=(
                profile_proxy_contract(None)
                if d["family"] == "mesh_profile_proxy"
                else VERSION
            ),
            label=d["label"],
            definition=d["definition"],
            formula=d["formula"],
            method=d["method"],
            unit=d["unit"],
            status="unavailable",
            value=None,
            revision=1,
            withheld_reason=(
                "No current eligible turned-view mesh proposal, or a review/removal takes precedence. No anatomy was substituted."
                if d["family"] == "mesh_profile_proxy"
                else "No compatible reviewed source. Visible points can be reviewed optionally; no evaluated same-quantity estimator is available."
            ),
        )
        for d in current_reading_registry()["readings"]
        if d["family"] in {"reviewed_frontal", "reviewed_profile", "mesh_profile_proxy"}
        and d["key"] not in keys
    )
    records = (
        automatic
        + reviewed
        + proxy_records
        + absent
        + expanded_frontal
        + expanded_profiles
    )
    public_observations(records)
    return records


def archive_observations(axes, frontal=None, profiles=None):
    """Photo-free numerical snapshots respect the existing geometry opt-in."""
    result = []
    for row in public_observations(current_observations(axes, frontal, profiles)):
        row["geometry_retained"] = False
        row["score_eligible"] = False
        for source in row["sources"]:
            source["points"] = []
        result.append(Observation.model_validate_json(json.dumps(row)))
    return tuple(result)


def profile_proxy_attempts(axes):
    """Disclose routing outcomes with ordinals; private source IDs/hashes stay local."""
    harmony = axes.get("harmony")
    details = harmony.details if harmony else {}
    controls = details.get("profile_controls", {})
    result = []
    for proposal in details.get("profile_proposals", []):
        number = proposal.get("photo_number")
        if type(number) is not int or not 1 <= number <= 10000:
            continue
        control = controls.get(proposal.get("photo_id"))
        result.append(
            dict(
                photo_number=number,
                method="mesh-profile-proxy-v1",
                status=control or proposal["status"],
                reason=(
                    "A current review or explicit removal takes precedence."
                    if control
                    else proposal["reason"]
                ),
                facing=proposal.get("facing"),
                pose=proposal.get("pose"),
                coordinate_frame="Original stored image; no mirroring or leveling applied to measurements.",
                facing_convention="Nose direction in the displayed image, not anatomical left/right.",
                routing_is_accuracy_evidence=False,
                inference_orientation=proposal.get("inference_orientation"),
            )
        )
    return result


def render_observations(records) -> str:
    """Render only validated public records, including unavailable cards."""
    from .consultation_findings import evidence_anchor

    validated = tuple(Observation.model_validate_json(json.dumps(r)) for r in records)
    public = public_observations(validated)
    rows = []
    for row in public:
        value = (
            "Unavailable"
            if row["value"] is None
            else f"{row['value']:.{row['display_decimals']}f} {row['unit']}"
        )
        source = (
            ", ".join(
                f"Photo {s['photo_number']} · {s['view'].replace('_', ' ')} · review {s['review_revision']}"
                for s in row["sources"]
            )
            or "No compatible source"
        )
        details = (
            row["withheld_reason"]
            or row["score_exclusion_reason"]
            or "Compatible score input"
        )
        rows.append(
            f'<tr id="{evidence_anchor(row)}">'
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in (
                    row["label"],
                    value,
                    row["status"],
                    row["method"],
                    source,
                    details,
                )
            )
            + "</tr>"
        )
    return (
        '<section class="current-observations"><h2>Measurement evidence</h2><p>Every supported automatic reading keeps a card. Missing inputs stay unavailable; mesh proxies are distinct from anatomical measurements. Reviewed readings retain their own source and revision.</p><div class="table-scroll"><table><thead><tr><th>Reading</th><th>Value</th><th>Status</th><th>Method</th><th>Source</th><th>Eligibility or missing reason</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div></section>"
    )
