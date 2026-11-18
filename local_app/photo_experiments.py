"""One bounded, reversible illustration kept separate from measured report data."""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET

from .previews import make_preview

VERSION = "photo-experiments-v1"
MAX_ENTRY_BYTES = 8_000_000


def report_basis(payload: dict) -> str:
    base = {
        k: v
        for k, v in payload.items()
        if k not in {"report_questions", "photo_experiment"}
    }
    if "consultation" in base:
        base["consultation"] = {
            k: v
            for k, v in base["consultation"].items()
            if k not in {"reviewer_record", "reviewer_context_binding"}
        }
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(base):
        digest.update(chunk.encode())
    return "report-" + digest.hexdigest()


def seal_entry(entry: dict) -> dict:
    encoded = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > MAX_ENTRY_BYTES:
        raise ValueError("This illustration is too large to retain in the report.")
    return {**entry, "id": "experiment-" + hashlib.sha256(encoded.encode()).hexdigest()}


def bind_svg(content: bytes, *, basis: str, photo_number: int) -> str:
    root = ET.fromstring(content)
    node = root.find("{*}metadata")
    if node is None or node.text is None:
        raise ValueError("An illustration needs its source metadata.")
    metadata = json.loads(node.text)
    metadata.update(base_report_revision=basis, photo_number=photo_number)
    node.text = json.dumps(metadata, sort_keys=True)
    return ET.tostring(root, encoding="unicode")


def _safe_svg(content: str) -> tuple[dict, str]:
    if "<?" in content or "<!DOCTYPE" in content.upper():
        raise ValueError(
            "Illustration processing instructions and document types are not supported."
        )
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise ValueError("Invalid saved illustration.") from error
    tags = {"svg", "metadata", "image", "g", "rect", "path", "text", "circle", "line"}
    attributes = {
        "xmlns",
        "width",
        "height",
        "viewBox",
        "x",
        "y",
        "x1",
        "x2",
        "y1",
        "y2",
        "cx",
        "cy",
        "r",
        "rx",
        "fill",
        "fill-opacity",
        "stroke",
        "stroke-width",
        "d",
        "id",
        "font-size",
        "href",
    }
    if root.tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError("Invalid illustration namespace.")
    for node in root.iter():
        if (
            node.tag not in {"{http://www.w3.org/2000/svg}" + t for t in tags}
            or set(node.attrib) - attributes
        ):
            raise ValueError("Unsupported saved illustration content.")
        if any(
            "url(" in value.lower() or "javascript:" in value.lower()
            for value in node.attrib.values()
        ):
            raise ValueError("External illustration resources are not supported.")
        for field in ("fill", "stroke"):
            if field in node.attrib and not re.fullmatch(
                r"none|#[0-9a-fA-F]{3,8}", node.attrib[field]
            ):
                raise ValueError("Unsupported illustration paint value.")
        if "href" in node.attrib and not node.attrib["href"].startswith(
            "data:image/jpeg;base64,"
        ):
            raise ValueError(
                "Illustrations may contain only embedded source image data."
            )
    metadata = root.find("{*}metadata")
    if metadata is None or metadata.text is None:
        raise ValueError("An illustration needs its source metadata.")
    decoded = json.loads(metadata.text)
    if not isinstance(decoded, dict):
        raise ValueError("Illustration metadata must be an object.")
    return decoded, " ".join(node.text or "" for node in root.findall("{*}text"))


def public_experiment(state: dict | None, current_basis: str | None = None) -> dict:
    if state is not None and not isinstance(state, dict):
        raise ValueError("Invalid saved experiment state.")
    if not state or state.get("entry") is None:
        return {
            "revision": (state or {}).get("revision", 0),
            "entry": None,
            "binding_status": "not_supplied",
        }
    entry = state["entry"]
    if not isinstance(entry, dict):
        raise ValueError("Invalid saved experiment entry.")
    raw = {k: v for k, v in entry.items() if k != "id"}
    if (
        seal_entry(raw) != entry
        or entry.get("version") != VERSION
        or entry.get("kind") not in {"hair", "eyewear", "geometry"}
    ):
        raise ValueError("The saved illustration does not match its retained evidence.")
    if (
        type(entry.get("photo_number")) is not int
        or not 1 <= entry["photo_number"] <= 10000
    ):
        raise ValueError("Invalid illustration source photo.")
    if not isinstance(entry.get("source_sha256"), str) or not isinstance(
        entry.get("base_report_revision"), str
    ):
        raise ValueError("The illustration has no valid source/report binding.")
    if not re.fullmatch(
        r"[0-9a-f]{64}", entry.get("source_sha256", "")
    ) or not re.fullmatch(
        r"report-[0-9a-f]{64}", entry.get("base_report_revision", "")
    ):
        raise ValueError("The illustration has no valid source/report binding.")
    artifacts = {}
    for field in ("original_svg", "simulation_svg"):
        if not isinstance(entry.get(field), str):
            raise ValueError("The saved illustration has no SVG artifact.")
        metadata, visible = _safe_svg(entry[field])
        artifacts[field] = metadata
        if (
            metadata.get("source_sha256") != entry["source_sha256"]
            or metadata.get("base_report_revision") != entry["base_report_revision"]
            or metadata.get("photo_number") != entry["photo_number"]
            or "Illustrative only" not in visible
        ):
            raise ValueError(
                "The illustration lacks matching source metadata or a visible label."
            )
    if entry["kind"] == "geometry":
        from .geometry_experiments import validate_geometry

        validate_geometry(entry, artifacts)
    elif entry.get("measurements") != []:
        raise ValueError("A styling overlay cannot supply new geometry readings.")
    return {
        "revision": state.get("revision", 0),
        "entry": deepcopy(entry),
        "binding_status": (
            "current"
            if current_basis == entry["base_report_revision"]
            else "historical" if current_basis else "saved"
        ),
    }


def styling_experiment(
    source: bytes,
    *,
    photo_number: int,
    basis: str,
    style: str,
    x: float,
    y: float,
    scale: float,
) -> dict:
    if style not in {"hair", "eyewear"}:
        raise ValueError("Choose a hair or eyewear illustration.")
    preview = make_preview(source, style=style, x=x, y=y, scale=scale)
    original = make_preview(source, style=style, original=True)
    entry = seal_entry(
        dict(
            version=VERSION,
            kind=style,
            photo_number=photo_number,
            source_sha256=preview.source_sha256,
            base_report_revision=basis,
            parameters=dict(x=x, y=y, scale=scale),
            original_svg=bind_svg(
                original.content, basis=basis, photo_number=photo_number
            ),
            simulation_svg=bind_svg(
                preview.content, basis=basis, photo_number=photo_number
            ),
            measurements=[],
            label="Illustrative simulation only. Styling geometry is hand-authored, not a fit prescription or tested improvement. Source measurements are unchanged.",
        )
    )
    return public_experiment({"entry": entry}, basis)["entry"]


def update_experiment(state: dict, *, expected: int, entry: dict | None) -> dict:
    revision = state.get("revision", 0)
    if type(expected) is not int or expected != revision:
        raise ValueError("This experiment form is stale; reload it first.")
    if revision >= 1000:
        raise ValueError("This session has reached its experiment revision limit.")
    result = {"revision": revision + 1, "entry": entry}
    public_experiment(result)
    return deepcopy(result)


def render_experiment(
    state: dict | None, *, action: str | None = None, prefix: str = ""
) -> str:
    payload = public_experiment(state)
    entry = payload["entry"]
    esc = html.escape
    output = f'<section id="{prefix}photo-experiment"><h2>Photo experiments</h2>'
    if action:
        output += f'<p><a href="{esc(action,quote=True)}">Try a reversible photo experiment</a></p>'
    if entry is None:
        return (
            output
            + "<p>No illustration saved. Original photo and measured report are unchanged.</p></section>"
        )
    status = (state or {}).get("binding_status", payload["binding_status"])
    if status == "historical":
        output += "<p>This illustration belongs to an earlier report. Its retained original and simulated constructions stay separate from current evidence.</p>"
    output += f'<p>{esc(entry["label"])}</p><p>Source photo {entry["photo_number"]}. Source SHA-256: {entry["source_sha256"]}.</p><div class="axis-grid experiment-pair">'
    for field, label in (
        ("original_svg", "Original photo display"),
        ("simulation_svg", "Illustrative simulation"),
    ):
        encoded = base64.b64encode(entry[field].encode()).decode()
        output += f'<figure><img width="420" class="experiment-image" alt="{label}" src="data:image/svg+xml;base64,{encoded}"><figcaption>{label}. Embedded display may be resized; original stored photo bytes are preserved.</figcaption></figure>'
    output += "</div>"
    if entry["measurements"]:
        output += "<p>All supported construction dependencies are recalculated. These simulated points and readings never replace observed anatomy, reference percentiles or scores.</p><table><thead><tr><th>Construction</th><th>Original points</th><th>Simulated points</th><th>Difference</th></tr></thead><tbody>"
        for row in entry["measurements"]:
            output += f'<tr id="{prefix}{entry["id"]}-{row["id"]}"><td>{esc(row["label"])} ({esc(row["unit"])})</td><td>{esc(str(row["before"]))}</td><td>{esc(str(row["after"]))}</td><td>{esc(str(row["delta"]))}</td></tr>'
        output += "</tbody></table>"
    return output + "</section>"
