"""Validate imported report display inputs and audit generated static HTML."""

from __future__ import annotations

import base64
from html.parser import HTMLParser
from math import isfinite
import re
from urllib.parse import urlsplit


class _StaticReport(HTMLParser):
    """Fail closed on active content even in a separately opened HTML download."""

    tags = frozenset(
        "html head meta title style body main header footer nav aside section article div span p h1 h2 h3 h4 h5 h6 a ul ol li table caption thead tbody tfoot tr th td details summary figure figcaption img svg g polyline polygon line circle path rect ellipse text pre code em strong small b i br hr dl dt dd sup sub".split()
    )
    attrs = frozenset(
        "charset name content http-equiv lang class id role width height viewbox points x y x1 y1 x2 y2 cx cy r rx d fill stroke stroke-width fill-opacity font-size xmlns style href src alt rel colspan rowspan scope open".split()
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.style_count = 0
        self.in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self.tags:
            raise ValueError("Saved report contains unsupported HTML.")
        if tag == "style":
            self.style_count += 1
            self.in_style = True
            if self.style_count > 1 or attrs:
                raise ValueError("Saved report contains an additional stylesheet.")
        for key, value in attrs:
            if key not in self.attrs and not key.startswith("aria-"):
                raise ValueError("Saved report contains an unsupported HTML attribute.")
            value = value or ""
            if key == "href":
                address = urlsplit(value)
                if not value.startswith("#") and not (
                    address.scheme == "https" and address.netloc
                ):
                    raise ValueError("Saved report contains an unsupported link.")
            if key == "src":
                if tag != "img" or not re.fullmatch(
                    r"data:image/(?:jpeg|svg\+xml);base64,[A-Za-z0-9+/]*={0,2}", value
                ):
                    raise ValueError(
                        "Saved report images must be embedded static image data."
                    )
            if key == "http-equiv" and value.lower() != "content-security-policy":
                raise ValueError("Saved report cannot redirect or refresh.")
            if key == "style" and not re.fullmatch(
                r"(?:\s*(?:left|width|height|max-width)\s*:\s*[0-9.]+(?:%|px)?\s*;?)+",
                value,
            ):
                raise ValueError("Saved report contains unsupported inline styling.")
            if key in {"fill", "stroke"} and not re.fullmatch(
                r"none|#[a-fA-F0-9]{3,8}", value
            ):
                raise ValueError("Saved report contains unsupported SVG paint.")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag not in self.tags:
            raise ValueError("Saved report contains an unsupported closing tag.")
        if tag == "style":
            self.in_style = False

    def handle_data(self, data: str) -> None:
        if self.in_style and re.search(
            r"url\s*\(|@import|expression\s*\(|-moz-binding|behavior\s*:", data, re.I
        ):
            raise ValueError("Saved report contains unsupported stylesheet resources.")


def validate_display_fields(value: object) -> None:
    """Check image/ordinal/hash fields in current and recursively saved contexts."""
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("Saved report numbers must be finite.")
    if isinstance(value, list):
        for item in value:
            validate_display_fields(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "id" and (
                not isinstance(item, str)
                or not re.fullmatch(r"[A-Za-z0-9_.:-]{0,200}", item)
            ):
                raise ValueError("Invalid saved report identifier.")
            if key == "jpeg_b64":
                if not isinstance(item, str) or not 1 <= len(item) <= 8_000_000:
                    raise ValueError("Invalid saved crop.")
                try:
                    raw = base64.b64decode(item, validate=True)
                except ValueError as error:
                    raise ValueError("Invalid saved crop encoding.") from error
                if not raw.startswith(b"\xff\xd8\xff"):
                    raise ValueError("Saved crops must use JPEG data.")
            if key == "photo_number" and (
                type(item) is not int or not 1 <= item <= 10000
            ):
                raise ValueError("Invalid saved photo number.")
            if key == "source_sha256" and (
                not isinstance(item, str) or not re.fullmatch("[a-f0-9]{64}", item)
            ):
                raise ValueError("Invalid saved source hash.")
            validate_display_fields(item)


def render_saved_report(payload: dict) -> str:
    from .report import render_html_export
    from .report_questions import _load

    try:
        validate_display_fields(payload)
        for context in (payload.get("report_questions") or {}).values():
            if isinstance(context, dict) and "report_json" in context:
                validate_display_fields(_load(context))
        content = render_html_export(payload)
        parser = _StaticReport(convert_charrefs=True)
        parser.feed(content)
        parser.close()
        return content
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("The saved report display contract is unsupported.") from error
