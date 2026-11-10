#!/usr/bin/env python3
"""Compatibility launcher for the private loopback facial-analysis interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from local_app.server import serve


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the facial-analysis interface locally on 127.0.0.1.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="start the local server without opening a browser window",
    )
    options = parser.parse_args(argv)
    return serve(
        host="127.0.0.1",
        port=options.port,
        open_browser=not options.no_browser,
    )


if __name__ == "__main__":
    raise SystemExit(main())
