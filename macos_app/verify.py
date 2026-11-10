"""Verification helpers for the private macOS distribution."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
import hashlib
import html
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import plistlib
import re
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from typing import Any
from urllib.parse import quote, urlencode, urlsplit
import zipfile

from . import bootstrap, inventory
from .config import APP_VERSION, MEDIAPIPE_VERSION, MINIMUM_MACOS, MODEL_ASSETS

CommandRunner = Callable[[list[str]], tuple[int, str, str]]
PRIVATE_REPORT_KEYS = frozenset({"session_id", "photo_fingerprint"})
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".bmp",
        ".gif",
        ".heic",
        ".jpeg",
        ".jpg",
        ".pdf",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        ".git",
        ".env",
        "backlog.md",
        "ledger.json",
        "publishing.md",
        "report.html",
        "report.json",
        "state.md",
        "subject.json",
    }
)
EXPECTED_DEPENDENCIES = {
    "mediapipe": MEDIAPIPE_VERSION,
    "cv2": "4.13.0",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
}
EXPECTED_DISTRIBUTIONS = {
    "cryptography": "50.0.2",
    "pydantic": "2.14.1",
    "pydantic-core": "2.50.1",
    "annotated-types": "0.8.0",
    "typing-inspection": "0.4.4",
    "typing-extensions": "4.16.0",
    "limits": "5.8.0",
    "deprecated": "1.3.1",
    "wrapt": "2.5.1",
    "absl-py": "2.5.0",
    "certifi": "2026.7.22",
    "cffi": "2.1.1",
    "contourpy": "1.3.3",
    "cycler": "0.12.1",
    "flatbuffers": "25.12.19",
    "fonttools": "4.65.0",
    "kiwisolver": "1.5.1",
    "matplotlib": "3.11.0",
    "mediapipe": MEDIAPIPE_VERSION,
    "numpy": "2.4.6",
    "opencv-contrib-python": "4.13.0.92",
    "packaging": "26.3",
    "pillow": "12.3.0",
    "pip": "26.2.1",
    "pycparser": "3.0",
    "pyparsing": "3.3.3",
    "python-dateutil": "2.9.0.post0",
    "scipy": "1.17.1",
    "setuptools": "82.0.1",
    "six": "1.17.0",
    "sounddevice": "0.5.6",
}
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
BANK_SHA256 = "938f02dfaff4d32c914a89bd303f824ca7ced12a6a1b5a0b37280623f7d40540"
ARCHIVE_NAMES = {
    "arm64": "Facial-Analysis-macOS-arm64.zip",
    "x86_64": "Facial-Analysis-macOS-x86_64.zip",
}
ALLOWED_RUNTIME_MEDIA_PREFIXES = (
    "contents/resources/python/lib/python3.11/site-packages/matplotlib/mpl-data/images/",
    "contents/resources/python/lib/python3.11/site-packages/matplotlib/backends/web_backend/",
)
SUPPORTED_SMOKE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png"})
NETWORK_PROFILE = (
    "(version 1) (allow default) (deny network*) "
    '(allow network-inbound (local ip "localhost:*")) '
    '(allow network-outbound (remote ip "localhost:*"))'
)
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
STARTUP_TIMEOUT_SECONDS = 900
FONT_CACHE_FILE = re.compile(r"fontlist-v[0-9]+(?:\.[0-9]+)*\.json")
PROCESS_EXIT_TIMEOUT_SECONDS = 60
AXES = frozenset({"harmony", "dimorphism", "symmetry", "averageness"})


class VerificationError(RuntimeError):
    """A distribution failed a release-blocking verification."""


class _LoopbackBrowser:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.cookies: dict[str, str] = {}

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        request_headers = dict(headers or {})
        if self.cookies:
            request_headers["Cookie"] = "; ".join(
                f"{name}={value}" for name, value in self.cookies.items()
            )
        connection = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=STARTUP_TIMEOUT_SECONDS,
        )
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            payload = response.read()
            response_headers = {
                key.casefold(): value for key, value in response.getheaders()
            }
            for key, value in response.getheaders():
                if key.casefold() != "set-cookie":
                    continue
                pair = value.split(";", 1)[0]
                name, cookie_value = pair.split("=", 1)
                if "max-age=0" in value.casefold():
                    self.cookies.pop(name, None)
                else:
                    self.cookies[name] = cookie_value
            return response.status, response_headers, payload
        finally:
            connection.close()


def _run(command: list[str]) -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory(prefix="facial-analysis-command-") as root:
        command_root = Path(root)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=packaged_environment(command_root / "home", command_root / "tmp"),
        )
    return completed.returncode, completed.stdout, completed.stderr


def _require_success(
    command: list[str],
    *,
    runner: CommandRunner,
    expected_output: str | None = None,
) -> str:
    code, stdout, _stderr = runner(command)
    if code != 0 or (expected_output is not None and expected_output not in stdout):
        raise VerificationError("packaged verification command failed")
    return stdout


def verify_payload_manifest(payload: Path, manifest: Path) -> None:
    """Require exact payload membership and checksums."""
    try:
        bootstrap.verify_manifest(Path(payload), Path(manifest))
    except (OSError, bootstrap.ManifestIntegrityError) as error:
        raise VerificationError("packaged payload manifest did not verify") from error


@lru_cache(maxsize=1)
def _expected_payload_entries() -> tuple[tuple[str, int, str], ...]:
    root = Path(__file__).resolve().parents[1]
    try:
        files = inventory.audited_payload(root)
        return tuple(
            (
                relative.as_posix(),
                (root / relative).stat().st_size,
                _sha256(root / relative),
            )
            for relative in files
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise VerificationError("trusted payload inventory is unavailable") from error


def verify_payload_matches_source(manifest: Path) -> None:
    """Bind the packaged payload manifest to the audited local release source."""
    try:
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        entries = tuple(
            (entry["path"], entry["size"], entry["sha256"])
            for entry in payload["files"]
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise VerificationError("packaged payload manifest is unreadable") from error
    if entries != _expected_payload_entries():
        raise VerificationError("packaged payload differs from audited release source")


def validate_startup_url(value: str) -> str:
    """Accept only the exact numeric-loopback URL shape emitted by the app."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise VerificationError("packaged startup URL is invalid") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/"
        or parsed.query
        or (
            parsed.fragment
            and not re.fullmatch(r"history=[A-Za-z0-9_-]{40,96}", parsed.fragment)
        )
    ):
        raise VerificationError("packaged startup URL is not numeric loopback")
    return value


def _sanitize_report(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _sanitize_report(item)
            for key, item in value.items()
            if key not in PRIVATE_REPORT_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_report(item) for item in value]
    return value


# Cross-architecture agreement. The two builds carry byte-identical formulas,
# reference data and model weights (the manifest proves that), but Apple
# Silicon runs the landmarker natively and the Intel build runs it under
# Rosetta, and floating-point inference does not accumulate in the same order
# on two instruction sets. Measured 2026-09-21 on the nine-photo set: 111 of
# 120 report fields identical, the rest apart by at most 0.3 percentile points
# and 7e-4 relative. Zero tolerance therefore fails a correct release; the bar
# below is the instrument's own display precision -- whole percentile points
# in prose, one decimal in the export -- so a difference a reader could see is
# still a failure, and the measured maxima are returned so drift stays visible.
PERCENTILE_TOLERANCE_POINTS = 1.0
VALUE_RELATIVE_TOLERANCE = 1e-3
PERCENTILE_KEY_MARKERS = ("percentile", "interval")


def _report_leaves(
    value: Any, path: tuple[str, ...] = ()
) -> dict[tuple[str, ...], Any]:
    if isinstance(value, Mapping):
        leaves: dict[tuple[str, ...], Any] = {}
        for key, item in value.items():
            leaves.update(_report_leaves(item, (*path, str(key))))
        return leaves
    if isinstance(value, list):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(_report_leaves(item, (*path, f"[{index}]")))
        return leaves
    return {path: value}


def _is_percentile_scale(path: tuple[str, ...]) -> bool:
    return any(marker in part for part in path for marker in PERCENTILE_KEY_MARKERS)


def compare_sanitized_reports(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, float]:
    """Require the two architectures' reports to agree to display precision.

    Every non-numeric field, every integer and every bool must be identical.
    Percentile-scale numbers may differ by PERCENTILE_TOLERANCE_POINTS and other
    measured values by VALUE_RELATIVE_TOLERANCE. Returns the measured maxima.
    """
    left = _report_leaves(_sanitize_report(first))
    right = _report_leaves(_sanitize_report(second))
    if set(left) != set(right):
        raise VerificationError("packaged architecture reports differ")
    max_points = 0.0
    max_relative = 0.0
    for path, value in left.items():
        other = right[path]
        numeric = (
            isinstance(value, float)
            and isinstance(other, (int, float))
            and not isinstance(other, bool)
        ) or (
            isinstance(other, float)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
        if not numeric:
            if value != other:
                raise VerificationError("packaged architecture reports differ")
            continue
        difference = abs(float(value) - float(other))
        if _is_percentile_scale(path):
            max_points = max(max_points, difference)
            if difference > PERCENTILE_TOLERANCE_POINTS:
                raise VerificationError("packaged architecture reports differ")
            continue
        scale = max(abs(float(value)), abs(float(other)), 1e-12)
        relative = difference / scale
        max_relative = max(max_relative, relative)
        if relative > VALUE_RELATIVE_TOLERANCE:
            raise VerificationError("packaged architecture reports differ")
    return {
        "max_percentile_delta": round(max_points, 4),
        "max_relative_delta": float(f"{max_relative:.3g}"),
    }


def _is_forbidden_path(relative: Path) -> bool:
    lowered_parts = tuple(part.casefold() for part in relative.parts)
    if any(part in FORBIDDEN_NAMES for part in lowered_parts):
        return True
    joined = relative.as_posix().casefold()
    if relative.suffix.casefold() in FORBIDDEN_SUFFIXES and not any(
        joined.startswith(prefix) for prefix in ALLOWED_RUNTIME_MEDIA_PREFIXES
    ):
        return True
    return any(
        marker in joined
        for marker in (
            "subject-report",
            "smoke-photo",
            "private-photo",
            "credentials",
            "id_rsa",
        )
    )


def scan_extracted_app(app: Path) -> None:
    """Reject private/generated material and require the reference bank."""
    app = Path(app)
    payload = app / "Contents" / "Resources" / "payload"
    bank = payload / "bank" / "derived.npz"
    if not app.is_dir() or not payload.is_dir() or not bank.is_file():
        raise VerificationError("packaged application is incomplete")
    for path in app.rglob("*"):
        relative = path.relative_to(app)
        metadata = path.lstat()
        if _is_forbidden_path(relative):
            raise VerificationError("packaged application contains forbidden material")
        if not (
            stat.S_ISREG(metadata.st_mode)
            or stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
        ):
            raise VerificationError("packaged application contains an unsafe entry")


def verify_native_architecture(
    executable: Path,
    architecture: str,
    *,
    runner: CommandRunner = _run,
) -> None:
    """Require the native launcher to contain the requested architecture."""
    if architecture not in {"arm64", "x86_64"}:
        raise VerificationError("unsupported packaged architecture")
    stdout = _require_success(
        ["/usr/bin/file", "-b", "--", str(executable)], runner=runner
    )
    normalized = stdout.casefold().replace("x86-64", "x86_64")
    if architecture.casefold() not in normalized:
        raise VerificationError("packaged launcher has the wrong architecture")


def _architecture_command(architecture: str, command: Sequence[str]) -> list[str]:
    if architecture == "x86_64":
        return ["/usr/bin/arch", "-x86_64", *command]
    if architecture != "arm64":
        raise VerificationError("unsupported packaged architecture")
    return list(command)


def network_sandbox_command(
    architecture: str,
    command: Sequence[str],
    *,
    profile: str = NETWORK_PROFILE,
) -> list[str]:
    """Wrap a command in the macOS network sandbox, permitting loopback only."""
    sandbox = Path("/usr/bin/sandbox-exec")
    if not sandbox.is_file():
        raise VerificationError("the macOS network sandbox is unavailable")
    return [
        os.fspath(sandbox),
        "-p",
        profile,
        *_architecture_command(architecture, command),
    ]


def packaged_sandbox_profile(app: Path, home: Path, temp_root: Path) -> str:
    """Deny external networking, host-home reads, and writes outside private roots."""

    def literal(path: Path) -> str:
        return json.dumps(os.fspath(Path(path).resolve()))

    return " ".join(
        (
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            '(allow network-inbound (local ip "localhost:*"))',
            "(deny file-write*)",
            f"(allow file-write* (subpath {literal(home)}) (subpath {literal(temp_root)}))",
            f"(deny file-read* (subpath {literal(Path.home())}))",
            (
                f"(allow file-read* (subpath {literal(app)}) "
                f"(subpath {literal(home)}) (subpath {literal(temp_root)}))"
            ),
        )
    )


def smoke_photo_paths() -> tuple[Path, ...]:
    """Resolve the private smoke inputs without logging or persisting their names."""
    raw = os.environ.get("FA_MACOS_SMOKE_PHOTOS")
    if not raw:
        raise VerificationError("the private package smoke input is not configured")
    lexical = Path(raw).expanduser().absolute()
    current = Path(lexical.anchor)
    try:
        for part in lexical.parts[1:]:
            current /= part
            if current.is_symlink():
                if current == Path("/var") and current.lstat().st_uid == 0:
                    continue
                raise VerificationError("the private package smoke input is unsafe")
    except OSError as error:
        raise VerificationError(
            "the private package smoke input is unavailable"
        ) from error
    root = lexical.resolve()
    if not root.is_dir():
        raise VerificationError("the private package smoke input is unavailable")
    try:
        photos = tuple(
            sorted(
                (
                    path
                    for path in root.iterdir()
                    if path.is_file()
                    and not path.is_symlink()
                    and path.suffix.casefold() in SUPPORTED_SMOKE_SUFFIXES
                ),
                key=lambda path: path.name.casefold(),
            )
        )
    except OSError as error:
        raise VerificationError(
            "the private package smoke input is unreadable"
        ) from error
    if len(photos) != 9:
        raise VerificationError(
            "the private package smoke input must contain nine images"
        )
    return photos


def _normalized_words(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "_".join(
        value.strip().casefold().replace("/", " ").replace("-", " ").split()
    )


def load_smoke_demographics(path: Path) -> tuple[str, str, tuple[str, ...]]:
    """Load user-entered smoke demographics; never infer them from photographs."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise VerificationError("private smoke demographics are unavailable") from error
    if not isinstance(payload, dict):
        raise VerificationError("private smoke demographics are invalid")
    age = payload.get("age")
    if isinstance(age, bool) or not isinstance(age, int) or not 18 <= age <= 120:
        raise VerificationError("private smoke demographics are invalid")

    pronoun = _normalized_words(payload.get("pronoun"))
    if pronoun.startswith(("she", "her", "female", "woman")):
        sex = "female"
    elif pronoun.startswith(("he", "him", "male", "man")):
        sex = "male"
    elif pronoun.startswith(("they", "them", "nonbinary", "non_binary")):
        sex = "another"
    else:
        sex = "prefer_not_to_say"

    race = _normalized_words(payload.get("race"))
    aliases = {
        "white": "white",
        "black": "black",
        "east_asian": "east_asian",
        "southeast_asian": "southeast_asian",
        "south_east_asian": "southeast_asian",
        "indian": "indian",
        "middle_eastern": "middle_eastern",
        "latino": "latino_hispanic",
        "hispanic": "latino_hispanic",
        "latino_hispanic": "latino_hispanic",
        "mixed": "mixed",
        "multiple_ethnicities": "mixed",
    }
    ethnicity = aliases.get(race)
    if ethnicity is None:
        raise VerificationError("private smoke demographics are invalid")
    return str(age), sex, (ethnicity,)


def _multipart(
    fields: Sequence[tuple[str, str]],
    files: Sequence[tuple[str, str, bytes]],
) -> tuple[bytes, str]:
    boundary = "----facial-analysis-packaged-smoke"
    parts: list[bytes] = []
    for name, value in fields:
        parts.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            )
        )
    for name, filename, content in files:
        content_type = "image/png" if filename.endswith(".png") else "image/jpeg"
        parts.extend(
            (
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            )
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _csrf(page: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', page)
    if match is None:
        raise VerificationError("the packaged page omitted its request token")
    return match.group(1)


def _expect_status(actual: int, expected: int) -> None:
    if actual != expected:
        raise VerificationError("the packaged browser journey failed")


def _encoded_tokens(values: Sequence[str]) -> frozenset[bytes]:
    encoded: set[bytes] = set()
    for value in values:
        if not value:
            continue
        variants = {
            value,
            quote(value),
            quote(value, safe=""),
            html.escape(value, quote=True),
            json.dumps(value, ensure_ascii=True)[1:-1],
            base64.b64encode(value.encode("utf-8")).decode("ascii"),
            base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii"),
        }
        encoded.update(item.encode("utf-8") for item in variants if item)
    return frozenset(encoded)


def verify_export_privacy(
    serialized: bytes,
    sensitive_strings: Sequence[str],
    sensitive_blobs: Sequence[bytes],
) -> None:
    """Reject raw or reversibly encoded private metadata and image content."""
    blob_canaries: set[bytes] = set()
    for content in sensitive_blobs:
        if len(content) < 192:
            continue
        prefix = content[:192]
        blob_canaries.update(
            {
                prefix,
                base64.b64encode(prefix),
                base64.urlsafe_b64encode(prefix),
            }
        )
    if any(token in serialized for token in _encoded_tokens(sensitive_strings)) or any(
        blob in serialized for blob in blob_canaries
    ):
        raise VerificationError("the packaged export leaked private metadata")


def packaged_environment(home: Path, temp_root: Path) -> dict[str, str]:
    """Return the minimal environment allowed to reach packaged executables."""
    home = Path(home)
    temp_root = Path(temp_root)
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return {
        "HOME": os.fspath(home),
        "TMPDIR": os.fspath(temp_root),
        "PATH": SYSTEM_PATH,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        # The verifier's own dependency probes import matplotlib, and a bare
        # import already creates its config directory. Point them where the
        # app itself keeps that state, or the probe -- not the app -- leaves
        # ~/.matplotlib behind and the cleanup check reads it as a leak.
        "MPLCONFIGDIR": os.fspath(
            bootstrap.application_support(home) / bootstrap.FONT_CACHE_DIRNAME
        ),
    }


def _startup_url(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        raise VerificationError("the packaged process output is unavailable")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    descriptor = process.stdout.fileno()
    os.set_blocking(descriptor, False)
    buffer = b""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            events = selector.select(timeout=min(1.0, deadline - time.monotonic()))
            if events:
                try:
                    chunk = os.read(descriptor, 4096)
                except BlockingIOError:
                    chunk = b""
                buffer += chunk
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="replace")
                    if line.startswith("LOCAL_URL="):
                        return validate_startup_url(
                            line.removeprefix("LOCAL_URL=").strip()
                        )
            if process.poll() is not None:
                raise VerificationError(
                    "the packaged application exited during startup"
                )
    finally:
        selector.close()
    raise VerificationError("the packaged application did not become ready")


def stop_process(process: subprocess.Popen[str]) -> None:
    """Stop the owned process group progressively, waiting after every signal."""
    if process.poll() is None:
        for stop_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, stop_signal)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                continue
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# Exactly what launcher.swift passes: -I ignores the environment (including
# PYTHONDONTWRITEBYTECODE), so -B is what keeps bytecode out of the verified
# runtime. The verifier must launch the very same command or it verifies a
# different program (2026-09-21: -I alone wrote __pycache__ and failed start-up).
PACKAGED_INTERPRETER_FLAGS = ("-I", "-B")


def packaged_launch_command(resources: Path) -> list[str]:
    """The interpreter command the shipped launcher runs, for the verifier to mirror."""
    resources = Path(resources)
    return [
        os.fspath(resources / "python" / "bin" / "python3"),
        *PACKAGED_INTERPRETER_FLAGS,
        os.fspath(resources / "bootstrap.py"),
    ]


def _launch_packaged_server(
    app: Path,
    architecture: str,
    *,
    environment: Mapping[str, str],
    sandbox_profile: str,
) -> tuple[subprocess.Popen[str], str]:
    resources = Path(app) / "Contents" / "Resources"
    base_command = packaged_launch_command(resources)
    command = network_sandbox_command(
        architecture,
        base_command,
        profile=sandbox_profile,
    )
    process = subprocess.Popen(
        command,
        cwd=resources,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    try:
        return process, _startup_url(process)
    except BaseException:
        stop_process(process)
        raise


def _verify_network_denial(
    python: Path,
    architecture: str,
    environment: Mapping[str, str],
    sandbox_profile: str,
) -> None:
    probe = (
        "import socket,sys; s=socket.socket(); s.settimeout(1);"
        "\ntry: s.connect(('192.0.2.1',9))"
        "\nexcept OSError as e: sys.exit(0 if e.errno in (1,13) else 2)"
        "\nelse: sys.exit(3)"
    )
    command = network_sandbox_command(
        architecture,
        [os.fspath(python), "-I", "-B", "-c", probe],
        profile=sandbox_profile,
    )
    try:
        completed = subprocess.run(
            command,
            env=dict(environment),
            cwd=environment["TMPDIR"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise VerificationError("the packaged network-denial probe failed") from error
    if completed.returncode != 0:
        raise VerificationError("the packaged network sandbox was not enforced")


def _sandbox_runner(
    architecture: str,
    environment: Mapping[str, str],
    sandbox_profile: str,
) -> CommandRunner:
    def run(command: list[str]) -> tuple[int, str, str]:
        base_command = command
        if architecture == "x86_64" and command[:2] == [
            "/usr/bin/arch",
            "-x86_64",
        ]:
            base_command = command[2:]
        wrapped = network_sandbox_command(
            architecture,
            base_command,
            profile=sandbox_profile,
        )
        completed = subprocess.run(
            wrapped,
            check=False,
            capture_output=True,
            text=True,
            env=dict(environment),
            cwd=environment["TMPDIR"],
        )
        return completed.returncode, completed.stdout, completed.stderr

    return run


def _verify_host_home_denial(
    architecture: str,
    environment: Mapping[str, str],
    sandbox_profile: str,
) -> None:
    protected = Path(__file__).resolve().parents[1] / "README.md"
    if not protected.is_file():
        raise VerificationError("the filesystem-denial probe is unavailable")
    probe = (
        "import pathlib,sys;"
        "\ntry: pathlib.Path(sys.argv[1]).read_bytes()"
        "\nexcept PermissionError: sys.exit(0)"
        "\nexcept OSError as e: sys.exit(0 if e.errno in (1,13) else 2)"
        "\nelse: sys.exit(3)"
    )
    command = network_sandbox_command(
        architecture,
        ["/usr/bin/python3", "-I", "-c", probe, os.fspath(protected)],
        profile=sandbox_profile,
    )
    try:
        completed = subprocess.run(
            command,
            env=dict(environment),
            cwd=environment["TMPDIR"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise VerificationError(
            "the packaged filesystem-denial probe failed"
        ) from error
    if completed.returncode != 0:
        raise VerificationError("the packaged filesystem sandbox was not enforced")


def _verify_self_confinement(
    app: Path,
    architecture: str,
    environment: Mapping[str, str],
) -> None:
    """Launch exactly as the shipped launcher does -- with no verifier sandbox --
    and ask the kernel whether the serving process confined its own networking.

    The measurement library posts usage events to Google over NSURLSession, which
    no Python-level probe can see. The bootstrap therefore re-executes itself
    under the loopback-only sandbox before accepting photographs; this is the
    outside-the-process check that it really did (2026-09-21)."""
    resources = Path(app) / "Contents" / "Resources"
    command = _architecture_command(architecture, packaged_launch_command(resources))
    process = subprocess.Popen(
        command,
        cwd=resources,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    try:
        startup_url = _startup_url(process)
        if not bootstrap.outbound_network_denied(process.pid):
            raise VerificationError(
                "the packaged application did not confine its own networking"
            )
        parsed = urlsplit(startup_url)
        if parsed.hostname != "127.0.0.1" or parsed.port is None:
            raise VerificationError("the packaged application did not use loopback")
        status, _headers, _body = _LoopbackBrowser(
            parsed.hostname, parsed.port
        ).request("GET", "/")
        _expect_status(status, 200)
    finally:
        stop_process(process)
    if process.returncode != 0:
        raise VerificationError(
            "the self-confined packaged application did not exit cleanly"
        )


def verify_loopback_listener(
    process: subprocess.Popen[str],
    port: int,
    *,
    runner: CommandRunner = _run,
) -> None:
    code, stdout, _stderr = runner(
        [
            "/usr/sbin/lsof",
            "-nP",
            "-a",
            "-p",
            str(process.pid),
            "-iTCP",
            "-sTCP:LISTEN",
            "-Fn",
        ]
    )
    addresses = tuple(line[1:] for line in stdout.splitlines() if line.startswith("n"))
    expected = f"127.0.0.1:{port}"
    if code != 0 or addresses != (expected,):
        raise VerificationError("the packaged server is not bound only to loopback")


def _prepare_packaged_runtime(app: Path, home: Path) -> Path:
    """Install payload and pinned assets using trusted verifier code, not package code."""
    resources = Path(app) / "Contents" / "Resources"
    payload = resources / "payload"
    manifest = payload / "manifest.json"
    runtime = bootstrap.application_support(home) / "runtime" / f"v{APP_VERSION}"
    try:
        installed = bootstrap.install_runtime(payload, runtime, manifest)
        for name, download in MODEL_ASSETS.items():
            bootstrap.ensure_asset(installed / name, download)
    except (OSError, RuntimeError, ValueError) as error:
        raise VerificationError("the trusted packaged preflight failed") from error
    return installed


def _verify_private_roots(
    home: Path,
    temp_root: Path,
    installed: Path,
    *,
    expected_archives: Mapping[str, str] | None = None,
) -> None:
    """Allow only pinned runtime, font cache and explicitly verified encrypted saves."""
    verify_session_cleanup(temp_root)
    # One canonical form for every path compared below. `installed` arrives
    # resolved (/private/var/...) while a caller's `home` is usually the
    # unresolved /var/... macOS temp path; until 2026-09-21 that mismatch made
    # the first entry under home read as retained material, so this check
    # could never pass on macOS and no journey had ever reached a clean exit.
    home = Path(home).resolve()
    installed = Path(installed).resolve()
    try:
        if any(Path(temp_root).iterdir()):
            raise VerificationError(
                "the packaged application retained temporary material"
            )
        bootstrap.verify_manifest(
            installed,
            installed / "manifest.json",
            allowed_extra_files=frozenset(Path(name) for name in MODEL_ASSETS),
        )
        permitted_parents = set(installed.parents)
        permitted_parents.add(installed)
        font_cache = bootstrap.application_support(home) / bootstrap.FONT_CACHE_DIRNAME
        history = bootstrap.application_support(home) / "history"
        expected_archives = dict(expected_archives or {})
        if any(
            not re.fullmatch(r"[a-f0-9]{32}\.fae", name)
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
            for name, digest in expected_archives.items()
        ):
            raise VerificationError("invalid expected archive inventory")
        if history.is_symlink():
            raise VerificationError("history directory must not be a symlink")
        if history.exists():
            if (
                history.is_symlink()
                or not history.is_dir()
                or stat.S_IMODE(history.stat().st_mode) != 0o700
            ):
                raise VerificationError(
                    "history directory permissions or type did not verify"
                )
            if {p.name for p in history.iterdir()} - {".lock"} != set(
                expected_archives
            ):
                raise VerificationError("history contains undeclared saved material")
        elif expected_archives:
            raise VerificationError("expected saved history is missing")
        for path in home.rglob("*"):
            if path == history:
                continue
            if path.parent == history:
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or stat.S_IMODE(path.stat().st_mode) != 0o600
                ):
                    raise VerificationError(
                        "history file permissions or type did not verify"
                    )
                if path.name == ".lock" and path.stat().st_size == 0:
                    continue
                if (
                    path.name in expected_archives
                    and 33 <= path.stat().st_size <= 6_000_000
                    and _sha256(path) == expected_archives[path.name]
                ):
                    with path.open("rb") as stream:
                        if stream.read(5) == b"FAH1\0":
                            continue
                raise VerificationError(
                    "saved history does not match its explicit verification inventory"
                )
            if path == installed or installed in path.parents:
                continue
            if path in permitted_parents or path == font_cache:
                continue
            # The one write the app makes outside its pinned runtime: the
            # matplotlib font list, which names system fonts and nothing about
            # the person. Anything else under that directory is a leak.
            if (
                path.parent == font_cache
                and path.is_file()
                and not path.is_symlink()
                and FONT_CACHE_FILE.fullmatch(path.name)
            ):
                continue
            raise VerificationError(
                "the packaged application retained private material"
            )
    except (OSError, bootstrap.ManifestIntegrityError) as error:
        if isinstance(error, VerificationError):
            raise
        raise VerificationError("the packaged writable roots did not verify") from error


def verify_bank_guard(
    python: Path,
    payload: Path,
    architecture: str,
    *,
    runner: CommandRunner = _run,
) -> None:
    """Run the bank instrument guard inside the embedded interpreter."""
    script = (
        "import sys;"
        f"sys.path.insert(0,{os.fspath(Path(payload).resolve())!r});"
        "import bank_reference as bank;"
        "bank.cohort_inventory();"
        "message=bank._load().get('instrument_ok');"
        "assert message == 'instrument matches the bank', message;"
        "print(message)"
    )
    command = _architecture_command(
        architecture,
        [str(python), "-I", "-B", "-c", script],
    )
    _require_success(
        command,
        runner=runner,
        expected_output="instrument matches the bank",
    )


def verify_embedded_dependencies(
    python: Path,
    architecture: str,
    *,
    runner: CommandRunner = _run,
) -> None:
    """Require the frozen scientific dependency versions."""
    script = """
import importlib.metadata as metadata
import json
import re
import cv2, mediapipe, numpy, scipy
modules = {
    'cv2': cv2.__version__,
    'mediapipe': mediapipe.__version__,
    'numpy': numpy.__version__,
    'scipy': scipy.__version__,
}
distributions = {
    re.sub(r'[-_.]+', '-', item.metadata['Name']).lower(): item.version
    for item in metadata.distributions()
}
print(json.dumps({'modules': modules, 'distributions': distributions}, sort_keys=True))
"""
    stdout = _require_success(
        _architecture_command(architecture, [str(python), "-I", "-B", "-c", script]),
        runner=runner,
    )
    try:
        versions = json.loads(stdout.strip())
    except (TypeError, ValueError) as error:
        raise VerificationError("packaged dependency probe was unreadable") from error
    if versions != {
        "modules": EXPECTED_DEPENDENCIES,
        "distributions": EXPECTED_DISTRIBUTIONS,
    }:
        raise VerificationError("packaged dependency versions differ")


def verify_measurement_library_loads(
    python: Path,
    installed: Path,
    architecture: str,
    *,
    runner: CommandRunner = _run,
) -> None:
    """Require the packaged MediaPipe Tasks library to load and build a landmarker.

    Importing mediapipe proves only the Python half. The native Tasks library is
    dlopened on first use, and the Intel build shipped one that could not be
    found, then one that could not be loaded (2026-09-21) -- so the release gate
    constructs the app's own detector from the installed model asset, in the
    embedded interpreter, before any photograph is involved.
    """
    model = Path(installed) / "face_landmarker.task"
    script = f"""
import ctypes, importlib.resources as resources
from mediapipe.tasks.python import BaseOptions, vision
library = ctypes.CDLL(str(resources.files('mediapipe.tasks.c') / 'libmediapipe.dylib'))
assert hasattr(library, 'MpErrorFree'), 'Tasks library lacks MpErrorFree'
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path={os.fspath(model)!r}),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    output_facial_transformation_matrixes=True,
    output_face_blendshapes=True,
)
with vision.FaceLandmarker.create_from_options(options):
    print('landmarker ready')
"""
    code, stdout, _stderr = runner(
        _architecture_command(architecture, [str(python), "-I", "-B", "-c", script])
    )
    if code != 0 or "landmarker ready" not in stdout:
        raise VerificationError("the packaged measurement library did not load")


def verify_info_plist(app: Path, architecture: str) -> None:
    try:
        with (Path(app) / "Contents" / "Info.plist").open("rb") as stream:
            info = plistlib.load(stream)
    except (OSError, ValueError) as error:
        raise VerificationError("packaged Info.plist is unreadable") from error
    required = {
        "CFBundleShortVersionString": APP_VERSION,
        "LSMinimumSystemVersion": MINIMUM_MACOS,
        "FAArchitecture": architecture,
    }
    if any(info.get(key) != value for key, value in required.items()):
        raise VerificationError("packaged Info.plist differs from release settings")


def verify_codesign(app: Path, *, runner: CommandRunner = _run) -> None:
    _require_success(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)],
        runner=runner,
    )


def verify_file_modes(app: Path) -> None:
    launcher = Path(app) / "Contents" / "MacOS" / "Facial Analysis"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise VerificationError("packaged launcher is not executable")
    for path in (Path(app), *Path(app).rglob("*")):
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o022:
            raise VerificationError("packaged application is group/world writable")


def verify_session_cleanup(temp_root: Path) -> None:
    leftovers = tuple(Path(temp_root).glob(f"{bootstrap.SESSION_PREFIX}*"))
    if leftovers:
        raise VerificationError("packaged application left session material")


def parse_checksum_file(path: Path) -> dict[str, str]:
    """Parse a strict two-space-separated SHA256SUMS file."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise VerificationError("release checksum file is unreadable") from error
    result: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if match is None or match.group(2) in result:
            raise VerificationError("release checksum file is invalid")
        result[match.group(2)] = match.group(1)
    if not result:
        raise VerificationError("release checksum file is empty")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise VerificationError("release artifact is unreadable") from error
    return digest.hexdigest()


def verify_release_checksums(dist: Path) -> dict[str, str]:
    """Verify every strictly named artifact in SHA256SUMS.txt."""
    dist = Path(dist).resolve()
    checksum_path = dist / "SHA256SUMS.txt"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise VerificationError("release checksum file is unsafe")
    checksums = parse_checksum_file(checksum_path)
    try:
        entries = tuple(dist.iterdir())
    except OSError as error:
        raise VerificationError("release directory is unreadable") from error
    if any(not path.is_file() or path.is_symlink() for path in entries) or {
        path.name for path in entries
    } != {"SHA256SUMS.txt", *checksums}:
        raise VerificationError("release directory contains unexpected material")
    for name, expected in checksums.items():
        path = dist / name
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise VerificationError("release artifact checksum differs")
    return checksums


def _zip_relative_path(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise VerificationError("release archive contains an unsafe path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise VerificationError("release archive contains an unsafe path")
    return relative


def _contained_link(relative: PurePosixPath, value: bytes) -> str:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError("release archive contains an unsafe link") from error
    if not text or "\\" in text or "\0" in text or len(text) > 4096:
        raise VerificationError("release archive contains an unsafe link")
    target = PurePosixPath(text)
    if target.is_absolute():
        raise VerificationError("release archive contains an unsafe link")
    parts: list[str] = []
    for part in (relative.parent / target).parts:
        if part == "..":
            if not parts:
                raise VerificationError("release archive link escapes its root")
            parts.pop()
        elif part not in {"", "."}:
            parts.append(part)
    return text


def extract_release_archive(archive: Path, destination: Path) -> Path:
    """Safely extract a deterministic app ZIP while preserving contained links."""
    archive = Path(archive).resolve()
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise VerificationError("release extraction destination is not empty")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as stream:
            entries: list[tuple[zipfile.ZipInfo, PurePosixPath, int]] = []
            names: set[PurePosixPath] = set()
            total = 0
            for info in stream.infolist():
                relative = _zip_relative_path(info.filename)
                if relative in names:
                    raise VerificationError(
                        "release archive contains duplicate entries"
                    )
                names.add(relative)
                total += info.file_size
                if total > MAX_ARCHIVE_BYTES:
                    raise VerificationError("release archive is too large")
                mode = info.external_attr >> 16
                kind = stat.S_IFMT(mode)
                is_directory = info.is_dir() or kind == stat.S_IFDIR
                is_symlink = kind == stat.S_IFLNK
                if not (is_directory or is_symlink or kind in {0, stat.S_IFREG}):
                    raise VerificationError("release archive has an unsupported entry")
                entries.append((info, relative, mode))

            links: list[tuple[Path, str]] = []
            for info, relative, mode in entries:
                target = destination.joinpath(*relative.parts)
                kind = stat.S_IFMT(mode)
                if info.is_dir() or kind == stat.S_IFDIR:
                    target.mkdir(mode=0o755, parents=True, exist_ok=True)
                    os.chmod(target, stat.S_IMODE(mode) or 0o755)
                    continue
                target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                if kind == stat.S_IFLNK:
                    value = stream.read(info)
                    links.append((target, _contained_link(relative, value)))
                    continue
                with stream.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.chmod(target, stat.S_IMODE(mode) or 0o644)
            for target, link in links:
                os.symlink(link, target)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, VerificationError):
            raise
        raise VerificationError("release archive could not be extracted") from error
    return destination


def verify_extraction_root(destination: Path) -> Path:
    """Require one self-contained app bundle and no archive siblings."""
    destination = Path(destination).resolve()
    app = destination / "Facial Analysis.app"
    try:
        entries = tuple(destination.iterdir())
    except OSError as error:
        raise VerificationError("release extraction root is unreadable") from error
    if entries != (app,) or not app.is_dir() or app.is_symlink():
        raise VerificationError("release archive root contains unexpected material")
    app_root = app.resolve()
    for path in app.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(app_root)
        except (OSError, RuntimeError, ValueError) as error:
            raise VerificationError("release app contains an escaping link") from error
    return app


def verify_application(
    app: Path,
    architecture: str,
    *,
    runner: CommandRunner = _run,
) -> None:
    """Verify one extracted application's static and embedded-runtime contract."""
    app = Path(app)
    resources = app / "Contents" / "Resources"
    payload = resources / "payload"
    python = resources / "python" / "bin" / "python3"
    launcher = app / "Contents" / "MacOS" / "Facial Analysis"
    bank = payload / "bank" / "derived.npz"
    scan_extracted_app(app)
    verify_info_plist(app, architecture)
    verify_file_modes(app)
    verify_codesign(app, runner=runner)
    verify_native_architecture(launcher, architecture, runner=runner)
    verify_native_architecture(python, architecture, runner=runner)
    verify_payload_manifest(payload, payload / "manifest.json")
    verify_payload_matches_source(payload / "manifest.json")
    if _sha256(bank) != BANK_SHA256:
        raise VerificationError("packaged reference bank differs")


def verify_axis_outcomes(
    axes: Any,
    expected_statuses: Mapping[str, str] | None = None,
) -> None:
    """Match frozen smoke expectations; retain the measured-only release default."""
    expected = (
        dict.fromkeys(AXES, "measured")
        if expected_statuses is None
        else dict(expected_statuses)
    )
    allowed = {"measured", "estimated", "needs_photo"}
    if (
        set(expected) != AXES
        or any(
            not isinstance(value, str) or value not in allowed
            for value in expected.values()
        )
        or not any(value in {"measured", "estimated"} for value in expected.values())
    ):
        raise VerificationError("packaged outcome expectations are invalid")
    if not isinstance(axes, dict) or set(axes) != AXES:
        raise VerificationError("the packaged report is missing capabilities")
    if any(
        not isinstance(result, dict) or result.get("status") != expected[axis]
        for axis, result in axes.items()
    ):
        raise VerificationError(
            "the packaged report differs from its expected outcomes"
        )


def run_packaged_journey(
    app: Path,
    architecture: str,
    photos: Sequence[Path],
    demographics: tuple[str, str, tuple[str, ...]],
    *,
    home: Path,
    temp_root: Path,
    expected_archives: Mapping[str, str] | None = None,
    expected_axis_statuses: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one real packaged browser journey with networking and host-home denied."""
    app = Path(app).resolve()
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = packaged_environment(home, temp_root)
    installed = _prepare_packaged_runtime(app, home)
    sandbox_profile = packaged_sandbox_profile(app, home, temp_root)
    python = app / "Contents" / "Resources" / "python" / "bin" / "python3"
    sandbox_runner = _sandbox_runner(architecture, environment, sandbox_profile)
    verify_embedded_dependencies(python, architecture, runner=sandbox_runner)
    verify_measurement_library_loads(
        python, installed, architecture, runner=sandbox_runner
    )
    verify_bank_guard(python, installed, architecture, runner=sandbox_runner)
    _verify_network_denial(python, architecture, environment, sandbox_profile)
    _verify_host_home_denial(architecture, environment, sandbox_profile)
    _verify_self_confinement(app, architecture, environment)
    process: subprocess.Popen[str] | None = None
    session_root: Path | None = None
    try:
        process, startup_url = _launch_packaged_server(
            app,
            architecture,
            environment=environment,
            sandbox_profile=sandbox_profile,
        )
        parsed = urlsplit(startup_url)
        if parsed.hostname != "127.0.0.1" or parsed.port is None:
            raise VerificationError("the packaged application did not use loopback")
        verify_loopback_listener(process, parsed.port)
        browser = _LoopbackBrowser(parsed.hostname, parsed.port)
        status, _headers, landing_bytes = browser.request("GET", "/")
        _expect_status(status, 200)
        landing = landing_bytes.decode("utf-8")
        landing_token = _csrf(landing)
        age, sex, ethnicities = demographics
        candidates = tuple(temp_root.glob(f"{bootstrap.SESSION_PREFIX}*"))
        if len(candidates) != 1:
            raise VerificationError("the packaged session workspace is invalid")
        session_root = candidates[0]

        invalid_fields = [
            ("csrf_token", "invalid-token"),
            ("age", age),
            ("sex", sex),
            *(("ethnicity", ethnicity) for ethnicity in ethnicities),
        ]
        invalid_body, invalid_type = _multipart(invalid_fields, ())
        origin = f"http://127.0.0.1:{parsed.port}"
        status, _headers, _body = browser.request(
            "POST",
            "/sessions",
            body=invalid_body,
            headers={"Content-Type": invalid_type, "Origin": origin},
        )
        _expect_status(status, 403)
        hostile_fields = [
            ("csrf_token", landing_token),
            ("age", age),
            ("sex", sex),
            *(("ethnicity", ethnicity) for ethnicity in ethnicities),
        ]
        hostile_body, hostile_type = _multipart(hostile_fields, ())
        status, _headers, _body = browser.request(
            "POST",
            "/sessions",
            body=hostile_body,
            headers={
                "Content-Type": hostile_type,
                "Origin": "https://attacker.invalid",
            },
        )
        _expect_status(status, 403)
        if any(path.is_dir() for path in session_root.iterdir()):
            raise VerificationError("invalid packaged requests created a session")

        upload_files: list[tuple[str, str, bytes]] = []
        sensitive_names: list[str] = []
        transmitted_names: list[str] = []
        for index, path in enumerate(photos, start=1):
            try:
                content = Path(path).read_bytes()
            except OSError as error:
                raise VerificationError(
                    "a private smoke image is unreadable"
                ) from error
            if not content:
                raise VerificationError("a private smoke image is empty")
            suffix = Path(path).suffix.casefold()
            transmitted_name = f"smoke-{index:02d}{suffix}"
            upload_files.append(("photos", transmitted_name, content))
            sensitive_names.append(Path(path).name)
            transmitted_names.append(transmitted_name)

        fields = [("csrf_token", landing_token), ("age", age), ("sex", sex)]
        fields.extend(("ethnicity", ethnicity) for ethnicity in ethnicities)
        body, content_type = _multipart(fields, upload_files)
        status, headers, _body = browser.request(
            "POST",
            "/sessions",
            body=body,
            headers={"Content-Type": content_type, "Origin": origin},
        )
        _expect_status(status, 303)
        results_path = headers.get("location", "")
        match = re.fullmatch(
            r"/sessions/([A-Za-z0-9_-]{32,96})/results",
            results_path,
        )
        if match is None:
            raise VerificationError("the packaged session route is invalid")
        session_id = match.group(1)

        status, _headers, result_bytes = browser.request("GET", results_path)
        _expect_status(status, 200)
        result_page = result_bytes.decode("utf-8")
        result_token = _csrf(result_page)
        status, _headers, json_bytes = browser.request(
            "GET", f"/sessions/{session_id}/report.json"
        )
        _expect_status(status, 200)
        try:
            report = json.loads(json_bytes)
        except (TypeError, ValueError) as error:
            raise VerificationError("the packaged JSON export is invalid") from error
        if not isinstance(report, dict):
            raise VerificationError("the packaged JSON export is invalid")
        if report.get("accepted_photo_count") != len(photos):
            raise VerificationError("the packaged app did not accept every smoke image")
        verify_axis_outcomes(report.get("axes"), expected_axis_statuses)

        status, _headers, html_bytes = browser.request(
            "GET", f"/sessions/{session_id}/report.html"
        )
        _expect_status(status, 200)
        if b"@media print" not in html_bytes:
            raise VerificationError("the packaged HTML export is incomplete")
        serialized = json_bytes + b"\n" + html_bytes
        sensitive = [
            os.fspath(Path(photos[0]).parent.resolve()),
            os.fspath(session_root),
            session_id,
            landing_token,
            result_token,
            *sensitive_names,
            *transmitted_names,
        ]
        verify_export_privacy(
            serialized,
            sensitive,
            tuple(content for _field, _filename, content in upload_files),
        )

        end_body = urlencode({"csrf_token": result_token})
        status, _headers, _body = browser.request(
            "POST",
            f"/sessions/{session_id}/end",
            body=end_body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": origin,
            },
        )
        _expect_status(status, 200)
        try:
            process.wait(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise VerificationError("the packaged application did not quit") from error
        if process.returncode != 0:
            raise VerificationError("the packaged application quit unsuccessfully")
        verify_session_cleanup(temp_root)
        return _sanitize_report(report)
    except BaseException as failure:
        # The journey itself failed. The writable roots are in whatever state the
        # abort left them, so the cleanup verdict below is context, not the
        # diagnosis -- attach it to the real error instead of replacing it.
        if process is not None:
            stop_process(process)
            process = None
        try:
            _verify_private_roots(
                home, temp_root, installed, expected_archives=expected_archives
            )
        except VerificationError as cleanup:
            failure.add_note(
                f"after the failure, cleanup verification also failed: {cleanup}"
            )
        raise
    finally:
        if process is not None:
            stop_process(process)
            try:
                _verify_private_roots(
                    home, temp_root, installed, expected_archives=expected_archives
                )
            except VerificationError:
                raise VerificationError(
                    "the packaged application retained session material"
                ) from None


def verify_distribution_archives(
    dist: Path,
    *,
    runner: CommandRunner = _run,
) -> dict[str, int]:
    """Verify both release archives exactly as a collaborator receives them."""
    dist = Path(dist).expanduser().resolve()
    checksums = verify_release_checksums(dist)
    if set(checksums) != set(ARCHIVE_NAMES.values()):
        raise VerificationError("release checksum inventory is incomplete")
    photos = smoke_photo_paths()
    subject_path = Path(__file__).resolve().parents[1] / "subject.json"
    demographics = load_smoke_demographics(subject_path)
    verified = 0
    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="facial-analysis-release-check-") as root:
        extraction_root = Path(root)
        home = extraction_root / "private-home"
        for architecture, name in ARCHIVE_NAMES.items():
            destination = extraction_root / architecture
            extract_release_archive(dist / name, destination)
            app = verify_extraction_root(destination)
            verify_application(app, architecture, runner=runner)
            reports.append(
                run_packaged_journey(
                    app,
                    architecture,
                    photos,
                    demographics,
                    home=home,
                    temp_root=extraction_root / f"private-temp-{architecture}",
                )
            )
            verified += 1
        agreement = compare_sanitized_reports(reports[0], reports[1])
    return {
        "architectures": verified,
        "archives": len(checksums),
        "parity": 1,
        "network_denial": verified,
        "exports": verified * 2,
        "sessions_deleted": verified,
        **agreement,
    }


__all__ = [
    "VerificationError",
    "compare_sanitized_reports",
    "extract_release_archive",
    "parse_checksum_file",
    "load_smoke_demographics",
    "network_sandbox_command",
    "run_packaged_journey",
    "scan_extracted_app",
    "smoke_photo_paths",
    "validate_startup_url",
    "verify_bank_guard",
    "verify_codesign",
    "verify_embedded_dependencies",
    "verify_export_privacy",
    "verify_extraction_root",
    "verify_file_modes",
    "verify_info_plist",
    "verify_loopback_listener",
    "verify_application",
    "verify_distribution_archives",
    "verify_native_architecture",
    "verify_payload_manifest",
    "verify_release_checksums",
    "verify_session_cleanup",
]
