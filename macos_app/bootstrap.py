"""Private runtime installation and startup for the packaged macOS app.

This file is deliberately executable as a standalone script. The native launcher
runs ``python3 Contents/Resources/bootstrap.py`` before the verified application
payload has been installed or placed on ``sys.path``.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from types import FrameType, ModuleType
from typing import TYPE_CHECKING
from urllib.request import urlopen

if TYPE_CHECKING:
    from macos_app.config import Download

if __package__:
    from .config import APP_VERSION, MODEL_ASSETS
else:
    APP_VERSION = None
    MODEL_ASSETS = None


SESSION_PREFIX = "facial-analysis-packaged-"
SESSION_OWNER_FILE = ".owner-lock"
# matplotlib builds a font cache on its first import (12.6 s cold on an M3 Pro,
# 0.3 s warm) and writes it to ~/.matplotlib by default. That is state outside
# the pinned runtime, so it lives under the app's own support directory
# instead, where the verifier can account for it: fontlist-*.json and nothing
# else. A user's own MPLCONFIGDIR is deliberately overridden inside the app --
# it may point anywhere, including outside the writable roots the app is
# verified against.
FONT_CACHE_DIRNAME = "matplotlib"
# The app confines ITSELF to loopback-only networking before it accepts a
# photograph. The measurement library (MediaPipe Tasks) carries a usage
# reporter that posts "solution events" to play.googleapis.com over
# NSURLSession -- outside Python, invisible to socket patches, and not
# switchable from here. The written promise is "no analysis-time network
# request", so it is enforced by the kernel rather than by trust: the serving
# phase re-executes under the macOS sandbox with every network operation
# denied except inbound loopback, and the release verifier asks the kernel,
# from outside the process, whether that confinement holds.
NETWORK_SANDBOX = "/usr/bin/sandbox-exec"
NETWORK_PROFILE = (
    "(version 1) (allow default) (deny network*) "
    '(allow network-inbound (local ip "localhost:*"))'
)
SANDBOX_LIBRARY = "/usr/lib/system/libsystem_sandbox.dylib"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
serve = None


class ManifestIntegrityError(RuntimeError):
    """The packaged runtime differs from its signed inventory."""


class AssetIntegrityError(RuntimeError):
    """A downloaded model asset does not match its pinned checksum."""


def outbound_network_denied(pid: int | None = None) -> bool:
    """Ask the kernel whether a process may open arbitrary outbound connections."""
    try:
        sandbox = ctypes.CDLL(SANDBOX_LIBRARY)
        check = sandbox.sandbox_check
    except (OSError, AttributeError):
        return False
    check.restype = ctypes.c_int
    check.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
    return check(os.getpid() if pid is None else pid, b"network-outbound", 0) == 1


def confine_network() -> None:
    """Re-execute this interpreter inside the loopback-only sandbox, once."""
    if outbound_network_denied():
        return
    if not Path(NETWORK_SANDBOX).is_file():
        raise RuntimeError("the macOS network sandbox is unavailable")
    sys.stdout.flush()
    sys.stderr.flush()
    # execv keeps the pid, the inherited pipes and the caller's exact
    # interpreter flags; sandbox-exec applies the profile and execs them.
    os.execv(NETWORK_SANDBOX, [NETWORK_SANDBOX, "-p", NETWORK_PROFILE, *sys.orig_argv])


def font_cache_dir(support: Path) -> Path:
    """Create (0700) and return the app-private matplotlib cache directory."""
    directory = Path(support) / FONT_CACHE_DIRNAME
    if directory.is_symlink():
        raise RuntimeError("the font cache directory is a symlink")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def application_support(home: Path | None = None) -> Path:
    """Return the owner-private application support directory."""
    base = Path.home() if home is None else Path(home)
    return (
        base.expanduser().resolve()
        / "Library"
        / "Application Support"
        / "Facial Analysis"
    )


def _absolute_lexical(path: Path) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _reject_unsafe_symlink_components(path: Path) -> None:
    """Reject user-controlled symlinks before any resolve or mutation."""
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        if not stat.S_ISLNK(metadata.st_mode):
            continue
        # macOS exposes /var as an immutable root-owned alias to /private/var.
        if current == Path("/var") and metadata.st_uid == 0:
            continue
        raise ManifestIntegrityError("refusing a symlinked runtime path")


def _trusted_path(path: Path) -> Path:
    _reject_unsafe_symlink_components(path)
    return _absolute_lexical(path).resolve(strict=False)


def _manifest_entries(manifest_path: Path) -> tuple[dict[str, object], ...]:
    _reject_unsafe_symlink_components(manifest_path)
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ManifestIntegrityError("runtime manifest is unreadable") from error
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ManifestIntegrityError("runtime manifest version is unsupported")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ManifestIntegrityError("runtime manifest has no file inventory")
    return tuple(files)


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ManifestIntegrityError("runtime manifest contains an unsafe path")
    relative = Path(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ManifestIntegrityError("runtime manifest contains an unsafe path")
    return relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_relative_to_root(root: Path, manifest_path: Path) -> Path | None:
    try:
        return _trusted_path(manifest_path).relative_to(root)
    except ValueError:
        return None


def verify_manifest(
    root: Path,
    manifest_path: Path,
    *,
    allowed_extra_files: frozenset[Path] = frozenset(),
) -> None:
    """Verify exact file membership, sizes, and hashes without following symlinks."""
    root = _trusted_path(root)
    manifest_path = _trusted_path(manifest_path)
    if not root.is_dir():
        raise ManifestIntegrityError("runtime payload is missing")

    expected: dict[Path, tuple[int, str]] = {}
    for raw in _manifest_entries(manifest_path):
        if not isinstance(raw, dict):
            raise ManifestIntegrityError("runtime manifest entry is invalid")
        relative = _safe_relative_path(raw.get("path"))
        size = raw.get("size")
        digest = raw.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in expected
        ):
            raise ManifestIntegrityError("runtime manifest entry is invalid")
        expected[relative] = (size, digest)

    manifest_relative = _manifest_relative_to_root(root, manifest_path)
    ignored = set(allowed_extra_files)
    if manifest_relative is not None:
        ignored.add(manifest_relative)

    allowed_directories: set[Path] = set()
    for relative in (*expected, *ignored):
        allowed_directories.update(relative.parents)
    allowed_directories.discard(Path("."))

    actual: set[Path] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ManifestIntegrityError("runtime payload contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            if relative not in ignored:
                actual.add(relative)
        elif stat.S_ISDIR(metadata.st_mode):
            if relative not in allowed_directories:
                raise ManifestIntegrityError(
                    "runtime payload contains an unlisted directory"
                )
        else:
            raise ManifestIntegrityError(
                "runtime payload contains an unsupported entry"
            )
    if actual != set(expected):
        raise ManifestIntegrityError(
            "runtime payload membership differs from its manifest"
        )

    for relative, (size, digest) in expected.items():
        path = root / relative
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != size:
            raise ManifestIntegrityError(
                "runtime payload metadata differs from its manifest"
            )
        if _sha256(path) != digest:
            raise ManifestIntegrityError(
                "runtime payload checksum differs from its manifest"
            )


def _copy_owner_only(source: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    for item in sorted(source.rglob("*"), key=lambda path: path.as_posix()):
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ManifestIntegrityError("runtime payload contains a symlink")
        relative = item.relative_to(source)
        target = destination / relative
        if stat.S_ISDIR(metadata.st_mode):
            target.mkdir(mode=0o700)
        elif stat.S_ISREG(metadata.st_mode):
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(item, target, follow_symlinks=False)
            os.chmod(target, 0o600)
        else:
            raise ManifestIntegrityError(
                "runtime payload contains an unsupported entry"
            )


def _model_asset_paths() -> frozenset[Path]:
    if not MODEL_ASSETS:
        return frozenset()
    return frozenset(Path(name) for name in MODEL_ASSETS)


def install_runtime(payload: Path, destination: Path, manifest_path: Path) -> Path:
    """Verify and atomically install one immutable owner-private runtime version."""
    payload = _trusted_path(payload)
    destination = _trusted_path(destination)
    manifest_path = _trusted_path(manifest_path)
    verify_manifest(payload, manifest_path)

    parent = _trusted_path(destination.parent)
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(parent, 0o700)
    if destination.exists():
        if not destination.is_dir():
            raise ManifestIntegrityError("runtime destination is unsafe")
        verify_manifest(
            destination,
            manifest_path,
            allowed_extra_files=_model_asset_paths() | {Path("manifest.json")},
        )
        _enforce_owner_modes(destination)
        return destination

    staging = parent / f".{destination.name}.install-{secrets.token_hex(8)}"
    try:
        _copy_owner_only(payload, staging)
        installed_manifest = staging / "manifest.json"
        verify_manifest(
            staging,
            installed_manifest if installed_manifest.is_file() else manifest_path,
        )
        try:
            os.rename(staging, destination)
        except FileExistsError:
            verify_manifest(
                destination,
                manifest_path,
                allowed_extra_files=_model_asset_paths() | {Path("manifest.json")},
            )
            shutil.rmtree(staging)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return destination


def _enforce_owner_modes(root: Path) -> None:
    os.chmod(root, 0o700)
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ManifestIntegrityError("installed runtime contains a symlink")
        os.chmod(path, 0o700 if stat.S_ISDIR(metadata.st_mode) else 0o600)


def ensure_asset(destination: Path, download: Download, opener=urlopen) -> Path:
    """Install one checksum-pinned asset with no partial file left behind."""
    raw_destination = _absolute_lexical(destination)
    _reject_unsafe_symlink_components(raw_destination)
    destination = raw_destination.resolve(strict=False)
    parent = _trusted_path(destination.parent)
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(parent, 0o700)
    if destination.exists():
        metadata = destination.lstat()
        if stat.S_ISREG(metadata.st_mode) and _sha256(destination) == download.sha256:
            os.chmod(destination, 0o600)
            return destination
        if stat.S_ISDIR(metadata.st_mode):
            raise AssetIntegrityError("asset destination is a directory")
        destination.unlink(missing_ok=True)

    temporary = parent / f".{destination.name}.download-{secrets.token_hex(8)}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        digest = hashlib.sha256()
        with (
            os.fdopen(descriptor, "wb") as output,
            opener(download.url, timeout=60) as response,
        ):
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != download.sha256:
            raise AssetIntegrityError("downloaded asset checksum mismatch")
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        return destination
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def cleanup_abandoned_sessions(temp_root: Path) -> int:
    """Remove stale application-prefixed roots without following symlinks."""
    temp_root = _trusted_path(temp_root)
    if not temp_root.exists():
        return 0
    removed = 0
    for path in sorted(temp_root.iterdir(), key=lambda item: item.name):
        if not path.name.startswith(SESSION_PREFIX):
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            path.unlink(missing_ok=True)
            removed += 1
            continue
        owner_file = path / SESSION_OWNER_FILE
        descriptor: int | None = None
        try:
            descriptor = os.open(
                owner_file,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if descriptor is not None:
                os.close(descriptor)
            continue
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        shutil.rmtree(path)
        removed += 1
    return removed


def verify_bank(runtime_root: Path) -> str:
    """Validate the installed bank in a clean interpreter, immune to module caches."""
    runtime_root = _trusted_path(runtime_root)
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(runtime_root)!r});"
        "import bank_reference as bank;"
        "bank.cohort_inventory();"
        "message=bank._load().get('instrument_ok');"
        "assert message == 'instrument matches the bank', message;"
        "print('@@' + message)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=runtime_root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    marker = "@@instrument matches the bank"
    if result.returncode != 0 or marker not in result.stdout.splitlines():
        raise RuntimeError("packaged bank instrument guard did not pass")
    return marker[2:]


def _manifest_path(resources: Path) -> Path:
    in_payload = resources / "payload" / "manifest.json"
    return in_payload if in_payload.is_file() else resources / "manifest.json"


def _load_config(runtime_root: Path) -> ModuleType:
    path = runtime_root / "macos_app" / "config.py"
    spec = importlib.util.spec_from_file_location(
        "_facial_analysis_release_config", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("packaged release configuration is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode
        sys.modules.pop(spec.name, None)
    return module


def _runtime_serve(runtime_root: Path):
    if serve is not None:
        return serve
    sys.path.insert(0, str(runtime_root))
    try:
        from local_app.server import serve as packaged_serve
    except Exception:
        try:
            sys.path.remove(str(runtime_root))
        except ValueError:
            pass
        raise
    return packaged_serve


def run_packaged_app(resources: Path, home: Path | None = None) -> int:
    """Preflight the complete private instrument before accepting photographs."""
    global APP_VERSION, MODEL_ASSETS

    resources = _trusted_path(resources)
    payload = resources / "payload"
    manifest = _manifest_path(resources)

    temp_root = _trusted_path(Path(tempfile.gettempdir()))
    temp_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    cleanup_abandoned_sessions(temp_root)

    previous_handlers: dict[int, signal.Handlers] = {}
    owner_descriptor: int | None = None
    session_root: Path | None = None
    installed: Path | None = None
    try:

        def interrupt(_signum: int, _frame: FrameType | None) -> None:
            raise KeyboardInterrupt

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt)

        verify_manifest(payload, manifest)
        source_config = _load_config(payload) if APP_VERSION is None else None
        if APP_VERSION is None:
            APP_VERSION = source_config.APP_VERSION
        if MODEL_ASSETS is None:
            MODEL_ASSETS = source_config.MODEL_ASSETS
        version = APP_VERSION
        configured_assets = MODEL_ASSETS

        support = application_support(home)
        runtime = support / "runtime" / f"v{version}"
        installed = install_runtime(payload, runtime, manifest)
        os.environ["MPLCONFIGDIR"] = os.fspath(font_cache_dir(support))
        for name, download in configured_assets.items():
            ensure_asset(installed / name, download)
        verify_bank(installed)
        confine_network()

        session_root = Path(tempfile.mkdtemp(prefix=SESSION_PREFIX, dir=temp_root))
        os.chmod(session_root, 0o700)
        owner_file = session_root / SESSION_OWNER_FILE
        owner_descriptor = os.open(
            owner_file,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        fcntl.flock(owner_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(owner_descriptor, str(os.getpid()).encode("ascii"))

        packaged_serve = _runtime_serve(installed)
        return packaged_serve(
            host="127.0.0.1",
            port=0,
            open_browser=False,
            base_dir=session_root,
            shutdown_after_end=True,
            history_dir=support / "history",
            on_ready=lambda url: print(f"LOCAL_URL={url}", flush=True),
        )
    except KeyboardInterrupt:
        return 0
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if owner_descriptor is not None:
            fcntl.flock(owner_descriptor, fcntl.LOCK_UN)
            os.close(owner_descriptor)
        if installed is not None:
            try:
                sys.path.remove(str(installed))
            except ValueError:
                pass
        if session_root is not None:
            if session_root.is_symlink():
                session_root.unlink(missing_ok=True)
            elif session_root.exists():
                shutil.rmtree(session_root)


def main() -> int:
    try:
        return run_packaged_app(Path(__file__).resolve().parent)
    except KeyboardInterrupt:
        return 0
    except Exception:
        print("Facial Analysis could not start safely.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AssetIntegrityError",
    "ManifestIntegrityError",
    "application_support",
    "cleanup_abandoned_sessions",
    "ensure_asset",
    "install_runtime",
    "run_packaged_app",
    "verify_bank",
    "verify_manifest",
]
