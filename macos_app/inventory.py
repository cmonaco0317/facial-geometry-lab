"""Derive and audit the private macOS application's runtime payload."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import zipfile

MEASUREMENT_ENTRY_POINTS = ("local_page", "assess_set", "bank_reference")
JSON_READERS = (
    Path("local_app/analysis.py"),
    Path("assess_set.py"),
    Path("bank_reference.py"),
)
REQUIRED_DATA = (
    Path("bank/derived.npz"),
    Path("bank/public_reference.json"),
    Path("mirror_perm.npy"),  # averageness's L/R landmark map; tracked, model-derived
    Path("LICENSE"),
    Path("NOTICE"),
)
# The packaged bootstrap runs standalone (no package), so it reads APP_VERSION and
# MODEL_ASSETS from payload/macos_app/config.py at start-up. Nothing in the
# measurement import graph pulls these in, so they are named explicitly; without
# them the 2026-09-21 build failed _load_config and exited before serving.
BOOTSTRAP_SOURCES = (Path("macos_app/__init__.py"), Path("macos_app/config.py"))
ALLOWED_UNTRACKED = frozenset({Path("bank/derived.npz")})
# The per-cohort landmark files are untracked (100 MB of build output) but the
# product reads them: averageness takes the cohort mean shape from them and
# symmetry gates the cohort pool on their stored pose. Until 2026-09-21 only
# derived.npz shipped, so the packaged app answered "reference could not be
# read" for two of its four outputs while the checkout measured all four --
# the exact parity gap the release verifier exists to catch. Which cohorts
# ship is DERIVED from bank/provenance.json, never globbed, and each file's
# stored row count must agree with the record that names it.
COHORT_ID = re.compile(r"[a-z0-9_]{1,80}")
COHORT_MEASUREMENTS = Path("measurements.npz")
NPY_MAGIC = b"\x93NUMPY"
DISCOVERY_MARKER = "@@RUNTIME-INVENTORY@@"
FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        "captures",
        "docs",
        "media",
        "photos",
        "reports",
        "tests",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        "backlog.md",
        "credentials.json",
        "ledger.json",
        "publishing.md",
        "state.md",
    }
)
PRIVATE_SUFFIXES = frozenset(
    {
        ".bmp",
        ".heic",
        ".htm",
        ".html",
        ".jpeg",
        ".jpg",
        ".key",
        ".mov",
        ".mp4",
        ".pdf",
        ".pem",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)
HASH_CHUNK_BYTES = 1024 * 1024


class PayloadAuditError(RuntimeError):
    """A candidate runtime payload contains unsafe or private material."""


def _sort_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(sorted(set(paths), key=lambda path: path.as_posix()))


def _root_path(root: Path) -> Path:
    lexical = Path(root).expanduser().absolute()
    if lexical.is_symlink():
        raise PayloadAuditError("runtime source root must not be a symlink")
    root = lexical.resolve()
    if not root.is_dir():
        raise PayloadAuditError("runtime source root is missing")
    return root


def _relative_path(root: Path, path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(root)
        except ValueError:
            try:
                candidate = candidate.resolve(strict=False).relative_to(root)
            except ValueError:
                raise PayloadAuditError(
                    "payload path escapes the source root"
                ) from None
    if (
        not candidate.parts
        or "\\" in candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise PayloadAuditError("payload path is unsafe")
    return candidate


def _measurement_inventory(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return runtime sources and the bank's derived measurement module set."""
    root = _root_path(root)
    code = f"""
import importlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
for name in {MEASUREMENT_ENTRY_POINTS!r}:
    importlib.import_module(name)
bank_instrument = importlib.import_module("bank_instrument")
sources = set()
for module in tuple(sys.modules.values()):
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        continue
    path = Path(raw_path).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError:
        continue
    if path.suffix != ".py":
        continue
    if "venv" in relative.parts or "site-packages" in relative.parts:
        continue
    sources.add(relative.as_posix())
fingerprint = bank_instrument.measurement_fingerprint(force=True)
modules = fingerprint.get("modules")
if not isinstance(modules, dict):
    raise RuntimeError("invalid measurement fingerprint")
sources.update(modules)
print({DISCOVERY_MARKER!r} + json.dumps({{
    "sources": sorted(sources),
    "measurement_modules": sorted(modules),
}}))
"""
    try:
        environment = {**os.environ, "PYTHONSAFEPATH": "1"}
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, str(root)],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise PayloadAuditError("measurement source discovery failed") from None
    records = [
        line.removeprefix(DISCOVERY_MARKER)
        for line in result.stdout.splitlines()
        if line.startswith(DISCOVERY_MARKER)
    ]
    if result.returncode != 0 or len(records) != 1:
        raise PayloadAuditError("measurement source discovery failed")
    try:
        raw_inventory = json.loads(records[0])
    except (TypeError, ValueError):
        raise PayloadAuditError("measurement source discovery failed") from None
    if not isinstance(raw_inventory, dict):
        raise PayloadAuditError("measurement source discovery failed")
    raw_sources = raw_inventory.get("sources")
    raw_modules = raw_inventory.get("measurement_modules")
    if (
        not isinstance(raw_sources, list)
        or not isinstance(raw_modules, list)
        or any(not isinstance(path, str) for path in (*raw_sources, *raw_modules))
    ):
        raise PayloadAuditError("measurement source discovery failed")

    sources = set()
    modules = set()
    for raw_path, destination in (
        *((path, sources) for path in raw_sources),
        *((path, modules) for path in raw_modules),
    ):
        relative = _relative_path(root, Path(raw_path))
        if relative.suffix != ".py":
            raise PayloadAuditError("measurement source discovery failed")
        _ensure_regular_file(root, relative)
        destination.add(relative)
    if not modules or not modules.issubset(sources):
        raise PayloadAuditError("measurement source discovery failed")
    return _sort_paths(sources), _sort_paths(modules)


def _local_import_targets(root: Path, relative: Path) -> set[Path]:
    """Project-local modules named by ANY import statement in one source file,
    module-level or inside a function. Static, so lazy imports count too."""
    try:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    package = relative.parent if relative.parent != Path(".") else None
    targets: set[Path] = set()

    def candidate(dotted: str) -> None:
        parts = dotted.split(".")
        for cut in range(len(parts), 0, -1):
            module = Path(*parts[:cut])
            for path in (module.with_suffix(".py"), module / "__init__.py"):
                if (root / path).is_file():
                    targets.add(path)
                    return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidate(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.parent if base is not None else None
                if base is None:
                    continue
                prefix = ".".join(base.parts)
                module = f"{prefix}.{node.module}" if node.module else prefix
                candidate(module)
                for alias in node.names:
                    candidate(f"{module}.{alias.name}")
            elif node.module:
                candidate(node.module)
                for alias in node.names:
                    candidate(f"{node.module}.{alias.name}")
    return targets


def unresolved_local_imports(root: Path, payload: Iterable[Path]) -> tuple[Path, ...]:
    """Project modules imported by payload sources that the payload lacks."""
    root = _root_path(root)
    files = {Path(path) for path in payload}
    missing: set[Path] = set()
    for relative in files:
        if relative.suffix != ".py":
            continue
        for target in _local_import_targets(root, relative):
            if target not in files:
                missing.add(target)
    return _sort_paths(missing)


def _import_closure(root: Path, sources: Iterable[Path]) -> tuple[Path, ...]:
    """Close the source set under project-local imports, lazy imports included.
    Import-time discovery missed `local_app/cohorts.py` (imported inside a
    function) and the 2026-09-21 build died on its first request."""
    root = _root_path(root)
    closed = {Path(path) for path in sources}
    frontier = list(closed)
    while frontier:
        relative = frontier.pop()
        for target in _local_import_targets(root, relative):
            if target not in closed:
                closed.add(target)
                frontier.append(target)
    return _sort_paths(closed)


def measurement_sources(root: Path) -> tuple[Path, ...]:
    """Return the runtime import graph: what a clean interpreter loads when the
    entry points are imported, closed under every import statement those files
    contain (so a module imported inside a function is shipped too)."""
    return _import_closure(root, _measurement_inventory(root)[0])


def bank_measurement_modules(root: Path) -> tuple[Path, ...]:
    """Return the exact derived module set fingerprinted into the reference bank."""
    return _measurement_inventory(root)[1]


def _tracked_files(root: Path) -> frozenset[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        raise PayloadAuditError("source root has no readable Git inventory") from None
    return frozenset(
        Path(os.fsdecode(item)) for item in result.stdout.split(b"\0") if item
    )


def _json_literals(root: Path, tracked: frozenset[Path]) -> set[Path]:
    results: set[Path] = set()
    for source in JSON_READERS:
        source_path = root / source
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            raise PayloadAuditError("runtime data reader is unavailable") from None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            if not value.endswith(".json") or "\n" in value or "\0" in value:
                continue
            relative = Path(value)
            if source == Path("bank_reference.py") and len(relative.parts) == 1:
                relative = Path("bank") / relative
            if relative in tracked and (root / relative).is_file():
                results.add(relative)
    return results


def _ensure_regular_file(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current /= part
        try:
            metadata = current.lstat()
        except OSError:
            raise PayloadAuditError("required runtime data is missing") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise PayloadAuditError("runtime payload contains a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise PayloadAuditError("runtime payload contains a non-file entry")


def _npy_rows(archive: Path, member: str) -> int:
    """Read one array's leading dimension from an .npz without loading it."""
    try:
        with zipfile.ZipFile(archive) as bundle, bundle.open(f"{member}.npy") as stream:
            magic = stream.read(len(NPY_MAGIC))
            if magic != NPY_MAGIC:
                raise PayloadAuditError("a cohort landmark file is not an .npy archive")
            major = stream.read(1)
            stream.read(1)  # minor version
            length_size = 2 if major == b"\x01" else 4
            header_length = int.from_bytes(stream.read(length_size), "little")
            header = ast.literal_eval(stream.read(header_length).decode("latin-1"))
    except (OSError, KeyError, ValueError, SyntaxError, zipfile.BadZipFile) as error:
        raise PayloadAuditError("a cohort landmark file is unreadable") from error
    shape = header.get("shape") if isinstance(header, dict) else None
    if not isinstance(shape, tuple) or not shape or not isinstance(shape[0], int):
        raise PayloadAuditError("a cohort landmark file has no row count")
    return shape[0]


def cohort_measurements(root: Path) -> tuple[Path, ...]:
    """Every cohort's landmark file, as the bank's own provenance record names them."""
    root = _root_path(root)
    from public_reference import validate_directory
    validate_directory(root / "bank")  # Validate allowlisted ECDF/aggregate shapes, counts and checksum.
    return ()


def _is_cohort_measurements(relative: Path) -> bool:
    return (
        len(relative.parts) == 3
        and relative.parts[0] == "bank"
        and COHORT_ID.fullmatch(relative.parts[1]) is not None
        and relative.name == COHORT_MEASUREMENTS.name
    )


def runtime_data(root: Path) -> tuple[Path, ...]:
    """Return static assets and aggregate records read by the runtime."""
    root = _root_path(root)
    tracked = _tracked_files(root)
    files = _json_literals(root, tracked)
    files.update(
        path
        for path in tracked
        if len(path.parts) == 2 and path.parts[0] == "bank" and path.suffix == ".json"
    )
    files.update(cohort_measurements(root))

    static = Path("local_app/static")
    static_path = root / static
    if not static_path.is_dir() or static_path.is_symlink():
        raise PayloadAuditError("runtime static assets are missing or unsafe")
    for path in static_path.rglob("*"):
        if path.is_file() or path.is_symlink():
            files.add(path.relative_to(root))

    files.update(REQUIRED_DATA)
    for relative in files:
        _ensure_regular_file(root, relative)
    return _sort_paths(files)


def _forbidden_path(relative: Path) -> bool:
    lowered_parts = tuple(part.lower() for part in relative.parts)
    name = lowered_parts[-1]
    return (
        any(part in FORBIDDEN_PARTS for part in lowered_parts)
        or name in FORBIDDEN_NAMES
        or name.startswith(".env.")
        or "credential" in name
        or Path(name).suffix in PRIVATE_SUFFIXES
    )


def _contains_private_content(path: Path) -> bool:
    previous = b""
    needles = (b"/Users/", b"-----BEGIN PRIVATE KEY-----")
    overlap = max(len(needle) for needle in needles) - 1
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            combined = previous + chunk
            if any(needle in combined for needle in needles):
                return True
            previous = combined[-overlap:]
    return False


def _audit_candidates(root: Path, candidates: Iterable[Path]) -> tuple[Path, ...]:
    tracked = _tracked_files(root)
    files = _sort_paths(_relative_path(root, path) for path in candidates)
    for relative in files:
        if _forbidden_path(relative):
            raise PayloadAuditError("runtime payload contains forbidden material")
        _ensure_regular_file(root, relative)
        if (
            relative not in tracked
            and relative not in ALLOWED_UNTRACKED
            and not _is_cohort_measurements(relative)
        ):
            raise PayloadAuditError("runtime payload contains an untracked file")
        if _contains_private_content(root / relative):
            raise PayloadAuditError("runtime payload contains private content")
    return files


def bootstrap_required_sources() -> tuple[Path, ...]:
    """Source files the standalone bootstrap imports outside the measurement graph."""
    return BOOTSTRAP_SOURCES


def bootstrap_sources(root: Path) -> tuple[Path, ...]:
    """Return the bootstrap's own configuration modules, each a real tracked file."""
    root = _root_path(root)
    for relative in BOOTSTRAP_SOURCES:
        _ensure_regular_file(root, relative)
    return _sort_paths(BOOTSTRAP_SOURCES)


def audited_payload(root: Path) -> tuple[Path, ...]:
    """Return the complete payload after a deny-by-default privacy audit."""
    root = _root_path(root)
    return _audit_candidates(
        root,
        (*measurement_sources(root), *bootstrap_sources(root), *runtime_data(root)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_destination(
    root: Path,
    lexical_root: Path,
    destination: Path,
) -> tuple[Path, Path]:
    raw = Path(destination).expanduser()
    raw_text = os.fspath(raw)
    if "\\" in raw_text or "\0" in raw_text or ".." in raw.parts:
        raise PayloadAuditError("manifest destination is unsafe")
    absolute = raw if raw.is_absolute() else lexical_root / raw
    for base in (lexical_root, root):
        try:
            relative = absolute.relative_to(base)
            break
        except ValueError:
            continue
    else:
        raise PayloadAuditError("manifest destination escapes the payload")
    absolute = root / relative
    if not relative.parts or _forbidden_path(relative):
        raise PayloadAuditError("manifest destination is unsafe")

    current = root
    for index, part in enumerate(relative.parts):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise PayloadAuditError("manifest destination is unsafe") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise PayloadAuditError("manifest destination is unsafe")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise PayloadAuditError("manifest destination is unsafe")
        if index == len(relative.parts) - 1 and not stat.S_ISREG(metadata.st_mode):
            raise PayloadAuditError("manifest destination is unsafe")

    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", "--", relative.as_posix()],
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError:
        raise PayloadAuditError(
            "manifest destination is not a build location"
        ) from None
    if ignored.returncode != 0:
        raise PayloadAuditError("manifest destination is not a build location")
    return absolute, relative


def write_manifest(
    root: Path,
    files: Sequence[Path],
    destination: Path,
    *,
    measurement_modules: Sequence[Path] = (),
    runtime_assets: Mapping[str, str] | None = None,
) -> None:
    """Write a stable version-1 manifest without inventorying the manifest itself."""
    lexical_root = Path(root).expanduser().absolute()
    root = _root_path(lexical_root)
    destination, destination_relative = _manifest_destination(
        root,
        lexical_root,
        destination,
    )

    relative_files = _sort_paths(_relative_path(root, path) for path in files)
    entries = []
    for relative in relative_files:
        if relative == destination_relative:
            continue
        _ensure_regular_file(root, relative)
        path = root / relative
        entries.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    module_paths = _sort_paths(
        _relative_path(root, path) for path in measurement_modules
    )
    if any(path not in relative_files or path.suffix != ".py" for path in module_paths):
        raise PayloadAuditError("measurement module inventory is invalid")
    asset_hashes = dict(sorted((runtime_assets or {}).items()))
    if any(
        not isinstance(name, str)
        or Path(name).name != name
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in asset_hashes.items()
    ):
        raise PayloadAuditError("runtime asset inventory is invalid")
    payload = {
        "version": 1,
        "files": entries,
        "measurement_modules": [path.as_posix() for path in module_paths],
        "runtime_assets": asset_hashes,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the macOS runtime inventory.")
    parser.add_argument(
        "--check", action="store_true", help="audit without listing files"
    )
    options = parser.parse_args(argv)
    if not options.check:
        parser.error("--check is required")

    root = Path(__file__).resolve().parents[1]
    try:
        sources = measurement_sources(root)
        data = runtime_data(root)
        payload = _audit_candidates(root, (*sources, *data))
    except PayloadAuditError:
        print("runtime inventory audit failed", file=sys.stderr)
        return 1
    print(f"source files: {len(sources)}")
    print(f"data files: {len(data)}")
    print(f"payload files: {len(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
