"""Build a separately named local Apple Silicon beta without replacing releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from . import build, verify
from .config import APP_VERSION, RELEASE_TAG


def beta_destination(output: Path) -> Path:
    root = build.REPOSITORY_ROOT.resolve() / "dist" / "macos"
    output = Path(output).absolute()
    if (
        root.is_symlink()
        or root.parent.is_symlink()
        or output.parent.is_symlink()
        or output.parent.resolve() != root
    ):
        raise ValueError(
            "Beta output must be a new direct child of dist/macos, without symlinks."
        )
    if output.exists() or output.is_symlink():
        raise ValueError(
            "Beta output already exists; preserve it and choose a new directory."
        )
    return output


def build_beta(output: Path, *, runtime_app: Path | None = None) -> dict:
    output = beta_destination(output)
    if runtime_app is None:
        python = build._prepare_python("arm64")
        build._install_dependencies("arm64", python, intel_wheel=None)
        build._remove_build_residue(python)
    else:
        runtime_app = Path(runtime_app).resolve()
        verify.verify_codesign(runtime_app)
        python = runtime_app / "Contents" / "Resources" / "python"
        verify.verify_native_architecture(python / "bin" / "python3", "arm64")
        verify.verify_embedded_dependencies(python / "bin" / "python3", "arm64")
    payload = build._stage_payload()
    launcher = build._compile_launcher("arm64")
    build.BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="beta-stage-", dir=build.BUILD_ROOT
    ) as directory:
        release = Path(directory) / "release"
        release.mkdir()
        app = build.stage_application(
            architecture="arm64",
            python_root=python,
            payload_root=payload,
            bootstrap_path=Path(__file__).with_name("bootstrap.py"),
            launcher_path=launcher,
            destination=release,
        )
        build._remove_build_residue(app)
        build._assert_no_local_build_paths(app)
        build._codesign(app)
        verify.verify_application(app, "arm64")
        archive = (
            release
            / f"Facial-Analysis-{RELEASE_TAG.removeprefix('v')}-Apple-Silicon.zip"
        )
        build.write_deterministic_zip(app, archive)
        digest = build.sha256(archive)
        (release / "SHA256SUMS.txt").write_text(f"{digest}  {archive.name}\n")
        result = {
            "version": APP_VERSION,
            "tag": RELEASE_TAG,
            "architecture": "arm64",
            "archive": archive.name,
            "sha256": digest,
            "verification": "Static app, native architecture, ad-hoc signature, source/payload manifest and aggregate bank passed. Runtime journeys are separate.",
            "notarized": False,
            "published": False,
        }
        (release / "build.json").write_text(json.dumps(result, indent=2) + "\n")
        # Recheck the destination after the potentially lengthy build.
        beta_destination(output)
        release.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-app",
        type=Path,
        help="Reuse a signed local app's verified pinned runtime; no source photos are copied.",
    )
    options = parser.parse_args()
    print(
        json.dumps(
            build_beta(options.output, runtime_app=options.runtime_app), indent=2
        )
    )


if __name__ == "__main__":
    main()
