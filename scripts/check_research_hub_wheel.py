"""Check wheel assets, optionally checking a fresh core-only installed API.

Run with the installed interpreter's ``-I`` flag for --installed so the source
checkout cannot mask missing wheel modules. This checker uses only stdlib until
the explicitly requested installed-package check. It never loads a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import zipfile


OPTIONAL_ROOTS = {
    "torch", "transformers", "modal", "vllm", "diffusers", "PIL",
    "sklearn", "matplotlib", "sae_lens", "transformer_lens", "datasets",
    "circuit_tracer",
}


def required_assets(source: Path) -> dict[str, bytes]:
    """Discover pins and runtime recipes from source, independently of packaging."""
    assets = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for suffix in ("*.json", "*.lock", "*.in")
        for path in (source / "pipelines_v2").rglob(suffix)
        if "__pycache__" not in path.parts
    }
    if not assets:
        raise ValueError("No source lock/pin assets found; check --source-root")
    return assets


def check_wheel(wheel: Path, assets: dict[str, bytes]) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Wheel contains duplicate entries")
        for relative, expected in sorted(assets.items()):
            if relative not in names:
                raise ValueError(f"Required asset missing from wheel: {relative}")
            if archive.read(relative) != expected:
                raise ValueError(f"Wheel asset differs from source: {relative}")


def check_installed(source: Path, assets: dict[str, bytes]) -> str:
    if not sys.flags.isolated:
        raise ValueError("Use python -I for --installed to exclude checkout imports")
    package = importlib.import_module("pipelines_v2")
    package_root = Path(package.__file__).resolve().parent
    if package_root.is_relative_to(source):
        raise ValueError("Imported source checkout instead of installed wheel")
    importlib.import_module("pipelines_v2.api")
    for relative, expected in sorted(assets.items()):
        path = package_root.parent / relative
        if path.read_bytes() != expected:
            raise ValueError(f"Installed asset differs from source: {relative}")
        if path.suffix == ".json":
            json.loads(path.read_text())

    # Exercise the actual bundled-lock readers, without importing model stacks
    # or validating a foreign-platform runtime against this CI host.
    from pipelines_v2.runtime.locks import load_runtime_lock
    from pipelines_v2.engine.diffusers.artifacts import runtime_spec as diffusion_runtime
    from pipelines_v2.representation.crosscoder.reference import runtime_spec as crosscoder_runtime
    from pipelines_v2.operations.attribution import official_tool_lock
    from pipelines_v2.runtime.circuit_tracer import official_lock
    from pipelines_v2.operations.circuit_tracer.prepare import pair_lock

    for relative in assets:
        if relative.startswith("pipelines_v2/runtime/locks/") and relative.endswith(".json"):
            loaded = load_runtime_lock(Path(relative).stem)
            expected = json.loads(assets[relative])
            if loaded.requirements != expected["requirements"]:
                raise ValueError(f"Bundled runtime reader changed requirements: {relative}")
    expected_diffusion = json.loads(assets["pipelines_v2/engine/diffusers/runtime_lock.json"])
    if diffusion_runtime().lock.requirements != expected_diffusion["requirements"]:
        raise ValueError("SDXL runtime reader changed its requirements")
    expected_crosscoder = assets["pipelines_v2/representation/crosscoder/requirements-darwin-arm64.lock"].decode()
    if crosscoder_runtime().lock.requirements != expected_crosscoder:
        raise ValueError("Crosscoder runtime reader changed its requirements")
    for observed, relative in (
        (official_tool_lock(), "pipelines_v2/operations/attribution/official_mattr_lock.json"),
        (official_lock(), "pipelines_v2/operations/circuit_tracer/tool_lock.json"),
        (pair_lock(), "pipelines_v2/operations/circuit_tracer/pair_lock.json"),
    ):
        if observed != json.loads(assets[relative]):
            raise ValueError(f"Bundled tool/model reader changed its lock: {relative}")
    imported = sorted(OPTIONAL_ROOTS.intersection(name.split(".")[0] for name in sys.modules))
    if imported:
        raise ValueError(f"Core API eagerly imported optional runtimes: {imported}")
    return str(package_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    source = args.source_root.resolve()
    assets = required_assets(source)
    check_wheel(args.wheel, assets)
    package = check_installed(source, assets) if args.installed else None
    print(json.dumps({"status": "passed", "asset_count": len(assets),
                      "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                      "installed_package": package, "optional_import_check": args.installed}, indent=2))


if __name__ == "__main__":
    main()
