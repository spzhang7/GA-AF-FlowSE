"""Verify or refresh the checked-in FlowSE source subset.

The public repository keeps the pinned FlowSE source under ``flowse/``.  This
utility remains for maintainers who need to refresh that snapshot from the
commit recorded in ``flowse/UPSTREAM_SOURCE.json``; normal users do not need to
run it after cloning.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "flowse" / "UPSTREAM_SOURCE.json"
DEFAULT_DESTINATION = ROOT / "flowse"


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path in metadata: {value!r}")
    return path


def _read_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {"repository", "commit", "raw_base_url", "patch", "source_files"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"FlowSE metadata is missing keys: {sorted(missing)}")
    if not str(metadata["commit"]).strip():
        raise ValueError("FlowSE commit must be non-empty")
    source_files = metadata["source_files"]
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("FlowSE source_files must be a non-empty list")
    for relative in source_files:
        _safe_relative_path(str(relative))
    return metadata


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "GA-AF-FlowSE-fetch/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _missing_files(destination: Path, source_files: list[str]) -> list[str]:
    return [
        str(relative)
        for relative in source_files
        if not (destination / _safe_relative_path(str(relative))).is_file()
    ]


def _fetch(metadata: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    raw_base = str(metadata["raw_base_url"]).rstrip("/")
    for relative in sorted(metadata["source_files"]):
        relative = str(relative)
        target = destination / _safe_relative_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _download(f"{raw_base}/{relative}")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.replace(target)
        print(f"fetched {relative}")


def _apply_patch(metadata: dict, destination: Path) -> None:
    patch_path = ROOT / _safe_relative_path(str(metadata["patch"]))
    if not patch_path.is_file():
        raise FileNotFoundError(f"compatibility patch not found: {patch_path}")
    marker = destination / ".gaaf-flowse-patch-applied"
    if marker.is_file():
        print(f"patch already applied: {marker}")
        return
    command = [
        "git",
        "-c",
        "core.autocrlf=false",
        "apply",
        "--whitespace=nowarn",
        str(patch_path),
    ]
    check = subprocess.run(
        [*command[:4], "--check", *command[4:]],
        cwd=destination,
        text=True,
        capture_output=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "FlowSE compatibility patch does not apply cleanly:\n"
            + (check.stderr or check.stdout)
        )
    subprocess.run(command, cwd=destination, check=True)
    marker.write_text(
        json.dumps(
            {"upstream_commit": metadata["commit"], "patch": metadata["patch"]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"applied compatibility patch: {patch_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check that the required files exist without downloading or patching",
    )
    parser.add_argument(
        "--no-patch", action="store_true", help="fetch without applying the patch"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove the explicitly selected destination before fetching",
    )
    args = parser.parse_args()

    metadata = _read_metadata(args.metadata.resolve())
    destination = args.destination.resolve()
    if args.clean:
        if destination == DEFAULT_DESTINATION.resolve():
            raise ValueError(
                "--clean is disabled for the checked-in flowse/ directory; "
                "refresh it in a temporary directory and review the diff"
            )
        if destination == ROOT or len(destination.parts) < 3:
            raise ValueError(f"refusing to clean broad destination: {destination}")
        if destination.exists():
            shutil.rmtree(destination)

    marker = destination / ".gaaf-flowse-patch-applied"
    if destination.exists() and marker.is_file() and not args.clean:
        missing = _missing_files(destination, metadata["source_files"])
        if missing:
            raise RuntimeError(
                "existing FlowSE source is incomplete; refresh into a temporary "
                "destination and review the diff:\n"
                + "\n".join(missing)
            )
        print(f"FlowSE source already ready at {destination}")
        return 0

    if args.verify_only:
        missing = _missing_files(destination, metadata["source_files"])
        if missing:
            print("missing files:\n" + "\n".join(missing))
            return 1
        print(f"verified required FlowSE files at {destination}")
        return 0

    _fetch(metadata, destination)
    missing = _missing_files(destination, metadata["source_files"])
    if missing:
        raise RuntimeError("missing files after download: " + "; ".join(missing))
    if not args.no_patch:
        _apply_patch(metadata, destination)
    print(f"FlowSE source ready at {destination}")
    print(f"Set PYTHONPATH to {destination} before importing model/infer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
