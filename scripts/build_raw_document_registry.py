from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from filingdelta.core.config import REPO_ROOT
from filingdelta.ingestion.raw_registry import (
    DEFAULT_RAW_DIR,
    RawDocumentRegistry,
    scan_raw_document_registry,
)


DEFAULT_OUTPUT = Path("data/outputs/eval/raw_document_registry.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan local raw filing files and build a lightweight registry."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing raw filing files. Defaults to data/raw.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the registry JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> RawDocumentRegistry:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    registry = scan_raw_document_registry(raw_dir=args.raw_dir, repo_root=REPO_ROOT)

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(registry.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _print_summary(registry=registry, output_path=output_path)
    return registry


def _print_summary(*, registry: RawDocumentRegistry, output_path: Path) -> None:
    summary = registry.summary
    print(f"raw_dir: {registry.raw_dir}")
    print(f"output: {_display_path(output_path)}")
    print(
        "summary: "
        f"total_files={summary.total_files} "
        f"supported_files={summary.supported_files} "
        f"unsupported_files={summary.unsupported_files} "
        f"warning_count={summary.warning_count} "
        f"duplicate_checksum_groups={summary.duplicate_checksum_groups}"
    )
    if summary.warnings_by_type:
        warning_text = ", ".join(
            f"{warning}={count}" for warning, count in summary.warnings_by_type.items()
        )
        print(f"warnings: {warning_text}")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
