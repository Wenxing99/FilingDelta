from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from filingdelta.core.config import REPO_ROOT
from filingdelta.ingestion.parser_smoke import (
    ParserSmokeReport,
    load_raw_document_registry_json,
    run_parser_smoke_check,
)
from filingdelta.ingestion.raw_registry import DEFAULT_RAW_DIR, scan_raw_document_registry


DEFAULT_REGISTRY = Path("data/outputs/eval/raw_document_registry.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/parser_smoke_report.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a lightweight parser smoke check over raw registry entries."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help=(
            "Raw registry JSON to read. Defaults to "
            "data/outputs/eval/raw_document_registry.json when it exists."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Scan this raw directory instead of reading a registry JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the parser smoke report JSON.",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=3,
        help="Number of leading PDF pages to sample. Defaults to 3.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=1,
        help="Minimum extracted text characters required for a pass. Defaults to 1.",
    )
    parser.add_argument(
        "--fail-on-failure",
        action="store_true",
        help="Exit non-zero when any supported document fails the parser smoke check.",
    )
    return parser


def main(argv: list[str] | None = None) -> ParserSmokeReport:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    registry, registry_source = _load_or_scan_registry(args)

    report = run_parser_smoke_check(
        registry,
        repo_root=REPO_ROOT,
        sample_pages_per_pdf=args.sample_pages,
        min_text_chars=args.min_text_chars,
    )

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _print_summary(report=report, output_path=output_path, registry_source=registry_source)
    if args.fail_on_failure and report.summary.failed_count:
        raise SystemExit(1)
    return report


def _load_or_scan_registry(args: argparse.Namespace):
    if args.raw_dir is not None:
        raw_dir = _resolve_path(args.raw_dir)
        return (
            scan_raw_document_registry(raw_dir=raw_dir, repo_root=REPO_ROOT),
            f"scan:{_display_path(raw_dir)}",
        )

    registry_path = _resolve_path(args.registry or DEFAULT_REGISTRY)
    if registry_path.exists():
        return (
            load_raw_document_registry_json(registry_path, repo_root=REPO_ROOT),
            f"registry:{_display_path(registry_path)}",
        )

    if args.registry is not None:
        raise FileNotFoundError(f"Raw registry JSON not found: {registry_path}")

    return (
        scan_raw_document_registry(raw_dir=DEFAULT_RAW_DIR, repo_root=REPO_ROOT),
        f"scan:{DEFAULT_RAW_DIR.as_posix()}",
    )


def _print_summary(
    *,
    report: ParserSmokeReport,
    output_path: Path,
    registry_source: str,
) -> None:
    summary = report.summary
    print(f"registry_source: {registry_source}")
    print(f"raw_dir: {report.raw_dir}")
    print(f"output: {_display_path(output_path)}")
    print(
        "summary: "
        f"total_documents={summary.total_documents} "
        f"supported_documents={summary.supported_documents} "
        f"unsupported_documents={summary.unsupported_documents} "
        f"passed={summary.passed_count} "
        f"failed={summary.failed_count} "
        f"skipped={summary.skipped_count}"
    )
    if summary.error_counts:
        error_text = ", ".join(
            f"{error_type}={count}" for error_type, count in summary.error_counts.items()
        )
        print(f"errors: {error_text}")


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
