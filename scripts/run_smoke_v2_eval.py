from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from filingdelta.core.config import REPO_ROOT
from filingdelta.eval.smoke_v2 import (
    SMOKE_V2_TIER,
    build_builtin_placeholder_manifest_payload,
    build_smoke_v2_report,
    load_smoke_v2_manifest,
    load_smoke_v2_manifest_from_payload,
    select_smoke_v2_cases,
)


DEFAULT_MANIFEST = Path("data/outputs/eval/golden_queries_v2_smoke.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/smoke_v2_report.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or dry-run the FilingDelta smoke_v2 eval manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--case", dest="case_ids", action="append", default=[])
    parser.add_argument("--company", dest="companies", action="append", default=[])
    parser.add_argument("--industry", dest="industries", action="append", default=[])
    parser.add_argument("--intent", dest="intents", action="append", default=[])
    parser.add_argument("--tier", dest="tiers", action="append", default=[])
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--validate-only", action="store_true")
    mode_group.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--use-built-in-placeholders",
        action="store_true",
        help="Use two CMB/Tencent placeholder cases derived from existing eval cases.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    _configure_stdio()

    args = build_parser().parse_args(argv)
    mode = "validate_only" if args.validate_only else "dry_run"
    manifest = _load_manifest(args)
    cases = select_smoke_v2_cases(
        manifest.queries,
        case_ids=set(args.case_ids) or None,
        tiers=set(args.tiers) if args.tiers else {SMOKE_V2_TIER},
        companies=set(args.companies) or None,
        industries=set(args.industries) or None,
        intents=set(args.intents) or None,
    )
    if args.list_cases:
        _print_cases(cases)
        return None
    if not cases:
        raise SystemExit("No smoke_v2 cases selected.")

    report = build_smoke_v2_report(
        manifest=manifest,
        cases=cases,
        mode=mode,
        top_k=args.top_k,
    )
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = report["summary"]
    missing = report["documents"]["missing_document_keys"]
    print("report:", output_path, flush=True)
    print(
        f"mode={report['mode']} total_cases={summary['total_queries']} "
        f"missing_documents={len(missing)}",
        flush=True,
    )
    if missing:
        print("missing_document_keys:", ", ".join(missing), flush=True)
    return report


def _load_manifest(args: argparse.Namespace):
    if args.use_built_in_placeholders:
        payload = build_builtin_placeholder_manifest_payload()
        return load_smoke_v2_manifest_from_payload(payload, base_dir=REPO_ROOT)
    return load_smoke_v2_manifest(_resolve_path(args.manifest), base_dir=REPO_ROOT)


def _print_cases(cases) -> None:
    for case in cases:
        print(
            f"{case.case_id}\t{case.tier}\t{case.document_key}\t"
            f"{case.expected_document_evidence_intent}\t{case.query}",
            flush=True,
        )


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
