from __future__ import annotations

import argparse
import sys
from pathlib import Path

from filingdelta.services.small_doc_benchmark import (
    SmallDocBenchmarkProcessor,
    load_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FilingDelta on a small benchmark manifest of filing documents."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/small_doc_benchmark.json"),
        help="Path to a JSON manifest describing the benchmark filings.",
    )
    return parser


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest.resolve())
    run_result = SmallDocBenchmarkProcessor().run(manifest)

    print("benchmark_report:", run_result.report_path)
    print(
        "summary:",
        run_result.report.summary.total_documents,
        "docs /",
        run_result.report.summary.successful_documents,
        "success /",
        run_result.report.summary.failed_documents,
        "failed",
    )
    print("average_citation_coverage:", run_result.report.summary.average_citation_coverage)

    print("documents:")
    for document in run_result.report.documents:
        if not document.success:
            print(
                f"  - {document.entry.source_path.name}: FAILED "
                f"({document.error_type}: {document.error_message})"
            )
            continue

        print(
            f"  - {document.entry.source_path.name}: OK "
            f"pages={document.total_pages} chunks={document.chunk_count} "
            f"citation_coverage={document.citation_coverage}"
        )


if __name__ == "__main__":
    main()
