from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.eval.chat_quality import (
    CHAT_QUALITY_CASES,
    ChatQualityCase,
    chat_quality_result_to_json,
    evaluate_chat_quality,
    summarize_chat_quality_results,
)
from filingdelta.services.chat_qa import ChatQAService
from filingdelta.services.demo_documents import get_demo_document_sources


DEFAULT_OUTPUT = Path("data/outputs/eval/chat_quality_smoke.json")
DEFAULT_QDRANT_ROOT = Path("data/outputs/eval/qdrant_chat_quality_smoke")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FilingDelta chat quality smoke questions and write a JSON report."
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=[],
        help="Run a specific quality case id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit the number of selected cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the smoke report JSON.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=DEFAULT_QDRANT_ROOT,
        help="Persistent Qdrant path for the quality smoke run.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete --qdrant-path before running.",
    )
    parser.add_argument(
        "--temporary-qdrant",
        action="store_true",
        help="Use an isolated temporary Qdrant path instead of --qdrant-path.",
    )
    parser.add_argument(
        "--prewarm",
        action="store_true",
        help="Prewarm selected document indexes before asking questions.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print available quality cases without calling models.",
    )
    parser.add_argument(
        "--fail-on-quality-failure",
        action="store_true",
        help="Exit non-zero when any quality check fails.",
    )
    return parser


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    selected_cases = _select_cases(case_ids=args.case_ids, max_cases=args.max_cases)

    if args.list_cases:
        _print_cases(selected_cases)
        return

    asyncio.run(_run_smoke(args=args, selected_cases=selected_cases))


async def _run_smoke(
    *,
    args: argparse.Namespace,
    selected_cases: list[ChatQualityCase],
) -> None:
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.temporary_qdrant:
        with tempfile.TemporaryDirectory(prefix="filingdelta-chat-quality-") as temp_dir:
            await _run_with_qdrant_path(
                args=args,
                selected_cases=selected_cases,
                output_path=output_path,
                qdrant_path=Path(temp_dir),
                temporary_qdrant=True,
            )
        return

    qdrant_path = _resolve_path(args.qdrant_path)
    if args.fresh and qdrant_path.exists():
        shutil.rmtree(qdrant_path)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    await _run_with_qdrant_path(
        args=args,
        selected_cases=selected_cases,
        output_path=output_path,
        qdrant_path=qdrant_path,
        temporary_qdrant=False,
    )


async def _run_with_qdrant_path(
    *,
    args: argparse.Namespace,
    selected_cases: list[ChatQualityCase],
    output_path: Path,
    qdrant_path: Path,
    temporary_qdrant: bool,
) -> None:
    settings = Settings(
        FILINGDELTA_QDRANT_PATH=str(qdrant_path),
        FILINGDELTA_PARSE_PROVIDER="local",
        FILINGDELTA_USE_LLAMA_PARSE=False,
    )
    service = ChatQAService(settings=settings)
    sources = get_demo_document_sources()
    session_prefix = f"chat-quality-{uuid4().hex[:8]}"

    prewarm_results = []
    if args.prewarm:
        prewarm_results = await _prewarm_selected_documents(
            service=service,
            cases=selected_cases,
            sources=sources,
        )

    started = time.perf_counter()
    results = []
    case_payloads = []
    for index, case in enumerate(selected_cases, start=1):
        document_id, source = _resolve_document(case.document_name, sources)
        session_id = f"{session_prefix}-{index}-{case.case_id}-{document_id}"
        print(f"[{index}/{len(selected_cases)}] {case.case_id}: {case.question}", flush=True)

        case_started = time.perf_counter()
        answer = None
        error = None
        try:
            answer = await service.ask(
                document_id=document_id,
                source=source,
                question=case.question,
                session_id=session_id,
            )
        except Exception as caught:  # noqa: BLE001 - smoke script should report and continue.
            error = caught

        result = evaluate_chat_quality(
            case=case,
            answer=answer,
            wall_ms=_elapsed_ms(case_started),
            error=error,
        )
        results.append(result)
        case_payloads.append(
            {
                **chat_quality_result_to_json(result),
                "document_id": document_id,
                "answer_preview": _preview(answer.answer) if answer is not None else "",
                "citation_count": len(answer.citations) if answer is not None else 0,
                "citations": _citation_payload(answer) if answer is not None else [],
                "telemetry": answer.telemetry.model_dump(mode="json")
                if answer is not None and answer.telemetry is not None
                else None,
            }
        )
        print(_format_result_line(result), flush=True)

    summary = summarize_chat_quality_results(results)
    report = {
        "run": {
            "selected_case_ids": [case.case_id for case in selected_cases],
            "total_wall_ms": _elapsed_ms(started),
            "qdrant_path": str(qdrant_path),
            "temporary_qdrant": temporary_qdrant,
            "output_path": str(output_path),
            "prewarm_enabled": bool(args.prewarm),
            "prewarm_results": prewarm_results,
        },
        "summary": summary,
        "cases": case_payloads,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report:", output_path, flush=True)
    print(
        "summary:",
        f"passed={summary['passed_count']}/{summary['total_cases']}",
        f"failed={summary['failed_count']}",
        flush=True,
    )

    if args.fail_on_quality_failure and summary["failed_count"]:
        raise SystemExit(1)


async def _prewarm_selected_documents(
    *,
    service: ChatQAService,
    cases: list[ChatQualityCase],
    sources: dict[str, Any],
) -> list[dict[str, Any]]:
    documents = []
    seen_document_ids: set[str] = set()
    for case in cases:
        document_id, source = _resolve_document(case.document_name, sources)
        if document_id in seen_document_ids:
            continue
        seen_document_ids.add(document_id)
        documents.append((document_id, source))

    results = []
    for document_id, source in documents:
        print(f"prewarm {source.source_path.name}", flush=True)
        started = time.perf_counter()
        try:
            await service.prewarm_document(document_id=document_id, source=source)
            result = {
                "document_id": document_id,
                "source_path": str(source.source_path),
                "succeeded": True,
                "wall_ms": _elapsed_ms(started),
                "error": None,
            }
        except Exception as error:  # noqa: BLE001 - smoke script should report and continue.
            result = {
                "document_id": document_id,
                "source_path": str(source.source_path),
                "succeeded": False,
                "wall_ms": _elapsed_ms(started),
                "error": f"{type(error).__name__}: {error}",
            }
        print(
            f"  prewarm {'ok' if result['succeeded'] else 'failed'} "
            f"wall={result['wall_ms']}ms",
            flush=True,
        )
        results.append(result)
    return results


def _select_cases(
    *,
    case_ids: list[str],
    max_cases: int | None,
) -> list[ChatQualityCase]:
    selected = list(CHAT_QUALITY_CASES)
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if case.case_id in wanted]
        missing = sorted(wanted - {case.case_id for case in selected})
        if missing:
            raise SystemExit(f"Unknown quality case id(s): {', '.join(missing)}")

    if max_cases is not None:
        selected = selected[: max(0, max_cases)]
    if not selected:
        raise SystemExit("No quality cases selected.")
    return selected


def _resolve_document(document_name: str, sources: dict[str, Any]):
    for document_id, source in sources.items():
        if source.source_path.name == document_name:
            return document_id, source
    available = "\n".join(f"- {source.source_path.name}" for source in sources.values())
    raise FileNotFoundError(
        f"Could not find demo document named {document_name!r}. Available documents:\n{available}"
    )


def _citation_payload(answer) -> list[dict[str, object]]:
    return [
        {
            "source_type": citation.source_type,
            "page_number": citation.page_number,
            "title": citation.title,
            "url": citation.url,
        }
        for citation in answer.citations
    ]


def _format_result_line(result) -> str:
    failed_checks = [check.check_id for check in result.failed_checks]
    if result.passed:
        return f"  pass route={result.actual_route} wall={result.wall_ms}ms"
    if not result.succeeded:
        return f"  failed wall={result.wall_ms}ms error={result.error}"
    return (
        f"  quality-fail route={result.actual_route} wall={result.wall_ms}ms "
        f"checks={','.join(failed_checks)}"
    )


def _print_cases(cases: list[ChatQualityCase]) -> None:
    for case in cases:
        print(
            f"{case.case_id}\tdoc={case.document_name}\t"
            f"expected={case.expected_route}\t{case.question}",
            flush=True,
        )


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _preview(text: str, *, limit: int = 260) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


if __name__ == "__main__":
    main()
