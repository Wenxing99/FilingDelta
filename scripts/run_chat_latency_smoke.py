from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.schemas.chat import ChatAnswer
from filingdelta.services.chat_qa import ChatQAService
from filingdelta.services.demo_documents import get_demo_document_sources


DEFAULT_OUTPUT = Path("data/outputs/eval/chat_latency_smoke.json")
DEFAULT_QDRANT_ROOT = Path("data/outputs/eval/qdrant_chat_latency_smoke")

CMB_ANNUAL = "招商银行2025年度报告.pdf"
TCEHY_ANNUAL = "腾讯控股2025年度报告.pdf"


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    case_set: str
    document_name: str
    question: str
    expected_route: str | None = None
    notes: str = ""


SMOKE_CASES: tuple[SmokeCase, ...] = (
    SmokeCase(
        case_id="cmb-real-estate-risk",
        case_set="quick",
        document_name=CMB_ANNUAL,
        question="招商银行如何管控房地产风险？",
        expected_route="document_only",
        notes="Narrative risk question that should stay inside the filing.",
    ),
    SmokeCase(
        case_id="cmb-customer-deposits",
        case_set="quick",
        document_name=CMB_ANNUAL,
        question="招商银行客户存款有什么变化？",
        expected_route="document_only",
        notes="Document-only operating / balance-sheet discussion.",
    ),
    SmokeCase(
        case_id="tcehy-ai-ads",
        case_set="quick",
        document_name=TCEHY_ANNUAL,
        question="腾讯如何描述 AI 广告能力？",
        expected_route="document_only",
        notes="Narrative AI / advertising question expected to use section_text.",
    ),
    SmokeCase(
        case_id="tcehy-capex",
        case_set="quick",
        document_name=TCEHY_ANNUAL,
        question="腾讯2025年资本开支是多少？",
        expected_route="document_only",
        notes="Metric-heavy question expected to stay on page_text.",
    ),
    SmokeCase(
        case_id="cmb-roe-concept",
        case_set="mixed",
        document_name=CMB_ANNUAL,
        question="什么是净资产收益率？结合当前文档里的披露解释它说明什么。",
        expected_route="mixed",
        notes="Concept plus filing facts; intentionally exercises external evidence.",
    ),
    SmokeCase(
        case_id="tcehy-capex-meaning",
        case_set="mixed",
        document_name=TCEHY_ANNUAL,
        question="资本开支增加通常意味着什么？结合腾讯这份文档回答。",
        expected_route="mixed",
        notes="Implication question; intentionally exercises mixed QA.",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FilingDelta chat latency smoke questions and write telemetry JSON."
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["quick"],
        help="Case sets to run. Use quick, mixed, or all. Default: quick.",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=[],
        help="Run a specific case id. Can be passed multiple times.",
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
        default=None,
        help="Optional persistent Qdrant path. Defaults to an isolated temporary path.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete --qdrant-path before running. Ignored for the default temporary path.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print available smoke cases without calling models.",
    )
    parser.add_argument(
        "--fail-on-route-mismatch",
        action="store_true",
        help="Exit non-zero when an expected route does not match.",
    )
    return parser


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    selected_cases = _select_cases(
        set_names=args.sets,
        case_ids=args.case_ids,
        max_cases=args.max_cases,
    )

    if args.list_cases:
        _print_cases(selected_cases)
        return

    asyncio.run(_run_smoke(args=args, selected_cases=selected_cases))


async def _run_smoke(*, args: argparse.Namespace, selected_cases: list[SmokeCase]) -> None:
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.qdrant_path is None:
        with tempfile.TemporaryDirectory(prefix="filingdelta-chat-smoke-") as temp_dir:
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
    selected_cases: list[SmokeCase],
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
    session_prefix = f"chat-smoke-{uuid4().hex[:8]}"

    started = time.perf_counter()
    results = []
    for index, case in enumerate(selected_cases, start=1):
        document_id, source = _resolve_document(case.document_name, sources)
        session_id = f"{session_prefix}-{document_id}"
        print(f"[{index}/{len(selected_cases)}] {case.case_id}: {case.question}", flush=True)

        case_started = time.perf_counter()
        try:
            answer = await service.ask(
                document_id=document_id,
                source=source,
                question=case.question,
                session_id=session_id,
            )
            result = _build_success_result(
                case=case,
                document_id=document_id,
                answer=answer,
                wall_ms=_elapsed_ms(case_started),
            )
            print(_format_success_line(result), flush=True)
        except Exception as error:  # noqa: BLE001 - smoke script should record failures and continue.
            result = _build_failure_result(
                case=case,
                document_id=document_id,
                error=error,
                wall_ms=_elapsed_ms(case_started),
            )
            print(_format_failure_line(result), flush=True)
        results.append(result)

    summary = _build_summary(results)
    report = {
        "run": {
            "selected_sets": list(args.sets),
            "selected_case_ids": [case.case_id for case in selected_cases],
            "total_wall_ms": _elapsed_ms(started),
            "qdrant_path": str(qdrant_path),
            "temporary_qdrant": temporary_qdrant,
            "output_path": str(output_path),
        },
        "summary": summary,
        "cases": results,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report:", output_path, flush=True)
    print(
        "summary:",
        f"ok={summary['succeeded_count']}/{summary['total_cases']}",
        f"route_mismatch={summary['route_mismatch_count']}",
        flush=True,
    )

    if summary["failed_count"] or (args.fail_on_route_mismatch and summary["route_mismatch_count"]):
        raise SystemExit(1)


def _select_cases(
    *,
    set_names: list[str],
    case_ids: list[str],
    max_cases: int | None,
) -> list[SmokeCase]:
    normalized_sets = {item.strip() for item in set_names if item.strip()}
    if "all" in normalized_sets:
        selected = list(SMOKE_CASES)
    else:
        known_sets = {case.case_set for case in SMOKE_CASES}
        unknown_sets = sorted(normalized_sets - known_sets)
        if unknown_sets:
            raise SystemExit(f"Unknown case set(s): {', '.join(unknown_sets)}")
        selected = [case for case in SMOKE_CASES if case.case_set in normalized_sets]

    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if case.case_id in wanted]
        missing = sorted(wanted - {case.case_id for case in selected})
        if missing:
            raise SystemExit(f"Unknown or unselected case id(s): {', '.join(missing)}")

    if max_cases is not None:
        selected = selected[: max(0, max_cases)]
    if not selected:
        raise SystemExit("No smoke cases selected.")
    return selected


def _resolve_document(document_name: str, sources: dict[str, Any]):
    for document_id, source in sources.items():
        if source.source_path.name == document_name:
            return document_id, source
    available = "\n".join(f"- {source.source_path.name}" for source in sources.values())
    raise FileNotFoundError(
        f"Could not find demo document named {document_name!r}. Available documents:\n{available}"
    )


def _build_success_result(
    *,
    case: SmokeCase,
    document_id: str,
    answer: ChatAnswer,
    wall_ms: int,
) -> dict[str, Any]:
    telemetry = answer.telemetry
    telemetry_payload = telemetry.model_dump(mode="json") if telemetry else None
    route_match = case.expected_route is None or answer.route == case.expected_route
    citations = [
        {
            "source_type": citation.source_type,
            "page_number": citation.page_number,
            "title": citation.title,
            "url": citation.url,
        }
        for citation in answer.citations
    ]
    return {
        **asdict(case),
        "document_id": document_id,
        "succeeded": True,
        "route": answer.route,
        "expected_route_matched": route_match,
        "retrieval_mode": answer.retrieval_mode,
        "wall_ms": wall_ms,
        "telemetry": telemetry_payload,
        "answer_chars": len(answer.answer),
        "answer_preview": _preview(answer.answer),
        "citation_count": len(answer.citations),
        "citations": citations,
        "error": None,
    }


def _build_failure_result(
    *,
    case: SmokeCase,
    document_id: str,
    error: Exception,
    wall_ms: int,
) -> dict[str, Any]:
    return {
        **asdict(case),
        "document_id": document_id,
        "succeeded": False,
        "route": None,
        "expected_route_matched": False,
        "retrieval_mode": None,
        "wall_ms": wall_ms,
        "telemetry": None,
        "answer_chars": 0,
        "answer_preview": "",
        "citation_count": 0,
        "citations": [],
        "error": f"{type(error).__name__}: {error}",
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [result for result in results if result["succeeded"]]
    route_mismatches = [
        result
        for result in succeeded
        if result["expected_route"] and not result["expected_route_matched"]
    ]
    route_counts: dict[str, int] = {}
    for result in succeeded:
        route = str(result["route"])
        route_counts[route] = route_counts.get(route, 0) + 1

    return {
        "total_cases": len(results),
        "succeeded_count": len(succeeded),
        "failed_count": len(results) - len(succeeded),
        "route_counts": route_counts,
        "route_mismatch_count": len(route_mismatches),
        "route_mismatch_cases": [result["case_id"] for result in route_mismatches],
        "total_wall_ms": sum(int(result["wall_ms"]) for result in results),
        "avg_wall_ms": round(
            sum(int(result["wall_ms"]) for result in results) / len(results),
            2,
        )
        if results
        else 0.0,
    }


def _format_success_line(result: dict[str, Any]) -> str:
    telemetry = result.get("telemetry") or {}
    steps = telemetry.get("steps") or {}
    return (
        f"  ok route={result['route']} expected={result['expected_route']} "
        f"wall={result['wall_ms']}ms total={telemetry.get('total_latency_ms', 0):.0f}ms "
        f"index={_step_ms(steps, 'index_build_ms')} "
        f"retrieval={_step_ms(steps, 'document_retrieval_ms')} "
        f"external={_step_ms(steps, 'external_search_ms')} "
        f"answerer={_step_ms(steps, 'answerer_ms')} "
        f"summarizer={_step_ms(steps, 'memory_summarizer_ms')} "
        f"citations={result['citation_count']}"
    )


def _format_failure_line(result: dict[str, Any]) -> str:
    return f"  failed wall={result['wall_ms']}ms error={result['error']}"


def _step_ms(steps: dict[str, Any], key: str) -> str:
    value = steps.get(key)
    if value is None:
        return "-"
    return f"{float(value):.0f}ms"


def _print_cases(cases: list[SmokeCase]) -> None:
    for case in cases:
        print(
            f"{case.case_id}\tset={case.case_set}\tdoc={case.document_name}\t"
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
