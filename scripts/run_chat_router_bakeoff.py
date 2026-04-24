from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Protocol

from filingdelta.agents.chat_router import ChatRouterAgent
from filingdelta.agents.chat_router_direct import DirectJsonChatRouterAgent
from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.eval.router_bakeoff import (
    ROUTER_BAKEOFF_CASES,
    RouterBakeoffCase,
    RouterBakeoffResult,
    build_router_result,
    result_to_json,
    summarize_router_results,
)
from filingdelta.schemas.chat import ChatRouteDecision
from filingdelta.schemas.filing import FilingDocument, ParserKind
from filingdelta.services.demo_documents import get_demo_document_sources


DEFAULT_OUTPUT = Path("data/outputs/eval/chat_router_bakeoff.json")
METHODOLOGY_NOTES = [
    (
        "direct-json and direct-json-object use the Chat Completions default "
        "temperature because gpt-5-nano rejects custom temperature values on "
        "that endpoint; treat them as exploratory backend comparisons rather "
        "than strict deterministic replacements for the LlamaIndex router."
    )
]


class RouterBackend(Protocol):
    async def route(self, *, question: str, document: FilingDocument) -> ChatRouteDecision:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare top-level chat router backends on fixed golden router cases."
    )
    parser.add_argument(
        "--routers",
        nargs="+",
        default=["llamaindex", "direct-json"],
        choices=["llamaindex", "direct-json", "direct-json-object"],
        help="Router backends to evaluate.",
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
        help="Path to write the bake-off report JSON.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List router cases and exit without API calls.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero if any router has route or boolean mismatches.",
    )
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    selected_cases = _select_cases(case_ids=args.case_ids, max_cases=args.max_cases)

    if args.list_cases:
        for case in selected_cases:
            print(
                f"{case.case_id}\t{case.document_name}\t"
                f"expected={case.expected_route}\t{case.question}"
            )
        return

    settings = Settings()
    sources = get_demo_document_sources()
    documents = _build_case_documents(cases=selected_cases, sources=sources)
    routers = _build_routers(router_names=args.routers, settings=settings)

    started = time.perf_counter()
    results: list[RouterBakeoffResult] = []
    for router_name, router in routers.items():
        print(f"router={router_name}", flush=True)
        for case in selected_cases:
            document = documents[case.case_id]
            result = await _run_router_case(
                router_name=router_name,
                router=router,
                case=case,
                document=document,
            )
            results.append(result)
            _print_result(result)

    report = {
        "run": {
            "routers": list(routers.keys()),
            "case_ids": [case.case_id for case in selected_cases],
            "output_path": str((REPO_ROOT / args.output).resolve()),
            "total_wall_ms": round((time.perf_counter() - started) * 1000),
            "methodology_notes": METHODOLOGY_NOTES,
        },
        "summary": summarize_router_results(results),
        "cases": [result_to_json(result) for result in results],
    }

    output_path = (REPO_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"report: {output_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if args.fail_on_mismatch and _has_mismatch(results):
        raise SystemExit(1)


async def _run_router_case(
    *,
    router_name: str,
    router: RouterBackend,
    case: RouterBakeoffCase,
    document: FilingDocument,
) -> RouterBakeoffResult:
    started = time.perf_counter()
    try:
        decision = await router.route(question=case.question, document=document)
        return build_router_result(
            router_name=router_name,
            case=case,
            decision=decision,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except Exception as error:  # noqa: BLE001 - eval script should capture backend failures.
        return build_router_result(
            router_name=router_name,
            case=case,
            decision=None,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            error=error,
        )


def _build_routers(*, router_names: list[str], settings: Settings) -> dict[str, RouterBackend]:
    routers: dict[str, RouterBackend] = {}
    for router_name in router_names:
        if router_name == "llamaindex":
            routers[router_name] = ChatRouterAgent(settings=settings)
        elif router_name == "direct-json":
            routers[router_name] = DirectJsonChatRouterAgent(settings=settings)
        elif router_name == "direct-json-object":
            routers[router_name] = DirectJsonChatRouterAgent(
                settings=settings,
                response_mode="json_object",
            )
    return routers


def _select_cases(*, case_ids: list[str], max_cases: int | None) -> list[RouterBakeoffCase]:
    cases = list(ROUTER_BAKEOFF_CASES)
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case.case_id in wanted]
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise SystemExit("No router cases selected.")
    return cases


def _build_case_documents(
    *,
    cases: list[RouterBakeoffCase],
    sources,
) -> dict[str, FilingDocument]:
    documents: dict[str, FilingDocument] = {}
    for case in cases:
        source = _resolve_source(case.document_name, sources)
        documents[case.case_id] = FilingDocument(
            document_id=source.source_path.stem.lower().replace(" ", "_"),
            company_name=source.company_name,
            ticker=source.ticker,
            market=source.market,
            doc_type=source.doc_type,
            fiscal_period=source.fiscal_period,
            language=source.language,
            source_path=source.source_path,
            parser_kind=ParserKind.FALLBACK,
            total_pages=0,
        )
    return documents


def _resolve_source(document_name: str, sources):
    for source in sources.values():
        if source.source_path.name == document_name:
            return source
    raise SystemExit(f"Document not found in demo sources: {document_name}")


def _print_result(result: RouterBakeoffResult) -> None:
    status = "ok" if result.fully_matched else "mismatch" if result.succeeded else "failed"
    print(
        f"  {status} {result.case_id} "
        f"expected={result.expected_route} actual={result.actual_route} "
        f"external={result.actual_needs_external_background} "
        f"risk={result.actual_needs_risk_reasoning} "
        f"latency={result.latency_ms}ms",
        flush=True,
    )


def _has_mismatch(results: list[RouterBakeoffResult]) -> bool:
    return any(not result.fully_matched for result in results)


if __name__ == "__main__":
    asyncio.run(main())
