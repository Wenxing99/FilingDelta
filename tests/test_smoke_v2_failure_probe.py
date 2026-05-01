from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from filingdelta.eval.failure_probe import (
    build_gold_page_coverage,
    classify_failure_category,
    rank_expected_pages,
    render_failure_probe_markdown,
)
from filingdelta.eval.retrieval_diagnosis import RankSource, RetrievalCandidate
from filingdelta.schemas.chat import RetrievedChunk
from filingdelta.schemas.filing import (
    EvidenceKind,
    EvidenceMetadata,
    EvidenceUnit,
    FilingDocument,
    ParsedFiling,
    ParsedPage,
    ParserKind,
)


def test_gold_page_coverage_classifies_evidence_units() -> None:
    parsed = _parsed_filing(
        [
            ParsedPage(page_number=10, text="收入和分部经营数据"),
            ParsedPage(page_number=11, text="渠道效率说明"),
        ]
    )
    evidence_units = [
        _evidence_unit(page=10, kind=EvidenceKind.PAGE_TEXT, text="收入和分部经营数据"),
        _evidence_unit(
            page=10,
            kind=EvidenceKind.TABLE_ROW,
            text="Table row: 营业收入",
            row_label="营业收入",
            metric_tags=["revenue"],
        ),
        _evidence_unit(
            page=11,
            kind=EvidenceKind.SECTION_TEXT,
            text="渠道效率说明",
            section_title="渠道效率",
            section_type="business_review",
        ),
    ]

    coverage = build_gold_page_coverage(
        parsed_filing=parsed,
        evidence_units=evidence_units,
        expected_pages=[10, 11],
    )

    assert coverage["all_expected_pages_parsed"] is True
    assert coverage["any_page_text_on_gold_page"] is True
    assert coverage["any_table_row_on_gold_page"] is True
    assert coverage["any_section_text_on_gold_page"] is True
    assert coverage["pages"][0]["table_row_labels"] == ["营业收入"]
    assert coverage["pages"][1]["section_headings"] == ["渠道效率"]


def test_failure_probe_classifies_router_missing_evidence_and_low_rank_separately() -> None:
    coverage_with_table = _coverage(
        all_parsed=True,
        any_evidence=True,
        any_page_text=True,
        any_table_row=True,
        any_section_text=False,
    )
    router = classify_failure_category(
        expected_intent="metric_attribution",
        live_observed_intent="business_narrative",
        gold_page_coverage=coverage_with_table,
        mode_rankings=_rankings(semantic_hit=False, bm25_hit=True, hybrid_hit=True),
        top_false_positives={},
    )
    missing_evidence = classify_failure_category(
        expected_intent="metric_value",
        live_observed_intent="metric_value",
        gold_page_coverage=_coverage(
            all_parsed=True,
            any_evidence=False,
            any_page_text=False,
            any_table_row=False,
            any_section_text=False,
        ),
        mode_rankings=_rankings(semantic_hit=False, bm25_hit=False, hybrid_hit=False),
        top_false_positives={},
    )
    low_rank = classify_failure_category(
        expected_intent="metric_value",
        live_observed_intent="metric_value",
        gold_page_coverage=coverage_with_table,
        mode_rankings={
            "semantic_only": _ranking(status="ranked", best_rank=9, hit=False),
            "bm25_only": _ranking(status="not_in_top_candidates", best_rank=None, hit=False),
            "hybrid_rrf": _ranking(status="not_in_top_candidates", best_rank=None, hit=False),
        },
        top_false_positives={},
    )

    assert router["failure_category"] == "page_rescued_but_live_intent_mismatch"
    assert missing_evidence["failure_category"] == "gold_page_evidence_missing"
    assert low_rank["failure_category"] == "gold_page_low_rank"


def test_failure_probe_report_does_not_modify_expected_pages() -> None:
    expected_pages = [30, 47]
    parsed = _parsed_filing([ParsedPage(page_number=30, text="客户存款")])

    coverage = build_gold_page_coverage(
        parsed_filing=parsed,
        evidence_units=[],
        expected_pages=expected_pages,
    )
    ranking = rank_expected_pages(
        candidates=[
            RetrievalCandidate(
                chunk=_chunk(page=47),
                score=0.5,
                rank_sources=(RankSource(source="semantic", rank=1, score=0.5),),
            )
        ],
        expected_pages=expected_pages,
        final_top_k=6,
    )

    assert expected_pages == [30, 47]
    assert coverage["expected_pages"] == [30, 47]
    assert ranking["per_expected_page"] == {"30": "not_in_top_candidates", "47": 1}


def test_failure_probe_markdown_shows_required_case_fields() -> None:
    report = {
        "manifest_path": "manifest.json",
        "retrieval_diagnosis_path": "diagnosis.json",
        "pilot_report_path": "pilot.json",
        "cases": [
            {
                "query_id": "OTA-01",
                "query": "收入分别是多少？",
                "expected_pages": [7, 8],
                "expected_intent": "metric_value",
                "live_observed_intent": "metric_value",
                "pilot_status": "failed",
                "failure_category": "table_extraction_gap",
                "recommended_next_fix": "扩展 table_row 抽取并加回归测试。",
                "gold_page_coverage": _coverage(
                    all_parsed=True,
                    any_evidence=True,
                    any_page_text=True,
                    any_table_row=False,
                    any_section_text=False,
                ),
                "mode_rankings": _rankings(False, False, False),
                "top_false_positive_pages": {
                    "semantic_only": [],
                    "bm25_only": [],
                    "hybrid_rrf": [],
                },
            }
        ],
    }

    rendered = render_failure_probe_markdown(report)

    assert "收入分别是多少？" in rendered
    assert "7, 8" in rendered
    assert "table_extraction_gap" in rendered
    assert "扩展 table_row 抽取并加回归测试。" in rendered


def test_failure_probe_target_query_ids_follow_selected_cases() -> None:
    module = _load_failure_probe_runner()
    selected_cases = [
        SimpleNamespace(case_id="海尔智家_2025_annual_report-14186f9f::HA-03"),
        SimpleNamespace(case_id="阿里巴巴_2025_annual_report-8ab12348::BABA-01"),
    ]

    assert module._selected_query_ids_from_cases(selected_cases) == ["HA-03", "BABA-01"]


def _parsed_filing(pages: list[ParsedPage]) -> ParsedFiling:
    return ParsedFiling(
        document=FilingDocument(
            document_id="doc",
            company_name="测试公司",
            source_path=Path("source.pdf"),
            parser_kind=ParserKind.PYMUPDF,
            total_pages=len(pages),
        ),
        pages=pages,
    )


def _evidence_unit(
    *,
    page: int,
    kind: EvidenceKind,
    text: str,
    row_label: str | None = None,
    section_title: str | None = None,
    section_type: str | None = None,
    metric_tags: list[str] | None = None,
) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=f"{kind.value}-{page}",
        text=text,
        metadata=EvidenceMetadata(
            document_id="doc",
            source_path=Path("source.pdf"),
            page_number=page,
            parser_kind=ParserKind.PYMUPDF,
            chunk_kind=kind,
            row_label=row_label,
            section_title=section_title,
            section_type=section_type,
            metric_tags=metric_tags or [],
        ),
    )


def _chunk(*, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{page}",
        document_id="doc",
        page_number=page,
        source_path=Path("source.pdf"),
        text="客户存款",
        chunk_kind="page_text",
    )


def _coverage(
    *,
    all_parsed: bool,
    any_evidence: bool,
    any_page_text: bool,
    any_table_row: bool,
    any_section_text: bool,
) -> dict:
    return {
        "expected_pages": [1],
        "all_expected_pages_parsed": all_parsed,
        "any_expected_page_parsed": all_parsed,
        "any_evidence_unit_on_gold_page": any_evidence,
        "any_page_text_on_gold_page": any_page_text,
        "any_section_text_on_gold_page": any_section_text,
        "any_table_row_on_gold_page": any_table_row,
        "pages": [
            {
                "page_number": 1,
                "parsed_page_exists": all_parsed,
                "has_page_text": any_page_text,
                "has_section_text": any_section_text,
                "has_table_row": any_table_row,
                "table_row_labels": [],
                "section_headings": [],
                "metric_tags": [],
                "page_snippet": "snippet",
            }
        ],
    }


def _load_failure_probe_runner():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_smoke_v2_failure_probe.py"
    spec = importlib.util.spec_from_file_location("run_smoke_v2_failure_probe", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rankings(
    semantic_hit: bool,
    bm25_hit: bool,
    hybrid_hit: bool,
) -> dict[str, dict]:
    return {
        "semantic_only": _ranking(status="ranked" if semantic_hit else "not_in_top_candidates", best_rank=1 if semantic_hit else None, hit=semantic_hit),
        "bm25_only": _ranking(status="ranked" if bm25_hit else "not_in_top_candidates", best_rank=1 if bm25_hit else None, hit=bm25_hit),
        "hybrid_rrf": _ranking(status="ranked" if hybrid_hit else "not_in_top_candidates", best_rank=1 if hybrid_hit else None, hit=hybrid_hit),
    }


def _ranking(*, status: str, best_rank: int | None, hit: bool) -> dict:
    return {
        "status": status,
        "best_rank": best_rank,
        "best_final_rank": best_rank if hit else None,
        "final_top_k_hit": hit,
        "per_expected_page": {"1": best_rank or "not_in_top_candidates"},
    }
