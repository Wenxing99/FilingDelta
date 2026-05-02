from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filingdelta.eval.smoke_v2 import SmokeV2Case, SmokeV2Observation
from filingdelta.schemas.chat import RetrievedChunk
from filingdelta.schemas.filing import EvidenceKind, EvidenceUnit


DIAGNOSIS_MODES = ("semantic_only", "bm25_only", "hybrid_rrf")
ORIGINAL_PILOT_FAILED_QUERY_IDS = {
    "OTA-01",
    "SPORTS-01",
    "NEV-01",
    "HA-02",
    "HA-03",
    "BAIJIU-01",
    "HYDRO-01",
    "BABA-01",
}
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_RRF_K = 60
DIAGNOSIS_SCOPE_ID = "expected_intent_diagnosis/page_hit_only"
DIAGNOSIS_STRATEGY_SOURCE = "manifest_expected_intent"

_ASCII_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-_][A-Za-z]+)*|\d+(?:[.,]\d+)*%?")
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")


@dataclass(frozen=True)
class RankSource:
    source: str
    rank: int
    score: float | None = None


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk: RetrievedChunk
    score: float
    rank_sources: tuple[RankSource, ...]


@dataclass(frozen=True)
class DiagnosisStrategy:
    primary_chunk_kind: str
    fallback_chunk_kinds: tuple[str, ...] = ()
    include_fallback_when_primary_found: bool = False
    primary_top_k: int = 6
    fallback_top_k: int = 6


class BM25Index:
    def __init__(
        self,
        chunks: list[RetrievedChunk],
        *,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> None:
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._term_frequencies = [Counter(tokenize_for_bm25(chunk.text)) for chunk in chunks]
        self._doc_lengths = [sum(frequencies.values()) for frequencies in self._term_frequencies]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._idf = _build_idf(self._term_frequencies)

    def search(self, query: str, *, top_k: int) -> list[RetrievalCandidate]:
        query_terms = tokenize_for_bm25(query)
        if not query_terms or not self._chunks:
            return []

        scored: list[tuple[float, int, RetrievedChunk]] = []
        for index, chunk in enumerate(self._chunks):
            score = self._score_doc(query_terms, doc_index=index)
            if score <= 0:
                continue
            scored.append((score, index, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievalCandidate(
                chunk=chunk,
                score=score,
                rank_sources=(RankSource(source="bm25", rank=rank, score=score),),
            )
            for rank, (score, _, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def _score_doc(self, query_terms: list[str], *, doc_index: int) -> float:
        frequencies = self._term_frequencies[doc_index]
        doc_length = self._doc_lengths[doc_index]
        if doc_length <= 0 or self._avg_doc_length <= 0:
            return 0.0

        score = 0.0
        query_counts = Counter(query_terms)
        for term, query_count in query_counts.items():
            term_frequency = frequencies.get(term, 0)
            if term_frequency <= 0:
                continue
            denominator = term_frequency + self._k1 * (
                1 - self._b + self._b * doc_length / self._avg_doc_length
            )
            score += self._idf.get(term, 0.0) * (
                term_frequency * (self._k1 + 1) / denominator
            ) * query_count
        return score


def tokenize_for_bm25(text: str) -> list[str]:
    tokens: list[str] = []
    for match in re.finditer(r"[\u3400-\u9fff\uf900-\ufaff]+|[A-Za-z]+(?:[-_][A-Za-z]+)*|\d+(?:[.,]\d+)*%?", text):
        token = match.group(0)
        if _CJK_RE.fullmatch(token):
            tokens.extend(_cjk_bigrams(token))
        elif _ASCII_TOKEN_RE.fullmatch(token):
            tokens.append(token.lower())
    return tokens


def rank_semantic_chunks(chunks: list[RetrievedChunk], *, top_k: int) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            chunk=chunk,
            score=chunk.score or 0.0,
            rank_sources=(
                RankSource(source="semantic", rank=rank, score=chunk.score),
            ),
        )
        for rank, chunk in enumerate(chunks[:top_k], start=1)
    ]


def reciprocal_rank_fusion(
    semantic_candidates: list[RetrievalCandidate],
    bm25_candidates: list[RetrievalCandidate],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievalCandidate]:
    by_key: dict[tuple[object, ...], dict[str, Any]] = {}
    insertion_order: dict[tuple[object, ...], int] = {}
    next_order = 0

    for source_name, candidates in (
        ("semantic", semantic_candidates),
        ("bm25", bm25_candidates),
    ):
        seen_in_source: set[tuple[object, ...]] = set()
        for rank, candidate in enumerate(candidates, start=1):
            key = dedupe_key(candidate.chunk)
            if key in seen_in_source:
                continue
            seen_in_source.add(key)
            if key not in by_key:
                by_key[key] = {
                    "chunk": candidate.chunk,
                    "score": 0.0,
                    "rank_sources": {},
                }
                insertion_order[key] = next_order
                next_order += 1
            by_key[key]["score"] += 1 / (rrf_k + rank)
            by_key[key]["rank_sources"][source_name] = RankSource(
                source=source_name,
                rank=rank,
                score=candidate.score,
            )

    fused = [
        RetrievalCandidate(
            chunk=payload["chunk"],
            score=payload["score"],
            rank_sources=tuple(
                payload["rank_sources"][source]
                for source in ("semantic", "bm25")
                if source in payload["rank_sources"]
            ),
        )
        for payload in by_key.values()
    ]
    fused.sort(key=lambda candidate: (-candidate.score, insertion_order[dedupe_key(candidate.chunk)]))
    return fused


def diagnosis_strategy_for_case(case: SmokeV2Case) -> DiagnosisStrategy:
    intent = case.expected_document_evidence_intent
    if intent == "metric_value":
        return DiagnosisStrategy(
            primary_chunk_kind=EvidenceKind.TABLE_ROW.value,
            fallback_chunk_kinds=(EvidenceKind.PAGE_TEXT.value,),
            include_fallback_when_primary_found=True,
            primary_top_k=8,
            fallback_top_k=4,
        )
    if intent == "metric_attribution":
        return DiagnosisStrategy(
            primary_chunk_kind=EvidenceKind.SECTION_TEXT.value,
            fallback_chunk_kinds=(EvidenceKind.TABLE_ROW.value, EvidenceKind.PAGE_TEXT.value),
            include_fallback_when_primary_found=True,
            primary_top_k=4,
            fallback_top_k=3,
        )
    if intent == "business_narrative":
        return DiagnosisStrategy(
            primary_chunk_kind=EvidenceKind.SECTION_TEXT.value,
            fallback_chunk_kinds=(EvidenceKind.PAGE_TEXT.value,),
            include_fallback_when_primary_found=False,
        )
    return DiagnosisStrategy(primary_chunk_kind=EvidenceKind.PAGE_TEXT.value)


def combine_strategy_candidates(
    *,
    primary_candidates: list[RetrievalCandidate],
    fallback_candidates: list[RetrievalCandidate],
    strategy: DiagnosisStrategy,
    final_top_k: int,
) -> list[RetrievalCandidate]:
    if not fallback_candidates:
        return dedupe_candidates(primary_candidates)[:final_top_k]
    if primary_candidates and not strategy.include_fallback_when_primary_found:
        return dedupe_candidates(primary_candidates)[:final_top_k]
    return dedupe_candidates([*primary_candidates, *fallback_candidates])[:final_top_k]


def evidence_units_to_retrieved_chunks(
    *,
    document_id: str,
    evidence_units: list[EvidenceUnit],
) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=unit.evidence_id,
            document_id=document_id,
            page_number=unit.metadata.page_number,
            source_path=Path(unit.metadata.source_path),
            text=unit.text,
            score=None,
            chunk_kind=unit.metadata.chunk_kind.value,
            section_title=unit.metadata.section_title,
            section_type=unit.metadata.section_type,
            table_id=unit.metadata.table_id,
            row_label=unit.metadata.row_label,
            metric_tags=unit.metadata.metric_tags,
            period_hint=unit.metadata.period_hint,
        )
        for unit in evidence_units
    ]


def build_mode_result(
    *,
    case: SmokeV2Case,
    mode: str,
    candidates: list[RetrievalCandidate],
    final_top_k: int,
    retrieval_ms: int,
) -> dict[str, Any]:
    final_candidates = candidates[:final_top_k]
    retrieved_pages = [
        candidate.chunk.page_number
        for candidate in final_candidates
        if candidate.chunk.page_number is not None
    ]
    hit_pages = sorted(set(case.expected_pages).intersection(retrieved_pages))
    supporting_hit_pages = sorted(set(case.supporting_pages).intersection(retrieved_pages))
    page_match_status = _page_match_status(
        primary_hit=bool(hit_pages),
        supporting_hit=bool(supporting_hit_pages),
    )
    return {
        "mode": mode,
        "hit": bool(hit_pages),
        "expected_pages": list(case.expected_pages),
        "supporting_pages": list(case.supporting_pages),
        "retrieved_pages": retrieved_pages,
        "hit_pages": hit_pages,
        "supporting_hit": bool(supporting_hit_pages),
        "supporting_hit_pages": supporting_hit_pages,
        "page_match_status": page_match_status,
        "retrieval_ms": retrieval_ms,
        "observed": observation_from_candidates(
            case=case,
            candidates=final_candidates,
            retrieval_mode=mode,
            latency_ms=retrieval_ms,
        ).__dict__,
        "top_results": [
            candidate_to_json(candidate, rank=rank)
            for rank, candidate in enumerate(final_candidates, start=1)
        ],
    }


def _page_match_status(*, primary_hit: bool, supporting_hit: bool) -> str:
    if primary_hit:
        return "primary_hit"
    if supporting_hit:
        return "partial_support_only"
    return "miss"


def build_live_pilot_context_from_query(query_result: dict[str, Any]) -> dict[str, Any]:
    expected = query_result.get("expected") or {}
    observed = query_result.get("observed") or {}
    scores = query_result.get("scores") or {}
    return {
        "pilot_status": query_result.get("status") or "unknown",
        "pilot_expected_intent": expected.get("document_evidence_intent"),
        "pilot_observed_intent": observed.get("document_evidence_intent"),
        "pilot_intent_hit": scores.get("intent_hit"),
        "pilot_page_hit_at_6": scores.get("page_hit@6"),
        "pilot_route_hit": scores.get("route_hit"),
        "pilot_failure_reasons": list(query_result.get("failure_reasons") or []),
    }


def unknown_live_pilot_context() -> dict[str, Any]:
    return {
        "pilot_status": "unknown",
        "pilot_expected_intent": None,
        "pilot_observed_intent": None,
        "pilot_intent_hit": None,
        "pilot_page_hit_at_6": None,
        "pilot_route_hit": None,
        "pilot_failure_reasons": [],
    }


def enrich_case_result_with_diagnosis_context(
    case_result: dict[str, Any],
    *,
    live_pilot: dict[str, Any] | None,
) -> dict[str, Any]:
    enriched = dict(case_result)
    enriched["diagnosis_scope"] = {
        "id": DIAGNOSIS_SCOPE_ID,
        "strategy_source": DIAGNOSIS_STRATEGY_SOURCE,
        "scoring": "page_hit_only",
        "full_live_pilot_rescue_claimed": False,
    }
    enriched["strategy_source"] = DIAGNOSIS_STRATEGY_SOURCE
    enriched["live_pilot"] = live_pilot or unknown_live_pilot_context()
    enriched["diagnosis_notes"] = _case_diagnosis_notes(enriched)
    return enriched


def build_original_failed_rescue(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rescue_rows: list[dict[str, Any]] = []
    by_query_id = {result["query_id"]: result for result in case_results}
    for query_id in sorted(ORIGINAL_PILOT_FAILED_QUERY_IDS):
        result = by_query_id.get(query_id)
        if result is None:
            rescue_rows.append(
                {
                    "query_id": query_id,
                    "case_id": None,
                    "semantic_only_hit": False,
                    "bm25_only_hit": False,
                    "hybrid_rrf_hit": False,
                    "pilot_status": "unknown",
                    "pilot_observed_intent": None,
                    "pilot_intent_hit": None,
                    "live_intent_mismatch": None,
                    "rescue_status": "not_in_selected_cases",
                    "diagnosis_scope": DIAGNOSIS_SCOPE_ID,
                    "full_live_pilot_rescue_claimed": False,
                }
            )
            continue
        semantic_hit = result["modes"]["semantic_only"]["hit"]
        bm25_hit = result["modes"]["bm25_only"]["hit"]
        hybrid_hit = result["modes"]["hybrid_rrf"]["hit"]
        live_pilot = result.get("live_pilot") or unknown_live_pilot_context()
        rescue_status = _page_rescue_status(
            semantic_hit=semantic_hit,
            bm25_hit=bm25_hit,
            hybrid_hit=hybrid_hit,
            live_intent_mismatch=_live_intent_mismatch(result),
        )
        rescue_rows.append(
            {
                "query_id": query_id,
                "case_id": result["id"],
                "semantic_only_hit": semantic_hit,
                "bm25_only_hit": bm25_hit,
                "hybrid_rrf_hit": hybrid_hit,
                "pilot_status": live_pilot.get("pilot_status", "unknown"),
                "pilot_observed_intent": live_pilot.get("pilot_observed_intent"),
                "pilot_intent_hit": live_pilot.get("pilot_intent_hit"),
                "live_intent_mismatch": _live_intent_mismatch(result),
                "rescue_status": rescue_status,
                "diagnosis_scope": DIAGNOSIS_SCOPE_ID,
                "full_live_pilot_rescue_claimed": False,
            }
        )
    return rescue_rows


def observation_from_candidates(
    *,
    case: SmokeV2Case,
    candidates: list[RetrievalCandidate],
    retrieval_mode: str,
    latency_ms: int | None = None,
) -> SmokeV2Observation:
    chunks = [candidate.chunk for candidate in candidates]
    return SmokeV2Observation(
        executed=True,
        route=case.expected_route,
        document_evidence_intent=case.expected_document_evidence_intent,
        retrieval_mode=retrieval_mode,
        retrieved_evidence_kinds=tuple(chunk.chunk_kind for chunk in chunks if chunk.chunk_kind),
        citation_pages=tuple(chunk.page_number for chunk in chunks if chunk.page_number),
        retrieved_row_labels=tuple(chunk.row_label for chunk in chunks if chunk.row_label),
        retrieved_metric_tags=tuple(sorted({tag for chunk in chunks for tag in chunk.metric_tags})),
        retrieved_section_types=tuple(chunk.section_type for chunk in chunks if chunk.section_type),
        latency_ms=latency_ms,
    )


def candidate_to_json(candidate: RetrievalCandidate, *, rank: int) -> dict[str, Any]:
    chunk = candidate.chunk
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "chunk_kind": chunk.chunk_kind,
        "page_number": chunk.page_number,
        "row_label": chunk.row_label,
        "section_title": chunk.section_title,
        "section_type": chunk.section_type,
        "metric_tags": chunk.metric_tags,
        "score": candidate.score,
        "rank_sources": [
            {"source": source.source, "rank": source.rank, "score": source.score}
            for source in candidate.rank_sources
        ],
        "preview": preview_text(chunk.text),
    }


def dedupe_candidates(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    deduped: list[RetrievalCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        key = dedupe_key(candidate.chunk)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def dedupe_key(chunk: RetrievedChunk) -> tuple[object, ...]:
    return (
        chunk.chunk_kind,
        chunk.page_number,
        chunk.row_label,
        normalize_for_match(chunk.text)[:180] if not chunk.row_label else "",
    )


def preview_text(text: str, *, limit: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def render_diagnosis_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    pilot_report = report.get("live_pilot_report") or {}
    lines = [
        "# Golden Queries v2 Retrieval Diagnosis",
        "",
        "## 摘要",
        "",
        f"- Manifest：`{report['manifest_path']}`",
        f"- Case 总数：`{summary['total_cases']}`",
        f"- semantic_only page hit：`{summary['mode_hits']['semantic_only']}`",
        f"- bm25_only page hit：`{summary['mode_hits']['bm25_only']}`",
        f"- hybrid_rrf page hit：`{summary['mode_hits']['hybrid_rrf']}`",
        f"- 诊断口径：`{DIAGNOSIS_SCOPE_ID}`；strategy 来自 manifest expected intent，不是 live router observed intent。",
        "- 下表的 rescued 只表示 expected-intent/page-hit-only 诊断命中 gold page；不等于 full live pilot rescue，也不代表 answer synthesis 通过。",
        f"- Live pilot context：`{pilot_report.get('status', 'unknown')}` "
        f"`{pilot_report.get('path', '-')}`",
        "",
        "## 原 Pilot 失败 Case 救回情况",
        "",
        "| Query ID | pilot_status | pilot_observed_intent | pilot_intent_hit | semantic_page | bm25_page | hybrid_page | page-hit-only 结论 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for rescue in report["original_failed_case_rescue"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{escape_markdown_table(rescue['query_id'])}`",
                    escape_markdown_table(rescue.get("pilot_status") or "unknown"),
                    escape_markdown_table(rescue.get("pilot_observed_intent") or "unknown"),
                    _tri_state_label(rescue.get("pilot_intent_hit")),
                    _hit_label(rescue["semantic_only_hit"]),
                    _hit_label(rescue["bm25_only_hit"]),
                    _hit_label(rescue["hybrid_rrf_hit"]),
                    escape_markdown_table(rescue["rescue_status"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 14 条 Case 对照",
            "",
            "| Case | 公司 | expected_intent | live_observed_intent | expected_pages | semantic_only | bm25_only | hybrid_rrf |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for case_result in report["cases"]:
        modes = case_result["modes"]
        live_pilot = case_result.get("live_pilot") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{escape_markdown_table(case_result['id'])}`",
                    escape_markdown_table(case_result.get("company") or "-"),
                    escape_markdown_table(case_result.get("expected_document_evidence_intent") or "-"),
                    escape_markdown_table(live_pilot.get("pilot_observed_intent") or "unknown"),
                    ", ".join(str(page) for page in case_result["expected_pages"]) or "-",
                    _mode_cell(modes["semantic_only"]),
                    _mode_cell(modes["bm25_only"]),
                    _mode_cell(modes["hybrid_rrf"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Top Snippets", ""])
    for case_result in report["cases"]:
        lines.append(f"### `{case_result['id']}`")
        if case_result.get("diagnosis_notes"):
            lines.append(
                "- diagnosis_notes="
                f"`{escape_markdown_table(', '.join(case_result['diagnosis_notes']))}`"
            )
        for mode in DIAGNOSIS_MODES:
            mode_result = case_result["modes"][mode]
            lines.append(
                f"- `{mode}` retrieved_pages="
                f"`{', '.join(str(page) for page in mode_result['retrieved_pages']) or '-'}`"
            )
            for top in mode_result["top_results"][:3]:
                source_bits = ", ".join(
                    f"{source['source']}#{source['rank']}"
                    for source in top["rank_sources"]
                )
                lines.append(
                    "  - "
                    f"rank={top['rank']} page={top['page_number']} "
                    f"kind={top['chunk_kind']} source={source_bits}: "
                    f"{escape_markdown_table(top['preview'])}"
                )
        lines.append("")

    lines.extend(
        [
            "## 口径说明",
            "",
            "- 本诊断只比较 retrieval 排序，不运行 answer synthesis。",
            "- `expected_pages` 直接来自当前 smoke manifest；本报告不修改 gold。",
            "- `semantic_only` 使用正式 live chat strategy 的 top_k 分配；BM25 / hybrid 使用诊断对照的 top20 候选。",
            "- BM25 corpus 按当前 document 和 chunk_kind 构造，避免跨文档污染。",
            "- `hybrid_rrf` 在同一 evidence-kind 内融合 semantic top20 与 BM25 top20，RRF k=60。",
            "- 原 pilot 失败表里的 `page_rescued_*` 只说明 retrieval page 命中；若 live pilot intent 不一致，会标记 `page_rescued_but_live_intent_mismatch`。",
            "",
        ]
    )
    return "\n".join(lines)


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def escape_markdown_table(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _build_idf(term_frequencies: list[Counter[str]]) -> dict[str, float]:
    doc_count = len(term_frequencies)
    document_frequency: Counter[str] = Counter()
    for frequencies in term_frequencies:
        document_frequency.update(frequencies.keys())
    return {
        term: math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        for term, df in document_frequency.items()
    }


def _cjk_bigrams(text: str) -> list[str]:
    if len(text) <= 1:
        return [text]
    return [text[index : index + 2] for index in range(len(text) - 1)]


def _mode_cell(mode_result: dict[str, Any]) -> str:
    pages = ", ".join(str(page) for page in mode_result["retrieved_pages"]) or "-"
    return f"{_hit_label(mode_result['hit'])} `{escape_markdown_table(pages)}`"


def _hit_label(hit: bool) -> str:
    return "hit" if hit else "miss"


def _tri_state_label(value: object) -> str:
    if value is True:
        return "hit"
    if value is False:
        return "miss"
    return "unknown"


def _page_rescue_status(
    *,
    semantic_hit: bool,
    bm25_hit: bool,
    hybrid_hit: bool,
    live_intent_mismatch: bool | None,
) -> str:
    page_rescued_modes = []
    if bm25_hit and not semantic_hit:
        page_rescued_modes.append("bm25_only")
    if hybrid_hit and not semantic_hit:
        page_rescued_modes.append("hybrid_rrf")
    if page_rescued_modes and live_intent_mismatch is True:
        return "page_rescued_but_live_intent_mismatch"
    if page_rescued_modes:
        return "expected_intent_page_rescued_by_" + "+".join(page_rescued_modes)
    if semantic_hit:
        return "semantic_only_page_hit_in_rerun"
    return "not_page_rescued"


def _case_diagnosis_notes(case_result: dict[str, Any]) -> list[str]:
    notes = [DIAGNOSIS_SCOPE_ID, "not_full_live_pilot_rescue"]
    if _live_intent_mismatch(case_result):
        notes.append("live_intent_mismatch")
        if _has_expected_intent_page_rescue(case_result):
            notes.append("page_rescued_but_live_intent_mismatch")
    return notes


def _has_expected_intent_page_rescue(case_result: dict[str, Any]) -> bool:
    modes = case_result["modes"]
    return (
        not modes["semantic_only"]["hit"]
        and (modes["bm25_only"]["hit"] or modes["hybrid_rrf"]["hit"])
    )


def _live_intent_mismatch(case_result: dict[str, Any]) -> bool | None:
    expected_intent = case_result.get("expected_document_evidence_intent")
    live_pilot = case_result.get("live_pilot") or {}
    observed_intent = live_pilot.get("pilot_observed_intent")
    if not expected_intent or not observed_intent:
        return None
    return observed_intent != expected_intent
