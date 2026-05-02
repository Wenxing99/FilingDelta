from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llama_index.core.callbacks import CallbackManager

from filingdelta.retrieval.retriever import DocumentChunkRetriever
from filingdelta.schemas.chat import RetrievedChunk
from filingdelta.schemas.filing import EvidenceKind, EvidenceUnit


DEFAULT_PAGE_TEXT_HYBRID_SEMANTIC_TOP_K = 5
DEFAULT_PAGE_TEXT_HYBRID_BM25_TOP_K = 5
DEFAULT_PAGE_TEXT_HYBRID_RRF_K = 20
DEFAULT_PAGE_TEXT_HYBRID_ALPHA_SEMANTIC = 0.4
DEFAULT_PAGE_TEXT_HYBRID_FINAL_TOP_K = 6
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

_ASCII_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-_][A-Za-z]+)*|\d+(?:[.,]\d+)*%?")
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")


@dataclass(frozen=True)
class PageTextHybridConfig:
    semantic_top_k: int = DEFAULT_PAGE_TEXT_HYBRID_SEMANTIC_TOP_K
    bm25_top_k: int = DEFAULT_PAGE_TEXT_HYBRID_BM25_TOP_K
    rrf_k: int = DEFAULT_PAGE_TEXT_HYBRID_RRF_K
    alpha_semantic: float = DEFAULT_PAGE_TEXT_HYBRID_ALPHA_SEMANTIC
    final_top_k: int = DEFAULT_PAGE_TEXT_HYBRID_FINAL_TOP_K


@dataclass(frozen=True)
class RankSource:
    source: Literal["semantic", "bm25"]
    rank: int
    score: float | None = None


@dataclass(frozen=True)
class HybridCandidate:
    chunk: RetrievedChunk
    score: float
    rank_sources: tuple[RankSource, ...]


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

    def search(self, query: str, *, top_k: int) -> list[HybridCandidate]:
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
            HybridCandidate(
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


def retrieve_page_text_hybrid(
    *,
    retriever: DocumentChunkRetriever,
    document_id: str,
    question: str,
    page_text_chunks: list[RetrievedChunk],
    config: PageTextHybridConfig | None = None,
    callback_manager: CallbackManager | None = None,
) -> list[RetrievedChunk]:
    active_config = config or PageTextHybridConfig()
    corpus = _filter_page_text_corpus(document_id=document_id, chunks=page_text_chunks)
    if not corpus:
        return []

    semantic_chunks = retriever.retrieve(
        document_id=document_id,
        question=question,
        top_k=active_config.semantic_top_k,
        chunk_kind=EvidenceKind.PAGE_TEXT.value,
        callback_manager=callback_manager,
    )
    semantic_candidates = rank_semantic_chunks(
        _filter_page_text_corpus(document_id=document_id, chunks=semantic_chunks),
        top_k=active_config.semantic_top_k,
    )
    bm25_candidates = BM25Index(corpus).search(question, top_k=active_config.bm25_top_k)
    fused = weighted_reciprocal_rank_fusion(
        semantic_candidates=semantic_candidates,
        bm25_candidates=bm25_candidates,
        rrf_k=active_config.rrf_k,
        alpha_semantic=active_config.alpha_semantic,
    )
    return [
        candidate.chunk.model_copy(update={"score": candidate.score})
        for candidate in fused[: active_config.final_top_k]
    ]


def evidence_units_to_page_text_chunks(
    *,
    document_id: str,
    evidence_units: list[EvidenceUnit],
) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for unit in evidence_units:
        if unit.metadata.chunk_kind != EvidenceKind.PAGE_TEXT:
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=unit.evidence_id,
                document_id=document_id,
                page_number=unit.metadata.page_number,
                source_path=Path(unit.metadata.source_path),
                text=unit.text,
                score=None,
                chunk_kind=EvidenceKind.PAGE_TEXT.value,
                section_title=unit.metadata.section_title,
                section_type=unit.metadata.section_type,
                table_id=unit.metadata.table_id,
                row_label=unit.metadata.row_label,
                metric_tags=unit.metadata.metric_tags,
                period_hint=unit.metadata.period_hint,
            )
        )
    return chunks


def rank_semantic_chunks(chunks: list[RetrievedChunk], *, top_k: int) -> list[HybridCandidate]:
    return [
        HybridCandidate(
            chunk=chunk,
            score=chunk.score or 0.0,
            rank_sources=(
                RankSource(source="semantic", rank=rank, score=chunk.score),
            ),
        )
        for rank, chunk in enumerate(chunks[:top_k], start=1)
    ]


def weighted_reciprocal_rank_fusion(
    *,
    semantic_candidates: list[HybridCandidate],
    bm25_candidates: list[HybridCandidate],
    rrf_k: int,
    alpha_semantic: float,
) -> list[HybridCandidate]:
    if not 0 <= alpha_semantic <= 1:
        raise ValueError("alpha_semantic must be between 0 and 1.")

    by_key: dict[tuple[object, ...], dict[str, object]] = {}
    insertion_order: dict[tuple[object, ...], int] = {}
    next_order = 0

    for source_name, source_weight, candidates in (
        ("semantic", alpha_semantic, semantic_candidates),
        ("bm25", 1 - alpha_semantic, bm25_candidates),
    ):
        if source_weight <= 0:
            continue
        seen_in_source: set[tuple[object, ...]] = set()
        for rank, candidate in enumerate(candidates, start=1):
            key = _candidate_key(candidate.chunk)
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
            by_key[key]["score"] = float(by_key[key]["score"]) + source_weight / (rrf_k + rank)
            rank_sources = by_key[key]["rank_sources"]
            assert isinstance(rank_sources, dict)
            rank_sources[source_name] = RankSource(
                source=source_name,  # type: ignore[arg-type]
                rank=rank,
                score=candidate.score,
            )

    fused = []
    for key, payload in by_key.items():
        rank_sources = payload["rank_sources"]
        assert isinstance(rank_sources, dict)
        fused.append(
            HybridCandidate(
                chunk=payload["chunk"],  # type: ignore[arg-type]
                score=float(payload["score"]),
                rank_sources=tuple(
                    rank_sources[source]
                    for source in ("semantic", "bm25")
                    if source in rank_sources
                ),
            )
        )
    fused.sort(key=lambda candidate: (-candidate.score, insertion_order[_candidate_key(candidate.chunk)]))
    return fused


def tokenize_for_bm25(text: str) -> list[str]:
    tokens: list[str] = []
    for match in re.finditer(
        r"[\u3400-\u9fff\uf900-\ufaff]+|[A-Za-z]+(?:[-_][A-Za-z]+)*|\d+(?:[.,]\d+)*%?",
        text,
    ):
        token = match.group(0)
        if _CJK_RE.fullmatch(token):
            tokens.extend(_cjk_bigrams(token))
        elif _ASCII_TOKEN_RE.fullmatch(token):
            tokens.append(token.lower())
    return tokens


def _filter_page_text_corpus(
    *,
    document_id: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    return [
        chunk
        for chunk in chunks
        if chunk.document_id == document_id and chunk.chunk_kind == EvidenceKind.PAGE_TEXT.value
    ]


def _candidate_key(chunk: RetrievedChunk) -> tuple[object, ...]:
    return (
        chunk.document_id,
        chunk.chunk_id,
        chunk.chunk_kind,
        chunk.page_number,
    )


def _build_idf(term_frequencies: list[Counter[str]]) -> dict[str, float]:
    doc_count = len(term_frequencies)
    document_frequencies: Counter[str] = Counter()
    for frequencies in term_frequencies:
        document_frequencies.update(frequencies.keys())
    return {
        term: math.log(1 + (doc_count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequencies.items()
    }


def _cjk_bigrams(text: str) -> list[str]:
    chars = list(text)
    if len(chars) <= 1:
        return chars
    return ["".join(chars[index : index + 2]) for index in range(len(chars) - 1)]
