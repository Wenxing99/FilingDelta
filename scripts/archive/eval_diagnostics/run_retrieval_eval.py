from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.ingestion.pipeline import FilingIngestionPipeline
from filingdelta.retrieval.indexer import DocumentChunkIndexer
from filingdelta.retrieval.retriever import DocumentChunkRetriever
from filingdelta.schemas.filing import FilingDocType, FilingSource, Market


DEFAULT_MANIFEST = Path("data/outputs/eval/golden_queries_v1_1.json")
DEFAULT_REPORT = Path("data/outputs/eval/retrieval_eval_baseline.json")
DEFAULT_QDRANT_PATH = Path("data/outputs/eval/qdrant_baseline")


@dataclass(frozen=True)
class EvalDocument:
    document_key: str
    source: FilingSource


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    query_set: str
    document_key: str
    query: str
    expected_pages: set[int]
    query_aliases: list[str]
    notes: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline document-scoped retrieval eval for FilingDelta."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to golden-query manifest JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to write the eval report JSON.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=DEFAULT_QDRANT_PATH,
        help="Local Qdrant path for this eval run.",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["core_retrieval"],
        help="Query sets to run, for example: core_retrieval qualitative_sidecar.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete and recreate the eval Qdrant path before indexing.",
    )
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="Reuse the existing eval Qdrant index and skip document indexing.",
    )
    return parser


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    manifest_path = _resolve_path(args.manifest)
    output_path = _resolve_path(args.output)
    qdrant_path = _resolve_path(args.qdrant_path)

    manifest = _load_json(manifest_path)
    top_k = int(args.top_k or manifest.get("default_top_k") or 6)
    selected_sets = set(args.sets)
    if args.fresh and args.reuse_index:
        raise SystemExit("--fresh and --reuse-index cannot be used together.")

    documents = _load_documents(manifest)
    queries = [
        query for query in _load_queries(manifest) if query.query_set in selected_sets
    ]
    if not queries:
        raise SystemExit(f"No queries found for sets: {sorted(selected_sets)}")

    if args.fresh and qdrant_path.exists():
        shutil.rmtree(qdrant_path)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        FILINGDELTA_QDRANT_PATH=str(qdrant_path.relative_to(REPO_ROOT))
        if qdrant_path.is_relative_to(REPO_ROOT)
        else str(qdrant_path)
    )
    client = QdrantClient(path=str(qdrant_path))
    indexer = DocumentChunkIndexer(settings=settings, client=client)
    retriever = DocumentChunkRetriever(settings=settings, client=client)
    pipeline = FilingIngestionPipeline(settings=settings)

    used_document_keys = sorted({query.document_key for query in queries})
    document_results: dict[str, dict[str, Any]] = {}
    document_ids: dict[str, str] = {}

    eval_started = time.perf_counter()
    for document_key in used_document_keys:
        document = documents[document_key]
        print(f"parsing/indexing {document_key}: {document.source.source_path}")
        parse_started = time.perf_counter()
        ingestion = pipeline.run(document.source)
        parse_ms = _elapsed_ms(parse_started)

        document_id = ingestion.parsed_filing.document.document_id
        document_ids[document_key] = document_id

        index_ms = None
        if not args.reuse_index:
            index_started = time.perf_counter()
            indexer.index_document(document_id=document_id, chunks=ingestion.chunks)
            index_ms = _elapsed_ms(index_started)

        document_results[document_key] = {
            "document_id": document_id,
            "source_path": str(document.source.source_path),
            "total_pages": ingestion.parsed_filing.document.total_pages,
            "chunk_count": len(ingestion.chunks),
            "parse_ms": parse_ms,
            "index_ms": index_ms,
            "reused_index": args.reuse_index,
        }

    query_results = []
    for query in queries:
        document_id = document_ids[query.document_key]
        retrieval_started = time.perf_counter()
        chunks = retriever.retrieve(
            document_id=document_id,
            question=query.query,
            top_k=top_k,
        )
        retrieval_ms = _elapsed_ms(retrieval_started)
        retrieved_pages = [
            chunk.page_number for chunk in chunks if chunk.page_number is not None
        ]
        retrieved_page_set = set(retrieved_pages)
        hit_pages = sorted(query.expected_pages & retrieved_page_set)
        page_hit = bool(hit_pages)

        query_results.append(
            {
                "id": query.query_id,
                "set": query.query_set,
                "document_key": query.document_key,
                "document_id": document_id,
                "query": query.query,
                "expected_pages": sorted(query.expected_pages),
                "retrieved_pages": retrieved_pages,
                "hit_pages": hit_pages,
                f"page_hit@{top_k}": page_hit,
                "retrieval_ms": retrieval_ms,
                "top_chunks": [
                    {
                        "rank": index + 1,
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.page_number,
                        "score": chunk.score,
                        "preview": _preview(chunk.text),
                    }
                    for index, chunk in enumerate(chunks)
                ],
                "notes": query.notes,
            }
        )
        status = "HIT" if page_hit else "MISS"
        print(
            f"{status} {query.query_id}: expected={sorted(query.expected_pages)} "
            f"retrieved={retrieved_pages}"
        )

    summary = _build_summary(query_results, top_k=top_k)
    report = {
        "manifest_path": str(manifest_path),
        "manifest_version": manifest.get("version"),
        "page_numbering": manifest.get("page_numbering"),
        "top_k": top_k,
        "selected_sets": sorted(selected_sets),
        "total_latency_ms": _elapsed_ms(eval_started),
        "summary": summary,
        "documents": document_results,
        "queries": query_results,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("report:", output_path)
    print(
        "summary:",
        f"{summary[f'page_hit@{top_k}_count']}/{summary['total_queries']} "
        f"page_hit@{top_k} = {summary[f'page_hit@{top_k}_rate']:.3f}",
    )
    if summary["miss_cases"]:
        print("miss_cases:", ", ".join(summary["miss_cases"]))
    client.close()


def _load_documents(manifest: dict[str, Any]) -> dict[str, EvalDocument]:
    documents: dict[str, EvalDocument] = {}
    for payload in manifest.get("documents", []):
        source_path = _resolve_path(Path(payload["source_path"]))
        source = FilingSource(
            source_path=source_path,
            company_name=payload["company_name"],
            ticker=payload.get("ticker"),
            market=Market(payload.get("market") or Market.OTHER.value),
            doc_type=FilingDocType(payload.get("doc_type") or FilingDocType.OTHER.value),
            fiscal_period=payload.get("fiscal_period"),
            language=payload.get("language") or "zh",
        )
        document_key = str(payload["document_key"])
        documents[document_key] = EvalDocument(document_key=document_key, source=source)
    return documents


def _load_queries(manifest: dict[str, Any]) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    for payload in manifest.get("queries", []):
        expected_pages = {int(page) for page in payload.get("expected_pages", [])}
        queries.append(
            EvalQuery(
                query_id=str(payload["id"]),
                query_set=str(payload["set"]),
                document_key=str(payload["document_key"]),
                query=str(payload["query"]),
                expected_pages=expected_pages,
                query_aliases=[str(alias) for alias in payload.get("query_aliases", [])],
                notes=str(payload.get("notes") or ""),
            )
        )
    return queries


def _build_summary(query_results: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    hit_key = f"page_hit@{top_k}"
    total = len(query_results)
    hit_count = sum(1 for result in query_results if result[hit_key])
    by_document: dict[str, dict[str, Any]] = {}
    for result in query_results:
        bucket = by_document.setdefault(
            result["document_key"],
            {"total_queries": 0, f"{hit_key}_count": 0, f"{hit_key}_rate": 0.0},
        )
        bucket["total_queries"] += 1
        if result[hit_key]:
            bucket[f"{hit_key}_count"] += 1

    for bucket in by_document.values():
        bucket[f"{hit_key}_rate"] = (
            bucket[f"{hit_key}_count"] / bucket["total_queries"]
            if bucket["total_queries"]
            else 0.0
        )

    return {
        "total_queries": total,
        f"{hit_key}_count": hit_count,
        f"{hit_key}_rate": hit_count / total if total else 0.0,
        "miss_cases": [result["id"] for result in query_results if not result[hit_key]],
        "by_document": by_document,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _preview(text: str, *, limit: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


if __name__ == "__main__":
    main()
