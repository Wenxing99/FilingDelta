from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from filingdelta.core.config import REPO_ROOT
from filingdelta.eval.document_registry import (
    DocumentRegistryError,
    RawDocumentRegistry,
    load_raw_document_registry,
)
from filingdelta.eval.table_row_retrieval import TABLE_ROW_RETRIEVAL_CASES
from filingdelta.schemas.filing import EvidenceKind


SmokeV2Route = Literal["document_only", "concept_only", "mixed", "unsupported"]
SmokeV2Intent = Literal["metric_value", "metric_attribution", "business_narrative", "fallback"]
SmokeV2Mode = Literal["validate_only", "dry_run", "live_retrieval"]

SMOKE_V2_TIER = "smoke_v2"
DEFAULT_TOP_K = 6
REQUIRED_QUERY_FIELDS = (
    "id",
    "tier",
    "document_key",
    "query",
    "expected_route",
    "expected_document_evidence_intent",
    "primary_evidence_kind",
    "secondary_evidence_kinds",
    "expected_pages",
    "forbidden_failure_modes",
    "mvp_status",
)
ALLOWED_ROUTES = {"document_only", "concept_only", "mixed", "unsupported"}
ALLOWED_DOCUMENT_INTENTS = {
    "metric_value",
    "metric_attribution",
    "business_narrative",
    "fallback",
}
ALLOWED_EVIDENCE_KINDS = {kind.value for kind in EvidenceKind}
KNOWN_HYGIENE_CHECKS = {"no_raw_metadata", "no_empty_parentheses", "unit_period_present"}

_EMPTY_CITATION_MARKER_RE = re.compile(r"\(\s*(?:[,;]\s*)*\)|\uff08\s*(?:[\u3001,\uff0c;\uff1b]\s*)*\uff09")
_RAW_METADATA_RE = re.compile(
    r"\b(?:DOC|WEB)_\d+\b|(?:\[|\()?Chunk\s+\d+(?:\]|\))?|"
    r"\b(?:source|score|chunk_kind|section_type|row_label)\s*=",
    flags=re.IGNORECASE,
)
_UNIT_RE = re.compile(
    r"%|percent|percentage point|ppt|bps|RMB|CNY|HKD|USD|yuan|million|billion|"
    r"TEU|ton|kWh|MWh|GWh|MW|GW",
    flags=re.IGNORECASE,
)
_PERIOD_RE = re.compile(r"\b20\d{2}\b|\bFY\s?\d{2,4}\b|annual|reporting period", flags=re.IGNORECASE)


class SmokeV2ManifestError(ValueError):
    """Raised when a smoke_v2 manifest cannot be loaded or validated."""


@dataclass(frozen=True)
class SmokeV2Case:
    case_id: str
    tier: str
    document_key: str
    query: str
    expected_route: SmokeV2Route
    expected_document_evidence_intent: SmokeV2Intent
    primary_evidence_kind: str
    secondary_evidence_kinds: tuple[str, ...] = ()
    expected_pages: tuple[int, ...] = ()
    query_aliases: tuple[str, ...] = ()
    company: str | None = None
    industry: str | None = None
    expected_row_labels: tuple[str, ...] = ()
    expected_metric_tags: tuple[str, ...] = ()
    expected_section_types: tuple[str, ...] = ()
    expected_document_area_ids: tuple[str, ...] = ()
    expected_answer_field_ids: tuple[str, ...] = ()
    forbidden_failure_modes: tuple[str, ...] = ()
    answer_hygiene_checks: tuple[str, ...] = ()
    mvp_status: str = "immediate"
    notes: str = ""

    @property
    def expected_evidence_kinds(self) -> tuple[str, ...]:
        return (self.primary_evidence_kind, *self.secondary_evidence_kinds)


@dataclass(frozen=True)
class SmokeV2Manifest:
    version: str
    documents: RawDocumentRegistry
    queries: tuple[SmokeV2Case, ...]
    default_top_k: int = DEFAULT_TOP_K
    suite: str = "golden_queries_v2"
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SmokeV2Observation:
    executed: bool = False
    route: str | None = None
    document_evidence_intent: str | None = None
    retrieval_mode: str | None = None
    retrieved_evidence_kinds: tuple[str, ...] = ()
    citation_pages: tuple[int, ...] = ()
    retrieved_row_labels: tuple[str, ...] = ()
    retrieved_metric_tags: tuple[str, ...] = ()
    retrieved_section_types: tuple[str, ...] = ()
    answer_field_ids: tuple[str, ...] = ()
    answer_text: str | None = None
    latency_ms: int | None = None
    error: str | None = None


def load_smoke_v2_manifest(path: Path, *, base_dir: Path = REPO_ROOT) -> SmokeV2Manifest:
    manifest_path = path if path.is_absolute() else (base_dir / path).resolve()
    payload = _load_manifest_payload(manifest_path)
    return load_smoke_v2_manifest_from_payload(
        payload,
        base_dir=base_dir,
        source_path=manifest_path,
    )


def load_smoke_v2_manifest_from_payload(
    payload: dict[str, Any],
    *,
    base_dir: Path = REPO_ROOT,
    source_path: Path | None = None,
) -> SmokeV2Manifest:
    if not isinstance(payload, dict):
        raise SmokeV2ManifestError("Manifest must be a JSON/YAML object.")

    try:
        documents = load_raw_document_registry(payload.get("documents", []), base_dir=base_dir)
    except DocumentRegistryError as exc:
        raise SmokeV2ManifestError(str(exc)) from exc
    default_top_k = _coerce_positive_int(
        payload.get("default_top_k", DEFAULT_TOP_K),
        context="default_top_k",
    )
    queries = _load_cases(payload.get("queries", []), documents=documents)
    return SmokeV2Manifest(
        version=str(payload.get("version") or "smoke_v2_draft"),
        suite=str(payload.get("suite") or "golden_queries_v2"),
        default_top_k=default_top_k,
        documents=documents,
        queries=tuple(queries),
        source_path=source_path,
        metadata=dict(payload.get("metadata") or {}),
    )


def select_smoke_v2_cases(
    cases: tuple[SmokeV2Case, ...] | list[SmokeV2Case],
    *,
    case_ids: set[str] | None = None,
    tiers: set[str] | None = None,
    companies: set[str] | None = None,
    industries: set[str] | None = None,
    intents: set[str] | None = None,
) -> list[SmokeV2Case]:
    selected = list(cases)
    if case_ids:
        selected = [case for case in selected if case.case_id in case_ids]
        missing = sorted(case_ids - {case.case_id for case in selected})
        if missing:
            raise SmokeV2ManifestError(f"Unknown case id(s): {', '.join(missing)}")
    if tiers:
        selected = [case for case in selected if case.tier in tiers]
    if companies:
        selected = [case for case in selected if case.company in companies]
    if industries:
        selected = [case for case in selected if case.industry in industries]
    if intents:
        selected = [
            case
            for case in selected
            if case.expected_document_evidence_intent in intents
        ]
    return selected


def build_smoke_v2_report(
    *,
    manifest: SmokeV2Manifest,
    cases: list[SmokeV2Case],
    mode: SmokeV2Mode,
    top_k: int | None = None,
    observations: dict[str, SmokeV2Observation] | None = None,
) -> dict[str, Any]:
    effective_top_k = _coerce_positive_int(
        manifest.default_top_k if top_k is None else top_k,
        context="top_k",
    )
    used_document_keys = {case.document_key for case in cases}
    missing_documents = manifest.documents.missing_documents(used_document_keys)
    if mode == "live_retrieval":
        query_results = _build_live_retrieval_results(
            cases=cases,
            observations=observations or {},
            top_k=effective_top_k,
        )
    else:
        query_results = [
            build_smoke_v2_case_result(
                case=case,
                observation=SmokeV2Observation(),
                top_k=effective_top_k,
                status="validated" if mode == "validate_only" else "dry_run_skipped",
                skip_reason=None if mode == "validate_only" else _dry_run_skip_reason(case, manifest),
            )
            for case in cases
        ]

    return {
        "version": "smoke_v2_eval_v0",
        "mode": mode,
        "manifest_path": str(manifest.source_path) if manifest.source_path else None,
        "manifest_version": manifest.version,
        "suite": manifest.suite,
        "top_k": effective_top_k,
        "summary": summarize_smoke_v2_results(query_results, top_k=effective_top_k),
        "documents": {
            "registry": manifest.documents.to_json(),
            "used_document_keys": sorted(used_document_keys),
            "missing_document_keys": [document.document_key for document in missing_documents],
        },
        "queries": query_results,
    }


def build_smoke_v2_case_result(
    *,
    case: SmokeV2Case,
    observation: SmokeV2Observation,
    top_k: int,
    status: str = "evaluated",
    skip_reason: str | None = None,
    score_answer_quality: bool = True,
) -> dict[str, Any]:
    effective_top_k = _coerce_positive_int(top_k, context="top_k")
    executed = (
        observation.executed
        or status == "evaluated"
        or _observation_has_execution_signal(observation)
    )
    hygiene = evaluate_answer_hygiene(
        answer_text=observation.answer_text,
        check_ids=case.answer_hygiene_checks,
    )
    scores = _score_case(
        case=case,
        observation=observation,
        hygiene=hygiene,
        top_k=effective_top_k,
        executed=executed,
        score_answer_quality=score_answer_quality,
    )
    return {
        "id": case.case_id,
        "tier": case.tier,
        "document_key": case.document_key,
        "company": case.company,
        "industry": case.industry,
        "query": case.query,
        "expected": {
            "route": case.expected_route,
            "document_evidence_intent": case.expected_document_evidence_intent,
            "primary_evidence_kind": case.primary_evidence_kind,
            "secondary_evidence_kinds": list(case.secondary_evidence_kinds),
            "pages": list(case.expected_pages),
            "row_labels": list(case.expected_row_labels),
            "metric_tags": list(case.expected_metric_tags),
            "section_types": list(case.expected_section_types),
            "document_area_ids": list(case.expected_document_area_ids),
            "answer_field_ids": list(case.expected_answer_field_ids),
            "forbidden_failure_modes": list(case.forbidden_failure_modes),
            "answer_hygiene_checks": list(case.answer_hygiene_checks),
        },
        "observed": {
            "route": observation.route,
            "document_evidence_intent": observation.document_evidence_intent,
            "retrieval_mode": observation.retrieval_mode,
            "retrieved_evidence_kinds": list(observation.retrieved_evidence_kinds),
            "citation_pages": list(observation.citation_pages),
            "retrieved_row_labels": list(observation.retrieved_row_labels),
            "retrieved_metric_tags": list(observation.retrieved_metric_tags),
            "retrieved_section_types": list(observation.retrieved_section_types),
            "answer_field_ids": list(observation.answer_field_ids),
            "latency_ms": observation.latency_ms,
            "answer_hygiene": hygiene,
            "error": observation.error,
        },
        "scores": scores,
        "status": status,
        "skip_reason": skip_reason,
        "failure_reasons": [],
        "notes": case.notes,
    }


def evaluate_answer_hygiene(
    *,
    answer_text: str | None,
    check_ids: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for check_id in check_ids:
        if check_id not in KNOWN_HYGIENE_CHECKS:
            results[check_id] = {
                "passed": None,
                "message": f"Unknown hygiene check: {check_id}",
            }
            continue

        if answer_text is None:
            results[check_id] = {
                "passed": None,
                "message": "No answer text was produced.",
            }
            continue

        if check_id == "no_raw_metadata":
            passed = _RAW_METADATA_RE.search(answer_text) is None
            message = "answer does not leak internal source/chunk metadata"
        elif check_id == "no_empty_parentheses":
            passed = _EMPTY_CITATION_MARKER_RE.search(answer_text) is None
            message = "answer does not contain empty citation parentheses"
        elif check_id == "unit_period_present":
            passed = bool(_UNIT_RE.search(answer_text) and _PERIOD_RE.search(answer_text))
            message = "answer includes a visible unit and reporting period cue"

        results[check_id] = {"passed": passed, "message": message}
    return results


def summarize_smoke_v2_results(query_results: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    effective_top_k = _coerce_positive_int(top_k, context="top_k")
    score_keys = (
        "route_hit",
        "intent_hit",
        f"evidence_kind_hit@{effective_top_k}",
        f"page_hit@{effective_top_k}",
        f"table_row_label_hit@{effective_top_k}",
        f"metric_tag_hit@{effective_top_k}",
        f"section_type_hit@{effective_top_k}",
        "required_fields_present",
        "citation_anchor_valid",
        "forbidden_failure_absent",
        "output_hygiene_passed",
    )
    summary: dict[str, Any] = {
        "total_queries": len(query_results),
        "status_counts": _count_values(result["status"] for result in query_results),
    }
    for score_key in score_keys:
        scored = [result for result in query_results if result["scores"].get(score_key) is not None]
        passed = [result for result in scored if result["scores"][score_key] is True]
        summary[f"{score_key}_scored"] = len(scored)
        summary[f"{score_key}_passed"] = len(passed)
        summary[f"{score_key}_rate"] = len(passed) / len(scored) if scored else None
    return summary


def render_smoke_v2_markdown_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    top_k = int(report["top_k"])
    page_key = f"page_hit@{top_k}"
    evidence_key = f"evidence_kind_hit@{top_k}"
    lines = [
        "# Golden Queries v2 Smoke Pilot Summary",
        "",
        "## 摘要",
        "",
        f"- 运行模式：`{report['mode']}`",
        f"- Manifest：`{report.get('manifest_path') or '-'}`",
        f"- Manifest version：`{report.get('manifest_version') or '-'}`",
        f"- Top K：`{top_k}`",
        f"- Case 总数：`{summary['total_queries']}`",
        f"- 状态统计：`{json.dumps(summary['status_counts'], ensure_ascii=False)}`",
        (
            f"- 页码命中：`{summary[f'{page_key}_passed']}/"
            f"{summary[f'{page_key}_scored']}`"
        ),
        (
            f"- 证据类型命中：`{summary[f'{evidence_key}_passed']}/"
            f"{summary[f'{evidence_key}_scored']}`"
        ),
        "",
        "## Case 结果",
        "",
        "| Case | 公司 | 状态 | 期望页 | 命中页 | 证据类型 | 失败原因 |",
        "|---|---|---:|---|---|---|---|",
    ]
    for result in report["queries"]:
        expected_pages = ", ".join(str(page) for page in result["expected"]["pages"]) or "-"
        observed_pages = ", ".join(
            str(page) for page in result["observed"]["citation_pages"]
        ) or "-"
        evidence_kinds = ", ".join(result["observed"]["retrieved_evidence_kinds"]) or "-"
        failure_reasons = "; ".join(result.get("failure_reasons") or []) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_escape_markdown_table(result['id'])}`",
                    _escape_markdown_table(result.get("company") or "-"),
                    f"`{_escape_markdown_table(result['status'])}`",
                    _escape_markdown_table(expected_pages),
                    _escape_markdown_table(observed_pages),
                    _escape_markdown_table(evidence_kinds),
                    _escape_markdown_table(failure_reasons),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 本摘要只反映当前 manifest 中 14 条 anchor-confirmed case 的 live retrieval 结果。",
            "- 本轮没有修改 manifest gold 页码，也没有把失败 case 改成通过。",
            "- `required_fields_present`、`forbidden_failure_absent`、`output_hygiene_passed` 需要答案合成/字段判定，本轮 live retrieval-only pilot 不计入通过判定。",
            "",
        ]
    )
    return "\n".join(lines)


def build_builtin_placeholder_manifest_payload() -> dict[str, Any]:
    cases = {case.case_id: case for case in TABLE_ROW_RETRIEVAL_CASES}
    cmb_case = cases["CMB-DEP-01"]
    tencent_case = cases["TCEHY-CAPEX-01"]
    return {
        "version": "smoke_v2_placeholder_v0",
        "suite": "golden_queries_v2",
        "default_top_k": DEFAULT_TOP_K,
        "documents": [
            {
                "document_key": "cmb_2025_annual",
                "source_path": "data/raw/\u62db\u5546\u94f6\u884c2025\u5e74\u5ea6\u62a5\u544a.pdf",
                "company_name": "\u62db\u5546\u94f6\u884c",
                "ticker": "600036",
                "market": "a_share",
                "doc_type": "annual_report",
                "fiscal_period": "2025 annual report",
                "language": "zh",
                "industry": "banking",
            },
            {
                "document_key": "tcehy_2025_annual",
                "source_path": "data/raw/\u817e\u8baf\u63a7\u80a12025\u5e74\u5ea6\u62a5\u544a.pdf",
                "company_name": "\u817e\u8baf\u63a7\u80a1",
                "ticker": "0700",
                "market": "h_share",
                "doc_type": "annual_report",
                "fiscal_period": "2025 annual report",
                "language": "zh",
                "industry": "internet",
            },
        ],
        "queries": [
            _placeholder_query_payload(
                case_id=cmb_case.case_id,
                document_key=cmb_case.document_key,
                query=cmb_case.query,
                expected_pages=cmb_case.expected_pages,
                expected_row_labels=cmb_case.expected_row_labels,
                expected_metric_tags=cmb_case.expected_metric_tags,
                company="\u62db\u5546\u94f6\u884c",
                industry="banking",
                expected_answer_field_ids=[
                    "customer_deposit_balance",
                    "demand_deposit_share",
                ],
                forbidden_failure_modes=[
                    "answers only company-level deposits when group-level deposits are asked",
                ],
            ),
            _placeholder_query_payload(
                case_id=tencent_case.case_id,
                document_key=tencent_case.document_key,
                query=tencent_case.query,
                expected_pages=tencent_case.expected_pages,
                expected_row_labels=tencent_case.expected_row_labels,
                expected_metric_tags=tencent_case.expected_metric_tags,
                company="\u817e\u8baf\u63a7\u80a1",
                industry="internet",
                expected_answer_field_ids=["capex_value", "period", "unit"],
                forbidden_failure_modes=[
                    "uses planned capex or prior-year capex without a period label",
                ],
            ),
        ],
    }


def _load_cases(
    queries_payload: list[dict[str, Any]],
    *,
    documents: RawDocumentRegistry,
) -> list[SmokeV2Case]:
    if not isinstance(queries_payload, list):
        raise SmokeV2ManifestError("Manifest field 'queries' must be a list.")

    cases: list[SmokeV2Case] = []
    seen_ids: set[str] = set()
    for index, payload in enumerate(queries_payload):
        if not isinstance(payload, dict):
            raise SmokeV2ManifestError(f"Query entry #{index + 1} must be an object.")
        _validate_required_query_fields(payload, index=index)

        case_id = str(payload["id"])
        if case_id in seen_ids:
            raise SmokeV2ManifestError(f"Duplicate query id: {case_id}")
        seen_ids.add(case_id)

        document_key = str(payload["document_key"])
        try:
            documents.require(document_key)
        except DocumentRegistryError as exc:
            raise SmokeV2ManifestError(str(exc)) from exc

        expected_route = str(payload["expected_route"])
        if expected_route not in ALLOWED_ROUTES:
            raise SmokeV2ManifestError(f"{case_id}: invalid expected_route: {expected_route}")

        expected_intent = str(payload["expected_document_evidence_intent"])
        if expected_intent not in ALLOWED_DOCUMENT_INTENTS:
            raise SmokeV2ManifestError(
                f"{case_id}: invalid expected_document_evidence_intent: {expected_intent}"
            )

        primary_evidence_kind = str(payload["primary_evidence_kind"])
        if primary_evidence_kind not in ALLOWED_EVIDENCE_KINDS:
            raise SmokeV2ManifestError(
                f"{case_id}: invalid primary_evidence_kind: {primary_evidence_kind}"
            )
        secondary_evidence_kinds = _str_tuple(payload.get("secondary_evidence_kinds", []))
        invalid_secondary = [
            kind for kind in secondary_evidence_kinds if kind not in ALLOWED_EVIDENCE_KINDS
        ]
        if invalid_secondary:
            raise SmokeV2ManifestError(
                f"{case_id}: invalid secondary_evidence_kinds: {invalid_secondary}"
            )
        answer_hygiene_checks = _str_tuple(payload.get("answer_hygiene_checks", []))
        invalid_hygiene_checks = [
            check_id for check_id in answer_hygiene_checks if check_id not in KNOWN_HYGIENE_CHECKS
        ]
        if invalid_hygiene_checks:
            raise SmokeV2ManifestError(
                f"{case_id}: invalid answer_hygiene_checks: {invalid_hygiene_checks}"
            )

        cases.append(
            SmokeV2Case(
                case_id=case_id,
                tier=str(payload["tier"]),
                document_key=document_key,
                query=str(payload["query"]),
                expected_route=expected_route,  # type: ignore[arg-type]
                expected_document_evidence_intent=expected_intent,  # type: ignore[arg-type]
                primary_evidence_kind=primary_evidence_kind,
                secondary_evidence_kinds=secondary_evidence_kinds,
                expected_pages=_int_tuple(payload.get("expected_pages", [])),
                query_aliases=_str_tuple(payload.get("query_aliases", [])),
                company=_optional_str(payload.get("company")),
                industry=_optional_str(payload.get("industry")),
                expected_row_labels=_str_tuple(payload.get("expected_row_labels", [])),
                expected_metric_tags=_str_tuple(payload.get("expected_metric_tags", [])),
                expected_section_types=_str_tuple(payload.get("expected_section_types", [])),
                expected_document_area_ids=_str_tuple(payload.get("expected_document_area_ids", [])),
                expected_answer_field_ids=_str_tuple(payload.get("expected_answer_field_ids", [])),
                forbidden_failure_modes=_str_tuple(payload.get("forbidden_failure_modes", [])),
                answer_hygiene_checks=answer_hygiene_checks,
                mvp_status=str(payload.get("mvp_status") or "immediate"),
                notes=str(payload.get("notes") or ""),
            )
        )
    return cases


def _score_case(
    *,
    case: SmokeV2Case,
    observation: SmokeV2Observation,
    hygiene: dict[str, dict[str, object]],
    top_k: int,
    executed: bool,
    score_answer_quality: bool,
) -> dict[str, object]:
    return {
        "route_hit": _optional_equal(
            observation.route,
            case.expected_route,
            executed=executed,
        ),
        "intent_hit": _optional_equal(
            observation.document_evidence_intent,
            case.expected_document_evidence_intent,
            executed=executed,
        ),
        f"evidence_kind_hit@{top_k}": _optional_intersection_hit(
            observed=observation.retrieved_evidence_kinds[:top_k],
            expected=(case.primary_evidence_kind,),
            required=bool(case.primary_evidence_kind),
            executed=executed,
        ),
        f"page_hit@{top_k}": _optional_intersection_hit(
            observed=observation.citation_pages[:top_k],
            expected=case.expected_pages,
            required=bool(case.expected_pages),
            executed=executed,
        ),
        f"table_row_label_hit@{top_k}": _optional_intersection_hit(
            observed=observation.retrieved_row_labels[:top_k],
            expected=case.expected_row_labels,
            required=bool(case.expected_row_labels),
            executed=executed,
        ),
        f"metric_tag_hit@{top_k}": _optional_intersection_hit(
            observed=observation.retrieved_metric_tags[:top_k],
            expected=case.expected_metric_tags,
            required=bool(case.expected_metric_tags),
            executed=executed,
        ),
        f"section_type_hit@{top_k}": _optional_intersection_hit(
            observed=observation.retrieved_section_types[:top_k],
            expected=case.expected_section_types,
            required=bool(case.expected_section_types),
            executed=executed,
        ),
        "required_fields_present": _optional_required_subset(
            observed=observation.answer_field_ids,
            expected=case.expected_answer_field_ids,
            required=bool(case.expected_answer_field_ids),
            executed=executed and score_answer_quality,
        ),
        "citation_anchor_valid": _citation_anchor_valid(
            observation.citation_pages,
            executed=executed,
        ),
        "forbidden_failure_absent": _forbidden_failure_absent(
            answer_text=observation.answer_text,
            forbidden_failure_modes=case.forbidden_failure_modes,
            executed=executed and score_answer_quality,
        ),
        "output_hygiene_passed": _output_hygiene_passed(
            hygiene,
            executed=executed and score_answer_quality,
        ),
    }


def _build_live_retrieval_results(
    *,
    cases: list[SmokeV2Case],
    observations: dict[str, SmokeV2Observation],
    top_k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        observation = observations.get(case.case_id)
        if observation is None:
            observation = SmokeV2Observation(
                executed=True,
                error="live retrieval observation was not produced",
            )
        result = build_smoke_v2_case_result(
            case=case,
            observation=observation,
            top_k=top_k,
            status="evaluated",
            score_answer_quality=False,
        )
        failure_reasons = _live_retrieval_failure_reasons(result=result, top_k=top_k)
        if observation.error:
            result["status"] = "error"
        else:
            result["status"] = "failed" if failure_reasons else "passed"
        result["failure_reasons"] = failure_reasons
        results.append(result)
    return results


def _live_retrieval_failure_reasons(*, result: dict[str, Any], top_k: int) -> list[str]:
    observed = result["observed"]
    expected = result["expected"]
    scores = result["scores"]
    reasons: list[str] = []
    if observed.get("error"):
        reasons.append(f"runner error: {observed['error']}")
    if scores["route_hit"] is False:
        reasons.append(
            f"route mismatch: expected {expected['route']}, got {observed['route']}"
        )
    if scores["intent_hit"] is False:
        reasons.append(
            "intent mismatch: expected "
            f"{expected['document_evidence_intent']}, got "
            f"{observed['document_evidence_intent']}"
        )
    evidence_key = f"evidence_kind_hit@{top_k}"
    if scores[evidence_key] is False:
        reasons.append(
            "primary evidence kind miss: expected "
            f"{expected['primary_evidence_kind']}, got "
            f"{observed['retrieved_evidence_kinds']}"
        )
    page_key = f"page_hit@{top_k}"
    if scores[page_key] is False:
        reasons.append(
            f"page miss: expected {expected['pages']}, got {observed['citation_pages']}"
        )
    if scores["citation_anchor_valid"] is False:
        reasons.append("no valid document citation page returned")
    return reasons


def _load_manifest_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:
            raise SmokeV2ManifestError(
                "YAML manifests require PyYAML, which is not a current repo dependency. "
                "Use JSON or add PyYAML intentionally."
            ) from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise SmokeV2ManifestError(f"Unsupported manifest suffix: {path.suffix}")


def _validate_required_query_fields(payload: dict[str, Any], *, index: int) -> None:
    context = payload.get("id") or f"query #{index + 1}"
    missing = [field_name for field_name in REQUIRED_QUERY_FIELDS if field_name not in payload]
    if missing:
        raise SmokeV2ManifestError(f"{context}: missing required field(s): {', '.join(missing)}")


def _placeholder_query_payload(
    *,
    case_id: str,
    document_key: str,
    query: str,
    expected_pages: tuple[int, ...],
    expected_row_labels: tuple[str, ...],
    expected_metric_tags: tuple[str, ...],
    company: str,
    industry: str,
    expected_answer_field_ids: list[str],
    forbidden_failure_modes: list[str],
) -> dict[str, Any]:
    return {
        "id": case_id,
        "tier": SMOKE_V2_TIER,
        "company": company,
        "industry": industry,
        "document_key": document_key,
        "query": query,
        "query_aliases": [],
        "expected_route": "document_only",
        "expected_document_evidence_intent": "metric_value",
        "primary_evidence_kind": EvidenceKind.TABLE_ROW.value,
        "secondary_evidence_kinds": [EvidenceKind.PAGE_TEXT.value],
        "expected_pages": list(expected_pages),
        "expected_row_labels": list(expected_row_labels),
        "expected_metric_tags": list(expected_metric_tags),
        "expected_section_types": [],
        "expected_document_area_ids": ["financial_summary"],
        "expected_answer_field_ids": expected_answer_field_ids,
        "forbidden_failure_modes": forbidden_failure_modes,
        "answer_hygiene_checks": [
            "no_raw_metadata",
            "no_empty_parentheses",
            "unit_period_present",
        ],
        "mvp_status": "immediate",
        "notes": "Placeholder smoke_v2 case derived from the existing table_row retrieval eval set.",
    }


def _dry_run_skip_reason(case: SmokeV2Case, manifest: SmokeV2Manifest) -> str:
    document = manifest.documents.require(case.document_key)
    if not document.exists:
        return f"source document not found: {document.source_path}"
    return "dry run requested; live router/retriever execution is not wired in this skeleton"


def _optional_equal(observed: object | None, expected: object, *, executed: bool) -> bool | None:
    if observed is None:
        return False if executed else None
    return observed == expected


def _optional_intersection_hit(
    *,
    observed: tuple[object, ...],
    expected: tuple[object, ...],
    required: bool,
    executed: bool,
) -> bool | None:
    if not required:
        return None
    if not observed:
        return False if executed else None
    return bool(set(expected).intersection(observed))


def _optional_required_subset(
    *,
    observed: tuple[object, ...],
    expected: tuple[object, ...],
    required: bool,
    executed: bool,
) -> bool | None:
    if not required:
        return None
    if not observed:
        return False if executed else None
    return set(expected).issubset(set(observed))


def _citation_anchor_valid(citation_pages: tuple[int, ...], *, executed: bool) -> bool | None:
    if not citation_pages:
        return False if executed else None
    return all(page > 0 for page in citation_pages)


def _forbidden_failure_absent(
    *,
    answer_text: str | None,
    forbidden_failure_modes: tuple[str, ...],
    executed: bool,
) -> bool | None:
    if not forbidden_failure_modes:
        return None
    if answer_text is None:
        return False if executed else None
    return all(failure not in answer_text for failure in forbidden_failure_modes)


def _output_hygiene_passed(
    hygiene: dict[str, dict[str, object]],
    *,
    executed: bool,
) -> bool | None:
    if not hygiene:
        return None
    passed_values = [result["passed"] for result in hygiene.values()]
    if any(value is None for value in passed_values):
        return False if executed else None
    return all(value is True for value in passed_values)


def _coerce_positive_int(value: object, *, context: str) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise SmokeV2ManifestError(f"{context} must be an integer >= 1.") from exc
    if coerced < 1:
        raise SmokeV2ManifestError(f"{context} must be an integer >= 1.")
    return coerced


def _observation_has_execution_signal(observation: SmokeV2Observation) -> bool:
    return any(
        (
            observation.route is not None,
            observation.document_evidence_intent is not None,
            observation.retrieval_mode is not None,
            bool(observation.retrieved_evidence_kinds),
            bool(observation.citation_pages),
            bool(observation.retrieved_row_labels),
            bool(observation.retrieved_metric_tags),
            bool(observation.retrieved_section_types),
            bool(observation.answer_field_ids),
            observation.answer_text is not None,
            observation.latency_ms is not None,
            observation.error is not None,
        )
    )


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _int_tuple(value: object) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        raise SmokeV2ManifestError(f"Expected a list of integers, got {type(value).__name__}.")
    return tuple(int(item) for item in value)


def _str_tuple(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        raise SmokeV2ManifestError(f"Expected a list of strings, got {type(value).__name__}.")
    return tuple(str(item) for item in value)


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _escape_markdown_table(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
