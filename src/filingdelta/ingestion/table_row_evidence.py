from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from uuid import NAMESPACE_URL, uuid5

from filingdelta.schemas.filing import EvidenceKind, EvidenceMetadata, EvidenceUnit, ParsedFiling, ParsedPage


_MAX_ROWS_PER_DOCUMENT = 120
_MAX_ROWS_PER_LABEL = 24
_MAX_CONTEXT_LINES = 16


@dataclass(frozen=True)
class _MetricRowDefinition:
    row_label: str
    metric_tags: tuple[str, ...]
    aliases: tuple[str, ...]
    banking_only: bool = False


_METRIC_ROW_DEFINITIONS: tuple[_MetricRowDefinition, ...] = (
    _MetricRowDefinition(
        row_label="客户存款",
        metric_tags=("customer_deposits", "deposits"),
        aliases=("客户存款", "客戶存款", "存款总额", "存款總額"),
        banking_only=True,
    ),
    _MetricRowDefinition(
        row_label="活期存款",
        metric_tags=("demand_deposits", "customer_deposits", "deposits"),
        aliases=("活期存款", "活期占比", "活期佔比"),
        banking_only=True,
    ),
    _MetricRowDefinition(
        row_label="定期存款",
        metric_tags=("time_deposits", "customer_deposits", "deposits"),
        aliases=("定期存款", "存款定期化", "存款定期化趋势", "存款定期化趨勢"),
        banking_only=True,
    ),
    _MetricRowDefinition(
        row_label="公司客户存款",
        metric_tags=("company_customer_deposits", "customer_deposits", "deposits"),
        aliases=("公司客户存款", "公司客戶存款"),
        banking_only=True,
    ),
    _MetricRowDefinition(
        row_label="零售客户存款",
        metric_tags=("retail_customer_deposits", "customer_deposits", "deposits"),
        aliases=("零售客户存款", "零售客戶存款"),
        banking_only=True,
    ),
    _MetricRowDefinition(
        row_label="营业收入",
        metric_tags=("revenue", "income_statement"),
        aliases=("营业收入", "營業收入", "收入", "revenue"),
    ),
    _MetricRowDefinition(
        row_label="归属股东净利润",
        metric_tags=("net_profit", "profit"),
        aliases=(
            "归属于本行股东的净利润",
            "归属于本公司股东的净利润",
            "歸屬於本公司權益持有人應佔盈利",
            "本公司權益持有人應佔盈利",
            "净利润",
            "淨利潤",
        ),
    ),
    _MetricRowDefinition(
        row_label="净资产收益率",
        metric_tags=("roe", "profitability_ratio"),
        aliases=("净资产收益率", "淨資產收益率", "ROE", "ROAE"),
    ),
    _MetricRowDefinition(
        row_label="资本开支",
        metric_tags=("capital_expenditure", "capex"),
        aliases=("资本开支", "資本開支", "资本性支出", "資本性支出", "capex"),
    ),
    _MetricRowDefinition(
        row_label="不良贷款率",
        metric_tags=("npl_ratio", "asset_quality"),
        aliases=("不良贷款率", "不良貸款率"),
    ),
    _MetricRowDefinition(
        row_label="拨备覆盖率",
        metric_tags=("provision_coverage", "asset_quality"),
        aliases=("拨备覆盖率", "撥備覆蓋率"),
    ),
)


def build_table_row_evidence(parsed_filing: ParsedFiling) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    seen: set[tuple[int, str, str]] = set()
    row_label_counts: dict[str, int] = {}
    period_hint = _infer_period_hint(parsed_filing.document.fiscal_period)
    is_bank_document = _is_bank_document(parsed_filing)

    for page in parsed_filing.pages:
        lines = _clean_lines(page.markdown or page.text)
        if not lines:
            continue
        page_table_score = _score_table_like_page(lines)
        for line_index, line in enumerate(lines):
            definition = _match_metric_row(line, is_bank_document=is_bank_document)
            if definition is None:
                continue
            if row_label_counts.get(definition.row_label, 0) >= _MAX_ROWS_PER_LABEL:
                continue
            context_lines = _collect_context_lines(
                lines=lines,
                line_index=line_index,
                page_table_score=page_table_score,
            )
            if not _has_numeric_signal(context_lines):
                continue

            context_text = "\n".join(context_lines)
            seen_key = (page.page_number, definition.row_label, _normalize_for_match(context_text))
            if seen_key in seen:
                continue
            seen.add(seen_key)

            units.append(
                EvidenceUnit(
                    evidence_id=_table_row_evidence_id(
                        document_id=parsed_filing.document.document_id,
                        page_number=page.page_number,
                        line_index=line_index,
                        row_label=definition.row_label,
                    ),
                    text=_compose_table_row_text(
                        page=page,
                        row_label=definition.row_label,
                        metric_tags=definition.metric_tags,
                        context_lines=context_lines,
                        period_hint=period_hint,
                    ),
                    metadata=EvidenceMetadata(
                        document_id=parsed_filing.document.document_id,
                        source_path=parsed_filing.document.source_path,
                        page_number=page.page_number,
                        page_end=page.page_number,
                        parser_kind=parsed_filing.document.parser_kind,
                        chunk_kind=EvidenceKind.TABLE_ROW,
                        section_type=_infer_section_type(definition.metric_tags),
                        table_id=f"{parsed_filing.document.document_id}:page:{page.page_number}",
                        row_label=definition.row_label,
                        metric_tags=list(definition.metric_tags),
                        period_hint=period_hint,
                    ),
                )
            )
            row_label_counts[definition.row_label] = row_label_counts.get(definition.row_label, 0) + 1
            if len(units) >= _MAX_ROWS_PER_DOCUMENT:
                return units

    return units


def _match_metric_row(line: str, *, is_bank_document: bool) -> _MetricRowDefinition | None:
    if not _line_can_be_table_row_label(line):
        return None
    normalized = _normalize_for_match(line)
    if not normalized:
        return None
    matches = [
        definition
        for definition in _METRIC_ROW_DEFINITIONS
        if (is_bank_document or not definition.banking_only)
        and any(_alias_matches_line(alias=alias, normalized_line=normalized) for alias in definition.aliases)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: max(len(_normalize_for_match(alias)) for alias in item.aliases))


def _alias_matches_line(*, alias: str, normalized_line: str) -> bool:
    normalized_alias = _normalize_for_match(alias)
    if normalized_alias in {"收入", "revenue", "净利润", "淨利潤", "roe", "roae"}:
        return normalized_line == normalized_alias or (
            normalized_line.startswith(normalized_alias) and len(normalized_line) <= 18
        )
    return normalized_alias in normalized_line


def _line_can_be_table_row_label(line: str) -> bool:
    normalized = _normalize_for_match(line)
    if not normalized:
        return False
    if len(normalized) > 90:
        return False
    if len(normalized) > 45 and re.search(r"[，。；:：]", line):
        return False
    return True


def _is_bank_document(parsed_filing: ParsedFiling) -> bool:
    name = _normalize_for_match(parsed_filing.document.company_name)
    return "银行" in name or "銀行" in name or "bank" in name


def _collect_context_lines(
    *,
    lines: list[str],
    line_index: int,
    page_table_score: int,
) -> list[str]:
    before = 6 if page_table_score else 3
    after = 9 if page_table_score else 5
    start = max(0, line_index - before)
    end = min(len(lines), line_index + after + 1)
    context = lines[start:end]

    if len(context) > _MAX_CONTEXT_LINES:
        row_offset = line_index - start
        left = max(0, row_offset - 5)
        right = min(len(context), left + _MAX_CONTEXT_LINES)
        context = context[left:right]
    return context


def _compose_table_row_text(
    *,
    page: ParsedPage,
    row_label: str,
    metric_tags: tuple[str, ...],
    context_lines: list[str],
    period_hint: str | None,
) -> str:
    parts = [
        f"Table row: {row_label}",
        f"Metric tags: {', '.join(metric_tags)}",
        f"Page: {page.page_number}",
    ]
    if period_hint:
        parts.append(f"Period: {period_hint}")
    parts.append("Context:")
    parts.extend(context_lines)
    return "\n".join(parts)


def _infer_section_type(metric_tags: tuple[str, ...]) -> str:
    if any(tag in metric_tags for tag in ("npl_ratio", "provision_coverage", "asset_quality")):
        return "risk_asset_quality"
    return "financial_summary"


def _infer_period_hint(fiscal_period: str | None) -> str | None:
    if not fiscal_period:
        return None
    normalized = unicodedata.normalize("NFKC", fiscal_period)
    year_match = re.search(r"(20\d{2})", normalized)
    if not year_match:
        return None
    year = year_match.group(1)
    if "三季度" in normalized or "第三季度" in normalized or "1-9月" in normalized:
        return f"q3_{year}_ytd"
    if "半年" in normalized or "中期" in normalized or "1-6月" in normalized:
        return f"h1_{year}"
    return f"fy{year}"


def _score_table_like_page(lines: list[str]) -> int:
    text = _normalize_for_match(" ".join(lines[:80]))
    score = 0
    for token in (
        "主要会计数据",
        "主要會計數據",
        "主要财务指标",
        "主要財務指標",
        "财务报表",
        "財務報表",
        "客户存款",
        "客戶存款",
        "人民币百万元",
        "人民幣百萬元",
        "上年末",
        "同比增减",
        "同比變動",
    ):
        if _normalize_for_match(token) in text:
            score += 1
    return score


def _has_numeric_signal(lines: list[str]) -> bool:
    joined = " ".join(lines)
    numbers = re.findall(r"\(?-?\d[\d,]*(?:\.\d+)?%?\)?", joined)
    if len(numbers) >= 2:
        return True
    return bool(numbers) and any("%" in line or "亿元" in line or "百万元" in line for line in lines)


def _clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = unicodedata.normalize("NFKC", raw_line).strip()
        if not line:
            continue
        lines.append(line)
    return lines


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", normalized).lower()


def _table_row_evidence_id(
    *,
    document_id: str,
    page_number: int,
    line_index: int,
    row_label: str,
) -> str:
    stable_key = f"{document_id}:table_row:{page_number}:{line_index}:{row_label}"
    return str(uuid5(NAMESPACE_URL, stable_key))
