from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
import re
import unicodedata

from filingdelta.financial_facts.catalog import get_metric_definition
from filingdelta.financial_facts.normalization import (
    extract_fiscal_year,
    normalize_numeric_value,
    normalize_unit,
)
from filingdelta.financial_facts.schemas import FinancialFact, ReviewStatus
from filingdelta.schemas.facts import ExtractedFactField, HeadlineMetricFacts
from filingdelta.schemas.filing import Citation


_HEADLINE_FIELD_TO_CANONICAL = {
    "revenue": "revenue",
    "net_profit": "net_profit_attributable",
    "total_assets": "total_assets",
    "total_liabilities": "total_liabilities",
}


@dataclass(frozen=True)
class _UnitResolution:
    status: str
    unit_source: str
    currency: str | None
    scale: float | None
    unit_raw: str
    normalized_value_allowed: bool
    reason: str
    matched_number_span: tuple[int, int] | None = None
    note_parts: tuple[str, ...] = ()


@dataclass(frozen=True)
class _UnitToken:
    start: int
    end: int
    raw: str
    currency: str | None
    scale: float | None


def convert_headline_metric_facts(facts: HeadlineMetricFacts) -> list[FinancialFact]:
    converted: list[FinancialFact] = []
    fiscal_period = _field_text(facts.fiscal_period) or ""
    unit_raw = _field_text(facts.unit) or ""
    company_name = _field_text(facts.company_name)

    for source_field, metric_id in _HEADLINE_FIELD_TO_CANONICAL.items():
        metric_def = get_metric_definition(metric_id)
        if metric_def is None:
            continue
        fact_field = getattr(facts, source_field)
        value = normalize_numeric_value(fact_field.value)
        if value is None:
            continue

        global_unit = normalize_unit(unit_raw)
        evidence_page, evidence_quote = _best_page_quote(fact_field)
        unit_resolution = _resolve_unit_for_fact(
            value=value,
            global_unit=global_unit,
            evidence_quote=evidence_quote,
        )
        fiscal_year = extract_fiscal_year(fiscal_period)
        status, notes = _review_status(
            value=value,
            unit_clear=bool(unit_resolution.currency and unit_resolution.scale),
            fiscal_period=fiscal_period,
            fiscal_year=fiscal_year,
            evidence_page=evidence_page,
            evidence_quote=evidence_quote,
            unit_resolution=unit_resolution,
        )
        normalized = (
            value * unit_resolution.scale
            if unit_resolution.normalized_value_allowed and unit_resolution.scale is not None
            else None
        )
        converted.append(
            FinancialFact(
                fact_id=_build_fact_id(
                    document_id=facts.document_id,
                    metric_id=metric_def.metric_id,
                    fiscal_period=fiscal_period,
                    period_type=metric_def.period_type,
                ),
                document_id=facts.document_id,
                company_name=company_name,
                source_path=Path(facts.source_path),
                metric_id=metric_def.metric_id,
                metric_label=metric_def.label,
                source_metric_name=source_field,
                period_type=metric_def.period_type,
                fiscal_period=fiscal_period,
                fiscal_year=fiscal_year,
                value=value,
                unit_raw=unit_resolution.unit_raw,
                currency=unit_resolution.currency,
                scale=unit_resolution.scale,
                normalized_value=normalized,
                normalized_unit=unit_resolution.currency
                if unit_resolution.currency and unit_resolution.scale
                else None,
                evidence_page=evidence_page,
                evidence_quote=evidence_quote,
                review_status=status,
                notes=notes,
            )
        )

    return converted


def _field_text(field: ExtractedFactField) -> str | None:
    value = field.value
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _best_page_quote(field: ExtractedFactField) -> tuple[int | None, str]:
    quote = (field.evidence_quote or "").strip()
    if field.evidence_page is not None and quote:
        return int(field.evidence_page), quote

    citation = _first_complete_citation(field.citations)
    if citation is not None:
        return citation.page_number, citation.quote.strip()

    return field.evidence_page, quote


def _first_complete_citation(citations: list[Citation]) -> Citation | None:
    for citation in citations:
        if citation.page_number is not None and citation.quote.strip():
            return citation
    return None


def _review_status(
    *,
    value: float | None,
    unit_clear: bool,
    fiscal_period: str,
    fiscal_year: int | None,
    evidence_page: int | None,
    evidence_quote: str,
    unit_resolution: _UnitResolution,
) -> tuple[ReviewStatus, str | None]:
    missing: list[str] = []
    if value is None:
        missing.append("value")
    if not unit_clear:
        missing.append("unit_currency_or_scale")
    if not fiscal_period.strip():
        missing.append("fiscal_period")
    if fiscal_year is None:
        missing.append("fiscal_year")
    if evidence_page is None:
        missing.append("evidence_page")
    if not evidence_quote.strip():
        missing.append("evidence_quote")
    if unit_resolution.status == "needs_review":
        missing.append(unit_resolution.reason)

    if missing:
        return "needs_review", _join_notes(["missing: " + ", ".join(missing), _unit_note(unit_resolution)])
    return "verified", _unit_note(unit_resolution)


def _resolve_unit_for_fact(
    *,
    value: float,
    global_unit: object,
    evidence_quote: str,
) -> _UnitResolution:
    global_currency = getattr(global_unit, "currency", None)
    global_scale = getattr(global_unit, "scale", None)
    global_raw = getattr(global_unit, "unit_raw", "") or ""

    quote = _normalize_quote_text(evidence_quote)
    if quote:
        matched_resolutions = _matched_quote_unit_resolutions(
            quote=quote,
            value=value,
            global_currency=global_currency,
            global_scale=global_scale,
        )
        if matched_resolutions.status == "resolved":
            resolution = matched_resolutions.resolution
            assert resolution is not None
            if global_currency and resolution.currency and resolution.currency != global_currency:
                return _needs_review_unit(
                    reason="currency_conflict",
                    unit_raw=resolution.unit_raw,
                    currency=resolution.currency,
                    scale=resolution.scale,
                    span=resolution.matched_number_span,
                )
            if global_scale is not None and resolution.scale != global_scale:
                return _with_reason(
                    resolution,
                    "quote_local_unit_global_ignored",
                    extra_note=f"global_unit_ignored={global_raw}",
                )
            return resolution
        if matched_resolutions.status == "needs_review":
            return _needs_review_unit(
                reason=matched_resolutions.reason,
                unit_raw=global_raw,
                currency=global_currency,
                scale=global_scale,
                span=matched_resolutions.span,
            )
        if not matched_resolutions.value_found and _quote_has_unit_signal(quote):
            return _needs_review_unit(
                reason="value_unbound",
                unit_raw=global_raw,
                currency=global_currency,
                scale=global_scale,
                span=None,
            )

    return _UnitResolution(
        status="fallback_global",
        unit_source="global",
        currency=global_currency,
        scale=global_scale,
        unit_raw=global_raw,
        normalized_value_allowed=bool(global_currency and global_scale),
        reason="global_fallback",
    )


@dataclass(frozen=True)
class _MatchedQuoteResolution:
    status: str
    reason: str
    value_found: bool
    resolution: _UnitResolution | None = None
    span: tuple[int, int] | None = None


def _matched_quote_unit_resolutions(
    *,
    quote: str,
    value: float,
    global_currency: str | None,
    global_scale: float | None,
) -> _MatchedQuoteResolution:
    matched_spans = _matched_number_spans(quote, value)
    if not matched_spans:
        return _MatchedQuoteResolution(status="fallback_global", reason="no_matching_value", value_found=False)

    resolutions: list[_UnitResolution] = []
    unbound_spans: list[tuple[int, int]] = []
    for span in matched_spans:
        local_resolution = _resolve_local_quote_unit(
            quote=quote,
            span=span,
            global_currency=global_currency,
            global_scale=global_scale,
        )
        if local_resolution.status == "needs_review":
            return _MatchedQuoteResolution(
                status="needs_review",
                reason=local_resolution.reason,
                value_found=True,
                span=span,
            )
        if local_resolution.status == "resolved":
            resolutions.append(local_resolution)
        else:
            unbound_spans.append(span)

    if not resolutions:
        return _MatchedQuoteResolution(status="fallback_global", reason="no_bound_quote_unit", value_found=True)
    if unbound_spans:
        return _MatchedQuoteResolution(
            status="needs_review",
            reason="unit_ambiguous",
            value_found=True,
            span=unbound_spans[0],
        )
    first = resolutions[0]
    if any(
        resolution.currency != first.currency or resolution.scale != first.scale
        for resolution in resolutions[1:]
    ):
        return _MatchedQuoteResolution(
            status="needs_review",
            reason="unit_ambiguous",
            value_found=True,
            span=first.matched_number_span,
        )
    return _MatchedQuoteResolution(
        status="resolved",
        reason="quote_local_unit",
        value_found=True,
        resolution=first,
        span=first.matched_number_span,
    )


def _resolve_local_quote_unit(
    *,
    quote: str,
    span: tuple[int, int],
    global_currency: str | None,
    global_scale: float | None,
) -> _UnitResolution:
    window_start = max(0, span[0] - 24)
    window_end = min(len(quote), span[1] + 24)
    tokens = _unit_tokens(quote[window_start:window_end], offset=window_start)
    if not tokens:
        return _UnitResolution(
            status="fallback_global",
            unit_source="global",
            currency=global_currency,
            scale=None,
            unit_raw="",
            normalized_value_allowed=False,
            reason="no_bound_quote_unit",
            matched_number_span=span,
        )

    scale_tokens = [token for token in tokens if token.scale is not None]
    currency_tokens = [token for token in tokens if token.currency is not None]
    if not scale_tokens:
        nearest_currency = _nearest_unique_token(currency_tokens, span)
        if nearest_currency is None:
            return _needs_review_unit(
                reason="unit_ambiguous",
                unit_raw="",
                currency=global_currency,
                scale=global_scale,
                span=span,
            )
        if global_currency and nearest_currency.currency != global_currency:
            return _needs_review_unit(
                reason="currency_conflict",
                unit_raw=nearest_currency.raw,
                currency=global_currency,
                scale=global_scale,
                span=span,
            )
        if global_scale is not None:
            return _UnitResolution(
                status="fallback_global",
                unit_source="global",
                currency=global_currency or nearest_currency.currency,
                scale=global_scale,
                unit_raw=nearest_currency.raw,
                normalized_value_allowed=bool(global_currency or nearest_currency.currency),
                reason="currency_only_global_fallback",
                matched_number_span=span,
            )
        return _needs_review_unit(
            reason="unit_ambiguous",
            unit_raw=nearest_currency.raw,
            currency=nearest_currency.currency,
            scale=None,
            span=span,
        )

    nearest_scale = _nearest_unique_token(scale_tokens, span)
    if nearest_scale is None:
        return _needs_review_unit(
            reason="unit_ambiguous",
            unit_raw="",
            currency=global_currency,
            scale=None,
            span=span,
        )

    tied_scales = _tokens_at_distance(scale_tokens, span, _token_distance(nearest_scale, span))
    if len({token.scale for token in tied_scales}) > 1:
        return _needs_review_unit(
            reason="unit_ambiguous",
            unit_raw=nearest_scale.raw,
            currency=global_currency,
            scale=None,
            span=span,
        )

    nearest_currency = _nearest_unique_token(currency_tokens, span)
    currency = nearest_currency.currency if nearest_currency else global_currency
    unit_raw = _quote_unit_raw(scale_token=nearest_scale, currency_token=nearest_currency)
    return _UnitResolution(
        status="resolved",
        unit_source="quote",
        currency=currency,
        scale=nearest_scale.scale,
        unit_raw=unit_raw,
        normalized_value_allowed=bool(currency and nearest_scale.scale),
        reason="quote_local_unit",
        matched_number_span=span,
    )


def _matched_number_spans(quote: str, value: float) -> list[tuple[int, int]]:
    target = _decimal_from_number(value)
    if target is None:
        return []
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[-+]?\d[\d,\s]*(?:\.\d+)?", quote):
        token_value = _decimal_from_text(match.group(0))
        if token_value is not None and token_value == target:
            spans.append(match.span())
    return spans


def _unit_tokens(text: str, *, offset: int = 0) -> list[_UnitToken]:
    candidates: list[_UnitToken] = []
    for pattern, currency, scale in _UNIT_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidates.append(
                _UnitToken(
                    start=offset + match.start(),
                    end=offset + match.end(),
                    raw=match.group(0),
                    currency=currency,
                    scale=scale,
                )
            )
    return _longest_non_overlapping_tokens(candidates)


def _longest_non_overlapping_tokens(tokens: list[_UnitToken]) -> list[_UnitToken]:
    selected: list[_UnitToken] = []
    for token in sorted(tokens, key=lambda item: (-(item.end - item.start), item.start)):
        if any(not (token.end <= existing.start or token.start >= existing.end) for existing in selected):
            continue
        selected.append(token)
    return sorted(selected, key=lambda item: item.start)


def _nearest_unique_token(tokens: list[_UnitToken], span: tuple[int, int]) -> _UnitToken | None:
    if not tokens:
        return None
    distances = [(token, _token_distance(token, span)) for token in tokens]
    min_distance = min(distance for _token, distance in distances)
    nearest = [token for token, distance in distances if distance == min_distance]
    if len({(token.currency, token.scale) for token in nearest}) > 1:
        return None
    return nearest[0]


def _tokens_at_distance(
    tokens: list[_UnitToken],
    span: tuple[int, int],
    distance: int,
) -> list[_UnitToken]:
    return [token for token in tokens if _token_distance(token, span) == distance]


def _token_distance(token: _UnitToken, span: tuple[int, int]) -> int:
    if token.end <= span[0]:
        return span[0] - token.end
    if token.start >= span[1]:
        return token.start - span[1]
    return 0


def _quote_has_unit_signal(quote: str) -> bool:
    return bool(_unit_tokens(quote))


def _quote_unit_raw(
    *,
    scale_token: _UnitToken,
    currency_token: _UnitToken | None,
) -> str:
    if currency_token is not None and currency_token.raw not in scale_token.raw:
        return f"{currency_token.raw} {scale_token.raw}"
    return scale_token.raw


def _with_reason(
    resolution: _UnitResolution,
    reason: str,
    *,
    extra_note: str | None = None,
) -> _UnitResolution:
    return _UnitResolution(
        status=resolution.status,
        unit_source=resolution.unit_source,
        currency=resolution.currency,
        scale=resolution.scale,
        unit_raw=resolution.unit_raw,
        normalized_value_allowed=resolution.normalized_value_allowed,
        reason=reason,
        matched_number_span=resolution.matched_number_span,
        note_parts=resolution.note_parts + ((extra_note,) if extra_note else ()),
    )


def _needs_review_unit(
    *,
    reason: str,
    unit_raw: str,
    currency: str | None,
    scale: float | None,
    span: tuple[int, int] | None,
) -> _UnitResolution:
    return _UnitResolution(
        status="needs_review",
        unit_source="none",
        currency=currency,
        scale=scale,
        unit_raw=unit_raw,
        normalized_value_allowed=False,
        reason=reason,
        matched_number_span=span,
    )


def _normalize_quote_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def _decimal_from_number(value: float) -> Decimal | None:
    try:
        return Decimal(str(value)).normalize()
    except InvalidOperation:
        return None


def _decimal_from_text(text: str) -> Decimal | None:
    cleaned = re.sub(r"[\s,]", "", _normalize_quote_text(text))
    if not cleaned:
        return None
    try:
        return Decimal(cleaned).normalize()
    except InvalidOperation:
        return None


def _unit_note(unit_resolution: _UnitResolution) -> str | None:
    if unit_resolution.status == "fallback_global":
        return None
    parts = [f"unit_source={unit_resolution.unit_source}", f"unit_reason={unit_resolution.reason}"]
    if unit_resolution.currency:
        parts.append(f"currency={unit_resolution.currency}")
    if unit_resolution.scale is not None:
        parts.append(f"scale={unit_resolution.scale:g}")
    if unit_resolution.unit_raw:
        parts.append(f"unit_raw={unit_resolution.unit_raw}")
    parts.extend(unit_resolution.note_parts)
    return "; ".join(parts)


def _join_notes(parts: list[str | None]) -> str | None:
    clean_parts = [part for part in parts if part]
    return "; ".join(clean_parts) or None


def _build_fact_id(
    *,
    document_id: str,
    metric_id: str,
    fiscal_period: str,
    period_type: str,
) -> str:
    payload = "|".join((document_id, metric_id, fiscal_period, period_type))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{document_id}:{metric_id}:{digest}"


_UNIT_PATTERNS: tuple[tuple[str, str | None, float | None], ...] = (
    (r"人民币\s*百万元", "CNY", 1_000_000.0),
    (r"人民幣\s*百萬元", "CNY", 1_000_000.0),
    (r"人民币\s*千元", "CNY", 1_000.0),
    (r"人民幣\s*千元", "CNY", 1_000.0),
    (r"人民币\s*万元", "CNY", 10_000.0),
    (r"人民幣\s*萬(?:元)?", "CNY", 10_000.0),
    (r"人民币\s*亿元", "CNY", 100_000_000.0),
    (r"人民幣\s*億(?:元)?", "CNY", 100_000_000.0),
    (r"HKD\s*million", "HKD", 1_000_000.0),
    (r"USD\s*million", "USD", 1_000_000.0),
    (r"RMB\s*million", "CNY", 1_000_000.0),
    (r"CNY\s*million", "CNY", 1_000_000.0),
    (r"百万元", None, 1_000_000.0),
    (r"百萬元", None, 1_000_000.0),
    (r"million", None, 1_000_000.0),
    (r"亿元", None, 100_000_000.0),
    (r"億(?:元)?", None, 100_000_000.0),
    (r"万元", None, 10_000.0),
    (r"萬(?:元)?", None, 10_000.0),
    (r"千元", None, 1_000.0),
    (r"港元", "HKD", 1.0),
    (r"港币", "HKD", 1.0),
    (r"港幣", "HKD", 1.0),
    (r"美元", "USD", 1.0),
    (r"人民币", "CNY", None),
    (r"人民幣", "CNY", None),
    (r"\bRMB\b", "CNY", None),
    (r"\bCNY\b", "CNY", None),
    (r"\bHKD\b", "HKD", None),
    (r"\bUSD\b", "USD", None),
    (r"元", None, 1.0),
)
