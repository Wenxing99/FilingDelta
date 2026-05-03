from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PeriodType = Literal["period", "end_of_period"]


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    label: str
    period_type: PeriodType
    aliases: tuple[str, ...]


def _normalize_alias(value: str) -> str:
    return " ".join(value.strip().casefold().replace("_", " ").split())


CANONICAL_METRICS: dict[str, MetricDefinition] = {
    "revenue": MetricDefinition(
        metric_id="revenue",
        label="营业收入",
        period_type="period",
        aliases=(
            "revenue",
            "revenues",
            "total revenue",
            "total revenues",
            "operating revenue",
            "营业收入",
            "營業收入",
            "收入",
        ),
    ),
    "net_profit_attributable": MetricDefinition(
        metric_id="net_profit_attributable",
        label="归母净利润",
        period_type="period",
        aliases=(
            "net_profit",
            "net profit",
            "net profit attributable",
            "profit attributable",
            "profit attributable to shareholders",
            "profit attributable to owners",
            "net profit attributable to parent",
            "归母净利润",
            "归属于母公司股东的净利润",
            "归属于上市公司股东的净利润",
            "归属于本行股东的净利润",
            "歸母淨利潤",
            "歸屬於母公司股東的淨利潤",
            "歸屬於上市公司股東的淨利潤",
            "歸屬於本行股東的淨利潤",
        ),
    ),
    "total_assets": MetricDefinition(
        metric_id="total_assets",
        label="总资产",
        period_type="end_of_period",
        aliases=(
            "total_assets",
            "total assets",
            "assets total",
            "总资产",
            "資產總額",
            "总资产额",
            "總資產",
        ),
    ),
    "total_liabilities": MetricDefinition(
        metric_id="total_liabilities",
        label="总负债",
        period_type="end_of_period",
        aliases=(
            "total_liabilities",
            "total liabilities",
            "liabilities total",
            "总负债",
            "負債總額",
            "总负债额",
            "總負債",
        ),
    ),
}

_ALIAS_TO_METRIC_ID = {
    _normalize_alias(alias): metric.metric_id
    for metric in CANONICAL_METRICS.values()
    for alias in (metric.metric_id, *metric.aliases)
}


def canonicalize_metric_id(metric_or_alias: str) -> str | None:
    return _ALIAS_TO_METRIC_ID.get(_normalize_alias(metric_or_alias))


def get_metric_definition(metric_or_alias: str) -> MetricDefinition | None:
    metric_id = canonicalize_metric_id(metric_or_alias)
    if metric_id is None:
        return None
    return CANONICAL_METRICS[metric_id]
