from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class UnitNormalization:
    unit_raw: str
    currency: str | None
    scale: float | None
    normalized_unit: str | None

    @property
    def is_clear(self) -> bool:
        return bool(self.currency and self.scale)


def normalize_numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def normalize_unit(unit: str | None) -> UnitNormalization:
    raw = (unit or "").strip()
    normalized = raw.casefold().replace(" ", "")
    currency = _detect_currency(raw, normalized)
    scale = _detect_scale(raw, normalized)
    normalized_unit = currency if currency and scale else None
    return UnitNormalization(
        unit_raw=raw,
        currency=currency,
        scale=scale,
        normalized_unit=normalized_unit,
    )


def normalized_value(value: float | None, unit: UnitNormalization) -> float | None:
    if value is None or unit.scale is None:
        return None
    return value * unit.scale


def extract_fiscal_year(fiscal_period: str | None) -> int | None:
    if not fiscal_period:
        return None
    match = re.search(r"(20\d{2}|19\d{2})", fiscal_period)
    if not match:
        return None
    return int(match.group(1))


def _detect_currency(raw: str, normalized: str) -> str | None:
    if re.search(r"\b(hkd|hk\$|港元|港币|港幣)\b", raw, re.IGNORECASE) or any(
        token in normalized for token in ("hkd", "hk$", "港元", "港币", "港幣")
    ):
        return "HKD"
    if re.search(r"\b(usd|us\$|美元)\b", raw, re.IGNORECASE) or any(
        token in normalized for token in ("usd", "us$", "美元")
    ):
        return "USD"
    if any(
        token in normalized
        for token in (
            "rmb",
            "cny",
            "人民币",
            "人民幣",
            "元",
            "äººæ°‘",
            "äººæ°‘å¹£",
        )
    ):
        return "CNY"
    return None


def _detect_scale(raw: str, normalized: str) -> float | None:
    if any(token in normalized for token in ("百万元", "百萬", "million", "ç™¾ä¸‡")):
        return 1_000_000.0
    if any(token in normalized for token in ("千元", "thousand", "åƒ")):
        return 1_000.0
    if any(token in normalized for token in ("亿元", "億元", "亿", "億", "äº¿")):
        return 100_000_000.0
    if any(token in normalized for token in ("万元", "萬元", "万", "萬", "ä¸‡")):
        return 10_000.0
    if raw and any(token in normalized for token in ("元", "rmb", "cny", "hkd", "usd")):
        return 1.0
    return None
