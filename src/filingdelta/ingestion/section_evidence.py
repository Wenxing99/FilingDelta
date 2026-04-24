from __future__ import annotations

from dataclasses import dataclass
import re
from uuid import NAMESPACE_URL, uuid5

from filingdelta.ingestion.section_taxonomy import infer_section_type
from filingdelta.schemas.filing import EvidenceKind, EvidenceMetadata, EvidenceUnit, ParsedFiling


_CHAPTER_HEADING_RE = re.compile(r"^第[一二三四五六七八九十百千万0-9]+章\s*.+$")
_NUMBERED_HEADING_RE = re.compile(
    r"^\d+(?:\.\d+){1,3}(?:\s+|(?=[A-Za-z\u4e00-\u9fff]))[A-Za-z\u4e00-\u9fff].*$"
)
_SINGLE_NUMBER_HEADING_RE = re.compile(
    r"^\d{1,2}(?:\s+|(?=[A-Za-z\u4e00-\u9fff]))[A-Za-z\u4e00-\u9fff].*$"
)
_SPLIT_NUMBER_PREFIX_RE = re.compile(r"^\d+(?:\.\d+){0,3}\.?$")
_FRAME_NUMBER_RE = re.compile(r"^\d{1,4}$")
_FRAME_WRAPPER_RE = re.compile(r"^\d{1,3}\s+.+$")
_INLINE_TOPIC_RE = re.compile(r"^(?P<title>[^。]{2,24})。(?P<body>.+)$")
_INLINE_TOPIC_TITLE_RE = re.compile(r"^[A-Za-z\u4e00-\u9fff\s\-/]+$")
_COLON_TOPIC_RE = re.compile(r"^(?P<title>[^：:]{2,36})[：:]\s*$")
_LEDE_HEADING_RE = re.compile(r"^(?P<title>展望(?:20\d{2}|二零[一二三四五六七八九十]{2})年)[，,。](?P<body>.+)$")
_HEADING_PREFIX_RE = re.compile(r"^(?P<prefix>\d+(?:\.\d+){0,3})(?:\s+|)(?P<rest>.+)$")
_REPORT_LABEL_HINTS = ("年度报告", "年度報告", "年報", "annual report")
_GENERIC_HEADINGS = {
    "管理层讨论与分析",
    "管理層討論及分析",
    "财务表现摘要",
    "財務表現摘要",
    "业务回顾",
    "業務回顧",
    "主席報告",
    "主席报告",
    "業績",
    "业绩",
    "經營資料",
    "经营资料",
    "企業管治報告",
    "企业管治报告",
}
_INLINE_BODY_PREFIXES = (
    "截至",
    "报告期",
    "報告期",
    "二零",
    "202",
    "本公司",
    "本集团",
    "本集團",
    "我們",
    "下表",
    "主要",
    "年內",
)
_INLINE_MAX_TITLE_CHARS = 12
_INLINE_MIN_BODY_CHARS = 24
_SKIP_GENERIC_BODY_LINE_GAP = 2
_NUMERIC_VALUE_PREFIXES = (
    "%",
    "％",
    "月",
    "日",
    "楼",
    "樓",
    "页",
    "頁",
    "亿",
    "億",
    "万元",
    "萬元",
    "亿元",
    "億元",
    "百万元",
    "百萬元",
    "次",
    "倍",
    "户",
    "戶",
    "点",
    "點",
)
_PLAIN_HEADING_SUFFIXES = (
    "风险",
    "風險",
    "风险管控",
    "風險管控",
    "风险管理",
    "風險管理",
    "应对措施",
    "應對措施",
    "前景展望",
    "业务回顾",
    "業務回顧",
    "重点表现",
    "重點表現",
    "经营资料",
    "經營資料",
    "业绩",
    "業績",
    "收入",
    "毛利",
    "开支",
    "開支",
    "股息",
    "分红",
    "分紅",
    "贷款",
    "貸款",
    "存款",
    "機遇",
)
_PLAIN_HEADING_KEYWORDS = (
    "ai",
    "人工智能",
    "视频号",
    "視頻號",
    "广告",
    "廣告",
    "混元",
    "房地产",
    "房地產",
    "地方政府",
    "零售贷款",
    "零售貸款",
    "消费信贷",
    "消費信貸",
    "小微贷款",
    "小微貸款",
    "金融科技",
    "企业服务",
    "企業服務",
)
_NORMALIZE_TITLE_PREFIXES = ("以下为", "以下為", "其中", "有关", "有關")
_NORMALIZE_TITLE_YEAR_RE = re.compile(r"^(?:20\d{2}|二零[一二三四五六七八九十]{2})年")


@dataclass(frozen=True)
class _Heading:
    title: str
    body_seed: str | None = None


def build_section_evidence(
    parsed_filing: ParsedFiling,
    *,
    min_body_chars: int = 60,
    max_text_chars: int = 2200,
) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    active_heading: _Heading | None = None
    document = parsed_filing.document

    for page in parsed_filing.pages:
        lines = _clean_page_lines(
            page.markdown or page.text,
            company_name=document.company_name,
        )
        if not lines:
            continue

        segments, last_heading = _segment_page(lines, min_body_chars=min_body_chars)
        if segments:
            for segment_index, (heading, body_lines) in enumerate(segments):
                text = _compose_section_text(
                    heading=heading,
                    body_lines=body_lines,
                    max_text_chars=max_text_chars,
                )
                if len(_normalize_for_match(text)) < min_body_chars:
                    continue
                section_type = infer_section_type(heading.title, text)
                if section_type == "other":
                    continue
                units.append(
                    EvidenceUnit(
                        evidence_id=_section_evidence_id(
                            document_id=document.document_id,
                            page_number=page.page_number,
                            section_index=segment_index,
                            heading_title=heading.title,
                        ),
                        text=text,
                        metadata=EvidenceMetadata(
                            document_id=document.document_id,
                            source_path=document.source_path,
                            page_number=page.page_number,
                            page_end=page.page_number,
                            parser_kind=document.parser_kind,
                            chunk_kind=EvidenceKind.SECTION_TEXT,
                            section_title=heading.title,
                            section_type=section_type,
                        ),
                    )
                )
            active_heading = last_heading
            continue

        if active_heading is None:
            continue
        if _is_generic_heading(active_heading.title):
            replacement = _match_local_replacement_heading(lines)
            if replacement is None or _is_generic_heading(replacement.title):
                continue
            active_heading = replacement

        text = _compose_section_text(
            heading=active_heading,
            body_lines=lines,
            max_text_chars=max_text_chars,
        )
        if len(_normalize_for_match(text)) < min_body_chars:
            continue
        section_type = infer_section_type(active_heading.title, text)
        if section_type == "other":
            continue
        units.append(
            EvidenceUnit(
                evidence_id=_section_evidence_id(
                    document_id=document.document_id,
                    page_number=page.page_number,
                    section_index=0,
                    heading_title=active_heading.title,
                ),
                text=text,
                metadata=EvidenceMetadata(
                    document_id=document.document_id,
                    source_path=document.source_path,
                    page_number=page.page_number,
                    page_end=page.page_number,
                    parser_kind=document.parser_kind,
                    chunk_kind=EvidenceKind.SECTION_TEXT,
                    section_title=active_heading.title,
                    section_type=section_type,
                ),
            )
        )

    return units


def _segment_page(
    lines: list[str],
    *,
    min_body_chars: int,
) -> tuple[list[tuple[_Heading, list[str]]], _Heading | None]:
    headings: list[tuple[int, _Heading]] = []
    index = 0
    while index < len(lines):
        heading, consumed_lines = _match_heading(lines, line_index=index)
        if heading is not None:
            headings.append((index, heading))
            index += consumed_lines
            continue
        index += 1

    if not headings:
        return [], None

    segments: list[tuple[_Heading, list[str]]] = []
    for heading_index, (line_index, heading) in enumerate(headings):
        next_index = headings[heading_index + 1][0] if heading_index + 1 < len(headings) else len(lines)
        if (
            _is_generic_heading(heading.title)
            and heading_index + 1 < len(headings)
            and next_index - line_index <= _SKIP_GENERIC_BODY_LINE_GAP + 1
        ):
            continue
        body_lines = list(lines[line_index + 1 : next_index])
        heading, body_lines = _refine_heading_from_body(heading, body_lines)
        if heading.body_seed:
            body_lines.insert(0, heading.body_seed)
        if len(_normalize_for_match("\n".join(body_lines))) < min_body_chars:
            continue
        segments.append((heading, body_lines))

    return segments, headings[-1][1]


def _match_heading(lines: list[str], *, line_index: int) -> tuple[_Heading | None, int]:
    candidate = lines[line_index].strip()
    next_line = lines[line_index + 1].strip() if line_index + 1 < len(lines) else ""
    if not candidate:
        return None, 1
    if _CHAPTER_HEADING_RE.match(candidate):
        return _Heading(title=candidate), 1
    split_heading = _match_split_number_heading(candidate, next_line)
    if split_heading is not None:
        return split_heading, 2
    if _looks_like_numbered_heading(candidate):
        return _Heading(title=candidate), 1
    if candidate in _GENERIC_HEADINGS:
        return _Heading(title=candidate), 1
    colon_heading = _match_colon_heading(candidate)
    if colon_heading is not None:
        return colon_heading, 1
    plain_heading = _match_plain_heading(candidate, next_line=next_line)
    if plain_heading is not None:
        return plain_heading, 1
    lede_heading = _match_lede_heading(candidate)
    if lede_heading is not None:
        return lede_heading, 1
    inline_heading = _match_inline_topic_heading(candidate)
    if inline_heading is not None:
        return inline_heading, 1
    return None, 1


def _match_split_number_heading(candidate: str, next_line: str) -> _Heading | None:
    if not _SPLIT_NUMBER_PREFIX_RE.match(candidate):
        return None
    if not next_line:
        return None
    next_title = _normalize_heading_title(next_line)
    if not next_title:
        return None
    if not _looks_like_plain_heading(next_title):
        return None
    prefix = candidate if candidate.endswith(".") else f"{candidate}."
    return _Heading(title=f"{prefix} {next_title}")


def _match_colon_heading(line: str) -> _Heading | None:
    match = _COLON_TOPIC_RE.match(line)
    if match is None:
        return None
    title = _normalize_heading_title(match.group("title"))
    if not title or not _looks_like_plain_heading(title):
        return None
    return _Heading(title=title)


def _match_plain_heading(candidate: str, *, next_line: str) -> _Heading | None:
    title = _normalize_heading_title(candidate)
    if not title or not next_line:
        return None
    if not _looks_like_plain_heading(title):
        return None
    if _looks_like_heading(next_line):
        return None
    if len(_normalize_for_match(next_line)) < _INLINE_MIN_BODY_CHARS:
        return None
    return _Heading(title=title)


def _match_lede_heading(line: str) -> _Heading | None:
    match = _LEDE_HEADING_RE.match(line)
    if match is None:
        return None
    title = _normalize_heading_title(match.group("title"))
    body = line.strip()
    if not title or len(_normalize_for_match(match.group("body"))) < _INLINE_MIN_BODY_CHARS:
        return None
    return _Heading(title=title, body_seed=body)


def _match_inline_topic_heading(line: str) -> _Heading | None:
    match = _INLINE_TOPIC_RE.match(line)
    if match is None:
        return None

    title = match.group("title").strip(" ：:;,.，、")
    body = line.strip()
    body_prefix = match.group("body").strip()
    if not title or len(title) > _INLINE_MAX_TITLE_CHARS:
        return None
    if any(character.isdigit() for character in title):
        return None
    if not _INLINE_TOPIC_TITLE_RE.match(title):
        return None
    if not body_prefix.startswith(_INLINE_BODY_PREFIXES):
        return None
    if len(_normalize_for_match(match.group("body"))) < _INLINE_MIN_BODY_CHARS:
        return None
    if title.endswith(("如下", "分别", "情况", "说明")):
        return None
    return _Heading(title=title, body_seed=body)


def _looks_like_numbered_heading(candidate: str) -> bool:
    if not (_NUMBERED_HEADING_RE.match(candidate) or _SINGLE_NUMBER_HEADING_RE.match(candidate)):
        return False

    match = _HEADING_PREFIX_RE.match(candidate)
    if match is None:
        return False

    rest = match.group("rest").strip()
    if not rest:
        return False
    if rest.startswith(_NUMERIC_VALUE_PREFIXES):
        return False
    return True


def _looks_like_heading(candidate: str) -> bool:
    if not candidate:
        return False
    if _CHAPTER_HEADING_RE.match(candidate):
        return True
    if _SPLIT_NUMBER_PREFIX_RE.match(candidate):
        return True
    if _looks_like_numbered_heading(candidate):
        return True
    if candidate in _GENERIC_HEADINGS:
        return True
    if _COLON_TOPIC_RE.match(candidate):
        return True
    if _INLINE_TOPIC_RE.match(candidate):
        return True
    return _looks_like_plain_heading(_normalize_heading_title(candidate))


def _looks_like_plain_heading(candidate: str) -> bool:
    clean = _normalize_heading_title(candidate)
    if not clean:
        return False
    if len(clean) < 2 or len(clean) > 24:
        return False
    if any(token in clean for token in ("。", "；", ";", "，", ",", "：", ":")):
        return False
    if re.search(r"\d", clean) and not clean.startswith(("展望20", "展望二零")):
        return False
    if clean in _GENERIC_HEADINGS:
        return False
    if clean.endswith(_PLAIN_HEADING_SUFFIXES):
        return True
    normalized = _normalize_for_match(clean)
    return any(keyword in normalized for keyword in (_normalize_for_match(item) for item in _PLAIN_HEADING_KEYWORDS))


def _normalize_heading_title(candidate: str) -> str:
    title = candidate.strip().strip(" ：:;,.，、")
    for prefix in _NORMALIZE_TITLE_PREFIXES:
        if title.startswith(prefix):
            title = title[len(prefix) :].strip()
    title = _NORMALIZE_TITLE_YEAR_RE.sub("", title).strip()
    if title.startswith(("我們", "我们")):
        title = title[2:].strip()
    return title.strip(" ：:;,.，、")


def _is_generic_heading(title: str) -> bool:
    clean = " ".join(title.split())
    if clean in _GENERIC_HEADINGS:
        return True
    if _CHAPTER_HEADING_RE.match(clean):
        return True
    return any(token in clean for token in _GENERIC_HEADINGS)


def _refine_heading_from_body(heading: _Heading, body_lines: list[str]) -> tuple[_Heading, list[str]]:
    if not _is_generic_heading(heading.title) or not body_lines:
        return heading, body_lines

    replacement = _match_local_replacement_heading(body_lines)
    if replacement is None:
        return heading, body_lines
    return replacement, body_lines[1:]


def _match_local_replacement_heading(lines: list[str]) -> _Heading | None:
    if not lines:
        return None
    first_line = lines[0].strip()
    next_line = lines[1].strip() if len(lines) > 1 else ""
    return (
        _match_colon_heading(first_line)
        or _match_plain_heading(first_line, next_line=next_line)
        or _match_lede_heading(first_line)
        or _match_inline_topic_heading(first_line)
    )


def _clean_page_lines(text: str, *, company_name: str) -> list[str]:
    raw_lines = [line.strip() for line in text.splitlines()]
    compact_lines = [line for line in raw_lines if line]
    if not compact_lines:
        return []

    company_marker = company_name.replace(" ", "")
    cleaned: list[str] = []
    last_line = ""
    total_lines = len(compact_lines)
    for index, line in enumerate(compact_lines):
        normalized = line.replace(" ", "")
        if _is_frame_line(
            normalized_line=normalized,
            raw_line=line,
            line_index=index,
            total_lines=total_lines,
            company_marker=company_marker,
        ):
            continue
        if line == last_line:
            continue
        cleaned.append(line)
        last_line = line
    return cleaned


def _is_frame_line(
    *,
    normalized_line: str,
    raw_line: str,
    line_index: int,
    total_lines: int,
    company_marker: str,
) -> bool:
    near_top = line_index <= 4
    near_bottom = line_index >= total_lines - 2

    if _FRAME_NUMBER_RE.match(normalized_line) and (near_top or near_bottom):
        return True
    if near_top and _FRAME_WRAPPER_RE.match(raw_line) and any(token in raw_line for token in _GENERIC_HEADINGS):
        return True
    if near_top and company_marker and company_marker in normalized_line:
        return True
    if near_top and raw_line.endswith(("有限公司", "股份有限公司")):
        return True
    if near_top and any(hint in raw_line.lower() for hint in _REPORT_LABEL_HINTS):
        return True
    return False


def _compose_section_text(
    *,
    heading: _Heading,
    body_lines: list[str],
    max_text_chars: int,
) -> str:
    if heading.body_seed:
        text = "\n".join(body_lines).strip()
    else:
        text = f"{heading.title}\n" + "\n".join(body_lines).strip()
    text = text.strip()
    if len(text) <= max_text_chars:
        return text
    return text[: max_text_chars - 3].rstrip() + "..."


def _section_evidence_id(
    *,
    document_id: str,
    page_number: int,
    section_index: int,
    heading_title: str,
) -> str:
    stable_key = f"{document_id}:section_text:{page_number}:{section_index}:{heading_title}"
    return str(uuid5(NAMESPACE_URL, stable_key))


def _normalize_for_match(text: str) -> str:
    return "".join(text.lower().split())
