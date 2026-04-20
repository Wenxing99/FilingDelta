from filingdelta.ingestion.fact_extractors import _extract_preferred_roe


def test_extract_preferred_roe_prefers_roae_over_nearby_growth_rate() -> None:
    text = (
        "报告期内，本集团实现营业收入3,375.32亿元，同比增长0.01%，其中，净利息收入2,155.93亿元，"
        "同比增长2.04%，非利息净收入1,219.39亿元，同比下降3.38%；实现归属于本行股东的净利润"
        "1,501.81亿元，同比增长1.21%；归属于本行股东的平均总资产收益率(ROAA)和归属于本行"
        "普通股股东的平均净资产收益率(ROAE)分别为1.19%和13.44%，同比分别下降0.09和1.05个百分点。"
    )

    assert _extract_preferred_roe(text) == (13.44, "分别为1.19%和13.44%", 4)


def test_extract_preferred_roe_handles_parser_inserted_spaces_in_pair_phrase() -> None:
    text = (
        "归属于本行股东的平均总资产收益率(ROAA)和归属于本行普通股股东的平均净资产收益率(ROAE)"
        "分 别为1.19%和13.44%，同比分别下降0.09和1.05个百分点。"
    )

    assert _extract_preferred_roe(text) == (13.44, "分 别为1.19%和13.44%", 4)


def test_extract_preferred_roe_rejects_growth_rate_when_roe_value_is_missing() -> None:
    text = "实现归属于本行股东的净利润1,501.81亿元，同比增长1.21%；归属于本行普通股股东的平均净资产收益率(ROAE)"

    assert _extract_preferred_roe(text) is None


def test_extract_preferred_roe_supports_table_like_anchor_value_layout() -> None:
    text = "归属于本行普通股股东的平均净资产收益率(ROAE)\n13.44%"

    assert _extract_preferred_roe(text) == (
        13.44,
        "归属于本行普通股股东的平均净资产收益率(ROAE) 13.44%",
        4,
    )
