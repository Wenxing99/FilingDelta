from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionTaxonomyRule:
    section_type: str
    display_title: str
    keywords: tuple[str, ...]


SECTION_TAXONOMY_RULES: tuple[SectionTaxonomyRule, ...] = (
    SectionTaxonomyRule(
        section_type="financial_summary",
        display_title="财务表现摘要",
        keywords=(
            "financial highlights",
            "financial summary",
            "主要财务数据",
            "主要財務數據",
            "财务摘要",
            "財務摘要",
            "营业收入",
            "營業收入",
            "收入",
            "净利润",
            "淨利潤",
            "毛利",
            "销售及市场推广开支",
            "銷售及市場推廣開支",
            "一般及行政开支",
            "一般及行政開支",
            "利息收入",
            "利息支出",
            "所得税开支",
            "所得稅開支",
            "财务成本",
            "財務成本",
            "每股盈利",
            "profit attributable",
            "operating profit",
        ),
    ),
    SectionTaxonomyRule(
        section_type="dividend",
        display_title="股息 / 分红",
        keywords=(
            "dividend",
            "股息",
            "分红",
            "派息",
            "现金股息",
            "末期股息",
            "每股现金分红",
        ),
    ),
    SectionTaxonomyRule(
        section_type="operating_metrics",
        display_title="经营数据 / 关键指标",
        keywords=(
            "总资产",
            "客户存款",
            "贷款和垫款",
            "经营活动产生的现金流量净额",
            "月活跃账户数",
            "订阅会员数",
            "不良贷款率",
            "拨备覆盖率",
            "active accounts",
            "arpu",
            "revenue growth",
        ),
    ),
    SectionTaxonomyRule(
        section_type="strategy_outlook",
        display_title="战略与展望",
        keywords=(
            "strategy",
            "strategic",
            "outlook",
            "plan",
            "目标",
            "战略",
            "戰略",
            "将继续",
            "计划",
            "計劃",
            "展望",
            "愿景",
            "願景",
            "best value",
            "ai first",
            "人工智能+",
            "数智化",
            "數智化",
        ),
    ),
    SectionTaxonomyRule(
        section_type="shareholder",
        display_title="股东情况",
        keywords=(
            "shareholder",
            "股东",
            "股東",
            "普通股股东总数",
            "普通股股東總數",
            "a股股东总数",
            "a股股東總數",
            "h股股东总数",
            "h股股東總數",
        ),
    ),
    SectionTaxonomyRule(
        section_type="risk_asset_quality",
        display_title="风险与资产质量",
        keywords=(
            "不良贷款",
            "不良貸款",
            "贷款损失准备",
            "貸款損失準備",
            "拨备覆盖率",
            "撥備覆蓋率",
            "风险",
            "風險",
            "资产质量",
            "資產質量",
            "npl",
            "impairment",
            "allowance",
            "房地产",
            "房地產",
            "地方政府隐性债务",
            "地方政府隱性債務",
            "合规",
            "合規",
            "财务风险",
            "財務風險",
            "信贷风险",
            "信貸風險",
            "流动性风险",
            "流動性風險",
        ),
    ),
    SectionTaxonomyRule(
        section_type="business_review",
        display_title="业务回顾",
        keywords=(
            "business review",
            "业务回顾",
            "广告",
            "游戏",
            "视频号",
            "雲服務",
            "云服务",
            "金融科技",
            "wechat",
            "qq",
            "mini program",
            "cloud",
        ),
    ),
    SectionTaxonomyRule(
        section_type="product_user_metrics",
        display_title="用户与产品数据",
        keywords=(
            "月活跃账户数",
            "月活躍賬戶數",
            "月活跃",
            "月活躍",
            "活躍賬戶數",
            "活跃账户数",
            "使用时长",
            "使用時長",
            "用户时长",
            "用戶時長",
            "總用戶使用時長",
            "总用户使用时长",
            "subscription",
            "会员数",
            "會員數",
            "订阅会员数",
            "訂閱會員數",
            "wechat",
            "qq",
            "视频号",
            "視頻號",
            "小游戏",
            "mini games",
            "video accounts",
            "users",
        ),
    ),
    SectionTaxonomyRule(
        section_type="sustainability",
        display_title="可持续发展",
        keywords=(
            "esg",
            "可持续",
            "公益",
            "碳中和",
            "慈善",
            "绿色电力",
            "sustainability",
            "climate",
        ),
    ),
    SectionTaxonomyRule(
        section_type="governance",
        display_title="公司治理",
        keywords=(
            "corporate governance",
            "governance",
            "董事会",
            "监事会",
            "审计",
            "委员会",
            "董事",
            "治理",
        ),
    ),
)

SECTION_KEYWORDS_BY_TITLE: dict[str, tuple[str, ...]] = {
    rule.display_title: rule.keywords for rule in SECTION_TAXONOMY_RULES
}
SECTION_TITLES = tuple(rule.display_title for rule in SECTION_TAXONOMY_RULES)
SECTION_TYPE_BY_TITLE = {
    rule.display_title: rule.section_type for rule in SECTION_TAXONOMY_RULES
}


def infer_section_type(*texts: str) -> str:
    normalized_text = " ".join(_normalize_for_match(text) for text in texts if text)
    if not normalized_text:
        return "other"

    best_match = ("other", 0)
    for rule in SECTION_TAXONOMY_RULES:
        score = sum(
            1
            for keyword in rule.keywords
            if _normalize_for_match(keyword) and _normalize_for_match(keyword) in normalized_text
        )
        if score > best_match[1]:
            best_match = (rule.section_type, score)
    return best_match[0]


def _normalize_for_match(text: str) -> str:
    return "".join(text.lower().split())
