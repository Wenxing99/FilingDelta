from pathlib import Path

from filingdelta.agents.answerer import _build_retrieved_context
from filingdelta.schemas.chat import RetrievedChunk
from filingdelta.schemas.filing import EvidenceKind


def test_build_retrieved_context_includes_table_row_metadata() -> None:
    context, ref_map = _build_retrieved_context(
        [
            RetrievedChunk(
                chunk_id="row-1",
                document_id="doc-test",
                page_number=30,
                source_path=Path("dummy.pdf"),
                text="客户存款总额\n98,361.30\n8.13%",
                chunk_kind=EvidenceKind.TABLE_ROW.value,
                row_label="客户存款",
                metric_tags=["customer_deposits", "deposits"],
                period_hint="fy2025",
            )
        ]
    )

    assert ref_map == {"DOC_1": "row-1"}
    assert "Evidence kind: table_row" in context
    assert "Table row: 客户存款" in context
    assert "Metric tags: customer_deposits, deposits" in context
    assert "Period hint: fy2025" in context
