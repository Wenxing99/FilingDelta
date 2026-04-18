from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from filingdelta.core.config import Settings, get_settings
from filingdelta.ingestion.fact_extractors import get_filing_fact_extractor
from filingdelta.ingestion.pipeline import FilingIngestionPipeline, IngestionResult
from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.filing import FilingSource
from filingdelta.storage.paths import ensure_data_dirs


class SingleFilingRunArtifacts(BaseModel):
    parsed_path: Path
    facts_path: Path


class SingleFilingRunResult(BaseModel):
    ingestion: IngestionResult
    facts: HeadlineMetricFacts
    artifacts: SingleFilingRunArtifacts


class SingleFilingProcessor:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pipeline = FilingIngestionPipeline(settings=self._settings)
        self._fact_extractor = get_filing_fact_extractor(settings=self._settings)

    def run(self, source: FilingSource) -> SingleFilingRunResult:
        paths = ensure_data_dirs()
        ingestion = self._pipeline.run(source)
        facts = self._fact_extractor.extract(source)

        document_id = ingestion.parsed_filing.document.document_id
        parsed_path = paths["parsed"] / f"{document_id}.parsed.json"
        facts_path = paths["outputs"] / f"{document_id}.headline_metrics.json"

        _write_model_json(parsed_path, ingestion.parsed_filing)
        _write_model_json(facts_path, facts)

        return SingleFilingRunResult(
            ingestion=ingestion,
            facts=facts,
            artifacts=SingleFilingRunArtifacts(
                parsed_path=parsed_path,
                facts_path=facts_path,
            ),
        )


def _write_model_json(path: Path, model: BaseModel) -> None:
    path.write_text(
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
