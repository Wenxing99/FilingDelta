from __future__ import annotations

import asyncio

from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from filingdelta.agents.reader import ReaderAgent
from filingdelta.agents.verifier import VerifierAgent
from filingdelta.core.config import Settings, get_settings
from filingdelta.ingestion.fact_extractors import get_filing_fact_extractor
from filingdelta.ingestion.pipeline import FilingIngestionPipeline
from filingdelta.schemas.filing import FilingSource
from filingdelta.schemas.workflow import SingleFilingWorkflowResult
from filingdelta.services.fact_citation_enrichment import enrich_headline_metric_citations
from filingdelta.workflows.events import (
    FactExtractionCompletedEvent,
    PreparedFilingEvent,
    ReaderCompletedEvent,
    WorkflowProgressEvent,
)


class SingleFilingWorkflow(Workflow):
    def __init__(
        self,
        settings: Settings | None = None,
        timeout: float | None = 90.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(timeout=timeout, verbose=verbose)
        self._settings = settings or get_settings()
        self._pipeline = FilingIngestionPipeline(settings=self._settings)
        self._fact_extractor = get_filing_fact_extractor(settings=self._settings)
        self._reader = ReaderAgent(settings=self._settings)
        self._verifier = VerifierAgent()

    @step
    async def orchestrate(self, ctx: Context, ev: StartEvent) -> PreparedFilingEvent:
        source = _coerce_source(ev.get("source"))
        ingestion = await asyncio.to_thread(self._pipeline.run, source)

        await ctx.store.set("source", source)
        await ctx.store.set("parsed_filing", ingestion.parsed_filing)
        await ctx.store.set("chunks", ingestion.chunks)

        ctx.write_event_to_stream(
            WorkflowProgressEvent(
                stage="orchestrate",
                message="Prepared filing and stored parsed pages and chunks.",
            )
        )

        return PreparedFilingEvent(
            document_id=ingestion.parsed_filing.document.document_id,
        )

    @step
    async def reader(self, ctx: Context, ev: PreparedFilingEvent) -> ReaderCompletedEvent:
        parsed_filing = await ctx.store.get("parsed_filing")
        chunks = await ctx.store.get("chunks")
        reader_drafts = await self._reader.read(parsed_filing, chunks)

        ctx.write_event_to_stream(
            WorkflowProgressEvent(
                stage="reader",
                message=f"Reader produced {len(reader_drafts.items)} summary items for {ev.document_id}.",
            )
        )

        return ReaderCompletedEvent(reader_drafts=reader_drafts)

    @step
    async def fact_extractor(
        self,
        ctx: Context,
        ev: PreparedFilingEvent,
    ) -> FactExtractionCompletedEvent:
        source = await ctx.store.get("source")
        parsed_filing = await ctx.store.get("parsed_filing")

        facts = await asyncio.to_thread(
            self._fact_extractor.extract,
            source,
            parsed_filing,
        )
        facts = await asyncio.to_thread(
            enrich_headline_metric_citations,
            parsed_filing,
            facts,
        )

        ctx.write_event_to_stream(
            WorkflowProgressEvent(
                stage="fact_extractor",
                message=f"FactExtractor produced headline metrics for {ev.document_id}.",
            )
        )

        return FactExtractionCompletedEvent(facts=facts)

    @step
    async def verifier(
        self,
        ctx: Context,
        ev: ReaderCompletedEvent | FactExtractionCompletedEvent,
    ) -> StopEvent | None:
        events = ctx.collect_events(
            ev,
            [ReaderCompletedEvent, FactExtractionCompletedEvent],
        )
        if events is None:
            return None

        reader_event, fact_event = events
        parsed_filing = await ctx.store.get("parsed_filing")
        chunks = await ctx.store.get("chunks")

        verification = self._verifier.verify(
            parsed_filing=parsed_filing,
            reader_drafts=reader_event.reader_drafts,
            facts=fact_event.facts,
        )

        result = SingleFilingWorkflowResult(
            document_id=parsed_filing.document.document_id,
            source_path=parsed_filing.document.source_path,
            parser_kind=parsed_filing.document.parser_kind,
            total_pages=parsed_filing.document.total_pages,
            chunk_count=len(chunks),
            reader_drafts=reader_event.reader_drafts,
            summary_items=verification.summary_items,
            headline_metrics=fact_event.facts,
            verification_issues=verification.issues,
            needs_human_review=verification.needs_human_review,
        )

        ctx.write_event_to_stream(
            WorkflowProgressEvent(
                stage="verifier",
                message=(
                    f"Verifier finished with {len(verification.issues)} issues; "
                    f"needs_human_review={verification.needs_human_review}."
                ),
            )
        )

        return StopEvent(result=result)


def _coerce_source(raw_source: object) -> FilingSource:
    if isinstance(raw_source, FilingSource):
        return raw_source
    return FilingSource.model_validate(raw_source)
