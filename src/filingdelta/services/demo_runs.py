from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from llama_index.core.workflow import StopEvent

from filingdelta.core.config import Settings, get_settings
from filingdelta.schemas.demo import DemoRun, DemoRunArtifactsTelemetry, DemoRunStageTelemetry, DemoRunTelemetry
from filingdelta.schemas.filing import FilingSource
from filingdelta.schemas.workflow import SingleFilingWorkflowResult
from filingdelta.services.chat_qa import get_chat_qa_service
from filingdelta.services.review_feedback import ReviewFeedbackService
from filingdelta.workflows.events import WorkflowProgressEvent
from filingdelta.workflows.single_filing import SingleFilingWorkflow


STAGE_LABELS: dict[str, str] = {
    "queued": "等待开始",
    "orchestrate": "解析文档",
    "reader": "提取重点",
    "fact_extractor": "抽取关键数据",
    "verifier": "核验引用",
    "done": "分析完成",
    "failed": "分析失败",
}

STAGE_INDEX: dict[str, int] = {
    "queued": 0,
    "orchestrate": 1,
    "reader": 2,
    "fact_extractor": 3,
    "verifier": 4,
    "done": 4,
    "failed": 0,
}

TIMED_STAGES = ("orchestrate", "reader", "fact_extractor", "verifier")


@dataclass
class _RunTelemetryTracker:
    started_at: float = field(default_factory=perf_counter)
    current_stage: str | None = None
    current_stage_started_at: float | None = None
    completed_total_ms: float | None = None
    stage_timings: dict[str, float | None] = field(
        default_factory=lambda: {stage: None for stage in TIMED_STAGES}
    )

    def transition(self, next_stage: str) -> None:
        if next_stage == self.current_stage:
            return

        now = perf_counter()
        self._finalize_current_stage(now)

        if next_stage in self.stage_timings:
            self.current_stage = next_stage
            self.current_stage_started_at = now
        else:
            if next_stage in {"done", "failed"} and self.completed_total_ms is None:
                self.completed_total_ms = round((now - self.started_at) * 1000, 2)
            self.current_stage = next_stage
            self.current_stage_started_at = None

    def snapshot(self) -> DemoRunStageTelemetry:
        now = perf_counter()
        stage_timings = dict(self.stage_timings)

        if self.current_stage in stage_timings and self.current_stage_started_at is not None:
            current_elapsed = round((now - self.current_stage_started_at) * 1000, 2)
            completed = stage_timings[self.current_stage] or 0.0
            stage_timings[self.current_stage] = round(completed + current_elapsed, 2)

        return DemoRunStageTelemetry(
            orchestrate_ms=stage_timings["orchestrate"],
            reader_ms=stage_timings["reader"],
            fact_extractor_ms=stage_timings["fact_extractor"],
            verifier_ms=stage_timings["verifier"],
            total_ms=self.completed_total_ms or round((now - self.started_at) * 1000, 2),
        )

    def _finalize_current_stage(self, now: float) -> None:
        if self.current_stage not in self.stage_timings or self.current_stage_started_at is None:
            return

        elapsed = round((now - self.current_stage_started_at) * 1000, 2)
        previous = self.stage_timings[self.current_stage] or 0.0
        self.stage_timings[self.current_stage] = round(previous + elapsed, 2)
        self.current_stage_started_at = None


class DemoRunManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._runs: dict[str, DemoRun] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._prewarm_tasks: set[asyncio.Task[None]] = set()
        self._sources: dict[str, FilingSource] = {}
        self._telemetry_trackers: dict[str, _RunTelemetryTracker] = {}
        self._lock = asyncio.Lock()
        self._review_feedback = ReviewFeedbackService(settings=self._settings)

    async def create_run(self, document_id: str, source: FilingSource) -> DemoRun:
        now = datetime.now(UTC)
        run_id = uuid4().hex
        run = DemoRun(
            run_id=run_id,
            stage="queued",
            stage_label=STAGE_LABELS["queued"],
            stage_index=STAGE_INDEX["queued"],
            progress_message="等待开始分析。",
            document_id=document_id,
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            self._runs[run_id] = run
            self._sources[run_id] = source
            self._telemetry_trackers[run_id] = _RunTelemetryTracker()

        task = asyncio.create_task(
            self._execute_run(run_id=run_id, document_id=document_id, source=source)
        )
        self._tasks[run_id] = task
        return run

    async def get_run(self, run_id: str) -> DemoRun:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Unknown run: {run_id}")
            return run.model_copy(deep=True)

    async def approve_issue(self, run_id: str, item_key: str) -> DemoRun:
        async with self._lock:
            current = self._runs.get(run_id)
            if current is None:
                raise KeyError(f"Unknown run: {run_id}")
            if current.result is None:
                raise ValueError("Run result is not ready yet.")
            if current.status != "succeeded":
                raise ValueError("Only succeeded runs can accept issue actions.")

            updated_result = await self._review_feedback.approve_issue(
                result=current.result,
                item_key=item_key,
            )
            telemetry = self._build_run_telemetry(
                run_id=run_id,
                result=updated_result,
                succeeded=current.status == "succeeded",
            )
            self._runs[run_id] = current.model_copy(
                update={
                    "updated_at": datetime.now(UTC),
                    "progress_message": "已手动确认一条待确认项。",
                    "result": updated_result,
                    "telemetry": telemetry,
                }
            )
            return self._runs[run_id].model_copy(deep=True)

    async def rerun_issue(self, run_id: str, item_key: str) -> DemoRun:
        async with self._lock:
            current = self._runs.get(run_id)
            source = self._sources.get(run_id)
            if current is None:
                raise KeyError(f"Unknown run: {run_id}")
            if source is None:
                raise KeyError(f"Missing source for run: {run_id}")
            if current.result is None:
                raise ValueError("Run result is not ready yet.")
            if current.status != "succeeded":
                raise ValueError("Only succeeded runs can accept issue actions.")

            self._runs[run_id] = current.model_copy(
                update={
                    "updated_at": datetime.now(UTC),
                    "progress_message": "正在重新处理待确认项。",
                }
            )

        updated_result = await self._review_feedback.rerun_issue(
            source=source,
            result=current.result,
            item_key=item_key,
        )

        async with self._lock:
            latest = self._runs[run_id]
            telemetry = self._build_run_telemetry(
                run_id=run_id,
                result=updated_result,
                succeeded=latest.status == "succeeded",
            )
            self._runs[run_id] = latest.model_copy(
                update={
                    "updated_at": datetime.now(UTC),
                    "progress_message": "已重新处理一条待确认项。",
                    "result": updated_result,
                    "telemetry": telemetry,
                }
            )
            return self._runs[run_id].model_copy(deep=True)

    async def rerun_feedback(self, run_id: str, feedback_category: str) -> DemoRun:
        async with self._lock:
            current = self._runs.get(run_id)
            source = self._sources.get(run_id)
            if current is None:
                raise KeyError(f"Unknown run: {run_id}")
            if source is None:
                raise KeyError(f"Missing source for run: {run_id}")
            if current.result is None:
                raise ValueError("Run result is not ready yet.")
            if current.status != "succeeded":
                raise ValueError("Only succeeded runs can accept feedback actions.")

            self._runs[run_id] = current.model_copy(
                update={
                    "updated_at": datetime.now(UTC),
                    "progress_message": _feedback_progress_message(feedback_category, "running"),
                }
            )

        updated_result = await self._review_feedback.rerun_feedback_category(
            source=source,
            result=current.result,
            feedback_category=feedback_category,
        )

        async with self._lock:
            latest = self._runs[run_id]
            telemetry = self._build_run_telemetry(
                run_id=run_id,
                result=updated_result,
                succeeded=latest.status == "succeeded",
            )
            self._runs[run_id] = latest.model_copy(
                update={
                    "updated_at": datetime.now(UTC),
                    "progress_message": _feedback_progress_message(feedback_category, "done"),
                    "result": updated_result,
                    "telemetry": telemetry,
                }
            )
            return self._runs[run_id].model_copy(deep=True)

    async def _set_stage(
        self,
        run_id: str,
        *,
        status: str,
        stage: str,
        progress_message: str,
        result=None,
        error_message: str | None = None,
    ) -> None:
        async with self._lock:
            current = self._runs[run_id]
            tracker = self._telemetry_trackers.get(run_id)
            if tracker is not None:
                tracker.transition(stage)
            next_result = result if result is not None else current.result
            telemetry = self._build_run_telemetry(
                run_id=run_id,
                result=next_result,
                succeeded=status == "succeeded",
            )
            self._runs[run_id] = current.model_copy(
                update={
                    "status": status,
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "stage_index": STAGE_INDEX[stage],
                    "progress_message": progress_message,
                    "updated_at": datetime.now(UTC),
                    "result": next_result,
                    "error_message": error_message,
                    "telemetry": telemetry,
                }
            )

    async def _execute_run(self, *, run_id: str, document_id: str, source: FilingSource) -> None:
        workflow = SingleFilingWorkflow(settings=self._settings, verbose=False)
        handler = workflow.run(source=source)
        await self._set_stage(
            run_id,
            status="running",
            stage="orchestrate",
            progress_message="开始解析文档。",
        )

        try:
            async for event in handler.stream_events():
                if isinstance(event, WorkflowProgressEvent):
                    await self._set_stage(
                        run_id,
                        status="running",
                        stage=event.stage,
                        progress_message=event.message,
                    )
                elif isinstance(event, StopEvent):
                    result = event.result
                    await self._set_stage(
                        run_id,
                        status="succeeded",
                        stage="done",
                        progress_message="分析完成。",
                        result=result,
                    )
                    self._schedule_chat_index_prewarm(
                        document_id=document_id,
                        source=source,
                    )
        except Exception as error:
            await self._set_stage(
                run_id,
                status="failed",
                stage="failed",
                progress_message="分析失败。",
                error_message=str(error),
            )
        finally:
            self._tasks.pop(run_id, None)

    def _schedule_chat_index_prewarm(self, *, document_id: str, source: FilingSource) -> None:
        task = asyncio.create_task(
            self._prewarm_chat_index_background(document_id=document_id, source=source)
        )
        self._prewarm_tasks.add(task)
        task.add_done_callback(self._prewarm_tasks.discard)

    async def _prewarm_chat_index_background(self, *, document_id: str, source: FilingSource) -> None:
        try:
            await get_chat_qa_service().prewarm_document(document_id=document_id, source=source)
        except Exception:
            return

    async def _drain_prewarm_tasks(self) -> None:
        tasks = list(self._prewarm_tasks)
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)

    def _build_run_telemetry(
        self,
        *,
        run_id: str,
        result: SingleFilingWorkflowResult | None,
        succeeded: bool,
    ) -> DemoRunTelemetry:
        tracker = self._telemetry_trackers.get(run_id)
        stage_timings = tracker.snapshot() if tracker is not None else DemoRunStageTelemetry()
        return DemoRunTelemetry(
            succeeded=succeeded,
            stage_timings=stage_timings,
            artifacts=_build_artifacts_telemetry(result),
        )


_demo_run_manager: DemoRunManager | None = None


def get_demo_run_manager() -> DemoRunManager:
    global _demo_run_manager
    if _demo_run_manager is None:
        _demo_run_manager = DemoRunManager()
    return _demo_run_manager


def _build_artifacts_telemetry(
    result: SingleFilingWorkflowResult | None,
) -> DemoRunArtifactsTelemetry:
    if result is None:
        return DemoRunArtifactsTelemetry()

    return DemoRunArtifactsTelemetry(
        total_pages=result.total_pages,
        chunk_count=result.chunk_count,
        summary_sections_count=len(result.summary_sections),
        summary_points_count=sum(len(section.points) for section in result.summary_sections),
        verification_issues_count=result.review.pending_confirmation_count,
        needs_human_review=result.needs_human_review,
    )


def _feedback_progress_message(feedback_category: str, phase: str) -> str:
    messages = {
        "citation": {
            "running": "正在重新处理引用回溯反馈。",
            "done": "已根据引用回溯反馈重新处理结果。",
        },
        "numeric": {
            "running": "正在重新处理数据准确度反馈。",
            "done": "已根据数据准确度反馈重新处理结果。",
        },
        "summary": {
            "running": "正在重新处理摘要信息反馈。",
            "done": "已根据摘要信息反馈重新处理结果。",
        },
    }
    category_messages = messages.get(feedback_category)
    if category_messages is None:
        raise ValueError(f"Unsupported feedback category: {feedback_category}")
    return category_messages[phase]
