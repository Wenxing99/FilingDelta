from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from llama_index.core.workflow import StopEvent

from filingdelta.core.config import Settings, get_settings
from filingdelta.schemas.demo import DemoRun
from filingdelta.schemas.filing import FilingSource
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


class DemoRunManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._runs: dict[str, DemoRun] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

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

        task = asyncio.create_task(self._execute_run(run_id=run_id, source=source))
        self._tasks[run_id] = task
        return run

    async def get_run(self, run_id: str) -> DemoRun:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Unknown run: {run_id}")
            return run.model_copy(deep=True)

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
            self._runs[run_id] = current.model_copy(
                update={
                    "status": status,
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "stage_index": STAGE_INDEX[stage],
                    "progress_message": progress_message,
                    "updated_at": datetime.now(UTC),
                    "result": result if result is not None else current.result,
                    "error_message": error_message,
                }
            )

    async def _execute_run(self, *, run_id: str, source: FilingSource) -> None:
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


_demo_run_manager: DemoRunManager | None = None


def get_demo_run_manager() -> DemoRunManager:
    global _demo_run_manager
    if _demo_run_manager is None:
        _demo_run_manager = DemoRunManager()
    return _demo_run_manager
