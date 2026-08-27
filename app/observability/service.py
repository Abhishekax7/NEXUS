from typing import Optional

from app.observability.collector import (
    TraceCollector,
)
from app.observability.models import (
    TraceSummary,
    WorkflowTrace,
)
from app.observability.store import (
    TraceStore,
)


class ObservabilityService:
    """
    Production observability interface
    for NEXUS workflow execution.

    Responsibilities:
    - create trace collectors
    - persist traces
    - retrieve historical traces
    - expose compact summaries
    """

    def __init__(
        self,
        store: TraceStore,
    ):
        self.store = store

    def start_run(
        self,
        run_id: str,
        *,
        task_count: int = 0,
    ) -> TraceCollector:
        collector = TraceCollector(
            run_id=run_id
        )

        collector.workflow_started(
            task_count=task_count
        )

        self.save(
            collector.trace
        )

        return collector

    def save(
        self,
        trace: WorkflowTrace,
    ) -> None:
        self.store.save(
            trace
        )

    def complete_run(
        self,
        collector: TraceCollector,
    ) -> WorkflowTrace:
        collector.workflow_completed()

        self.save(
            collector.trace
        )

        return collector.trace

    def fail_run(
        self,
        collector: TraceCollector,
        message: str,
    ) -> WorkflowTrace:
        collector.workflow_failed(
            message
        )

        self.save(
            collector.trace
        )

        return collector.trace

    def get_trace(
        self,
        run_id: str,
    ) -> Optional[
        WorkflowTrace
    ]:
        return self.store.get(
            run_id
        )

    def get_summary(
        self,
        run_id: str,
    ) -> Optional[
        TraceSummary
    ]:
        return self.store.summary(
            run_id
        )

    def recent_traces(
        self,
        limit: int = 10,
    ) -> list[
        WorkflowTrace
    ]:
        return self.store.list_recent(
            limit=limit
        )

    def delete_trace(
        self,
        run_id: str,
    ) -> bool:
        return self.store.delete(
            run_id
        )

    def clear_traces(
        self,
    ) -> int:
        return self.store.clear()
