from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Optional

from app.observability.models import (
    TraceEvent,
    TraceEventType,
    TraceStatus,
    TraceSummary,
    WorkflowTrace,
)


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


class TraceCollector:
    """
    Collect structured observability events
    for one NEXUS workflow execution.
    """

    def __init__(
        self,
        run_id: str,
    ):
        if not run_id:
            raise ValueError(
                "run_id cannot be empty."
            )

        self.trace = WorkflowTrace(
            run_id=run_id
        )

        self._workflow_started_at: Optional[
            float
        ] = None

        self._task_started_at: dict[
            str,
            float,
        ] = {}

        self._repair_started_at: Optional[
            float
        ] = None

        self._replan_started_at: Optional[
            float
        ] = None

    @property
    def run_id(
        self,
    ) -> str:
        return self.trace.run_id

    def _record(
        self,
        event_type: TraceEventType,
        status: TraceStatus,
        *,
        task_id: Optional[str] = None,
        agent_role: Optional[str] = None,
        artifact_id: Optional[str] = None,
        duration_ms: Optional[
            float
        ] = None,
        message: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            event_type=event_type,
            status=status,
            task_id=task_id,
            agent_role=agent_role,
            artifact_id=artifact_id,
            duration_ms=duration_ms,
            message=message,
            metadata=metadata or {},
        )

        self.trace.add_event(
            event
        )

        return event

    def workflow_started(
        self,
        *,
        task_count: int = 0,
    ) -> TraceEvent:
        if task_count < 0:
            raise ValueError(
                "task_count cannot be negative."
            )

        self._workflow_started_at = (
            perf_counter()
        )

        self.trace.started_at = (
            utc_now()
        )

        self.trace.status = (
            TraceStatus.STARTED
        )

        self.trace.task_count = (
            task_count
        )

        return self._record(
            TraceEventType.WORKFLOW_STARTED,
            TraceStatus.STARTED,
            metadata={
                "task_count": task_count,
            },
        )

    def workflow_completed(
        self,
    ) -> TraceEvent:
        duration_ms = None

        if (
            self._workflow_started_at
            is not None
        ):
            duration_ms = (
                perf_counter()
                - self._workflow_started_at
            ) * 1000

        self.trace.completed_at = (
            utc_now()
        )

        self.trace.status = (
            TraceStatus.COMPLETED
        )

        self.trace.total_duration_ms = (
            duration_ms
        )

        return self._record(
            TraceEventType.WORKFLOW_COMPLETED,
            TraceStatus.COMPLETED,
            duration_ms=duration_ms,
        )

    def workflow_failed(
        self,
        message: str,
    ) -> TraceEvent:
        duration_ms = None

        if (
            self._workflow_started_at
            is not None
        ):
            duration_ms = (
                perf_counter()
                - self._workflow_started_at
            ) * 1000

        self.trace.completed_at = (
            utc_now()
        )

        self.trace.status = (
            TraceStatus.FAILED
        )

        self.trace.total_duration_ms = (
            duration_ms
        )

        return self._record(
            TraceEventType.WORKFLOW_FAILED,
            TraceStatus.FAILED,
            duration_ms=duration_ms,
            message=message,
        )

    def task_started(
        self,
        *,
        task_id: str,
        agent_role: str,
    ) -> TraceEvent:
        self._task_started_at[
            task_id
        ] = perf_counter()

        return self._record(
            TraceEventType.TASK_STARTED,
            TraceStatus.STARTED,
            task_id=task_id,
            agent_role=agent_role,
        )

    def task_completed(
        self,
        *,
        task_id: str,
        agent_role: str,
        artifact_id: Optional[
            str
        ] = None,
    ) -> TraceEvent:
        started_at = (
            self._task_started_at.pop(
                task_id,
                None,
            )
        )

        duration_ms = None

        if started_at is not None:
            duration_ms = (
                perf_counter()
                - started_at
            ) * 1000

        self.trace.completed_task_count += 1

        return self._record(
            TraceEventType.TASK_COMPLETED,
            TraceStatus.COMPLETED,
            task_id=task_id,
            agent_role=agent_role,
            artifact_id=artifact_id,
            duration_ms=duration_ms,
        )

    def task_failed(
        self,
        *,
        task_id: str,
        agent_role: str,
        message: str,
    ) -> TraceEvent:
        started_at = (
            self._task_started_at.pop(
                task_id,
                None,
            )
        )

        duration_ms = None

        if started_at is not None:
            duration_ms = (
                perf_counter()
                - started_at
            ) * 1000

        self.trace.failed_task_count += 1

        return self._record(
            TraceEventType.TASK_FAILED,
            TraceStatus.FAILED,
            task_id=task_id,
            agent_role=agent_role,
            duration_ms=duration_ms,
            message=message,
        )

    def artifact_created(
        self,
        *,
        artifact_id: str,
        agent_role: str,
        artifact_type: Optional[
            str
        ] = None,
    ) -> TraceEvent:
        self.trace.artifact_count += 1

        metadata = {}

        if artifact_type is not None:
            metadata[
                "artifact_type"
            ] = artifact_type

        return self._record(
            TraceEventType.ARTIFACT_CREATED,
            TraceStatus.INFO,
            artifact_id=artifact_id,
            agent_role=agent_role,
            metadata=metadata,
        )

    def repair_started(
        self,
    ) -> TraceEvent:
        self._repair_started_at = (
            perf_counter()
        )

        return self._record(
            TraceEventType.REPAIR_STARTED,
            TraceStatus.STARTED,
        )

    def repair_completed(
        self,
        *,
        passed: bool,
        attempts: int,
    ) -> TraceEvent:
        duration_ms = None

        if (
            self._repair_started_at
            is not None
        ):
            duration_ms = (
                perf_counter()
                - self._repair_started_at
            ) * 1000

        self._repair_started_at = None

        self.trace.repair_count += 1

        return self._record(
            TraceEventType.REPAIR_COMPLETED,
            (
                TraceStatus.COMPLETED
                if passed
                else TraceStatus.FAILED
            ),
            duration_ms=duration_ms,
            metadata={
                "passed": passed,
                "attempts": attempts,
            },
        )

    def repair_failed(
        self,
        message: str,
    ) -> TraceEvent:
        duration_ms = None

        if (
            self._repair_started_at
            is not None
        ):
            duration_ms = (
                perf_counter()
                - self._repair_started_at
            ) * 1000

        self._repair_started_at = None

        self.trace.repair_count += 1

        return self._record(
            TraceEventType.REPAIR_FAILED,
            TraceStatus.FAILED,
            duration_ms=duration_ms,
            message=message,
        )

    def replan_started(
        self,
    ) -> TraceEvent:
        self._replan_started_at = (
            perf_counter()
        )

        return self._record(
            TraceEventType.REPLAN_STARTED,
            TraceStatus.STARTED,
        )

    def replan_completed(
        self,
        *,
        action: str,
    ) -> TraceEvent:
        duration_ms = None

        if (
            self._replan_started_at
            is not None
        ):
            duration_ms = (
                perf_counter()
                - self._replan_started_at
            ) * 1000

        self._replan_started_at = None

        self.trace.replan_count += 1

        return self._record(
            TraceEventType.REPLAN_COMPLETED,
            TraceStatus.COMPLETED,
            duration_ms=duration_ms,
            metadata={
                "action": action,
            },
        )

    def evaluation_completed(
        self,
        *,
        overall_score: float,
        regression_detected: bool = False,
    ) -> TraceEvent:
        return self._record(
            TraceEventType.EVALUATION_COMPLETED,
            TraceStatus.COMPLETED,
            metadata={
                "overall_score":
                    overall_score,
                "regression_detected":
                    regression_detected,
            },
        )

    def summary(
        self,
    ) -> TraceSummary:
        agents_used = sorted(
            {
                event.agent_role
                for event in self.trace.events
                if event.agent_role
            }
        )

        return TraceSummary(
            run_id=self.run_id,
            status=self.trace.status,
            total_events=len(
                self.trace.events
            ),
            total_duration_ms=(
                self.trace.total_duration_ms
            ),
            task_count=(
                self.trace.task_count
            ),
            completed_task_count=(
                self.trace
                .completed_task_count
            ),
            failed_task_count=(
                self.trace
                .failed_task_count
            ),
            repair_count=(
                self.trace.repair_count
            ),
            replan_count=(
                self.trace.replan_count
            ),
            artifact_count=(
                self.trace.artifact_count
            ),
            agents_used=agents_used,
        )
