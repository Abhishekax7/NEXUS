from typing import Optional

from app.core.state import (
    NexusState,
)

from app.events.bus import (
    EventBus,
)
from app.events.models import (
    EventSeverity,
    EventType,
    NexusEvent,
)

from app.governance.models import (
    ResourceUsage,
)

from app.governance.service import (
    GovernanceService,
)

from app.jobs.models import (
    JobExecutionResult,
    JobNotFoundError,
    JobPriority,
    JobSnapshot,
    JobStateError,
    JobStatus,
    WorkflowJob,
)

from app.jobs.queue import (
    PriorityJobQueue,
)

from app.jobs.worker import (
    WorkflowWorker,
)


class JobManager:
    """
    Coordinates NEXUS asynchronous
    workflow jobs.

    Responsibilities:

    - submit jobs
    - track workflow states
    - execute queued work
    - inspect job status
    - cancel queued jobs
    - retry failed jobs
    - enforce workflow governance
    - publish lifecycle telemetry
    """

    def __init__(
        self,
        queue: PriorityJobQueue,
        worker: WorkflowWorker,
        governance_service: Optional[
            GovernanceService
        ] = None,
        event_bus: Optional[
            EventBus
        ] = None,
    ):
        self.queue = queue
        self.worker = worker

        self.governance_service = (
            governance_service
        )

        self.event_bus = (
            event_bus
        )

        self._jobs: dict[
            str,
            WorkflowJob,
        ] = {}

        self._states: dict[
            str,
            NexusState,
        ] = {}

        self._results: dict[
            str,
            JobExecutionResult,
        ] = {}

    def _publish(
        self,
        *,
        event_type: EventType,
        message: str,
        run_id: Optional[
            str
        ] = None,
        job_id: Optional[
            str
        ] = None,
        severity: EventSeverity = (
            EventSeverity.INFO
        ),
        payload: Optional[
            dict
        ] = None,
    ) -> None:
        """
        Publish lifecycle telemetry when
        an EventBus is configured.

        Event streaming remains optional
        so existing JobManager consumers
        remain backwards compatible.
        """

        if self.event_bus is None:
            return

        self.event_bus.publish(
            NexusEvent(
                type=event_type,
                severity=severity,
                run_id=run_id,
                job_id=job_id,
                source="job-manager",
                message=message,
                payload=(
                    payload
                    or {}
                ),
            )
        )

    def submit(
        self,
        state: NexusState,
        *,
        priority: JobPriority = (
            JobPriority.NORMAL
        ),
        max_attempts: int = 1,
        metadata: Optional[
            dict
        ] = None,
    ) -> WorkflowJob:
        job = WorkflowJob(
            run_id=state.run_id,
            priority=priority,
            max_attempts=(
                max_attempts
            ),
            metadata=(
                metadata
                or {}
            ),
        )

        self._jobs[
            job.id
        ] = job

        self._states[
            job.id
        ] = state

        self.queue.enqueue(
            job
        )

        self._publish(
            event_type=(
                EventType.JOB_QUEUED
            ),
            run_id=state.run_id,
            job_id=job.id,
            message="Job queued.",
            payload={
                "priority":
                    job.priority.value,
                "max_attempts":
                    job.max_attempts,
            },
        )

        return job

    def get_job(
        self,
        job_id: str,
    ) -> WorkflowJob:
        job = self._jobs.get(
            job_id
        )

        if job is None:
            raise JobNotFoundError(
                "Job not found: "
                f"{job_id}"
            )

        return job

    def get_state(
        self,
        job_id: str,
    ) -> NexusState:
        state = self._states.get(
            job_id
        )

        if state is None:
            raise JobNotFoundError(
                "Workflow state not found "
                f"for job: {job_id}"
            )

        return state

    def get_result(
        self,
        job_id: str,
    ) -> Optional[
        JobExecutionResult
    ]:
        self.get_job(
            job_id
        )

        return self._results.get(
            job_id
        )

    def snapshot(
        self,
        job_id: str,
    ) -> JobSnapshot:
        job = self.get_job(
            job_id
        )

        terminal = (
            job.status
            in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }
        )

        return JobSnapshot(
            job_id=job.id,
            run_id=job.run_id,
            status=job.status,
            priority=job.priority,
            attempt=job.attempt,
            max_attempts=(
                job.max_attempts
            ),
            queued=(
                job.status
                == JobStatus.QUEUED
            ),
            running=(
                job.status
                == JobStatus.RUNNING
            ),
            terminal=terminal,
        )

    def _usage_for(
        self,
        state: NexusState,
    ) -> ResourceUsage:
        return ResourceUsage(
            tasks=len(
                state.tasks
            ),
            replans=int(
                state.metadata.get(
                    "replan_count",
                    0,
                )
            ),
            repairs=int(
                state.metadata.get(
                    "repair_count",
                    0,
                )
            ),
        )

    def _acquire_governance(
        self,
        state: NexusState,
    ) -> None:
        if (
            self.governance_service
            is None
        ):
            return

        self.governance_service.acquire(
            action="workflow.run",
            subject=state.run_id,
            usage=(
                self._usage_for(
                    state
                )
            ),
            context={
                "run_id":
                    state.run_id,
                "task_count":
                    len(
                        state.tasks
                    ),
            },
        )

    def _release_governance(
        self,
        state: NexusState,
    ) -> None:
        if (
            self.governance_service
            is None
        ):
            return

        self.governance_service.release(
            state.run_id
        )

    def _execute(
        self,
        job: WorkflowJob,
        state: NexusState,
    ) -> JobExecutionResult:
        """
        Shared governed execution path
        for execute_next() and
        execute_job().

        Publishes lifecycle events without
        changing the worker's execution
        semantics.
        """

        self._acquire_governance(
            state
        )

        self._publish(
            event_type=(
                EventType.JOB_STARTED
            ),
            run_id=state.run_id,
            job_id=job.id,
            message="Job execution started.",
            payload={
                "attempt":
                    job.attempt,
                "max_attempts":
                    job.max_attempts,
            },
        )

        try:
            result = (
                self.worker.execute(
                    job,
                    state,
                )
            )

        except Exception as exc:
            self._publish(
                event_type=(
                    EventType.JOB_FAILED
                ),
                severity=(
                    EventSeverity.ERROR
                ),
                run_id=state.run_id,
                job_id=job.id,
                message=(
                    "Job execution failed."
                ),
                payload={
                    "error":
                        str(exc),
                    "error_type":
                        type(exc).__name__,
                },
            )

            raise

        finally:
            self._release_governance(
                state
            )

        self._results[
            job.id
        ] = result

        if result.success:
            self._publish(
                event_type=(
                    EventType.JOB_COMPLETED
                ),
                run_id=state.run_id,
                job_id=job.id,
                message=(
                    "Job execution completed."
                ),
                payload={
                    "status":
                        job.status.value,
                    "attempt":
                        job.attempt,
                },
            )

        else:
            self._publish(
                event_type=(
                    EventType.JOB_FAILED
                ),
                severity=(
                    EventSeverity.ERROR
                ),
                run_id=state.run_id,
                job_id=job.id,
                message=(
                    "Job execution failed."
                ),
                payload={
                    "status":
                        job.status.value,
                    "attempt":
                        job.attempt,
                },
            )

        return result

    def execute_next(
        self,
    ) -> Optional[
        JobExecutionResult
    ]:
        job = self.queue.dequeue()

        if job is None:
            return None

        state = self.get_state(
            job.id
        )

        return self._execute(
            job,
            state,
        )

    def execute_job(
        self,
        job_id: str,
    ) -> JobExecutionResult:
        job = self.get_job(
            job_id
        )

        if (
            job.status
            != JobStatus.QUEUED
        ):
            raise JobStateError(
                "Only queued jobs can "
                "be executed."
            )

        if self.queue.contains(
            job_id
        ):
            self.queue.remove(
                job_id
            )

        state = self.get_state(
            job_id
        )

        return self._execute(
            job,
            state,
        )

    def cancel(
        self,
        job_id: str,
    ) -> WorkflowJob:
        job = self.get_job(
            job_id
        )

        if self.queue.contains(
            job_id
        ):
            self.queue.remove(
                job_id
            )

        cancelled = (
            self.worker.cancel(
                job
            )
        )

        self._publish(
            event_type=(
                EventType.JOB_CANCELLED
            ),
            severity=(
                EventSeverity.WARNING
            ),
            run_id=job.run_id,
            job_id=job.id,
            message="Job cancelled.",
            payload={
                "status":
                    cancelled.status.value,
            },
        )

        return cancelled

    def retry(
        self,
        job_id: str,
    ) -> WorkflowJob:
        job = self.get_job(
            job_id
        )

        prepared = (
            self.worker
            .prepare_retry(
                job
            )
        )

        self.queue.enqueue(
            prepared
        )

        self._results.pop(
            job_id,
            None,
        )

        self._publish(
            event_type=(
                EventType.JOB_RETRIED
            ),
            run_id=job.run_id,
            job_id=job.id,
            message=(
                "Job prepared for retry."
            ),
            payload={
                "attempt":
                    prepared.attempt,
                "max_attempts":
                    prepared.max_attempts,
            },
        )

        return prepared

    def pending_jobs(
        self,
    ) -> list[
        WorkflowJob
    ]:
        return (
            self.queue
            .queued_jobs()
        )

    def job_count(
        self,
    ) -> int:
        return len(
            self._jobs
        )

    def result_count(
        self,
    ) -> int:
        return len(
            self._results
        )
