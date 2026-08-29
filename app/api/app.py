import json

from typing import (
    Optional,
)

from fastapi import (
    FastAPI,
    HTTPException,
)

from fastapi.responses import (
    StreamingResponse,
)

from app.api.control_plane import (
    ControlPlaneError,
    NexusControlPlane,
    RunNotFoundError,
)

from app.api.schemas import (
    ApprovalDecisionRequest,
    ApprovalResponse,
    CreateRunRequest,
    EvaluationResponse,
    HealthResponse,
    JobExecutionResponse,
    JobListResponse,
    JobResponse,
    RecoveryResponse,
    ResumeRunRequest,
    ResumeRunResponse,
    RunResponse,
    RunSummaryResponse,
    SubmitJobRequest,
    TraceResponse,
)

from app.events.models import (
    EventFilter,
)

from app.jobs.models import (
    JobNotFoundError,
)


def create_app(
    control_plane: NexusControlPlane,
) -> FastAPI:
    app = FastAPI(
        title="NEXUS Control Plane",
        version="1.0.0",
        description=(
            "Production API for NEXUS "
            "agentic workflow control."
        ),
    )

    @app.get("/")
    def root():
        return {
            "name": "NEXUS",
            "description": "Autonomous AI Engineering System",
            "status": "online",
            "docs": "/docs",
            "health": "/healthz",
        }

    # ---------------------------------
    # Health
    # ---------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
    )
    def health():
        return HealthResponse()

    # ---------------------------------
    # Workflow runs
    # ---------------------------------

    @app.post(
        "/runs",
        response_model=RunResponse,
        status_code=201,
    )
    def create_run(
        request: CreateRunRequest,
    ):
        try:
            return control_plane.create_run(
                user_request=(
                    request.user_request
                ),
                metadata=request.metadata,
            )

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.get(
        "/runs/{run_id}",
        response_model=RunResponse,
    )
    def get_run(
        run_id: str,
    ):
        try:
            state = (
                control_plane.get_state(
                    run_id
                )
            )

            return (
                control_plane.run_response(
                    state
                )
            )

        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.get(
        "/runs/{run_id}/summary",
        response_model=RunSummaryResponse,
    )
    def get_run_summary(
        run_id: str,
    ):
        try:
            return (
                control_plane.run_summary(
                    run_id
                )
            )

        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.get(
        "/runs/{run_id}/recovery",
        response_model=RecoveryResponse,
    )
    def get_recovery(
        run_id: str,
    ):
        try:
            return (
                control_plane.recovery(
                    run_id
                )
            )

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/runs/{run_id}/resume",
        response_model=ResumeRunResponse,
    )
    def resume_run(
        run_id: str,
        request: ResumeRunRequest,
    ):
        try:
            return (
                control_plane.resume_run(
                    run_id,
                    allow_failed=(
                        request.allow_failed
                    ),
                )
            )

        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    # ---------------------------------
    # Observability
    # ---------------------------------

    @app.get(
        "/runs/{run_id}/trace",
        response_model=TraceResponse,
    )
    def get_trace(
        run_id: str,
    ):
        try:
            return (
                control_plane.trace(
                    run_id
                )
            )

        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    # ---------------------------------
    # Evaluation
    # ---------------------------------

    @app.get(
        "/runs/{run_id}/evaluation",
        response_model=EvaluationResponse,
    )
    def get_evaluation(
        run_id: str,
    ):
        try:
            return (
                control_plane.evaluation(
                    run_id
                )
            )

        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    # ---------------------------------
    # Human approvals
    # ---------------------------------

    @app.get(
        "/approvals",
        response_model=list[
            ApprovalResponse
        ],
    )
    def get_pending_approvals():
        try:
            return (
                control_plane
                .pending_approvals()
            )

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/approvals/{request_id}/approve",
        response_model=ApprovalResponse,
    )
    def approve(
        request_id: str,
        request: ApprovalDecisionRequest,
    ):
        try:
            return (
                control_plane.approve(
                    request_id,
                    reason=request.reason,
                    decided_by=(
                        request.decided_by
                    ),
                    metadata=(
                        request.metadata
                    ),
                )
            )

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/approvals/{request_id}/reject",
        response_model=ApprovalResponse,
    )
    def reject(
        request_id: str,
        request: ApprovalDecisionRequest,
    ):
        try:
            return (
                control_plane.reject(
                    request_id,
                    reason=request.reason,
                    decided_by=(
                        request.decided_by
                    ),
                    metadata=(
                        request.metadata
                    ),
                )
            )

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    # ---------------------------------
    # Asynchronous workflow jobs
    # ---------------------------------

    @app.post(
        "/jobs",
        response_model=JobResponse,
        status_code=201,
    )
    def submit_job(
        request: SubmitJobRequest,
    ):
        try:
            return (
                control_plane.submit_job(
                    request.run_id,
                    priority=(
                        request.priority
                    ),
                    max_attempts=(
                        request.max_attempts
                    ),
                    metadata=(
                        request.metadata
                    ),
                )
            )

        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.get(
        "/jobs",
        response_model=JobListResponse,
    )
    def list_jobs():
        try:
            snapshots = (
                control_plane
                .pending_jobs()
            )

            jobs = [
                JobResponse(
                    job_id=(
                        snapshot.job_id
                    ),
                    run_id=(
                        snapshot.run_id
                    ),
                    status=(
                        snapshot.status
                    ),
                    priority=(
                        snapshot.priority
                    ),
                    attempt=(
                        snapshot.attempt
                    ),
                    max_attempts=(
                        snapshot.max_attempts
                    ),
                    queued=(
                        snapshot.queued
                    ),
                    running=(
                        snapshot.running
                    ),
                    terminal=(
                        snapshot.terminal
                    ),
                )
                for snapshot
                in snapshots
            ]

            return JobListResponse(
                count=len(
                    jobs
                ),
                jobs=jobs,
            )

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.get(
        "/jobs/{job_id}",
        response_model=JobResponse,
    )
    def get_job(
        job_id: str,
    ):
        try:
            return (
                control_plane.get_job(
                    job_id
                )
            )

        except JobNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/jobs/execute-next",
        response_model=(
            JobExecutionResponse
        ),
    )
    def execute_next_job():
        try:
            result = (
                control_plane
                .execute_next_job()
            )

            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "No queued jobs "
                        "are available."
                    ),
                )

            return result

        except HTTPException:
            raise

        except ControlPlaneError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/jobs/{job_id}/execute",
        response_model=(
            JobExecutionResponse
        ),
    )
    def execute_job(
        job_id: str,
    ):
        try:
            return (
                control_plane.execute_job(
                    job_id
                )
            )

        except JobNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/jobs/{job_id}/cancel",
        response_model=JobResponse,
    )
    def cancel_job(
        job_id: str,
    ):
        try:
            return (
                control_plane.cancel_job(
                    job_id
                )
            )

        except JobNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @app.post(
        "/jobs/{job_id}/retry",
        response_model=JobResponse,
    )
    def retry_job(
        job_id: str,
    ):
        try:
            return (
                control_plane.retry_job(
                    job_id
                )
            )

        except JobNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    # ---------------------------------
    # Event telemetry
    # ---------------------------------

    @app.get(
        "/events",
    )
    def get_events(
        run_id: Optional[
            str
        ] = None,
        job_id: Optional[
            str
        ] = None,
        limit: int = 100,
    ):
        event_bus = getattr(
            control_plane.engine,
            "event_bus",
            None,
        )

        if event_bus is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Event streaming is not "
                    "configured."
                ),
            )

        if limit < 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "limit must be >= 1."
                ),
            )

        event_filter = EventFilter(
            run_id=run_id,
            job_id=job_id,
        )

        page = event_bus.history(
            event_filter,
            limit=limit,
        )

        return page.model_dump(
            mode="json"
        )

    @app.get(
        "/events/stream",
    )
    def stream_events(
        run_id: Optional[
            str
        ] = None,
        job_id: Optional[
            str
        ] = None,
    ):
        event_bus = getattr(
            control_plane.engine,
            "event_bus",
            None,
        )

        if event_bus is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Event streaming is not "
                    "configured."
                ),
            )

        subscription = (
            event_bus.subscribe(
                run_id=run_id,
                job_id=job_id,
            )
        )

        def event_generator():
            try:
                while True:
                    event = (
                        event_bus.get_event(
                            subscription.id,
                            timeout=15.0,
                        )
                    )

                    if event is None:
                        yield (
                            ": keep-alive\n\n"
                        )
                        continue

                    payload = json.dumps(
                        event.model_dump(
                            mode="json"
                        )
                    )

                    yield (
                        f"event: "
                        f"{event.type.value}\n"
                        f"data: {payload}\n\n"
                    )

            finally:
                try:
                    event_bus.unsubscribe(
                        subscription.id
                    )
                except Exception:
                    pass

        return StreamingResponse(
            event_generator(),
            media_type=(
                "text/event-stream"
            ),
            headers={
                "Cache-Control":
                    "no-cache",
                "Connection":
                    "keep-alive",
            },
        )

    return app
