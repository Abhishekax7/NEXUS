from fastapi import (
    FastAPI,
    HTTPException,
)

from app.api.control_plane import (
    ControlPlaneError,
    NexusControlPlane,
    RunNotFoundError,
)
from app.api.schemas import (
    ApprovalDecisionRequest,
    ApprovalResponse,
    EvaluationResponse,
    HealthResponse,
    RecoveryResponse,
    ResumeRunRequest,
    ResumeRunResponse,
    RunResponse,
    RunSummaryResponse,
    TraceResponse,
)


def create_app(
    control_plane: NexusControlPlane,
) -> FastAPI:
    """
    Build the NEXUS HTTP control-plane API.
    """

    app = FastAPI(
        title="NEXUS Control Plane",
        version="1.0.0",
        description=(
            "Production API for NEXUS "
            "agentic workflow control."
        ),
    )

    @app.get(
        "/health",
        response_model=HealthResponse,
    )
    def health():
        return HealthResponse()

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

    @app.get(
        "/runs/{run_id}/trace",
        response_model=TraceResponse,
    )
    def get_trace(
        run_id: str,
    ):
        try:
            return control_plane.trace(
                run_id
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
            return control_plane.approve(
                request_id,
                reason=request.reason,
                decided_by=(
                    request.decided_by
                ),
                metadata=request.metadata,
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
            return control_plane.reject(
                request_id,
                reason=request.reason,
                decided_by=(
                    request.decided_by
                ),
                metadata=request.metadata,
            )

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    return app
