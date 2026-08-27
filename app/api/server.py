from app.api.app import (
    create_app,
)
from app.api.control_plane import (
    NexusControlPlane,
)

from app.core.runtime import (
    build_nexus_engine,
)

from app.jobs.manager import (
    JobManager,
)
from app.jobs.queue import (
    PriorityJobQueue,
)
from app.jobs.worker import (
    WorkflowWorker,
)


def build_job_manager(
    engine,
) -> JobManager:
    """
    Build the production asynchronous
    job subsystem for NEXUS.

    The JobManager shares the same
    GovernanceService instance exposed
    by the production NexusEngine.
    """

    queue = PriorityJobQueue()

    worker = WorkflowWorker(
        engine=engine
    )

    return JobManager(
        queue=queue,
        worker=worker,
        governance_service=(
            getattr(
                engine,
                "governance_service",
                None,
            )
        ),
    )


def build_production_app():
    """
    Build the complete production
    NEXUS FastAPI application.

    Includes:
    - NexusEngine
    - governance
    - async job queue
    - workflow worker
    - job manager
    - control plane
    - FastAPI routes
    """

    engine = build_nexus_engine()

    job_manager = (
        build_job_manager(
            engine
        )
    )

    control_plane = NexusControlPlane(
        engine,
        job_manager=job_manager,
    )

    return create_app(
        control_plane
    )


app = build_production_app()
