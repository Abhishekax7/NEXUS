from app.api.app import (
    create_app,
)
from app.api.control_plane import (
    NexusControlPlane,
)

from app.core.runtime import (
    build_nexus_engine,
)

from app.events.bus import (
    EventBus,
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


def build_event_bus() -> EventBus:
    """
    Build the production real-time
    event bus used by NEXUS.
    """

    return EventBus(
        max_history=1000,
        subscriber_queue_size=100,
    )


def build_job_manager(
    engine,
    event_bus: EventBus | None = None,
) -> JobManager:
    """
    Build the production asynchronous
    job subsystem.

    The JobManager shares:
    - the production NexusEngine
    - the production GovernanceService
    - the production EventBus

    If no EventBus is supplied, one is
    created for backward compatibility.
    """

    if event_bus is None:
        event_bus = build_event_bus()

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
        event_bus=event_bus,
    )


def build_production_app():
    """
    Build the complete production
    NEXUS FastAPI application.

    Includes:
    - NexusEngine
    - governance
    - real-time EventBus
    - async job queue
    - workflow worker
    - JobManager
    - control plane
    - FastAPI routes
    """

    engine = build_nexus_engine()

    event_bus = build_event_bus()

    engine.event_bus = (
        event_bus
    )

    job_manager = (
        build_job_manager(
            engine,
            event_bus,
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
