from app.agents.base import BaseAgent
from app.agents.registry import AgentRegistry

from app.core.engine import NexusEngine
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState

from app.observability.models import (
    TraceEventType,
    TraceStatus,
)
from app.observability.service import (
    ObservabilityService,
)
from app.observability.store import (
    TraceStore,
)


class SuccessfulAgent(BaseAgent):
    role = AgentRole.CODER

    def execute(
        self,
        task,
        state,
    ):
        return Artifact(
            type=ArtifactType.CODE,
            name="generated_code",
            content={
                "files": [
                    {
                        "path": "app.py",
                        "content": "print('hello')",
                    }
                ]
            },
            created_by=self.role,
        )


class FailingAgent(BaseAgent):
    role = AgentRole.CODER

    def execute(
        self,
        task,
        state,
    ):
        raise RuntimeError(
            "intentional agent failure"
        )


def build_service(
    tmp_path,
):
    return ObservabilityService(
        store=TraceStore(
            db_path=str(
                tmp_path
                / "traces.db"
            )
        )
    )


def build_state():
    state = NexusState(
        user_request=(
            "Build a simple application."
        )
    )

    task = AgentTask(
        title="Implement application",
        description=(
            "Generate application code."
        ),
        assigned_agent=(
            AgentRole.CODER
        ),
    )

    state.add_task(
        task
    )

    return state, task


def build_success_registry():
    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        SuccessfulAgent,
    )

    return registry


def build_failure_registry():
    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        FailingAgent,
    )

    return registry


def event_types(
    trace,
):
    return [
        event.event_type
        for event
        in trace.events
    ]


def test_successful_engine_run_creates_trace(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    engine = NexusEngine(
        registry=(
            build_success_registry()
        ),
        observability_service=service,
    )

    state, _ = build_state()

    result = engine.run(
        state
    )

    trace = service.get_trace(
        state.run_id
    )

    assert trace is not None

    assert (
        trace.run_id
        == state.run_id
    )

    assert (
        trace.status
        == TraceStatus.COMPLETED
    )

    assert result.completed is True
    assert result.failed is False


def test_successful_trace_contains_core_events(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    engine = NexusEngine(
        registry=(
            build_success_registry()
        ),
        observability_service=service,
    )

    state, _ = build_state()

    engine.run(
        state
    )

    trace = service.get_trace(
        state.run_id
    )

    assert trace is not None

    types = event_types(
        trace
    )

    assert (
        TraceEventType.WORKFLOW_STARTED
        in types
    )

    assert (
        TraceEventType.TASK_STARTED
        in types
    )

    assert (
        TraceEventType.ARTIFACT_CREATED
        in types
    )

    assert (
        TraceEventType.TASK_COMPLETED
        in types
    )

    assert (
        TraceEventType.WORKFLOW_COMPLETED
        in types
    )


def test_successful_trace_event_order(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    engine = NexusEngine(
        registry=(
            build_success_registry()
        ),
        observability_service=service,
    )

    state, _ = build_state()

    engine.run(
        state
    )

    trace = service.get_trace(
        state.run_id
    )

    assert trace is not None

    types = event_types(
        trace
    )

    workflow_started_index = (
        types.index(
            TraceEventType.WORKFLOW_STARTED
        )
    )

    task_started_index = (
        types.index(
            TraceEventType.TASK_STARTED
        )
    )

    artifact_index = (
        types.index(
            TraceEventType.ARTIFACT_CREATED
        )
    )

    task_completed_index = (
        types.index(
            TraceEventType.TASK_COMPLETED
        )
    )

    workflow_completed_index = (
        types.index(
            TraceEventType.WORKFLOW_COMPLETED
        )
    )

    assert (
        workflow_started_index
        < task_started_index
        < artifact_index
        < task_completed_index
        < workflow_completed_index
    )


def test_successful_trace_counts_tasks_and_artifacts(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    engine = NexusEngine(
        registry=(
            build_success_registry()
        ),
        observability_service=service,
    )

    state, _ = build_state()

    engine.run(
        state
    )

    summary = service.get_summary(
        state.run_id
    )

    assert summary is not None

    assert (
        summary.task_count
        == 1
    )

    assert (
        summary.completed_task_count
        == 1
    )

    assert (
        summary.failed_task_count
        == 0
    )

    assert (
        summary.artifact_count
        == 1
    )

    assert (
        summary.agents_used
        == ["coder"]
    )


def test_state_contains_observability_metadata(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    engine = NexusEngine(
        registry=(
            build_success_registry()
        ),
        observability_service=service,
    )

    state, _ = build_state()

    result = engine.run(
        state
    )

    metadata = result.metadata[
        "observability"
    ]

    assert (
        metadata["run_id"]
        == state.run_id
    )

    assert (
        metadata["status"]
        == "completed"
    )

    assert (
        metadata["artifact_count"]
        == 1
    )

    assert (
        metadata["event_count"]
        >= 5
    )


def test_engine_exposes_last_trace_collector(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    engine = NexusEngine(
        registry=(
            build_success_registry()
        ),
        observability_service=service,
    )

    state, _ = build_state()

    engine.run(
        state
    )

    assert (
        engine.last_trace_collector
        is not None
    )

    assert (
        engine.last_trace_collector
        .run_id
        == state.run_id
    )


def test_failed_agent_creates_failed_trace(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    engine = NexusEngine(
        registry=(
            build_failure_registry()
        ),
        observability_service=service,
    )

    state, _ = build_state()

    try:
        engine.run(
            state
        )

    except RuntimeError:
        pass

    trace = service.get_trace(
        state.run_id
    )

    assert trace is not None

    assert (
        trace.status
        == TraceStatus.FAILED
    )

    assert state.failed is True


def test_failed_agent_records_task_failed(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    engine = NexusEngine(
        registry=(
            build_failure_registry()
        ),
        observability_service=service,
    )

    state, _ = build_state()

    try:
        engine.run(
            state
        )

    except RuntimeError:
        pass

    trace = service.get_trace(
        state.run_id
    )

    assert trace is not None

    types = event_types(
        trace
    )

    assert (
        TraceEventType.TASK_FAILED
        in types
    )

    assert (
        TraceEventType.WORKFLOW_FAILED
        in types
    )


def test_failure_event_contains_message(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    engine = NexusEngine(
        registry=(
            build_failure_registry()
        ),
        observability_service=service,
    )

    state, _ = build_state()

    try:
        engine.run(
            state
        )

    except RuntimeError:
        pass

    trace = service.get_trace(
        state.run_id
    )

    assert trace is not None

    failed_event = next(
        event
        for event
        in trace.events
        if (
            event.event_type
            == TraceEventType.TASK_FAILED
        )
    )

    assert (
        "intentional agent failure"
        in failed_event.message
    )


def test_engine_without_observability_preserves_old_behavior():
    engine = NexusEngine(
        registry=(
            build_success_registry()
        ),
        observability_service=None,
    )

    state, _ = build_state()

    result = engine.run(
        state
    )

    assert result.completed is True
    assert result.failed is False

    assert (
        engine.last_trace_collector
        is None
    )

    assert (
        "observability"
        not in result.metadata
    )
