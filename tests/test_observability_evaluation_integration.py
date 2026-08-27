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

from app.evaluation.benchmark import (
    BenchmarkEngine,
)
from app.evaluation.engine import (
    EvaluationEngine,
)
from app.evaluation.history import (
    EvaluationHistoryStore,
)
from app.evaluation.service import (
    EvaluationService,
)

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


class SimpleAgent(BaseAgent):
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
                        "content": "print('ok')",
                    }
                ]
            },
            created_by=self.role,
        )


def build_registry():
    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        SimpleAgent,
    )

    return registry


def build_state():
    state = NexusState(
        user_request=(
            "Build a simple application."
        )
    )

    task = AgentTask(
        title="Implement application",
        description="Generate code.",
        assigned_agent=AgentRole.CODER,
    )

    state.add_task(
        task
    )

    return state


def build_evaluation_service(
    tmp_path,
):
    return EvaluationService(
        evaluation_engine=(
            EvaluationEngine()
        ),
        history_store=(
            EvaluationHistoryStore(
                db_path=str(
                    tmp_path
                    / "evaluations.db"
                )
            )
        ),
        benchmark_engine=(
            BenchmarkEngine()
        ),
        auto_create_baseline=True,
    )


def build_observability_service(
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


def event_types(
    trace,
):
    return [
        event.event_type
        for event
        in trace.events
    ]


def test_evaluation_event_occurs_before_workflow_completed(
    tmp_path,
):
    evaluation_service = (
        build_evaluation_service(
            tmp_path
        )
    )

    observability_service = (
        build_observability_service(
            tmp_path
        )
    )

    engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            evaluation_service
        ),
        observability_service=(
            observability_service
        ),
    )

    state = build_state()

    engine.run(
        state
    )

    trace = (
        observability_service
        .get_trace(
            state.run_id
        )
    )

    assert trace is not None

    types = event_types(
        trace
    )

    evaluation_index = types.index(
        TraceEventType.EVALUATION_COMPLETED
    )

    workflow_completed_index = (
        types.index(
            TraceEventType.WORKFLOW_COMPLETED
        )
    )

    assert (
        evaluation_index
        < workflow_completed_index
    )


def test_evaluation_event_contains_score(
    tmp_path,
):
    evaluation_service = (
        build_evaluation_service(
            tmp_path
        )
    )

    observability_service = (
        build_observability_service(
            tmp_path
        )
    )

    engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            evaluation_service
        ),
        observability_service=(
            observability_service
        ),
    )

    state = build_state()

    engine.run(
        state
    )

    trace = (
        observability_service
        .get_trace(
            state.run_id
        )
    )

    assert trace is not None

    event = next(
        event
        for event
        in trace.events
        if (
            event.event_type
            == TraceEventType.EVALUATION_COMPLETED
        )
    )

    assert (
        "overall_score"
        in event.metadata
    )

    assert (
        0.0
        <= event.metadata[
            "overall_score"
        ]
        <= 100.0
    )


def test_evaluation_and_trace_share_same_run_id(
    tmp_path,
):
    evaluation_service = (
        build_evaluation_service(
            tmp_path
        )
    )

    observability_service = (
        build_observability_service(
            tmp_path
        )
    )

    engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            evaluation_service
        ),
        observability_service=(
            observability_service
        ),
    )

    state = build_state()

    engine.run(
        state
    )

    trace = (
        observability_service
        .get_trace(
            state.run_id
        )
    )

    evaluation = (
        evaluation_service
        .get_evaluation(
            state.run_id
        )
    )

    assert trace is not None
    assert evaluation is not None

    assert (
        trace.run_id
        == evaluation.run_id
        == state.run_id
    )


def test_both_evaluation_and_observability_metadata_exist(
    tmp_path,
):
    engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            build_evaluation_service(
                tmp_path
            )
        ),
        observability_service=(
            build_observability_service(
                tmp_path
            )
        ),
    )

    state = build_state()

    result = engine.run(
        state
    )

    assert (
        "evaluation"
        in result.metadata
    )

    assert (
        "observability"
        in result.metadata
    )

    assert (
        result.metadata[
            "observability"
        ][
            "run_id"
        ]
        == state.run_id
    )


def test_trace_finishes_completed_with_evaluation_enabled(
    tmp_path,
):
    observability_service = (
        build_observability_service(
            tmp_path
        )
    )

    engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            build_evaluation_service(
                tmp_path
            )
        ),
        observability_service=(
            observability_service
        ),
    )

    state = build_state()

    engine.run(
        state
    )

    trace = (
        observability_service
        .get_trace(
            state.run_id
        )
    )

    assert trace is not None

    assert (
        trace.status
        == TraceStatus.COMPLETED
    )

    assert (
        trace.completed_at
        is not None
    )


def test_first_run_evaluation_baseline_is_reflected_in_state(
    tmp_path,
):
    engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            build_evaluation_service(
                tmp_path
            )
        ),
        observability_service=(
            build_observability_service(
                tmp_path
            )
        ),
    )

    state = build_state()

    result = engine.run(
        state
    )

    assert (
        result.metadata[
            "evaluation_baseline_created"
        ]
        is True
    )

    assert (
        result.metadata[
            "evaluation_baseline_run_id"
        ]
        == state.run_id
    )
