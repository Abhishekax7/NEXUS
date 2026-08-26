import pytest

from app.agents.base import BaseAgent
from app.agents.registry import AgentRegistry
from app.core.engine import (
    NexusEngine,
    ReplanningLimitExceeded,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.replanning import (
    ProposedTask,
    ReplanAction,
    ReplanningDecision,
)
from app.core.state import NexusState


class FakeAgent(BaseAgent):
    def __init__(
        self,
        role,
    ):
        self.role = role
        self.calls = 0

    def execute(
        self,
        task,
        state,
    ):
        self.calls += 1

        return Artifact(
            type=ArtifactType.EVALUATION,
            name="fake_agent_output",
            created_by=self.role,
            content={
                "task": task.title,
                "success": True,
            },
        )


class SequenceReplanner:
    def __init__(
        self,
        decisions,
    ):
        self.decisions = list(
            decisions
        )
        self.calls = 0

    def decide(
        self,
        state,
    ):
        index = min(
            self.calls,
            len(self.decisions) - 1,
        )

        decision = self.decisions[
            index
        ]

        self.calls += 1

        return decision


def keep_plan():
    return ReplanningDecision(
        should_replan=False,
        reason=(
            "Current plan remains sufficient."
        ),
        action=ReplanAction.KEEP_PLAN,
        confidence=0.95,
        evidence=[
            "No additional work is required."
        ],
    )


def add_coder_task():
    return ReplanningDecision(
        should_replan=True,
        reason=(
            "Additional implementation "
            "work is required."
        ),
        action=ReplanAction.ADD_TASK,
        proposed_task=ProposedTask(
            title="Dynamic implementation",
            description=(
                "Implement newly discovered work."
            ),
            assigned_agent=AgentRole.CODER,
            depends_on_roles=[
                AgentRole.REQUIREMENTS,
            ],
            metadata={
                "source": "replanner",
            },
        ),
        confidence=0.95,
        evidence=[
            "New implementation work was discovered."
        ],
    )


def build_registry():
    registry = AgentRegistry()

    requirements_agent = FakeAgent(
        AgentRole.REQUIREMENTS
    )

    coder_agent = FakeAgent(
        AgentRole.CODER
    )

    registry.register(
        AgentRole.REQUIREMENTS,
        requirements_agent,
    )

    registry.register(
        AgentRole.CODER,
        coder_agent,
    )

    return (
        registry,
        requirements_agent,
        coder_agent,
    )


def build_state():
    state = NexusState(
        user_request=(
            "Build a dynamically planned system."
        )
    )

    requirements_task = AgentTask(
        title="Analyze requirements",
        description=(
            "Analyze the user request."
        ),
        assigned_agent=(
            AgentRole.REQUIREMENTS
        ),
    )

    state.add_task(
        requirements_task
    )

    return (
        state,
        requirements_task,
    )


def test_engine_without_replanner_preserves_old_behavior():
    (
        registry,
        requirements_agent,
        _,
    ) = build_registry()

    (
        state,
        requirements_task,
    ) = build_state()

    engine = NexusEngine(
        registry=registry
    )

    result = engine.run(
        state
    )

    assert result.completed is True
    assert result.failed is False

    assert (
        requirements_task.status
        == TaskStatus.COMPLETED
    )

    assert requirements_agent.calls == 1

    assert (
        result.metadata.get(
            "replan_count",
            0,
        )
        == 0
    )


def test_keep_plan_does_not_mutate_workflow():
    (
        registry,
        requirements_agent,
        _,
    ) = build_registry()

    state, _ = build_state()

    original_task_ids = set(
        state.tasks.keys()
    )

    replanner = SequenceReplanner(
        [
            keep_plan(),
        ]
    )

    engine = NexusEngine(
        registry=registry,
        replanner=replanner,
    )

    result = engine.run(
        state
    )

    assert result.completed is True

    assert (
        set(result.tasks.keys())
        == original_task_ids
    )

    assert (
        result.metadata.get(
            "replan_count",
            0,
        )
        == 0
    )

    assert requirements_agent.calls == 1
    assert replanner.calls == 1


def test_added_task_is_executed_by_engine():
    (
        registry,
        requirements_agent,
        coder_agent,
    ) = build_registry()

    state, _ = build_state()

    replanner = SequenceReplanner(
        [
            add_coder_task(),
            keep_plan(),
        ]
    )

    engine = NexusEngine(
        registry=registry,
        replanner=replanner,
        max_replans=2,
    )

    result = engine.run(
        state
    )

    assert result.completed is True
    assert result.failed is False

    assert requirements_agent.calls == 1
    assert coder_agent.calls == 1

    dynamic_tasks = [
        task
        for task in result.tasks.values()
        if (
            task.title
            == "Dynamic implementation"
        )
    ]

    assert len(dynamic_tasks) == 1

    dynamic_task = dynamic_tasks[0]

    assert (
        dynamic_task.status
        == TaskStatus.COMPLETED
    )


def test_replan_count_is_incremented():
    (
        registry,
        _,
        _,
    ) = build_registry()

    state, _ = build_state()

    replanner = SequenceReplanner(
        [
            add_coder_task(),
            keep_plan(),
        ]
    )

    engine = NexusEngine(
        registry=registry,
        replanner=replanner,
        max_replans=2,
    )

    result = engine.run(
        state
    )

    assert (
        result.metadata[
            "replan_count"
        ]
        == 1
    )


def test_replan_history_is_recorded():
    (
        registry,
        _,
        _,
    ) = build_registry()

    state, _ = build_state()

    replanner = SequenceReplanner(
        [
            add_coder_task(),
            keep_plan(),
        ]
    )

    engine = NexusEngine(
        registry=registry,
        replanner=replanner,
        max_replans=2,
    )

    result = engine.run(
        state
    )

    history = result.metadata[
        "replan_history"
    ]

    assert len(history) == 1

    assert (
        history[0]["action"]
        == "add_task"
    )

    assert (
        history[0]["added_task_id"]
        is not None
    )

    assert (
        history[0]["removed_task_id"]
        is None
    )


def test_replanner_runs_after_each_iteration():
    (
        registry,
        _,
        _,
    ) = build_registry()

    state, _ = build_state()

    replanner = SequenceReplanner(
        [
            add_coder_task(),
            keep_plan(),
        ]
    )

    engine = NexusEngine(
        registry=registry,
        replanner=replanner,
        max_replans=2,
    )

    result = engine.run(
        state
    )

    assert result.iteration == 2
    assert replanner.calls == 2


def test_replanning_limit_is_enforced():
    (
        registry,
        _,
        _,
    ) = build_registry()

    state, _ = build_state()

    replanner = SequenceReplanner(
        [
            add_coder_task(),
            ReplanningDecision(
                should_replan=True,
                reason=(
                    "More implementation "
                    "is required."
                ),
                action=ReplanAction.ADD_TASK,
                proposed_task=ProposedTask(
                    title="Second dynamic task",
                    description=(
                        "Perform more work."
                    ),
                    assigned_agent=(
                        AgentRole.CODER
                    ),
                    depends_on_roles=[
                        AgentRole.REQUIREMENTS
                    ],
                ),
                confidence=0.9,
                evidence=[
                    "More work was discovered."
                ],
            ),
        ]
    )

    engine = NexusEngine(
        registry=registry,
        replanner=replanner,
        max_replans=1,
    )

    with pytest.raises(
        ReplanningLimitExceeded,
        match="maximum replanning budget",
    ):
        engine.run(
            state
        )

    assert state.failed is True

    assert (
        state.metadata[
            "replan_count"
        ]
        == 1
    )


def test_zero_replan_budget_allows_keep_plan():
    (
        registry,
        _,
        _,
    ) = build_registry()

    state, _ = build_state()

    engine = NexusEngine(
        registry=registry,
        replanner=SequenceReplanner(
            [
                keep_plan(),
            ]
        ),
        max_replans=0,
    )

    result = engine.run(
        state
    )

    assert result.completed is True

    assert (
        result.metadata.get(
            "replan_count",
            0,
        )
        == 0
    )


def test_zero_replan_budget_rejects_mutation():
    (
        registry,
        _,
        _,
    ) = build_registry()

    state, _ = build_state()

    engine = NexusEngine(
        registry=registry,
        replanner=SequenceReplanner(
            [
                add_coder_task(),
            ]
        ),
        max_replans=0,
    )

    with pytest.raises(
        ReplanningLimitExceeded
    ):
        engine.run(
            state
        )

    assert state.failed is True


def test_negative_replan_budget_is_invalid():
    (
        registry,
        _,
        _,
    ) = build_registry()

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        NexusEngine(
            registry=registry,
            max_replans=-1,
        )
