from app.agents.base import BaseAgent
from app.agents.registry import AgentRegistry
from app.core.engine import NexusEngine
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.plan_mutator import PlanMutator
from app.core.replanning import (
    ProposedTask,
    ReplanAction,
    ReplanningDecision,
)
from app.core.state import NexusState


class RequirementsAgent(BaseAgent):
    def __init__(self):
        self.calls = 0

    def execute(
        self,
        task,
        state,
    ):
        self.calls += 1

        return Artifact(
            type=ArtifactType.REQUIREMENTS,
            name="requirements",
            created_by=AgentRole.REQUIREMENTS,
            content={
                "objective": (
                    "Build a secure API."
                ),
                "requirements": [
                    "Expose API endpoints.",
                    "Protect sensitive endpoints.",
                ],
            },
        )


class DynamicCoderAgent(BaseAgent):
    def __init__(self):
        self.calls = 0

    def execute(
        self,
        task,
        state,
    ):
        self.calls += 1

        return Artifact(
            type=ArtifactType.CODE,
            name="dynamic_code",
            created_by=AgentRole.CODER,
            content={
                "files": [
                    {
                        "path": "app.py",
                        "content": (
                            "def authenticate():\n"
                            "    return True\n"
                        ),
                    }
                ],
                "reason": (
                    "Authentication implementation "
                    "was dynamically added."
                ),
            },
        )


class EvidenceAwareReplanner:
    """
    Deterministic replanner used to prove the
    engine's end-to-end replanning behavior.

    First evaluation:
        requirements exist but no CODE artifact
        exists -> add implementation task.

    Second evaluation:
        CODE artifact exists -> keep plan.
    """

    def __init__(self):
        self.calls = 0

    def decide(
        self,
        state,
    ):
        self.calls += 1

        code_exists = any(
            artifact.type
            == ArtifactType.CODE
            for artifact
            in state.artifacts.values()
        )

        if not code_exists:
            return ReplanningDecision(
                should_replan=True,
                reason=(
                    "Requirements require protected "
                    "endpoints but implementation "
                    "work has not yet occurred."
                ),
                action=ReplanAction.ADD_TASK,
                proposed_task=ProposedTask(
                    title=(
                        "Implement authentication"
                    ),
                    description=(
                        "Implement authentication "
                        "for protected endpoints."
                    ),
                    assigned_agent=(
                        AgentRole.CODER
                    ),
                    depends_on_roles=[
                        AgentRole.REQUIREMENTS
                    ],
                    metadata={
                        "source": (
                            "dynamic_replanning"
                        ),
                        "trigger": (
                            "missing_implementation"
                        ),
                    },
                ),
                confidence=0.98,
                evidence=[
                    (
                        "Requirements request "
                        "protected endpoints."
                    ),
                    (
                        "No CODE artifact exists "
                        "in the current state."
                    ),
                ],
            )

        return ReplanningDecision(
            should_replan=False,
            reason=(
                "The dynamically required "
                "implementation now exists."
            ),
            action=ReplanAction.KEEP_PLAN,
            confidence=0.99,
            evidence=[
                (
                    "A CODE artifact now exists "
                    "for the implementation."
                )
            ],
        )


def build_registry():
    registry = AgentRegistry()

    requirements_agent = (
        RequirementsAgent()
    )

    coder_agent = (
        DynamicCoderAgent()
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


def build_initial_state():
    state = NexusState(
        user_request=(
            "Build a secure API with "
            "protected endpoints."
        )
    )

    requirements_task = AgentTask(
        title="Analyze requirements",
        description=(
            "Determine what the API requires."
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


def test_engine_dynamically_expands_and_executes_plan():
    (
        registry,
        requirements_agent,
        coder_agent,
    ) = build_registry()

    (
        state,
        requirements_task,
    ) = build_initial_state()

    replanner = (
        EvidenceAwareReplanner()
    )

    engine = NexusEngine(
        registry=registry,
        replanner=replanner,
        plan_mutator=PlanMutator(),
        max_replans=2,
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
    assert coder_agent.calls == 1

    dynamic_tasks = [
        task
        for task
        in result.tasks.values()
        if (
            task.title
            == "Implement authentication"
        )
    ]

    assert len(dynamic_tasks) == 1

    dynamic_task = dynamic_tasks[0]

    assert (
        dynamic_task.status
        == TaskStatus.COMPLETED
    )

    assert (
        requirements_task.id
        in dynamic_task.dependencies
    )

    assert (
        dynamic_task.metadata[
            "source"
        ]
        == "dynamic_replanning"
    )

    assert (
        result.metadata[
            "replan_count"
        ]
        == 1
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
        == dynamic_task.id
    )

    code_artifacts = [
        artifact
        for artifact
        in result.artifacts.values()
        if (
            artifact.type
            == ArtifactType.CODE
        )
    ]

    assert len(code_artifacts) == 1

    assert replanner.calls == 2


def test_dynamic_task_is_not_duplicated():
    (
        registry,
        _,
        coder_agent,
    ) = build_registry()

    state, _ = (
        build_initial_state()
    )

    replanner = (
        EvidenceAwareReplanner()
    )

    engine = NexusEngine(
        registry=registry,
        replanner=replanner,
        plan_mutator=PlanMutator(),
        max_replans=3,
    )

    result = engine.run(
        state
    )

    dynamic_tasks = [
        task
        for task
        in result.tasks.values()
        if (
            task.title
            == "Implement authentication"
        )
    ]

    assert len(dynamic_tasks) == 1
    assert coder_agent.calls == 1

    assert (
        result.metadata[
            "replan_count"
        ]
        == 1
    )


def test_replanning_preserves_audit_history():
    (
        registry,
        _,
        _,
    ) = build_registry()

    state, _ = (
        build_initial_state()
    )

    engine = NexusEngine(
        registry=registry,
        replanner=(
            EvidenceAwareReplanner()
        ),
        plan_mutator=PlanMutator(),
        max_replans=2,
    )

    result = engine.run(
        state
    )

    history = result.metadata.get(
        "replan_history"
    )

    assert history is not None
    assert len(history) == 1

    record = history[0]

    assert (
        record["action"]
        == ReplanAction.ADD_TASK.value
    )

    assert (
        record["added_task_id"]
        is not None
    )

    assert (
        record["removed_task_id"]
        is None
    )

    assert (
        record["replaced_task_id"]
        is None
    )
