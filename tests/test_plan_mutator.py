import pytest

from app.core.models import (
    AgentRole,
    AgentTask,
    TaskStatus,
)
from app.core.plan_mutator import (
    PlanMutator,
    PlanMutationError,
)
from app.core.replanning import (
    ProposedTask,
    ReplanAction,
    ReplanningDecision,
)
from app.core.state import NexusState


def build_state():
    state = NexusState(
        user_request="Build secure API"
    )

    requirements = AgentTask(
        title="Requirements",
        description="Analyze request",
        assigned_agent=AgentRole.REQUIREMENTS,
        status=TaskStatus.COMPLETED,
    )

    research = AgentTask(
        title="Research",
        description="Research stack",
        assigned_agent=AgentRole.RESEARCH,
        status=TaskStatus.COMPLETED,
        dependencies=[
            requirements.id
        ],
    )

    architecture = AgentTask(
        title="Architecture",
        description="Design system",
        assigned_agent=AgentRole.ARCHITECT,
        status=TaskStatus.PENDING,
        dependencies=[
            research.id
        ],
    )

    coder = AgentTask(
        title="Implementation",
        description="Generate code",
        assigned_agent=AgentRole.CODER,
        status=TaskStatus.PENDING,
        dependencies=[
            architecture.id
        ],
    )

    state.add_task(requirements)
    state.add_task(research)
    state.add_task(architecture)
    state.add_task(coder)

    return (
        state,
        requirements,
        research,
        architecture,
        coder,
    )


def test_keep_plan_does_not_change_state():
    (
        state,
        _,
        _,
        _,
        _,
    ) = build_state()

    original_ids = set(
        state.tasks.keys()
    )

    decision = ReplanningDecision(
        should_replan=False,
        reason="Plan is sufficient.",
        action=ReplanAction.KEEP_PLAN,
        confidence=0.95,
        evidence=[
            "All required tasks exist."
        ],
    )

    result = PlanMutator().apply(
        decision,
        state,
    )

    assert (
        result.action
        == ReplanAction.KEEP_PLAN
    )

    assert set(
        state.tasks.keys()
    ) == original_ids


def test_add_task_adds_new_task():
    (
        state,
        _,
        _,
        _,
        _,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=True,
        reason="Security remediation needed.",
        action=ReplanAction.ADD_TASK,
        proposed_task=ProposedTask(
            title="Security remediation",
            description="Fix authentication.",
            assigned_agent=AgentRole.CODER,
            depends_on_roles=[
                AgentRole.SECURITY
            ],
        ),
        confidence=0.9,
        evidence=[
            "Security finding exists."
        ],
    )

    security_task = AgentTask(
        title="Security",
        description="Review security",
        assigned_agent=AgentRole.SECURITY,
        status=TaskStatus.COMPLETED,
    )

    state.add_task(
        security_task
    )

    result = PlanMutator().apply(
        decision,
        state,
    )

    assert (
        result.action
        == ReplanAction.ADD_TASK
    )

    assert (
        result.added_task_id
        in state.tasks
    )

    added = state.tasks[
        result.added_task_id
    ]

    assert (
        added.assigned_agent
        == AgentRole.CODER
    )

    assert (
        security_task.id
        in added.dependencies
    )


def test_add_task_fails_when_dependency_role_missing():
    (
        state,
        _,
        _,
        _,
        _,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=True,
        reason="Need security task dependency.",
        action=ReplanAction.ADD_TASK,
        proposed_task=ProposedTask(
            title="Security remediation",
            description="Fix auth",
            assigned_agent=AgentRole.CODER,
            depends_on_roles=[
                AgentRole.SECURITY
            ],
        ),
        confidence=0.9,
        evidence=[
            "Security remediation needed."
        ],
    )

    with pytest.raises(
        PlanMutationError,
        match="No task exists for dependency role",
    ):
        PlanMutator().apply(
            decision,
            state,
        )


def test_remove_pending_task_without_dependents():
    (
        state,
        _,
        _,
        _,
        coder,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=True,
        reason="Implementation no longer needed.",
        action=ReplanAction.REMOVE_TASK,
        target_task_id=coder.id,
        confidence=0.85,
        evidence=[
            "Requirement removed."
        ],
    )

    result = PlanMutator().apply(
        decision,
        state,
    )

    assert (
        result.removed_task_id
        == coder.id
    )

    assert (
        coder.id
        not in state.tasks
    )


def test_remove_fails_when_task_has_dependents():
    (
        state,
        _,
        _,
        architecture,
        _,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=True,
        reason="Architecture obsolete.",
        action=ReplanAction.REMOVE_TASK,
        target_task_id=architecture.id,
        confidence=0.8,
        evidence=[
            "Architecture changed."
        ],
    )

    with pytest.raises(
        PlanMutationError,
        match="dependents",
    ):
        PlanMutator().apply(
            decision,
            state,
        )


def test_remove_rejects_completed_task():
    (
        state,
        requirements,
        _,
        _,
        _,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=True,
        reason="Remove old requirements.",
        action=ReplanAction.REMOVE_TASK,
        target_task_id=requirements.id,
        confidence=0.8,
        evidence=[
            "Requirements obsolete."
        ],
    )

    with pytest.raises(
        PlanMutationError,
        match="completed task",
    ):
        PlanMutator().apply(
            decision,
            state,
        )


def test_remove_rejects_running_task():
    (
        state,
        _,
        _,
        architecture,
        _,
    ) = build_state()

    architecture.status = (
        TaskStatus.RUNNING
    )

    decision = ReplanningDecision(
        should_replan=True,
        reason="Replace running architecture.",
        action=ReplanAction.REMOVE_TASK,
        target_task_id=architecture.id,
        confidence=0.8,
        evidence=[
            "Architecture issue."
        ],
    )

    with pytest.raises(
        PlanMutationError,
        match="running task",
    ):
        PlanMutator().apply(
            decision,
            state,
        )


def test_replace_task_rewires_dependents():
    (
        state,
        _,
        research,
        architecture,
        coder,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=True,
        reason="Architecture must be redesigned.",
        action=ReplanAction.REPLACE_TASK,
        target_task_id=architecture.id,
        proposed_task=ProposedTask(
            title="New architecture",
            description="Redesign system",
            assigned_agent=AgentRole.ARCHITECT,
            depends_on_roles=[
                AgentRole.RESEARCH
            ],
        ),
        confidence=0.95,
        evidence=[
            "Current design violates constraints."
        ],
    )

    result = PlanMutator().apply(
        decision,
        state,
    )

    assert (
        architecture.id
        not in state.tasks
    )

    assert (
        result.added_task_id
        in state.tasks
    )

    new_architecture = state.tasks[
        result.added_task_id
    ]

    assert (
        research.id
        in new_architecture.dependencies
    )

    assert (
        architecture.id
        not in coder.dependencies
    )

    assert (
        new_architecture.id
        in coder.dependencies
    )


def test_replace_rejects_completed_target():
    (
        state,
        requirements,
        _,
        _,
        _,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=True,
        reason="Replace requirements.",
        action=ReplanAction.REPLACE_TASK,
        target_task_id=requirements.id,
        proposed_task=ProposedTask(
            title="New requirements",
            description="Redo requirements",
            assigned_agent=AgentRole.REQUIREMENTS,
        ),
        confidence=0.9,
        evidence=[
            "Requirements changed."
        ],
    )

    with pytest.raises(
        PlanMutationError,
        match="completed task",
    ):
        PlanMutator().apply(
            decision,
            state,
        )


def test_replace_rejects_running_target():
    (
        state,
        _,
        _,
        architecture,
        _,
    ) = build_state()

    architecture.status = (
        TaskStatus.RUNNING
    )

    decision = ReplanningDecision(
        should_replan=True,
        reason="Replace architecture.",
        action=ReplanAction.REPLACE_TASK,
        target_task_id=architecture.id,
        proposed_task=ProposedTask(
            title="New architecture",
            description="Redo design",
            assigned_agent=AgentRole.ARCHITECT,
            depends_on_roles=[
                AgentRole.RESEARCH
            ],
        ),
        confidence=0.9,
        evidence=[
            "Design issue."
        ],
    )

    with pytest.raises(
        PlanMutationError,
        match="running task",
    ):
        PlanMutator().apply(
            decision,
            state,
        )


def test_replace_fails_when_dependency_role_missing():
    (
        state,
        _,
        _,
        architecture,
        _,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=True,
        reason="Replace architecture.",
        action=ReplanAction.REPLACE_TASK,
        target_task_id=architecture.id,
        proposed_task=ProposedTask(
            title="Replacement",
            description="Replacement task",
            assigned_agent=AgentRole.ARCHITECT,
            depends_on_roles=[
                AgentRole.SECURITY
            ],
        ),
        confidence=0.9,
        evidence=[
            "Security dependency required."
        ],
    )

    with pytest.raises(
        PlanMutationError,
        match="No task exists for dependency role",
    ):
        PlanMutator().apply(
            decision,
            state,
        )


def test_added_task_copies_metadata():
    (
        state,
        _,
        _,
        _,
        _,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=True,
        reason="Need additional testing.",
        action=ReplanAction.ADD_TASK,
        proposed_task=ProposedTask(
            title="Extra test",
            description="Run extra tests",
            assigned_agent=AgentRole.TESTER,
            metadata={
                "source": "replanner",
                "priority": "high",
            },
        ),
        confidence=0.9,
        evidence=[
            "More validation is required."
        ],
    )

    result = PlanMutator().apply(
        decision,
        state,
    )

    added = state.tasks[
        result.added_task_id
    ]

    assert (
        added.metadata["source"]
        == "replanner"
    )

    assert (
        added.metadata["priority"]
        == "high"
    )


def test_replace_result_tracks_old_and_new_ids():
    (
        state,
        _,
        _,
        architecture,
        _,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=True,
        reason="Replace architecture.",
        action=ReplanAction.REPLACE_TASK,
        target_task_id=architecture.id,
        proposed_task=ProposedTask(
            title="Replacement architecture",
            description="New design",
            assigned_agent=AgentRole.ARCHITECT,
            depends_on_roles=[
                AgentRole.RESEARCH
            ],
        ),
        confidence=0.9,
        evidence=[
            "Architecture correction needed."
        ],
    )

    result = PlanMutator().apply(
        decision,
        state,
    )

    assert (
        result.removed_task_id
        == architecture.id
    )

    assert (
        result.replaced_task_id
        == architecture.id
    )

    assert (
        result.added_task_id
        is not None
    )


def test_invalid_replanning_contract_is_wrapped():
    (
        state,
        _,
        _,
        _,
        _,
    ) = build_state()

    decision = ReplanningDecision(
        should_replan=False,
        reason="No change.",
        action=ReplanAction.ADD_TASK,
        proposed_task=ProposedTask(
            title="Invalid",
            description="Invalid",
            assigned_agent=AgentRole.CODER,
        ),
        confidence=0.8,
        evidence=[
            "No change needed."
        ],
    )

    with pytest.raises(
        PlanMutationError,
        match="KEEP_PLAN",
    ):
        PlanMutator().apply(
            decision,
            state,
        )
