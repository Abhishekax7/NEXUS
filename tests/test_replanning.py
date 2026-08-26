import pytest

from app.core.models import AgentRole
from app.core.replanning import (
    ProposedTask,
    ReplanAction,
    ReplanningDecision,
    ReplanningError,
    validate_replanning_decision,
)


def test_keep_plan_decision_is_valid():
    decision = ReplanningDecision(
        should_replan=False,
        reason="Current execution plan is sufficient.",
        action=ReplanAction.KEEP_PLAN,
        confidence=0.95,
        evidence=[
            "All required tasks are present."
        ],
    )

    validate_replanning_decision(
        decision
    )


def test_non_replanning_decision_must_keep_plan():
    decision = ReplanningDecision(
        should_replan=False,
        reason="No changes required.",
        action=ReplanAction.ADD_TASK,
        proposed_task=ProposedTask(
            title="Extra task",
            description="Unnecessary task",
            assigned_agent=AgentRole.CODER,
        ),
        confidence=0.8,
        evidence=[
            "Plan appears complete."
        ],
    )

    with pytest.raises(
        ReplanningError,
        match="must use KEEP_PLAN",
    ):
        validate_replanning_decision(
            decision
        )


def test_replanning_cannot_keep_plan():
    decision = ReplanningDecision(
        should_replan=True,
        reason="Plan must change.",
        action=ReplanAction.KEEP_PLAN,
        confidence=0.9,
        evidence=[
            "New requirement discovered."
        ],
    )

    with pytest.raises(
        ReplanningError,
        match="cannot use KEEP_PLAN",
    ):
        validate_replanning_decision(
            decision
        )


def test_add_task_requires_proposed_task():
    decision = ReplanningDecision(
        should_replan=True,
        reason="Additional implementation work is required.",
        action=ReplanAction.ADD_TASK,
        proposed_task=None,
        confidence=0.9,
        evidence=[
            "Security review found missing authentication."
        ],
    )

    with pytest.raises(
        ReplanningError,
        match="require proposed_task",
    ):
        validate_replanning_decision(
            decision
        )


def test_valid_add_task_decision():
    proposed_task = ProposedTask(
        title="Implement authentication",
        description=(
            "Add authentication and authorization "
            "to protected endpoints."
        ),
        assigned_agent=AgentRole.CODER,
        depends_on_roles=[
            AgentRole.ARCHITECT
        ],
        metadata={
            "reason": "security"
        },
    )

    decision = ReplanningDecision(
        should_replan=True,
        reason="Authentication is missing.",
        action=ReplanAction.ADD_TASK,
        proposed_task=proposed_task,
        confidence=0.95,
        evidence=[
            "Security report marked authentication as high risk."
        ],
    )

    validate_replanning_decision(
        decision
    )

    assert (
        decision.proposed_task.title
        == "Implement authentication"
    )

    assert (
        decision.proposed_task.assigned_agent
        == AgentRole.CODER
    )


def test_remove_task_requires_target_task_id():
    decision = ReplanningDecision(
        should_replan=True,
        reason="Task is no longer relevant.",
        action=ReplanAction.REMOVE_TASK,
        target_task_id=None,
        confidence=0.8,
        evidence=[
            "Requirement was removed."
        ],
    )

    with pytest.raises(
        ReplanningError,
        match="require target_task_id",
    ):
        validate_replanning_decision(
            decision
        )


def test_valid_remove_task_decision():
    decision = ReplanningDecision(
        should_replan=True,
        reason="Task is obsolete.",
        action=ReplanAction.REMOVE_TASK,
        target_task_id="task-123",
        confidence=0.85,
        evidence=[
            "New architecture removed the component."
        ],
    )

    validate_replanning_decision(
        decision
    )

    assert (
        decision.target_task_id
        == "task-123"
    )


def test_replace_task_requires_proposed_task():
    decision = ReplanningDecision(
        should_replan=True,
        reason="Architecture must be redesigned.",
        action=ReplanAction.REPLACE_TASK,
        target_task_id="task-architecture",
        proposed_task=None,
        confidence=0.95,
        evidence=[
            "Critic rejected the current architecture."
        ],
    )

    with pytest.raises(
        ReplanningError,
        match="require proposed_task",
    ):
        validate_replanning_decision(
            decision
        )


def test_replace_task_requires_target_task_id():
    decision = ReplanningDecision(
        should_replan=True,
        reason="Architecture must be redesigned.",
        action=ReplanAction.REPLACE_TASK,
        proposed_task=ProposedTask(
            title="Redesign architecture",
            description=(
                "Create a replacement architecture."
            ),
            assigned_agent=AgentRole.ARCHITECT,
        ),
        confidence=0.95,
        evidence=[
            "Current design violates constraints."
        ],
    )

    with pytest.raises(
        ReplanningError,
        match="require target_task_id",
    ):
        validate_replanning_decision(
            decision
        )


def test_valid_replace_task_decision():
    proposed_task = ProposedTask(
        title="Redesign architecture",
        description=(
            "Replace the current design with one "
            "that satisfies all constraints."
        ),
        assigned_agent=AgentRole.ARCHITECT,
        depends_on_roles=[
            AgentRole.RESEARCH
        ],
    )

    decision = ReplanningDecision(
        should_replan=True,
        reason="Existing architecture violates constraints.",
        action=ReplanAction.REPLACE_TASK,
        target_task_id="architecture-task",
        proposed_task=proposed_task,
        confidence=0.98,
        evidence=[
            "Critic rejected the current architecture.",
            "Security requirements are unmet.",
        ],
    )

    validate_replanning_decision(
        decision
    )

    assert (
        decision.target_task_id
        == "architecture-task"
    )

    assert (
        decision.proposed_task.assigned_agent
        == AgentRole.ARCHITECT
    )


def test_proposed_task_supports_dependencies():
    task = ProposedTask(
        title="Run security recheck",
        description=(
            "Re-run security analysis after implementation."
        ),
        assigned_agent=AgentRole.SECURITY,
        depends_on_roles=[
            AgentRole.CODER,
            AgentRole.TESTER,
        ],
    )

    assert (
        AgentRole.CODER
        in task.depends_on_roles
    )

    assert (
        AgentRole.TESTER
        in task.depends_on_roles
    )


def test_confidence_must_not_exceed_one():
    with pytest.raises(
        ValueError
    ):
        ReplanningDecision(
            should_replan=False,
            reason="Plan is good.",
            action=ReplanAction.KEEP_PLAN,
            confidence=1.5,
            evidence=[
                "Everything is present."
            ],
        )


def test_confidence_must_not_be_negative():
    with pytest.raises(
        ValueError
    ):
        ReplanningDecision(
            should_replan=False,
            reason="Plan is good.",
            action=ReplanAction.KEEP_PLAN,
            confidence=-0.1,
            evidence=[
                "Everything is present."
            ],
        )


def test_evidence_cannot_be_empty():
    with pytest.raises(
        ValueError
    ):
        ReplanningDecision(
            should_replan=False,
            reason="Plan is sufficient.",
            action=ReplanAction.KEEP_PLAN,
            confidence=0.9,
            evidence=[],
        )
