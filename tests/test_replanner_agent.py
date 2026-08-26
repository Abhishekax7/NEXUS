import json

import pytest

from app.agents.replanner import (
    ReplannerAgent,
    ReplannerGenerationError,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    TaskStatus,
)
from app.core.replanning import (
    ReplanAction,
)
from app.core.state import NexusState


class FakeLLM:
    def __init__(
        self,
        responses,
    ):
        self.responses = list(
            responses
        )

        self.calls = 0

        self.last_user_prompt = None

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
    ):
        self.last_user_prompt = (
            user_prompt
        )

        response = self.responses[
            min(
                self.calls,
                len(self.responses) - 1,
            )
        ]

        self.calls += 1

        return response


def keep_plan_response():
    return json.dumps(
        {
            "should_replan": False,
            "reason": (
                "The current workflow "
                "already covers the request."
            ),
            "action": "keep_plan",
            "proposed_task": None,
            "target_task_id": None,
            "confidence": 0.95,
            "evidence": [
                "All required work is already represented."
            ],
        }
    )


def add_task_response():
    return json.dumps(
        {
            "should_replan": True,
            "reason": (
                "Security review requires "
                "additional remediation."
            ),
            "action": "add_task",
            "proposed_task": {
                "title": (
                    "Implement authentication fix"
                ),
                "description": (
                    "Add missing authentication "
                    "controls."
                ),
                "assigned_agent": "coder",
                "depends_on_roles": [
                    "security",
                ],
                "metadata": {
                    "source": "security",
                },
            },
            "target_task_id": None,
            "confidence": 0.92,
            "evidence": [
                "Security review identified missing authentication."
            ],
        }
    )


def remove_task_response(
    target_task_id,
):
    return json.dumps(
        {
            "should_replan": True,
            "reason": (
                "The planned task is no "
                "longer necessary."
            ),
            "action": "remove_task",
            "proposed_task": None,
            "target_task_id": (
                target_task_id
            ),
            "confidence": 0.85,
            "evidence": [
                "The requirement was removed."
            ],
        }
    )


def replace_task_response(
    target_task_id,
):
    return json.dumps(
        {
            "should_replan": True,
            "reason": (
                "The planned architecture work "
                "must be replaced."
            ),
            "action": "replace_task",
            "proposed_task": {
                "title": (
                    "Redesign architecture"
                ),
                "description": (
                    "Create a replacement "
                    "architecture."
                ),
                "assigned_agent": "architect",
                "depends_on_roles": [
                    "research",
                ],
                "metadata": {
                    "reason": "critic_feedback",
                },
            },
            "target_task_id": (
                target_task_id
            ),
            "confidence": 0.97,
            "evidence": [
                "Current planned architecture violates constraints."
            ],
        }
    )


def build_state():
    state = NexusState(
        user_request=(
            "Build a secure API."
        )
    )

    requirements_task = AgentTask(
        title="Analyze requirements",
        description="Analyze request.",
        assigned_agent=(
            AgentRole.REQUIREMENTS
        ),
        status=TaskStatus.COMPLETED,
    )

    research_task = AgentTask(
        title="Research technologies",
        description="Research options.",
        assigned_agent=(
            AgentRole.RESEARCH
        ),
        status=TaskStatus.COMPLETED,
        dependencies=[
            requirements_task.id
        ],
    )

    architecture_task = AgentTask(
        title="Design architecture",
        description="Design system.",
        assigned_agent=(
            AgentRole.ARCHITECT
        ),
        status=TaskStatus.PENDING,
        dependencies=[
            research_task.id
        ],
    )

    coder_task = AgentTask(
        title="Implement system",
        description="Generate code.",
        assigned_agent=(
            AgentRole.CODER
        ),
        status=TaskStatus.PENDING,
        dependencies=[
            architecture_task.id
        ],
    )

    state.add_task(
        requirements_task
    )

    state.add_task(
        research_task
    )

    state.add_task(
        architecture_task
    )

    state.add_task(
        coder_task
    )

    return (
        state,
        requirements_task,
        research_task,
        architecture_task,
        coder_task,
    )


def test_replanner_keeps_valid_plan():
    state, *_ = build_state()

    agent = ReplannerAgent(
        llm_client=FakeLLM(
            [
                keep_plan_response()
            ]
        )
    )

    decision = agent.decide(
        state
    )

    assert (
        decision.should_replan
        is False
    )

    assert (
        decision.action
        == ReplanAction.KEEP_PLAN
    )


def test_replanner_returns_add_task():
    state, *_ = build_state()

    agent = ReplannerAgent(
        llm_client=FakeLLM(
            [
                add_task_response()
            ]
        )
    )

    decision = agent.decide(
        state
    )

    assert (
        decision.should_replan
        is True
    )

    assert (
        decision.action
        == ReplanAction.ADD_TASK
    )

    assert (
        decision.proposed_task
        is not None
    )

    assert (
        decision.proposed_task
        .assigned_agent
        == AgentRole.CODER
    )


def test_replanner_accepts_remove_pending_task():
    (
        state,
        _,
        _,
        architecture_task,
        _,
    ) = build_state()

    agent = ReplannerAgent(
        llm_client=FakeLLM(
            [
                remove_task_response(
                    architecture_task.id
                )
            ]
        )
    )

    decision = agent.decide(
        state
    )

    assert (
        decision.action
        == ReplanAction.REMOVE_TASK
    )

    assert (
        decision.target_task_id
        == architecture_task.id
    )


def test_replanner_accepts_replace_pending_task():
    (
        state,
        _,
        _,
        architecture_task,
        _,
    ) = build_state()

    agent = ReplannerAgent(
        llm_client=FakeLLM(
            [
                replace_task_response(
                    architecture_task.id
                )
            ]
        )
    )

    decision = agent.decide(
        state
    )

    assert (
        decision.action
        == ReplanAction.REPLACE_TASK
    )

    assert (
        decision.proposed_task
        is not None
    )


def test_replanner_rejects_missing_target():
    state, *_ = build_state()

    invalid = json.dumps(
        {
            "should_replan": True,
            "reason": "Remove task.",
            "action": "remove_task",
            "proposed_task": None,
            "target_task_id": (
                "missing-task"
            ),
            "confidence": 0.9,
            "evidence": [
                "Task should be removed."
            ],
        }
    )

    agent = ReplannerAgent(
        llm_client=FakeLLM(
            [
                invalid,
                keep_plan_response(),
            ]
        ),
        max_validation_retries=1,
    )

    decision = agent.decide(
        state
    )

    assert (
        decision.action
        == ReplanAction.KEEP_PLAN
    )


def test_replanner_rejects_completed_target():
    (
        state,
        requirements_task,
        *_,
    ) = build_state()

    agent = ReplannerAgent(
        llm_client=FakeLLM(
            [
                remove_task_response(
                    requirements_task.id
                ),
                keep_plan_response(),
            ]
        ),
        max_validation_retries=1,
    )

    decision = agent.decide(
        state
    )

    assert (
        decision.action
        == ReplanAction.KEEP_PLAN
    )


def test_replanner_rejects_running_target():
    (
        state,
        _,
        _,
        architecture_task,
        _,
    ) = build_state()

    architecture_task.status = (
        TaskStatus.RUNNING
    )

    agent = ReplannerAgent(
        llm_client=FakeLLM(
            [
                replace_task_response(
                    architecture_task.id
                ),
                keep_plan_response(),
            ]
        ),
        max_validation_retries=1,
    )

    decision = agent.decide(
        state
    )

    assert (
        decision.action
        == ReplanAction.KEEP_PLAN
    )


def test_replanner_retries_invalid_json():
    state, *_ = build_state()

    fake_llm = FakeLLM(
        [
            "not-json",
            keep_plan_response(),
        ]
    )

    agent = ReplannerAgent(
        llm_client=fake_llm,
        max_validation_retries=1,
    )

    decision = agent.decide(
        state
    )

    assert fake_llm.calls == 2

    assert (
        decision.action
        == ReplanAction.KEEP_PLAN
    )


def test_replanner_retries_invalid_contract():
    state, *_ = build_state()

    invalid = json.dumps(
        {
            "should_replan": False,
            "reason": (
                "No change required."
            ),
            "action": "add_task",
            "proposed_task": {
                "title": "Extra work",
                "description": "Extra task.",
                "assigned_agent": "coder",
                "depends_on_roles": [],
                "metadata": {},
            },
            "target_task_id": None,
            "confidence": 0.8,
            "evidence": [
                "Plan is sufficient."
            ],
        }
    )

    fake_llm = FakeLLM(
        [
            invalid,
            keep_plan_response(),
        ]
    )

    agent = ReplannerAgent(
        llm_client=fake_llm,
        max_validation_retries=1,
    )

    decision = agent.decide(
        state
    )

    assert fake_llm.calls == 2

    assert (
        decision.action
        == ReplanAction.KEEP_PLAN
    )


def test_replanner_fails_after_retry_limit():
    state, *_ = build_state()

    agent = ReplannerAgent(
        llm_client=FakeLLM(
            [
                "{}",
                "{}",
            ]
        ),
        max_validation_retries=1,
    )

    with pytest.raises(
        ReplannerGenerationError,
        match="could not be validated",
    ):
        agent.decide(
            state
        )


def test_replanner_prompt_contains_current_tasks():
    state, *_ = build_state()

    fake_llm = FakeLLM(
        [
            keep_plan_response()
        ]
    )

    agent = ReplannerAgent(
        llm_client=fake_llm
    )

    agent.decide(
        state
    )

    assert (
        "Analyze requirements"
        in fake_llm.last_user_prompt
    )

    assert (
        "Design architecture"
        in fake_llm.last_user_prompt
    )

    assert (
        "Implement system"
        in fake_llm.last_user_prompt
    )


def test_replanner_prompt_contains_user_request():
    state, *_ = build_state()

    fake_llm = FakeLLM(
        [
            keep_plan_response()
        ]
    )

    agent = ReplannerAgent(
        llm_client=fake_llm
    )

    agent.decide(
        state
    )

    assert (
        "Build a secure API"
        in fake_llm.last_user_prompt
    )


def test_replanner_rejects_dependency_free_critic_task():
    state, *_ = build_state()

    invalid = json.dumps(
        {
            "should_replan": True,
            "reason": (
                "Add another quality gate."
            ),
            "action": "add_task",
            "proposed_task": {
                "title": "Extra critic review",
                "description": (
                    "Review implementation."
                ),
                "assigned_agent": "critic",
                "depends_on_roles": [],
                "metadata": {},
            },
            "target_task_id": None,
            "confidence": 0.8,
            "evidence": [
                "Additional review requested."
            ],
        }
    )

    agent = ReplannerAgent(
        llm_client=FakeLLM(
            [
                invalid,
                keep_plan_response(),
            ]
        ),
        max_validation_retries=1,
    )

    decision = agent.decide(
        state
    )

    assert (
        decision.action
        == ReplanAction.KEEP_PLAN
    )


def test_replanner_snapshot_contains_status():
    (
        state,
        _,
        _,
        architecture_task,
        _,
    ) = build_state()

    fake_llm = FakeLLM(
        [
            keep_plan_response()
        ]
    )

    agent = ReplannerAgent(
        llm_client=fake_llm
    )

    agent.decide(
        state
    )

    assert (
        architecture_task.status.value
        in fake_llm.last_user_prompt
    )
