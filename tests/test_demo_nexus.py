import sys

import scripts.demo_nexus as demo_nexus

from app.agents.orchestrator import (
    OrchestratorAgent,
)

from app.core.models import (
    AgentRole,
)

from scripts.demo_nexus import (
    DEFAULT_REQUEST,
    build_demo_state,
    enum_value,
    short_id,
)


def test_default_request_is_not_empty():

    assert DEFAULT_REQUEST.strip()


def test_build_demo_state_uses_orchestrator_plan():

    state = build_demo_state(
        "Build a test service."
    )

    assert (
        state.user_request
        == "Build a test service."
    )

    assert (
        len(state.tasks)
        == 7
    )

    assert (
        len(state.execution_order)
        == 7
    )


def test_demo_plan_contains_expected_agents():

    state = build_demo_state(
        "Build a test service."
    )

    roles = {
        task.assigned_agent
        for task
        in state.tasks.values()
    }

    expected = {
        AgentRole.REQUIREMENTS,
        AgentRole.RESEARCH,
        AgentRole.ARCHITECT,
        AgentRole.CODER,
        AgentRole.TESTER,
        AgentRole.SECURITY,
        AgentRole.CRITIC,
    }

    assert expected.issubset(
        roles
    )


def test_demo_plan_contains_dependencies():

    state = build_demo_state(
        "Build a test service."
    )

    dependent_tasks = [
        task
        for task
        in state.tasks.values()
        if task.dependencies
    ]

    assert (
        len(dependent_tasks)
        >= 1
    )


def test_enum_value_handles_enum():

    assert (
        enum_value(
            AgentRole.CODER
        )
        == "coder"
    )


def test_enum_value_handles_plain_value():

    assert (
        enum_value(
            "plain"
        )
        == "plain"
    )


def test_short_id_returns_first_eight_chars():

    value = (
        "12345678-abcdefgh"
    )

    assert (
        short_id(value)
        == "12345678"
    )


def test_orchestrator_produces_requirements_first():

    state = (
        OrchestratorAgent()
        .create_initial_plan(
            "Build something."
        )
    )

    first_task = state.tasks[
        state.execution_order[0]
    ]

    assert (
        first_task.assigned_agent
        == AgentRole.REQUIREMENTS
    )


def _disable_demo_reporting(
    monkeypatch,
):

    reporting_functions = [
        "print_memory",
        "print_tools",
        "print_governance",
        "print_plan",
        "print_execution_summary",
        "print_task_results",
        "print_artifacts",
        "print_evaluation",
        "print_observability",
        "print_replanning",
        "print_checkpointing",
    ]

    for function_name in reporting_functions:

        monkeypatch.setattr(
            demo_nexus,
            function_name,
            lambda *args, **kwargs: None,
        )


class FakeFreshEngine:

    def __init__(
        self,
        result_state,
    ):

        self.result_state = (
            result_state
        )

        self.run_calls = []

    def run(
        self,
        state,
    ):

        self.run_calls.append(
            state
        )

        return self.result_state


class FakeResumeEngine:

    def __init__(
        self,
        restored_state,
    ):

        self.restored_state = (
            restored_state
        )

        self.restore_calls = []

        self.run_calls = []

    def restore_run(
        self,
        run_id,
        *,
        allow_failed=False,
    ):

        self.restore_calls.append(
            {
                "run_id": run_id,
                "allow_failed": allow_failed,
            }
        )

        return self.restored_state

    def run(
        self,
        state,
    ):

        self.run_calls.append(
            state
        )

        return state


class FakeRecoveryFailureEngine:

    def __init__(
        self,
    ):

        self.run_calls = []

    def restore_run(
        self,
        run_id,
        *,
        allow_failed=False,
    ):

        raise RuntimeError(
            "checkpoint unavailable"
        )

    def run(
        self,
        state,
    ):

        self.run_calls.append(
            state
        )

        return state


class FakeInterruptedEngine:

    def __init__(
        self,
    ):

        self.run_calls = []

    def run(
        self,
        state,
    ):

        self.run_calls.append(
            state
        )

        raise RuntimeError(
            "simulated interruption"
        )


def test_fresh_demo_builds_new_state_and_runs(
    monkeypatch,
):

    _disable_demo_reporting(
        monkeypatch
    )

    state = build_demo_state(
        "Build a fresh service."
    )

    fake_engine = FakeFreshEngine(
        state
    )

    monkeypatch.setattr(
        demo_nexus,
        "build_nexus_engine",
        lambda: fake_engine,
    )

    monkeypatch.setattr(
        demo_nexus,
        "build_demo_state",
        lambda request: state,
    )

    result = demo_nexus.run_demo(
        "Build a fresh service."
    )

    assert result == 0

    assert (
        len(fake_engine.run_calls)
        == 1
    )

    assert (
        fake_engine.run_calls[0]
        is state
    )


def test_resume_demo_restores_failed_run(
    monkeypatch,
):

    _disable_demo_reporting(
        monkeypatch
    )

    state = build_demo_state(
        "Build a recovered service."
    )

    fake_engine = FakeResumeEngine(
        state
    )

    monkeypatch.setattr(
        demo_nexus,
        "build_nexus_engine",
        lambda: fake_engine,
    )

    result = demo_nexus.run_demo(
        DEFAULT_REQUEST,
        resume_run_id="test-run-123",
    )

    assert result == 0

    assert (
        fake_engine.restore_calls
        == [
            {
                "run_id": "test-run-123",
                "allow_failed": True,
            }
        ]
    )


def test_resume_demo_runs_restored_state(
    monkeypatch,
):

    _disable_demo_reporting(
        monkeypatch
    )

    state = build_demo_state(
        "Continue workflow."
    )

    fake_engine = FakeResumeEngine(
        state
    )

    monkeypatch.setattr(
        demo_nexus,
        "build_nexus_engine",
        lambda: fake_engine,
    )

    result = demo_nexus.run_demo(
        DEFAULT_REQUEST,
        resume_run_id="recover-me",
    )

    assert result == 0

    assert (
        len(fake_engine.run_calls)
        == 1
    )

    assert (
        fake_engine.run_calls[0]
        is state
    )


def test_recovery_failure_does_not_execute_workflow(
    monkeypatch,
):

    _disable_demo_reporting(
        monkeypatch
    )

    fake_engine = (
        FakeRecoveryFailureEngine()
    )

    monkeypatch.setattr(
        demo_nexus,
        "build_nexus_engine",
        lambda: fake_engine,
    )

    result = demo_nexus.run_demo(
        DEFAULT_REQUEST,
        resume_run_id="missing-run",
    )

    assert result == 1

    assert (
        fake_engine.run_calls
        == []
    )


def test_interrupted_demo_prints_resume_command(
    monkeypatch,
    capsys,
):

    _disable_demo_reporting(
        monkeypatch
    )

    state = build_demo_state(
        "Interrupt this workflow."
    )

    state.run_id = (
        "resume-test-run"
    )

    fake_engine = (
        FakeInterruptedEngine()
    )

    monkeypatch.setattr(
        demo_nexus,
        "build_nexus_engine",
        lambda: fake_engine,
    )

    monkeypatch.setattr(
        demo_nexus,
        "build_demo_state",
        lambda request: state,
    )

    result = demo_nexus.run_demo(
        "Interrupt this workflow."
    )

    captured = (
        capsys.readouterr().out
    )

    assert result == 1

    assert (
        "simulated interruption"
        in captured
    )

    assert (
        "python scripts/demo_nexus.py "
        "--resume resume-test-run"
        in captured
    )


def test_main_forwards_resume_run_id(
    monkeypatch,
):

    captured = {}

    def fake_run_demo(
        user_request,
        resume_run_id=None,
    ):

        captured["user_request"] = (
            user_request
        )

        captured["resume_run_id"] = (
            resume_run_id
        )

        return 0

    monkeypatch.setattr(
        demo_nexus,
        "run_demo",
        fake_run_demo,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_nexus.py",
            "--resume",
            "cli-run-456",
        ],
    )

    result = demo_nexus.main()

    assert result == 0

    assert (
        captured["resume_run_id"]
        == "cli-run-456"
    )

    assert (
        captured["user_request"]
        == DEFAULT_REQUEST
    )
