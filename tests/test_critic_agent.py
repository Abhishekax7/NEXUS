import json

import pytest

from app.agents.critic import (
    CriticAgent,
    CriticGenerationError,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState
from app.memory.retriever import MemoryRetriever
from app.memory.store import MemoryStore


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.last_user_prompt = None

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
    ):
        self.last_user_prompt = user_prompt

        response = self.responses[
            min(
                self.calls,
                len(self.responses) - 1,
            )
        ]

        self.calls += 1

        return response


def create_artifact(
    artifact_type,
    name,
    content,
    created_by,
):
    return Artifact(
        type=artifact_type,
        name=name,
        content=content,
        created_by=created_by,
    )


def build_complete_state():
    state = NexusState(
        user_request=(
            "Build a secure FastAPI service "
            "with tests."
        )
    )

    requirements = create_artifact(
        ArtifactType.REQUIREMENTS,
        "requirements",
        {
            "objective": (
                "Build a secure tested "
                "FastAPI service."
            ),
            "functional_requirements": [
                "Expose a health endpoint.",
            ],
            "constraints": [
                "Use free tools.",
            ],
            "acceptance_criteria": [
                "Tests must pass.",
            ],
        },
        AgentRole.REQUIREMENTS,
    )

    architecture = create_artifact(
        ArtifactType.ARCHITECTURE,
        "architecture",
        {
            "architecture_style": (
                "Modular FastAPI architecture"
            ),
            "components": [
                {
                    "name": "API",
                    "responsibility": (
                        "Expose HTTP endpoints."
                    ),
                    "technology": "FastAPI",
                }
            ],
            "technology_stack": [
                "Python",
                "FastAPI",
            ],
        },
        AgentRole.ARCHITECT,
    )

    code = create_artifact(
        ArtifactType.CODE,
        "generated_code",
        {
            "files": [
                {
                    "path": "app.py",
                    "content": (
                        "from fastapi import FastAPI\n"
                        "app = FastAPI()\n"
                    ),
                }
            ],
            "test_commands": [
                "pytest -v",
            ],
        },
        AgentRole.CODER,
    )

    test_result = create_artifact(
        ArtifactType.TEST_RESULT,
        "test_result",
        {
            "passed": True,
            "total_commands": 1,
            "passed_commands": 1,
            "failed_commands": 0,
            "results": [
                {
                    "command": "pytest -v",
                    "exit_code": 0,
                    "stdout": "3 passed",
                    "stderr": "",
                    "timed_out": False,
                    "passed": True,
                }
            ],
            "failed_command_names": [],
            "summary": (
                "All generated test "
                "commands passed."
            ),
        },
        AgentRole.TESTER,
    )

    security = create_artifact(
        ArtifactType.SECURITY_REPORT,
        "security_review",
        {
            "passed": True,
            "risk_score": 10,
            "summary": (
                "No significant security "
                "issues found."
            ),
            "findings": [],
        },
        AgentRole.SECURITY,
    )

    artifacts = [
        requirements,
        architecture,
        code,
        test_result,
        security,
    ]

    for artifact in artifacts:
        state.add_artifact(
            artifact
        )

    return state, artifacts


def create_critic_task(
    artifacts,
):
    return AgentTask(
        title="Final quality gate",
        description=(
            "Evaluate the completed "
            "software workflow."
        ),
        assigned_agent=AgentRole.CRITIC,
        input_artifact_ids=[
            artifact.id
            for artifact in artifacts
        ],
    )


def build_retriever(
    tmp_path,
):
    store = MemoryStore(
        db_path=str(
            tmp_path
            / "memory.db"
        )
    )

    return store, MemoryRetriever(
        store
    )


def valid_accept_response():
    return json.dumps(
        {
            "verdict": "accept",
            "quality_score": 94,
            "summary": (
                "The implementation satisfies "
                "the required quality gates."
            ),
            "requirements_satisfied": True,
            "architecture_acceptable": True,
            "implementation_acceptable": True,
            "tests_acceptable": True,
            "security_acceptable": True,
            "issues": [],
            "strengths": [
                "Tests passed successfully.",
                "Security review found no "
                "significant issues.",
            ],
            "required_improvements": [],
            "final_recommendation": (
                "Accept the generated system."
            ),
        }
    )


def valid_revise_response():
    return json.dumps(
        {
            "verdict": "revise",
            "quality_score": 70,
            "summary": (
                "The implementation is usable "
                "but requires improvement."
            ),
            "requirements_satisfied": True,
            "architecture_acceptable": True,
            "implementation_acceptable": True,
            "tests_acceptable": True,
            "security_acceptable": False,
            "issues": [
                {
                    "category": "security",
                    "severity": "high",
                    "description": (
                        "A security concern "
                        "requires remediation."
                    ),
                    "recommendation": (
                        "Resolve the security "
                        "finding before release."
                    ),
                }
            ],
            "strengths": [
                "Core functionality is present.",
            ],
            "required_improvements": [
                "Resolve the security concern.",
            ],
            "final_recommendation": (
                "Revise and run the quality "
                "gate again."
            ),
        }
    )


def test_critic_agent_returns_evaluation():
    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    agent = CriticAgent(
        llm_client=FakeLLM(
            [
                valid_accept_response()
            ]
        )
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.type
        == ArtifactType.EVALUATION
    )

    assert (
        artifact.created_by
        == AgentRole.CRITIC
    )

    assert (
        artifact.content["verdict"]
        == "accept"
    )

    assert (
        artifact.content["quality_score"]
        == 94
    )


def test_critic_agent_records_metadata():
    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    agent = CriticAgent(
        llm_client=FakeLLM(
            [
                valid_accept_response()
            ]
        )
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata[
            "validation_attempts"
        ]
        == 1
    )

    assert (
        artifact.metadata["verdict"]
        == "accept"
    )

    assert (
        artifact.metadata[
            "quality_score"
        ]
        == 94
    )

    assert (
        artifact.metadata[
            "issue_count"
        ]
        == 0
    )


def test_critic_without_memory_preserves_old_behavior():
    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    agent = CriticAgent(
        llm_client=FakeLLM(
            [
                valid_accept_response()
            ]
        )
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata[
            "memory_augmented"
        ]
        is False
    )

    assert (
        artifact.metadata[
            "memory_context_count"
        ]
        == 0
    )


def test_critic_agent_accepts_revise_verdict():
    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    agent = CriticAgent(
        llm_client=FakeLLM(
            [
                valid_revise_response()
            ]
        )
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.content["verdict"]
        == "revise"
    )

    assert (
        artifact.content[
            "security_acceptable"
        ]
        is False
    )

    assert len(
        artifact.content[
            "required_improvements"
        ]
    ) == 1

    assert (
        artifact.metadata[
            "issue_count"
        ]
        == 1
    )


def test_critic_injects_previous_critic_feedback(
    tmp_path,
):
    store, retriever = build_retriever(
        tmp_path
    )

    store.save(
        run_id="old-run",
        memory_type="critic",
        key="previous_quality_gate",
        value={
            "verdict": "revise",
            "quality_score": 68,
            "summary": (
                "FastAPI service had weak "
                "input validation."
            ),
            "required_improvements": [
                "Add strict request validation."
            ],
        },
    )

    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    fake_llm = FakeLLM(
        [
            valid_accept_response()
        ]
    )

    agent = CriticAgent(
        llm_client=fake_llm,
        memory_retriever=retriever,
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata[
            "memory_augmented"
        ]
        is True
    )

    assert (
        artifact.metadata[
            "memory_context_count"
        ]
        >= 1
    )

    assert (
        "previous_quality_gate"
        in fake_llm.last_user_prompt
    )

    assert (
        "Add strict request validation"
        in fake_llm.last_user_prompt
    )


def test_critic_injects_previous_security_feedback(
    tmp_path,
):
    store, retriever = build_retriever(
        tmp_path
    )

    store.save(
        run_id="old-run",
        memory_type="security",
        key="previous_security_review",
        value={
            "risk_score": 75,
            "summary": (
                "FastAPI service exposed "
                "unsafe user input."
            ),
            "findings": [
                "Missing input validation."
            ],
        },
    )

    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    fake_llm = FakeLLM(
        [
            valid_accept_response()
        ]
    )

    agent = CriticAgent(
        llm_client=fake_llm,
        memory_retriever=retriever,
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata[
            "memory_augmented"
        ]
        is True
    )

    assert (
        "unsafe user input"
        in fake_llm.last_user_prompt
    )


def test_critic_injects_previous_failure_or_repair(
    tmp_path,
):
    store, retriever = build_retriever(
        tmp_path
    )

    store.save(
        run_id="old-run",
        memory_type="repair",
        key="fastapi_repair",
        value={
            "root_cause": (
                "FastAPI endpoint validation "
                "was missing."
            ),
            "failure_summary": (
                "API validation tests failed."
            ),
        },
    )

    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    fake_llm = FakeLLM(
        [
            valid_accept_response()
        ]
    )

    agent = CriticAgent(
        llm_client=fake_llm,
        memory_retriever=retriever,
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata[
            "memory_augmented"
        ]
        is True
    )

    assert (
        "fastapi_repair"
        in fake_llm.last_user_prompt
    )


def test_critic_excludes_current_run_memory(
    tmp_path,
):
    store, retriever = build_retriever(
        tmp_path
    )

    state, artifacts = (
        build_complete_state()
    )

    store.save(
        run_id=state.run_id,
        memory_type="critic",
        key="current_run_feedback",
        value={
            "summary": (
                "FastAPI current run feedback"
            ),
            "required_improvements": [
                "Current run only."
            ],
        },
    )

    task = create_critic_task(
        artifacts
    )

    fake_llm = FakeLLM(
        [
            valid_accept_response()
        ]
    )

    agent = CriticAgent(
        llm_client=fake_llm,
        memory_retriever=retriever,
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        "current_run_feedback"
        not in fake_llm.last_user_prompt
    )

    assert (
        artifact.metadata[
            "memory_context_count"
        ]
        == 0
    )


def test_critic_does_not_inject_irrelevant_memory(
    tmp_path,
):
    store, retriever = build_retriever(
        tmp_path
    )

    store.save(
        run_id="old-run",
        memory_type="critic",
        key="space_telescope_feedback",
        value={
            "summary": (
                "Quantum astrophysics "
                "telescope calibration."
            ),
            "required_improvements": [
                "Adjust telescope optics."
            ],
        },
    )

    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    fake_llm = FakeLLM(
        [
            valid_accept_response()
        ]
    )

    agent = CriticAgent(
        llm_client=fake_llm,
        memory_retriever=retriever,
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata[
            "memory_augmented"
        ]
        is False
    )

    assert (
        artifact.metadata[
            "memory_context_count"
        ]
        == 0
    )


def test_critic_respects_memory_limit(
    tmp_path,
):
    store, retriever = build_retriever(
        tmp_path
    )

    for index in range(6):
        store.save(
            run_id=f"old-run-{index}",
            memory_type="critic",
            key=f"fastapi_feedback_{index}",
            value={
                "summary": (
                    "FastAPI service quality "
                    "and security validation."
                ),
                "required_improvements": [
                    "Improve API validation."
                ],
            },
        )

    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    agent = CriticAgent(
        llm_client=FakeLLM(
            [
                valid_accept_response()
            ]
        ),
        memory_retriever=retriever,
        memory_limit=2,
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata[
            "memory_context_count"
        ]
        <= 2
    )


def test_critic_agent_retries_invalid_json():
    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    fake_llm = FakeLLM(
        [
            "not valid json",
            valid_accept_response(),
        ]
    )

    agent = CriticAgent(
        llm_client=fake_llm,
        max_validation_retries=2,
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert fake_llm.calls == 2

    assert (
        artifact.metadata[
            "validation_attempts"
        ]
        == 2
    )


def test_accept_requires_all_quality_gates():
    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    invalid = json.loads(
        valid_accept_response()
    )

    invalid[
        "security_acceptable"
    ] = False

    fake_llm = FakeLLM(
        [
            json.dumps(invalid),
            valid_accept_response(),
        ]
    )

    agent = CriticAgent(
        llm_client=fake_llm,
        max_validation_retries=2,
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert fake_llm.calls == 2

    assert (
        artifact.content[
            "security_acceptable"
        ]
        is True
    )


def test_accept_cannot_require_improvements():
    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    invalid = json.loads(
        valid_accept_response()
    )

    invalid[
        "required_improvements"
    ] = [
        "Fix something."
    ]

    fake_llm = FakeLLM(
        [
            json.dumps(invalid),
            valid_accept_response(),
        ]
    )

    agent = CriticAgent(
        llm_client=fake_llm,
        max_validation_retries=2,
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert fake_llm.calls == 2

    assert (
        artifact.content[
            "required_improvements"
        ]
        == []
    )


def test_revise_requires_improvements():
    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    invalid = json.loads(
        valid_revise_response()
    )

    invalid[
        "required_improvements"
    ] = []

    fake_llm = FakeLLM(
        [
            json.dumps(invalid),
            valid_revise_response(),
        ]
    )

    agent = CriticAgent(
        llm_client=fake_llm,
        max_validation_retries=2,
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert fake_llm.calls == 2

    assert (
        artifact.content["verdict"]
        == "revise"
    )


def test_critic_fails_when_required_artifact_missing():
    state, artifacts = (
        build_complete_state()
    )

    security_artifact = next(
        artifact
        for artifact in artifacts
        if (
            artifact.type
            == ArtifactType.SECURITY_REPORT
        )
    )

    del state.artifacts[
        security_artifact.id
    ]

    remaining = [
        artifact
        for artifact in artifacts
        if artifact.id
        != security_artifact.id
    ]

    task = create_critic_task(
        remaining
    )

    agent = CriticAgent(
        llm_client=FakeLLM(
            [
                valid_accept_response()
            ]
        )
    )

    with pytest.raises(
        CriticGenerationError,
        match="Missing required artifacts",
    ):
        agent.execute(
            task,
            state,
        )


def test_critic_fails_after_retry_limit():
    state, artifacts = (
        build_complete_state()
    )

    task = create_critic_task(
        artifacts
    )

    fake_llm = FakeLLM(
        [
            "invalid",
            "still invalid",
            "also invalid",
        ]
    )

    agent = CriticAgent(
        llm_client=fake_llm,
        max_validation_retries=2,
    )

    with pytest.raises(
        CriticGenerationError,
        match="could not be",
    ):
        agent.execute(
            task,
            state,
        )

    assert fake_llm.calls == 3
