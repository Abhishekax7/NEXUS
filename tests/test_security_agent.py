import json

import pytest

from app.agents.security import (
    SecurityAgent,
    SecurityGenerationError,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
    ):
        response = self.responses[
            min(
                self.calls,
                len(self.responses) - 1,
            )
        ]

        self.calls += 1

        return response


def create_code_artifact():
    return Artifact(
        type=ArtifactType.CODE,
        name="generated_code",
        content={
            "files": [
                {
                    "path": "app.py",
                    "content": (
                        "from fastapi import FastAPI\n"
                        "\n"
                        "app = FastAPI()\n"
                        "\n"
                        "@app.get('/health')\n"
                        "def health():\n"
                        "    return {'status': 'ok'}\n"
                    ),
                },
                {
                    "path": "requirements.txt",
                    "content": "fastapi\nuvicorn\n",
                },
            ],
            "test_commands": [
                "pytest -v",
            ],
        },
        created_by=AgentRole.CODER,
    )


def create_security_task(
    code_artifact,
):
    return AgentTask(
        title="Review security",
        description=(
            "Perform security review "
            "of generated code."
        ),
        assigned_agent=AgentRole.SECURITY,
        input_artifact_ids=[
            code_artifact.id
        ],
    )


def create_valid_response():
    return json.dumps(
        {
            "passed": True,
            "risk_score": 10,
            "summary": (
                "No significant security "
                "issues were identified."
            ),
            "findings": [],
            "reviewed_files": [
                "app.py",
                "requirements.txt",
            ],
            "positive_controls": [
                "No hard-coded credentials detected.",
                "The application exposes only "
                "a minimal health endpoint.",
            ],
            "recommended_actions": [
                "Pin dependency versions.",
                "Add production authentication "
                "before protected endpoints "
                "are introduced.",
            ],
        }
    )


def test_security_agent_returns_security_report():
    code_artifact = create_code_artifact()

    state = NexusState(
        user_request="Build API"
    )

    state.add_artifact(
        code_artifact
    )

    task = create_security_task(
        code_artifact
    )

    agent = SecurityAgent(
        llm_client=FakeLLM(
            [
                create_valid_response()
            ]
        )
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.type
        == ArtifactType.SECURITY_REPORT
    )

    assert (
        artifact.created_by
        == AgentRole.SECURITY
    )

    assert artifact.content["passed"] is True

    assert artifact.content["risk_score"] == 10


def test_security_agent_records_metadata():
    code_artifact = create_code_artifact()

    state = NexusState(
        user_request="Build API"
    )

    state.add_artifact(
        code_artifact
    )

    task = create_security_task(
        code_artifact
    )

    agent = SecurityAgent(
        llm_client=FakeLLM(
            [
                create_valid_response()
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
        artifact.metadata[
            "finding_count"
        ]
        == 0
    )

    assert (
        artifact.metadata[
            "risk_score"
        ]
        == 10
    )


def test_security_agent_accepts_real_finding():
    code_artifact = create_code_artifact()

    state = NexusState(
        user_request="Build API"
    )

    state.add_artifact(
        code_artifact
    )

    task = create_security_task(
        code_artifact
    )

    response = json.dumps(
        {
            "passed": False,
            "risk_score": 75,
            "summary": (
                "A high-risk security issue "
                "was identified."
            ),
            "findings": [
                {
                    "title": (
                        "Missing authentication"
                    ),
                    "severity": "high",
                    "category": (
                        "authentication"
                    ),
                    "affected_files": [
                        "app.py"
                    ],
                    "evidence": (
                        "The application defines "
                        "an endpoint without an "
                        "authentication control."
                    ),
                    "impact": (
                        "Future sensitive endpoints "
                        "could become publicly "
                        "accessible."
                    ),
                    "recommendation": (
                        "Add authentication and "
                        "authorization before "
                        "exposing protected routes."
                    ),
                }
            ],
            "reviewed_files": [
                "app.py",
                "requirements.txt",
            ],
            "positive_controls": [
                "No embedded credentials detected."
            ],
            "recommended_actions": [
                "Introduce authentication."
            ],
        }
    )

    agent = SecurityAgent(
        llm_client=FakeLLM(
            [response]
        )
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert artifact.content["passed"] is False

    assert (
        artifact.content["findings"][0][
            "severity"
        ]
        == "high"
    )

    assert (
        artifact.metadata["finding_count"]
        == 1
    )


def test_security_agent_retries_invalid_json():
    code_artifact = create_code_artifact()

    state = NexusState(
        user_request="Build API"
    )

    state.add_artifact(
        code_artifact
    )

    task = create_security_task(
        code_artifact
    )

    fake_llm = FakeLLM(
        [
            "not valid json",
            create_valid_response(),
        ]
    )

    agent = SecurityAgent(
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


def test_security_agent_retries_invalid_schema():
    code_artifact = create_code_artifact()

    state = NexusState(
        user_request="Build API"
    )

    state.add_artifact(
        code_artifact
    )

    task = create_security_task(
        code_artifact
    )

    invalid_response = json.dumps(
        {
            "passed": True,
            "risk_score": 10,
        }
    )

    fake_llm = FakeLLM(
        [
            invalid_response,
            create_valid_response(),
        ]
    )

    agent = SecurityAgent(
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


def test_security_agent_rejects_unknown_reviewed_file():
    code_artifact = create_code_artifact()

    state = NexusState(
        user_request="Build API"
    )

    state.add_artifact(
        code_artifact
    )

    task = create_security_task(
        code_artifact
    )

    invalid_response = json.dumps(
        {
            "passed": True,
            "risk_score": 5,
            "summary": "Review complete.",
            "findings": [],
            "reviewed_files": [
                "invented.py"
            ],
            "positive_controls": [
                "Input validation exists."
            ],
            "recommended_actions": [
                "Continue security testing."
            ],
        }
    )

    agent = SecurityAgent(
        llm_client=FakeLLM(
            [
                invalid_response,
                invalid_response,
            ]
        ),
        max_validation_retries=1,
    )

    with pytest.raises(
        SecurityGenerationError,
        match="could not be validated",
    ):
        agent.execute(
            task,
            state,
        )


def test_security_agent_rejects_unknown_affected_file():
    code_artifact = create_code_artifact()

    state = NexusState(
        user_request="Build API"
    )

    state.add_artifact(
        code_artifact
    )

    task = create_security_task(
        code_artifact
    )

    invalid_response = json.dumps(
        {
            "passed": False,
            "risk_score": 80,
            "summary": "Security issue found.",
            "findings": [
                {
                    "title": "Secret exposure",
                    "severity": "critical",
                    "category": "secrets",
                    "affected_files": [
                        "secret.py"
                    ],
                    "evidence": (
                        "A credential appears "
                        "to be embedded."
                    ),
                    "impact": (
                        "Credential compromise."
                    ),
                    "recommendation": (
                        "Move secrets into "
                        "environment variables."
                    ),
                }
            ],
            "reviewed_files": [
                "app.py"
            ],
            "positive_controls": [
                "Minimal API surface."
            ],
            "recommended_actions": [
                "Remove exposed secrets."
            ],
        }
    )

    agent = SecurityAgent(
        llm_client=FakeLLM(
            [
                invalid_response,
                invalid_response,
            ]
        ),
        max_validation_retries=1,
    )

    with pytest.raises(
        SecurityGenerationError,
        match="could not be validated",
    ):
        agent.execute(
            task,
            state,
        )


def test_security_agent_fails_without_code_artifact():
    state = NexusState(
        user_request="Build API"
    )

    task = AgentTask(
        title="Review security",
        description="Review generated code.",
        assigned_agent=AgentRole.SECURITY,
    )

    agent = SecurityAgent(
        llm_client=FakeLLM(
            [
                create_valid_response()
            ]
        )
    )

    with pytest.raises(
        SecurityGenerationError,
        match="CODE artifact not found",
    ):
        agent.execute(
            task,
            state,
        )


def test_security_agent_fails_after_retry_limit():
    code_artifact = create_code_artifact()

    state = NexusState(
        user_request="Build API"
    )

    state.add_artifact(
        code_artifact
    )

    task = create_security_task(
        code_artifact
    )

    fake_llm = FakeLLM(
        [
            "invalid",
            "still invalid",
            "also invalid",
        ]
    )

    agent = SecurityAgent(
        llm_client=fake_llm,
        max_validation_retries=2,
    )

    with pytest.raises(
        SecurityGenerationError,
        match="could not be validated",
    ):
        agent.execute(
            task,
            state,
        )

    assert fake_llm.calls == 3
