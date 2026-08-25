import json

import pytest

from app.agents.requirements import (
    RequirementsAgent,
    RequirementsGenerationError,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    ArtifactType,
)
from app.core.state import NexusState


class FakeLLMClient:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        return json.dumps(
            {
                "objective": "Build a RAG application",
                "functional_requirements": [
                    "Upload PDF documents",
                    "Answer questions from documents",
                ],
                "non_functional_requirements": [
                    "Responses should be fast",
                ],
                "constraints": [
                    "Use free tools",
                ],
                "assumptions": [
                    "Documents contain readable text",
                ],
                "acceptance_criteria": [
                    "Answers include source citations",
                ],
            }
        )


class RepairingFakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        self.calls += 1

        if self.calls == 1:
            return json.dumps(
                {
                    "objective": "Build a RAG application",
                    "functional_requirements": [
                        "Upload PDF documents",
                    ],
                }
            )

        return json.dumps(
            {
                "objective": "Build a RAG application",
                "functional_requirements": [
                    "Upload PDF documents",
                    "Answer questions from documents",
                ],
                "non_functional_requirements": [
                    "Responses should be fast",
                ],
                "constraints": [
                    "Use free tools",
                ],
                "assumptions": [
                    "Documents contain readable text",
                ],
                "acceptance_criteria": [
                    "Answers include source citations",
                ],
            }
        )


class AlwaysInvalidLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        return "{}"


def create_task() -> AgentTask:
    return AgentTask(
        title="Analyze requirements",
        description="Analyze the user request.",
        assigned_agent=AgentRole.REQUIREMENTS,
    )


def test_requirements_agent_returns_artifact():
    state = NexusState(
        user_request="Build a RAG application"
    )

    task = create_task()

    agent = RequirementsAgent(
        llm_client=FakeLLMClient()
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert artifact.type == ArtifactType.REQUIREMENTS
    assert artifact.created_by == AgentRole.REQUIREMENTS

    assert (
        artifact.content["objective"]
        == "Build a RAG application"
    )

    assert len(
        artifact.content["functional_requirements"]
    ) > 0

    assert len(
        artifact.content["non_functional_requirements"]
    ) > 0

    assert len(
        artifact.content["constraints"]
    ) > 0

    assert len(
        artifact.content["assumptions"]
    ) > 0

    assert len(
        artifact.content["acceptance_criteria"]
    ) > 0


def test_requirements_agent_repairs_invalid_output():
    fake_llm = RepairingFakeLLM()

    agent = RequirementsAgent(
        llm_client=fake_llm
    )

    state = NexusState(
        user_request="Build a RAG assistant"
    )

    task = create_task()

    artifact = agent.execute(
        task,
        state,
    )

    assert fake_llm.calls == 2

    assert (
        artifact.metadata["validation_attempts"]
        == 2
    )

    assert len(
        artifact.content["acceptance_criteria"]
    ) > 0


def test_requirements_agent_fails_after_retry_limit():
    agent = RequirementsAgent(
        llm_client=AlwaysInvalidLLM(),
        max_validation_retries=1,
    )

    state = NexusState(
        user_request="Build an application"
    )

    task = create_task()

    with pytest.raises(
        RequirementsGenerationError
    ):
        agent.execute(
            task,
            state,
        )
