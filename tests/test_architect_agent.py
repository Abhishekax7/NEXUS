import json

import pytest

from app.agents.architect import (
    ArchitectAgent,
    ArchitectureGenerationError,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState


class FakeArchitectLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        return json.dumps(
            {
                "architecture_style": "Layered modular architecture",
                "components": [
                    {
                        "name": "API Layer",
                        "responsibility": "Expose application endpoints",
                        "technology": "FastAPI",
                    },
                    {
                        "name": "Retrieval Layer",
                        "responsibility": "Retrieve relevant document chunks",
                        "technology": "FAISS",
                    },
                ],
                "data_flow": [
                    "User request enters through the API layer",
                    "Relevant chunks are retrieved",
                    "Retrieved context is passed to the generation layer",
                ],
                "technology_stack": [
                    "Python",
                    "FastAPI",
                    "FAISS",
                    "Groq",
                ],
                "interfaces": [
                    "REST API between client and backend",
                    "Internal interface between retrieval and generation",
                ],
                "security_considerations": [
                    "Validate uploaded files",
                    "Protect API credentials",
                ],
                "design_decisions": [
                    "Use modular components for maintainability",
                    "Use FAISS for free local vector retrieval",
                ],
            }
        )


class RepairingArchitectLLM:
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
                    "architecture_style": "Layered architecture",
                    "components": [],
                }
            )

        return json.dumps(
            {
                "architecture_style": "Layered modular architecture",
                "components": [
                    {
                        "name": "Backend",
                        "responsibility": "Handle application logic",
                        "technology": "FastAPI",
                    }
                ],
                "data_flow": [
                    "Request enters backend and is processed"
                ],
                "technology_stack": [
                    "Python",
                    "FastAPI",
                ],
                "interfaces": [
                    "REST API"
                ],
                "security_considerations": [
                    "Validate inputs"
                ],
                "design_decisions": [
                    "Use a modular backend architecture"
                ],
            }
        )


class AlwaysInvalidArchitectLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        return "{}"


def create_requirements_artifact() -> Artifact:
    return Artifact(
        type=ArtifactType.REQUIREMENTS,
        name="requirements_analysis",
        content={
            "objective": "Build a RAG application",
            "functional_requirements": [
                "Upload PDFs",
                "Answer questions with citations",
            ],
            "non_functional_requirements": [
                "Fast responses"
            ],
            "constraints": [
                "Use free tools"
            ],
            "assumptions": [
                "PDFs contain readable text"
            ],
            "acceptance_criteria": [
                "Answers contain citations"
            ],
        },
        created_by=AgentRole.REQUIREMENTS,
    )


def create_architect_task(
    requirements_artifact_id: str,
) -> AgentTask:
    return AgentTask(
        title="Design architecture",
        description="Design the software architecture.",
        assigned_agent=AgentRole.ARCHITECT,
        input_artifact_ids=[
            requirements_artifact_id
        ],
    )


def test_architect_agent_returns_architecture_artifact():
    state = NexusState(
        user_request="Build a RAG application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = create_architect_task(
        requirements.id
    )

    agent = ArchitectAgent(
        llm_client=FakeArchitectLLM()
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert artifact.type == ArtifactType.ARCHITECTURE
    assert artifact.created_by == AgentRole.ARCHITECT

    assert (
        artifact.content["architecture_style"]
        == "Layered modular architecture"
    )

    assert len(
        artifact.content["components"]
    ) > 0

    assert len(
        artifact.content["technology_stack"]
    ) > 0

    assert len(
        artifact.content["security_considerations"]
    ) > 0


def test_architect_agent_reads_requirements_artifact():
    state = NexusState(
        user_request="Build a RAG application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = create_architect_task(
        requirements.id
    )

    agent = ArchitectAgent(
        llm_client=FakeArchitectLLM()
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert artifact.type == ArtifactType.ARCHITECTURE


def test_architect_agent_repairs_invalid_output():
    fake_llm = RepairingArchitectLLM()

    state = NexusState(
        user_request="Build a RAG application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = create_architect_task(
        requirements.id
    )

    agent = ArchitectAgent(
        llm_client=fake_llm
    )

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
        artifact.content["components"]
    ) > 0


def test_architect_agent_fails_after_retry_limit():
    state = NexusState(
        user_request="Build an application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = create_architect_task(
        requirements.id
    )

    agent = ArchitectAgent(
        llm_client=AlwaysInvalidArchitectLLM(),
        max_validation_retries=1,
    )

    with pytest.raises(
        ArchitectureGenerationError
    ):
        agent.execute(
            task,
            state,
        )


def test_architect_agent_fails_without_requirements():
    state = NexusState(
        user_request="Build an application"
    )

    task = AgentTask(
        title="Design architecture",
        description="Design the system.",
        assigned_agent=AgentRole.ARCHITECT,
    )

    agent = ArchitectAgent(
        llm_client=FakeArchitectLLM()
    )

    with pytest.raises(
        ArchitectureGenerationError,
        match="Requirements artifact not found",
    ):
        agent.execute(
            task,
            state,
        )
