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
                    "Retrieved context is passed to generation",
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
                    "Protect credentials",
                ],
                "design_decisions": [
                    "Use modular components for maintainability",
                    "Use FAISS for free local vector retrieval",
                ],
                "research_influences": [
                    "FastAPI was selected based on the supplied research.",
                    "FAISS was selected because the research identified it as a free local vector-search option.",
                ],
            }
        )


class CapturingArchitectLLM:
    def __init__(self):
        self.last_user_prompt = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        self.last_user_prompt = user_prompt

        return FakeArchitectLLM().generate(
            system_prompt,
            user_prompt,
            json_mode,
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

        return FakeArchitectLLM().generate(
            system_prompt,
            user_prompt,
            json_mode,
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
                "Fast responses",
            ],
            "constraints": [
                "Use free tools",
            ],
            "assumptions": [
                "PDFs contain readable text",
            ],
            "acceptance_criteria": [
                "Answers contain citations",
            ],
        },
        created_by=AgentRole.REQUIREMENTS,
    )


def create_research_artifact() -> Artifact:
    return Artifact(
        type=ArtifactType.RESEARCH,
        name="technical_research",
        content={
            "research_question": (
                "What technologies should be used?"
            ),
            "findings": [
                "FastAPI is suitable for async APIs.",
                "FAISS supports free local vector retrieval.",
            ],
            "recommended_technologies": [
                "FastAPI",
                "FAISS",
            ],
            "tradeoffs": [
                "FAISS is lightweight but has fewer database features.",
            ],
            "risks": [
                "Local inference performance depends on hardware.",
            ],
            "sources": [
                {
                    "title": "FastAPI Documentation",
                    "url": "https://fastapi.tiangolo.com/",
                    "summary": "Official FastAPI documentation.",
                },
                {
                    "title": "FAISS Documentation",
                    "url": "https://faiss.ai/",
                    "summary": "FAISS vector-search documentation.",
                },
            ],
        },
        created_by=AgentRole.RESEARCH,
    )


def create_architect_task(
    requirements_artifact_id: str,
    research_artifact_id: str,
) -> AgentTask:
    return AgentTask(
        title="Design architecture",
        description="Design the software architecture.",
        assigned_agent=AgentRole.ARCHITECT,
        input_artifact_ids=[
            requirements_artifact_id,
            research_artifact_id,
        ],
    )


def build_state_and_task():
    state = NexusState(
        user_request="Build a RAG application"
    )

    requirements = create_requirements_artifact()
    research = create_research_artifact()

    state.add_artifact(requirements)
    state.add_artifact(research)

    task = create_architect_task(
        requirements.id,
        research.id,
    )

    return state, task


def test_architect_agent_returns_architecture_artifact():
    state, task = build_state_and_task()

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
        artifact.content["research_influences"]
    ) > 0


def test_architect_agent_is_grounded_in_both_inputs():
    state, task = build_state_and_task()

    agent = ArchitectAgent(
        llm_client=FakeArchitectLLM()
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata["grounded_in_requirements"]
        is True
    )

    assert (
        artifact.metadata["grounded_in_research"]
        is True
    )


def test_research_reaches_architect_prompt():
    state, task = build_state_and_task()

    fake_llm = CapturingArchitectLLM()

    agent = ArchitectAgent(
        llm_client=fake_llm
    )

    agent.execute(
        task,
        state,
    )

    assert fake_llm.last_user_prompt is not None

    assert (
        "FastAPI is suitable for async APIs."
        in fake_llm.last_user_prompt
    )

    assert (
        "FAISS supports free local vector retrieval."
        in fake_llm.last_user_prompt
    )

    assert (
        "Use free tools"
        in fake_llm.last_user_prompt
    )


def test_architect_agent_repairs_invalid_output():
    fake_llm = RepairingArchitectLLM()

    state, task = build_state_and_task()

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
        artifact.content["research_influences"]
    ) > 0


def test_architect_agent_fails_after_retry_limit():
    state, task = build_state_and_task()

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

    research = create_research_artifact()
    state.add_artifact(research)

    task = AgentTask(
        title="Design architecture",
        description="Design the system.",
        assigned_agent=AgentRole.ARCHITECT,
        input_artifact_ids=[
            research.id
        ],
    )

    agent = ArchitectAgent(
        llm_client=FakeArchitectLLM()
    )

    with pytest.raises(
        ArchitectureGenerationError,
        match="requirements artifact not found",
    ):
        agent.execute(
            task,
            state,
        )


def test_architect_agent_fails_without_research():
    state = NexusState(
        user_request="Build an application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = AgentTask(
        title="Design architecture",
        description="Design the system.",
        assigned_agent=AgentRole.ARCHITECT,
        input_artifact_ids=[
            requirements.id
        ],
    )

    agent = ArchitectAgent(
        llm_client=FakeArchitectLLM()
    )

    with pytest.raises(
        ArchitectureGenerationError,
        match="research artifact not found",
    ):
        agent.execute(
            task,
            state,
        )
