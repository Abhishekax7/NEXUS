import json

import pytest

from app.agents.research import (
    ResearchAgent,
    ResearchGenerationError,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState
from app.tools.web_search import SearchResult


class FakeSearchTool:
    def search(
        self,
        query: str,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                title="FastAPI Documentation",
                url="https://fastapi.tiangolo.com/",
                snippet="FastAPI is a modern Python web framework.",
            ),
            SearchResult(
                title="FAISS Documentation",
                url="https://faiss.ai/",
                snippet="FAISS provides efficient vector similarity search.",
            ),
        ]


class FakeResearchLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        return json.dumps(
            {
                "research_question": (
                    "What architecture and technologies "
                    "should be used for the requested system?"
                ),
                "findings": [
                    "FastAPI is suitable for an async API layer.",
                    "FAISS supports free local vector retrieval.",
                ],
                "recommended_technologies": [
                    "FastAPI",
                    "FAISS",
                    "Python",
                ],
                "tradeoffs": [
                    "FAISS is lightweight but lacks some database features.",
                ],
                "risks": [
                    "Local model inference may require significant compute.",
                ],
                "sources": [
                    {
                        "title": "FastAPI Documentation",
                        "url": "https://fastapi.tiangolo.com/",
                        "summary": (
                            "Official FastAPI documentation "
                            "for building Python APIs."
                        ),
                    },
                    {
                        "title": "FAISS Documentation",
                        "url": "https://faiss.ai/",
                        "summary": (
                            "Documentation for efficient "
                            "vector similarity search."
                        ),
                    },
                ],
            }
        )


class RepairingResearchLLM:
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
                    "research_question": "Research system architecture",
                    "findings": [
                        "Use a Python backend"
                    ],
                }
            )

        return json.dumps(
            {
                "research_question": "Research system architecture",
                "findings": [
                    "Use FastAPI for the backend."
                ],
                "recommended_technologies": [
                    "FastAPI",
                    "FAISS",
                ],
                "tradeoffs": [
                    "Local components reduce cost but increase setup complexity."
                ],
                "risks": [
                    "Local inference performance depends on hardware."
                ],
                "sources": [
                    {
                        "title": "FastAPI Documentation",
                        "url": "https://fastapi.tiangolo.com/",
                        "summary": "Official FastAPI documentation.",
                    }
                ],
            }
        )


class HallucinatingResearchLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        return json.dumps(
            {
                "research_question": "Research architecture",
                "findings": [
                    "Use modern Python tools."
                ],
                "recommended_technologies": [
                    "FastAPI"
                ],
                "tradeoffs": [
                    "Some complexity is introduced."
                ],
                "risks": [
                    "Deployment complexity."
                ],
                "sources": [
                    {
                        "title": "Invented Source",
                        "url": "https://invented-example.invalid/",
                        "summary": "Fabricated source.",
                    }
                ],
            }
        )


class AlwaysInvalidResearchLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        return "{}"


class EmptySearchTool:
    def search(
        self,
        query: str,
    ) -> list[SearchResult]:
        return []


def create_requirements_artifact() -> Artifact:
    return Artifact(
        type=ArtifactType.REQUIREMENTS,
        name="requirements_analysis",
        content={
            "objective": "Build a RAG application",
            "functional_requirements": [
                "Upload PDFs",
                "Answer questions",
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


def create_research_task(
    requirements_artifact_id: str,
) -> AgentTask:
    return AgentTask(
        title="Research solution",
        description="Research technologies and approaches.",
        assigned_agent=AgentRole.RESEARCH,
        input_artifact_ids=[
            requirements_artifact_id
        ],
    )


def test_research_agent_returns_research_artifact():
    state = NexusState(
        user_request="Build a RAG application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = create_research_task(
        requirements.id
    )

    agent = ResearchAgent(
        llm_client=FakeResearchLLM(),
        search_tool=FakeSearchTool(),
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert artifact.type == ArtifactType.RESEARCH
    assert artifact.created_by == AgentRole.RESEARCH

    assert len(
        artifact.content["findings"]
    ) > 0

    assert len(
        artifact.content["recommended_technologies"]
    ) > 0

    assert len(
        artifact.content["sources"]
    ) > 0


def test_research_agent_uses_supported_sources_only():
    state = NexusState(
        user_request="Build a RAG application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = create_research_task(
        requirements.id
    )

    agent = ResearchAgent(
        llm_client=FakeResearchLLM(),
        search_tool=FakeSearchTool(),
    )

    artifact = agent.execute(
        task,
        state,
    )

    allowed_urls = {
        "https://fastapi.tiangolo.com/",
        "https://faiss.ai/",
    }

    for source in artifact.content["sources"]:
        assert source["url"] in allowed_urls


def test_research_agent_repairs_invalid_output():
    fake_llm = RepairingResearchLLM()

    state = NexusState(
        user_request="Build a RAG application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = create_research_task(
        requirements.id
    )

    agent = ResearchAgent(
        llm_client=fake_llm,
        search_tool=FakeSearchTool(),
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


def test_research_agent_rejects_hallucinated_url():
    state = NexusState(
        user_request="Build a RAG application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = create_research_task(
        requirements.id
    )

    agent = ResearchAgent(
        llm_client=HallucinatingResearchLLM(),
        search_tool=FakeSearchTool(),
        max_validation_retries=0,
    )

    with pytest.raises(
        ResearchGenerationError
    ):
        agent.execute(
            task,
            state,
        )


def test_research_agent_fails_when_search_returns_nothing():
    state = NexusState(
        user_request="Build an application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = create_research_task(
        requirements.id
    )

    agent = ResearchAgent(
        llm_client=FakeResearchLLM(),
        search_tool=EmptySearchTool(),
    )

    with pytest.raises(
        ResearchGenerationError,
        match="Web research returned no results",
    ):
        agent.execute(
            task,
            state,
        )


def test_research_agent_fails_after_retry_limit():
    state = NexusState(
        user_request="Build an application"
    )

    requirements = create_requirements_artifact()
    state.add_artifact(requirements)

    task = create_research_task(
        requirements.id
    )

    agent = ResearchAgent(
        llm_client=AlwaysInvalidResearchLLM(),
        search_tool=FakeSearchTool(),
        max_validation_retries=1,
    )

    with pytest.raises(
        ResearchGenerationError
    ):
        agent.execute(
            task,
            state,
        )


def test_research_agent_fails_without_requirements():
    state = NexusState(
        user_request="Build an application"
    )

    task = AgentTask(
        title="Research solution",
        description="Research technical approaches.",
        assigned_agent=AgentRole.RESEARCH,
    )

    agent = ResearchAgent(
        llm_client=FakeResearchLLM(),
        search_tool=FakeSearchTool(),
    )

    with pytest.raises(
        ResearchGenerationError,
        match="Requirements artifact not found",
    ):
        agent.execute(
            task,
            state,
        )
