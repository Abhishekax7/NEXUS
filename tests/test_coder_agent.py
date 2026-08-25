import json

import pytest

from app.agents.coder import (
    CoderAgent,
    CodeGenerationError,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState


class FakeCoderLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        return json.dumps(
            {
                "project_name": "rag_assistant",
                "summary": "A simple modular RAG assistant.",
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
                        "purpose": "Application entry point.",
                    },
                    {
                        "path": "tests/test_app.py",
                        "content": (
                            "def test_placeholder():\n"
                            "    assert True\n"
                        ),
                        "purpose": "Initial test suite.",
                    },
                    {
                        "path": "requirements.txt",
                        "content": (
                            "fastapi\n"
                            "uvicorn\n"
                            "pytest\n"
                        ),
                        "purpose": "Project dependencies.",
                    },
                ],
                "dependencies": [
                    "fastapi",
                    "uvicorn",
                    "pytest",
                ],
                "run_commands": [
                    "uvicorn app:app --reload",
                ],
                "test_commands": [
                    "pytest -v",
                ],
                "implementation_notes": [
                    "Uses a modular FastAPI foundation.",
                    "Future retrieval components can be added separately.",
                ],
            }
        )


class CapturingCoderLLM:
    def __init__(self):
        self.last_user_prompt = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        self.last_user_prompt = user_prompt

        return FakeCoderLLM().generate(
            system_prompt,
            user_prompt,
            json_mode,
        )


class RepairingCoderLLM:
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
                    "project_name": "broken_project",
                    "summary": "Incomplete output",
                    "files": [],
                }
            )

        return FakeCoderLLM().generate(
            system_prompt,
            user_prompt,
            json_mode,
        )


class PathTraversalCoderLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        data = json.loads(
            FakeCoderLLM().generate(
                system_prompt,
                user_prompt,
                json_mode,
            )
        )

        data["files"][0]["path"] = "../secret.txt"

        return json.dumps(data)


class AbsolutePathCoderLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        data = json.loads(
            FakeCoderLLM().generate(
                system_prompt,
                user_prompt,
                json_mode,
            )
        )

        data["files"][0]["path"] = "/tmp/app.py"

        return json.dumps(data)


class DuplicatePathCoderLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        data = json.loads(
            FakeCoderLLM().generate(
                system_prompt,
                user_prompt,
                json_mode,
            )
        )

        data["files"][1]["path"] = "app.py"

        return json.dumps(data)


class AlwaysInvalidCoderLLM:
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
                "Which technologies should be used?"
            ),
            "findings": [
                "FastAPI works well for Python APIs.",
                "FAISS supports local vector retrieval.",
            ],
            "recommended_technologies": [
                "FastAPI",
                "FAISS",
            ],
            "tradeoffs": [
                "Local inference increases hardware requirements.",
            ],
            "risks": [
                "Large models may have high memory requirements.",
            ],
            "sources": [
                {
                    "title": "FastAPI Documentation",
                    "url": "https://fastapi.tiangolo.com/",
                    "summary": "Official FastAPI documentation.",
                }
            ],
        },
        created_by=AgentRole.RESEARCH,
    )


def create_architecture_artifact() -> Artifact:
    return Artifact(
        type=ArtifactType.ARCHITECTURE,
        name="architecture_design",
        content={
            "architecture_style": "Layered modular architecture",
            "components": [
                {
                    "name": "API Layer",
                    "responsibility": "Expose application endpoints",
                    "technology": "FastAPI",
                },
                {
                    "name": "Retrieval Layer",
                    "responsibility": "Retrieve document chunks",
                    "technology": "FAISS",
                },
            ],
            "data_flow": [
                "Request enters API layer",
                "Retrieval layer finds context",
                "Generation layer creates response",
            ],
            "technology_stack": [
                "Python",
                "FastAPI",
                "FAISS",
            ],
            "interfaces": [
                "REST API",
            ],
            "security_considerations": [
                "Validate uploaded files",
            ],
            "design_decisions": [
                "Use modular architecture",
            ],
            "research_influences": [
                "FastAPI and FAISS were selected based on research.",
            ],
        },
        created_by=AgentRole.ARCHITECT,
    )


def build_state_and_task():
    state = NexusState(
        user_request="Build a RAG application"
    )

    requirements = create_requirements_artifact()
    research = create_research_artifact()
    architecture = create_architecture_artifact()

    state.add_artifact(requirements)
    state.add_artifact(research)
    state.add_artifact(architecture)

    task = AgentTask(
        title="Implement solution",
        description="Generate application code.",
        assigned_agent=AgentRole.CODER,
        input_artifact_ids=[
            architecture.id,
        ],
    )

    return state, task


def test_coder_agent_returns_code_artifact():
    state, task = build_state_and_task()

    agent = CoderAgent(
        llm_client=FakeCoderLLM()
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert artifact.type == ArtifactType.CODE
    assert artifact.created_by == AgentRole.CODER

    assert (
        artifact.content["project_name"]
        == "rag_assistant"
    )

    assert len(
        artifact.content["files"]
    ) > 0

    assert len(
        artifact.content["dependencies"]
    ) > 0

    assert len(
        artifact.content["run_commands"]
    ) > 0


def test_coder_agent_is_grounded_in_architecture():
    state, task = build_state_and_task()

    fake_llm = CapturingCoderLLM()

    agent = CoderAgent(
        llm_client=fake_llm
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata["grounded_in_architecture"]
        is True
    )

    assert fake_llm.last_user_prompt is not None

    assert (
        "Layered modular architecture"
        in fake_llm.last_user_prompt
    )

    assert (
        "FAISS"
        in fake_llm.last_user_prompt
    )


def test_coder_agent_receives_requirements_and_research():
    state, task = build_state_and_task()

    fake_llm = CapturingCoderLLM()

    agent = CoderAgent(
        llm_client=fake_llm
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata["requirements_available"]
        is True
    )

    assert (
        artifact.metadata["research_available"]
        is True
    )

    assert (
        "Use free tools"
        in fake_llm.last_user_prompt
    )

    assert (
        "FastAPI works well for Python APIs."
        in fake_llm.last_user_prompt
    )


def test_coder_agent_repairs_invalid_output():
    fake_llm = RepairingCoderLLM()

    state, task = build_state_and_task()

    agent = CoderAgent(
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
        artifact.content["files"]
    ) > 0


def test_coder_agent_rejects_directory_traversal():
    state, task = build_state_and_task()

    agent = CoderAgent(
        llm_client=PathTraversalCoderLLM(),
        max_validation_retries=0,
    )

    with pytest.raises(
        CodeGenerationError,
        match="Parent directory traversal",
    ):
        agent.execute(
            task,
            state,
        )


def test_coder_agent_rejects_absolute_paths():
    state, task = build_state_and_task()

    agent = CoderAgent(
        llm_client=AbsolutePathCoderLLM(),
        max_validation_retries=0,
    )

    with pytest.raises(
        CodeGenerationError,
        match="Absolute file path",
    ):
        agent.execute(
            task,
            state,
        )


def test_coder_agent_rejects_duplicate_paths():
    state, task = build_state_and_task()

    agent = CoderAgent(
        llm_client=DuplicatePathCoderLLM(),
        max_validation_retries=0,
    )

    with pytest.raises(
        CodeGenerationError,
        match="Duplicate generated file path",
    ):
        agent.execute(
            task,
            state,
        )


def test_coder_agent_fails_after_retry_limit():
    state, task = build_state_and_task()

    agent = CoderAgent(
        llm_client=AlwaysInvalidCoderLLM(),
        max_validation_retries=1,
    )

    with pytest.raises(
        CodeGenerationError
    ):
        agent.execute(
            task,
            state,
        )


def test_coder_agent_fails_without_architecture():
    state = NexusState(
        user_request="Build an application"
    )

    task = AgentTask(
        title="Implement solution",
        description="Generate code.",
        assigned_agent=AgentRole.CODER,
    )

    agent = CoderAgent(
        llm_client=FakeCoderLLM()
    )

    with pytest.raises(
        CodeGenerationError,
        match="Architecture artifact not found",
    ):
        agent.execute(
            task,
            state,
        )
