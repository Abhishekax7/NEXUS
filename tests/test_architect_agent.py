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
from app.memory.retriever import MemoryRetriever
from app.memory.store import MemoryStore


class FakeArchitectLLM:
    def __init__(self):
        self.last_user_prompt = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        max_tokens=None,
    ) -> str:
        self.last_user_prompt = user_prompt

        return json.dumps(
            {
                "architecture_style":
                    "Layered modular architecture",

                "components": [
                    {
                        "name":
                            "API Layer",

                        "responsibility":
                            "Expose application endpoints",

                        "technology":
                            "FastAPI",
                    },
                    {
                        "name":
                            "Retrieval Layer",

                        "responsibility":
                            "Retrieve relevant "
                            "document chunks",

                        "technology":
                            "FAISS",
                    },
                ],

                "data_flow": [
                    "User request enters through "
                    "the API layer",

                    "Relevant chunks are retrieved",

                    "Retrieved context is passed "
                    "to generation",
                ],

                "technology_stack": [
                    "Python",
                    "FastAPI",
                    "FAISS",
                    "Groq",
                ],

                "interfaces": [
                    "REST API between client "
                    "and backend",

                    "Internal interface between "
                    "retrieval and generation",
                ],

                "security_considerations": [
                    "Validate uploaded files",
                    "Protect credentials",
                ],

                "design_decisions": [
                    "Use modular components for "
                    "maintainability",

                    "Use FAISS for free local "
                    "vector retrieval",
                ],

                "research_influences": [
                    "FastAPI was selected based "
                    "on the supplied research.",

                    "FAISS was selected because "
                    "the research identified it "
                    "as a free local "
                    "vector-search option.",
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
        max_tokens=None,
    ) -> str:
        self.calls += 1

        if self.calls == 1:
            return json.dumps(
                {
                    "architecture_style":
                        "Layered architecture",

                    "components": [],
                }
            )

        return FakeArchitectLLM().generate(
            system_prompt,
            user_prompt,
            json_mode,
            max_tokens,
        )


class AlwaysInvalidArchitectLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        max_tokens=None,
    ) -> str:
        return "{}"


def create_requirements_artifact() -> Artifact:
    return Artifact(
        type=ArtifactType.REQUIREMENTS,
        name="requirements_analysis",
        content={
            "objective":
                "Build a RAG application",

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
            "research_question":
                "What technologies should be used?",

            "findings": [
                "FastAPI is suitable for async APIs.",

                "FAISS supports free local "
                "vector retrieval.",
            ],

            "recommended_technologies": [
                "FastAPI",
                "FAISS",
            ],

            "tradeoffs": [
                "FAISS is lightweight but has "
                "fewer database features.",
            ],

            "risks": [
                "Local inference performance "
                "depends on hardware.",
            ],

            "sources": [
                {
                    "title":
                        "FastAPI Documentation",

                    "url":
                        "https://fastapi.tiangolo.com/",

                    "summary":
                        "Official FastAPI documentation.",
                },
                {
                    "title":
                        "FAISS Documentation",

                    "url":
                        "https://faiss.ai/",

                    "summary":
                        "FAISS vector-search documentation.",
                },
            ],
        },
        created_by=AgentRole.RESEARCH,
    )


def build_state_and_task():
    state = NexusState(
        user_request="Build a RAG application"
    )

    requirements = (
        create_requirements_artifact()
    )

    research = (
        create_research_artifact()
    )

    state.add_artifact(
        requirements
    )

    state.add_artifact(
        research
    )

    task = AgentTask(
        title="Design architecture",
        description=(
            "Design the software architecture."
        ),
        assigned_agent=(
            AgentRole.ARCHITECT
        ),
        input_artifact_ids=[
            requirements.id,
            research.id,
        ],
    )

    return state, task


def build_retriever(
    tmp_path,
):
    store = MemoryStore(
        db_path=str(
            tmp_path
            / "memory.db"
        )
    )

    return (
        store,
        MemoryRetriever(
            store
        ),
    )


def test_architect_agent_returns_architecture_artifact():
    state, task = (
        build_state_and_task()
    )

    agent = ArchitectAgent(
        llm_client=(
            FakeArchitectLLM()
        )
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.type
        == ArtifactType.ARCHITECTURE
    )

    assert (
        artifact.created_by
        == AgentRole.ARCHITECT
    )

    assert (
        artifact.content[
            "architecture_style"
        ]
        == "Layered modular architecture"
    )

    assert len(
        artifact.content[
            "components"
        ]
    ) > 0

    assert len(
        artifact.content[
            "technology_stack"
        ]
    ) > 0

    assert len(
        artifact.content[
            "research_influences"
        ]
    ) > 0


def test_architect_agent_is_grounded_in_both_inputs():
    state, task = (
        build_state_and_task()
    )

    agent = ArchitectAgent(
        llm_client=(
            FakeArchitectLLM()
        )
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata[
            "grounded_in_requirements"
        ]
        is True
    )

    assert (
        artifact.metadata[
            "grounded_in_research"
        ]
        is True
    )


def test_architect_without_memory_preserves_old_behavior():
    state, task = (
        build_state_and_task()
    )

    agent = ArchitectAgent(
        llm_client=(
            FakeArchitectLLM()
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


def test_research_reaches_architect_prompt():
    state, task = (
        build_state_and_task()
    )

    fake_llm = (
        FakeArchitectLLM()
    )

    agent = ArchitectAgent(
        llm_client=fake_llm
    )

    agent.execute(
        task,
        state,
    )

    assert (
        fake_llm.last_user_prompt
        is not None
    )

    assert (
        "FastAPI is suitable for async APIs."
        in fake_llm.last_user_prompt
    )

    assert (
        "FAISS supports free local "
        "vector retrieval."
        in fake_llm.last_user_prompt
    )

    assert (
        "Use free tools"
        in fake_llm.last_user_prompt
    )


def test_architect_injects_relevant_architecture_memory(
    tmp_path,
):
    store, retriever = (
        build_retriever(
            tmp_path
        )
    )

    store.save(
        run_id="old-run",
        memory_type="artifact",
        key="architecture",
        value={
            "artifact_id":
                "old-architecture",

            "name":
                "architecture_design",

            "type":
                "architecture",

            "created_by":
                "architect",

            "content": {
                "architecture_style":
                    "Modular FastAPI architecture",

                "technology_stack": [
                    "FastAPI",
                    "FAISS",
                ],

                "design_decisions": [
                    "Separate retrieval "
                    "from API layer."
                ],
            },
        },
    )

    state, task = (
        build_state_and_task()
    )

    fake_llm = (
        FakeArchitectLLM()
    )

    agent = ArchitectAgent(
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
        "Modular FastAPI architecture"
        in fake_llm.last_user_prompt
    )


def test_architect_injects_relevant_critic_feedback(
    tmp_path,
):
    store, retriever = (
        build_retriever(
            tmp_path
        )
    )

    store.save(
        run_id="old-run",
        memory_type="critic",
        key="old_quality_gate",
        value={
            "verdict":
                "revise",

            "quality_score":
                70,

            "summary": (
                "FastAPI design lacked "
                "input validation."
            ),

            "required_improvements": [
                "Add strict API input "
                "validation."
            ],
        },
    )

    state, task = (
        build_state_and_task()
    )

    fake_llm = (
        FakeArchitectLLM()
    )

    agent = ArchitectAgent(
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
        "Add strict API input validation"
        in fake_llm.last_user_prompt
    )


def test_architect_injects_relevant_security_feedback(
    tmp_path,
):
    store, retriever = (
        build_retriever(
            tmp_path
        )
    )

    store.save(
        run_id="old-run",
        memory_type="security",
        key="old_security_review",
        value={
            "risk_score":
                65,

            "summary": (
                "FastAPI upload endpoint "
                "needed stronger file "
                "validation."
            ),

            "findings": [
                "Validate uploaded "
                "file types."
            ],
        },
    )

    state, task = (
        build_state_and_task()
    )

    fake_llm = (
        FakeArchitectLLM()
    )

    agent = ArchitectAgent(
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
        "stronger file validation"
        in fake_llm.last_user_prompt
    )


def test_architect_excludes_current_run_memory(
    tmp_path,
):
    store, retriever = (
        build_retriever(
            tmp_path
        )
    )

    state, task = (
        build_state_and_task()
    )

    store.save(
        run_id=state.run_id,
        memory_type="critic",
        key="current_run_feedback",
        value={
            "summary": (
                "FastAPI FAISS "
                "validation feedback"
            ),

            "required_improvements": [
                "Current run only"
            ],
        },
    )

    fake_llm = (
        FakeArchitectLLM()
    )

    agent = ArchitectAgent(
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


def test_architect_does_not_inject_irrelevant_memory(
    tmp_path,
):
    store, retriever = (
        build_retriever(
            tmp_path
        )
    )

    store.save(
        run_id="old-run",
        memory_type="artifact",
        key="unrelated_system",
        value={
            "content": {
                "architecture_style": (
                    "Quantum telescope "
                    "processing"
                ),

                "technology_stack": [
                    "Astrophysics",
                ],
            }
        },
    )

    state, task = (
        build_state_and_task()
    )

    fake_llm = (
        FakeArchitectLLM()
    )

    agent = ArchitectAgent(
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


def test_architect_respects_memory_limit(
    tmp_path,
):
    store, retriever = (
        build_retriever(
            tmp_path
        )
    )

    for index in range(6):
        store.save(
            run_id=(
                f"old-run-{index}"
            ),
            memory_type="artifact",
            key=(
                f"fastapi_architecture_"
                f"{index}"
            ),
            value={
                "content": {
                    "architecture_style": (
                        "FastAPI FAISS "
                        "modular architecture"
                    ),

                    "technology_stack": [
                        "FastAPI",
                        "FAISS",
                    ],
                }
            },
        )

    state, task = (
        build_state_and_task()
    )

    agent = ArchitectAgent(
        llm_client=(
            FakeArchitectLLM()
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


def test_architect_agent_repairs_invalid_output():
    fake_llm = (
        RepairingArchitectLLM()
    )

    state, task = (
        build_state_and_task()
    )

    agent = ArchitectAgent(
        llm_client=fake_llm
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        fake_llm.calls
        == 2
    )

    assert (
        artifact.metadata[
            "validation_attempts"
        ]
        == 2
    )

    assert len(
        artifact.content[
            "research_influences"
        ]
    ) > 0


def test_architect_agent_fails_after_retry_limit():
    state, task = (
        build_state_and_task()
    )

    agent = ArchitectAgent(
        llm_client=(
            AlwaysInvalidArchitectLLM()
        ),
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

    research = (
        create_research_artifact()
    )

    state.add_artifact(
        research
    )

    task = AgentTask(
        title="Design architecture",
        description="Design the system.",
        assigned_agent=(
            AgentRole.ARCHITECT
        ),
        input_artifact_ids=[
            research.id
        ],
    )

    agent = ArchitectAgent(
        llm_client=(
            FakeArchitectLLM()
        )
    )

    with pytest.raises(
        ArchitectureGenerationError,
        match=(
            "requirements artifact "
            "not found"
        ),
    ):
        agent.execute(
            task,
            state,
        )


def test_architect_agent_fails_without_research():
    state = NexusState(
        user_request="Build an application"
    )

    requirements = (
        create_requirements_artifact()
    )

    state.add_artifact(
        requirements
    )

    task = AgentTask(
        title="Design architecture",
        description="Design the system.",
        assigned_agent=(
            AgentRole.ARCHITECT
        ),
        input_artifact_ids=[
            requirements.id
        ],
    )

    agent = ArchitectAgent(
        llm_client=(
            FakeArchitectLLM()
        )
    )

    with pytest.raises(
        ArchitectureGenerationError,
        match=(
            "research artifact not found"
        ),
    ):
        agent.execute(
            task,
            state,
        )
