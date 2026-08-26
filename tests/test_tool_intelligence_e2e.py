import json
from dataclasses import dataclass

from app.agents.research import ResearchAgent
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.runtime import (
    build_memory_manager,
    build_memory_retriever,
    build_tool_registry,
)
from app.core.state import NexusState
from app.tools.executor_runtime import (
    ToolExecutor,
)
from app.tools.runtime import ToolRuntime
from app.tools.selector import ToolSelector


@dataclass
class FakeSearchResult:
    title: str
    url: str
    snippet: str


class FakeWebSearch:
    def search(
        self,
        query,
    ):
        return [
            FakeSearchResult(
                title="FastAPI Documentation",
                url="https://fastapi.tiangolo.com/",
                snippet=(
                    "FastAPI is a modern "
                    "Python API framework."
                ),
            )
        ]


class CapturingResearchLLM:
    def __init__(
        self,
    ):
        self.last_user_prompt = None

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
    ):
        self.last_user_prompt = user_prompt

        return json.dumps(
            {
                "research_question": (
                    "What architecture should "
                    "be used?"
                ),
                "findings": [
                    "FastAPI is appropriate."
                ],
                "recommended_technologies": [
                    "FastAPI",
                    "Python",
                ],
                "tradeoffs": [
                    (
                        "FastAPI is lightweight "
                        "but requires disciplined "
                        "application structure."
                    )
                ],
                "risks": [
                    (
                        "Input validation must "
                        "be handled carefully."
                    )
                ],
                "sources": [
                    {
                        "title": (
                            "FastAPI Documentation"
                        ),
                        "url": (
                            "https://fastapi.tiangolo.com/"
                        ),
                        "summary": (
                            "Official FastAPI docs."
                        ),
                    }
                ],
            }
        )


class MemorySelectingLLM:
    def __init__(
        self,
    ):
        self.last_user_prompt = None

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
    ):
        self.last_user_prompt = user_prompt

        return json.dumps(
            {
                "use_tool": True,
                "tool_name": "search_memory",
                "arguments": {
                    "query": (
                        "FastAPI architecture "
                        "input validation"
                    ),
                    "limit": 3,
                },
                "reason": (
                    "Relevant prior NEXUS "
                    "experience may improve "
                    "the research."
                ),
                "confidence": 0.96,
            }
        )


def build_state():
    state = NexusState(
        user_request=(
            "Build a secure FastAPI "
            "application."
        )
    )

    requirements = Artifact(
        type=ArtifactType.REQUIREMENTS,
        name="requirements",
        content={
            "objective": (
                "Build a secure FastAPI "
                "application."
            ),
            "functional_requirements": [
                "Expose API endpoints.",
            ],
            "non_functional_requirements": [
                "Maintainable architecture.",
            ],
            "constraints": [
                "Use free tools.",
            ],
            "assumptions": [
                "Python is available.",
            ],
            "acceptance_criteria": [
                "API runs successfully.",
            ],
        },
        created_by=AgentRole.REQUIREMENTS,
    )

    state.add_artifact(
        requirements
    )

    task = AgentTask(
        title="Research technologies",
        description=(
            "Research architecture "
            "and implementation options."
        ),
        assigned_agent=AgentRole.RESEARCH,
        input_artifact_ids=[
            requirements.id
        ],
    )

    return state, task


def test_research_dynamically_uses_persistent_memory_tool(
    tmp_path,
):
    manager = build_memory_manager(
        memory_db_path=str(
            tmp_path
            / "memory.db"
        )
    )

    manager.store.save(
        run_id="previous-run",
        memory_type="critic",
        key="fastapi_validation_feedback",
        value={
            "summary": (
                "FastAPI architecture needed "
                "stronger input validation."
            ),
            "required_improvements": [
                (
                    "Add strict request "
                    "validation."
                )
            ],
        },
    )

    retriever = build_memory_retriever(
        manager
    )

    registry = build_tool_registry(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_retriever=retriever,
    )

    selector = ToolSelector(
        registry=registry,
        llm_client=MemorySelectingLLM(),
    )

    runtime = ToolRuntime(
        selector=selector,
        executor=ToolExecutor(
            registry=registry
        ),
    )

    research_llm = (
        CapturingResearchLLM()
    )

    agent = ResearchAgent(
        llm_client=research_llm,
        search_tool=FakeWebSearch(),
        tool_runtime=runtime,
    )

    state, task = build_state()

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.type
        == ArtifactType.RESEARCH
    )

    assert (
        artifact.metadata[
            "dynamic_tools_enabled"
        ]
        is True
    )

    assert (
        artifact.metadata[
            "dynamic_tool_used"
        ]
        is True
    )

    assert (
        artifact.metadata[
            "dynamic_tool_name"
        ]
        == "search_memory"
    )

    assert (
        artifact.metadata[
            "dynamic_tool_success"
        ]
        is True
    )

    assert (
        "fastapi_validation_feedback"
        in research_llm.last_user_prompt
    )

    assert (
        "stronger input validation"
        in research_llm.last_user_prompt
    )


def test_tool_selection_remains_allow_listed(
    tmp_path,
):
    manager = build_memory_manager(
        memory_db_path=str(
            tmp_path
            / "memory.db"
        )
    )

    retriever = build_memory_retriever(
        manager
    )

    registry = build_tool_registry(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_retriever=retriever,
    )

    names = set(
        registry.names()
    )

    assert names == {
        "list_workspace_files",
        "read_text_file",
        "search_memory",
    }


def test_tool_execution_is_read_only_by_default(
    tmp_path,
):
    registry = build_tool_registry(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_retriever=None,
    )

    for name in registry.names():
        capability = (
            registry.get_capability(
                name
            )
        )

        assert (
            capability.metadata.get(
                "read_only",
                False,
            )
            is True
        )
