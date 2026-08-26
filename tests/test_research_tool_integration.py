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
    build_nexus_engine,
)
from app.core.state import NexusState
from app.tools.executor_runtime import (
    ToolExecutor,
)
from app.tools.production import (
    build_production_tool_registry,
)
from app.tools.runtime import (
    ToolRuntime,
)
from app.tools.selector import (
    ToolSelector,
)


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
                title=(
                    "FastAPI Documentation"
                ),
                url=(
                    "https://fastapi.tiangolo.com/"
                ),
                snippet=(
                    "FastAPI is a modern Python "
                    "web framework."
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
        self.last_user_prompt = (
            user_prompt
        )

        return json.dumps(
            {
                "research_question": (
                    "What architecture and "
                    "technology should be used?"
                ),
                "findings": [
                    (
                        "FastAPI is suitable "
                        "for API development."
                    )
                ],
                "recommended_technologies": [
                    "FastAPI",
                    "Python",
                ],
                "tradeoffs": [
                    (
                        "FastAPI is lightweight "
                        "but still requires careful "
                        "application design."
                    )
                ],
                "risks": [
                    (
                        "Input validation and "
                        "dependency management "
                        "must be handled carefully."
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
                            "Official FastAPI "
                            "documentation."
                        ),
                    }
                ],
            }
        )


class ToolSelectorLLM:
    def __init__(
        self,
        response,
    ):
        self.response = response
        self.last_user_prompt = None

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
    ):
        self.last_user_prompt = (
            user_prompt
        )

        return self.response


def build_state():
    state = NexusState(
        user_request=(
            "Build a FastAPI application "
            "using free tools."
        )
    )

    requirements = Artifact(
        type=ArtifactType.REQUIREMENTS,
        name="requirements",
        content={
            "objective": (
                "Build a FastAPI application "
                "using free tools."
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
                "Application runs successfully.",
            ],
        },
        created_by=(
            AgentRole.REQUIREMENTS
        ),
    )

    state.add_artifact(
        requirements
    )

    task = AgentTask(
        title="Research technologies",
        description=(
            "Research implementation options."
        ),
        assigned_agent=(
            AgentRole.RESEARCH
        ),
        input_artifact_ids=[
            requirements.id
        ],
    )

    return (
        state,
        task,
    )


def test_engine_injects_same_tool_runtime_into_research_agent(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_db_path=str(
            tmp_path
            / "memory.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=True,
    )

    research_agent = (
        engine.registry.get_agent(
            AgentRole.RESEARCH
        )
    )

    assert (
        engine.tool_runtime
        is not None
    )

    assert (
        research_agent.tool_runtime
        is engine.tool_runtime
    )

    assert (
        research_agent.tool_runtime
        .selector.registry
        is engine.tool_registry
    )

    assert (
        research_agent.tool_runtime
        .executor.registry
        is engine.tool_registry
    )


def test_research_agent_uses_real_workspace_tool(
    tmp_path,
):
    workspace = (
        tmp_path
        / "workspace"
    )

    workspace.mkdir()

    (
        workspace
        / "architecture_notes.txt"
    ).write_text(
        (
            "Use FastAPI with modular "
            "service boundaries."
        ),
        encoding="utf-8",
    )

    tool_registry = (
        build_production_tool_registry(
            workspace_root=str(
                workspace
            )
        )
    )

    selector_llm = (
        ToolSelectorLLM(
            json.dumps(
                {
                    "use_tool": True,
                    "tool_name": (
                        "read_text_file"
                    ),
                    "arguments": {
                        "path": (
                            "architecture_notes.txt"
                        )
                    },
                    "reason": (
                        "Existing workspace notes "
                        "may contain relevant "
                        "architecture context."
                    ),
                    "confidence": 0.97,
                }
            )
        )
    )

    tool_runtime = ToolRuntime(
        selector=ToolSelector(
            registry=tool_registry,
            llm_client=selector_llm,
        ),
        executor=ToolExecutor(
            registry=tool_registry
        ),
    )

    research_llm = (
        CapturingResearchLLM()
    )

    agent = ResearchAgent(
        llm_client=research_llm,
        search_tool=FakeWebSearch(),
        tool_runtime=tool_runtime,
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
            "dynamic_tool_success"
        ]
        is True
    )

    assert (
        artifact.metadata[
            "dynamic_tool_name"
        ]
        == "read_text_file"
    )

    assert (
        "architecture_notes.txt"
        in research_llm.last_user_prompt
    )

    assert (
        "Use FastAPI with modular"
        in research_llm.last_user_prompt
    )


def test_research_agent_can_choose_no_tool(
    tmp_path,
):
    workspace = (
        tmp_path
        / "workspace"
    )

    tool_registry = (
        build_production_tool_registry(
            workspace_root=str(
                workspace
            )
        )
    )

    selector_llm = (
        ToolSelectorLLM(
            json.dumps(
                {
                    "use_tool": False,
                    "tool_name": None,
                    "arguments": {},
                    "reason": (
                        "No additional internal "
                        "context is required."
                    ),
                    "confidence": 0.94,
                }
            )
        )
    )

    tool_runtime = ToolRuntime(
        selector=ToolSelector(
            registry=tool_registry,
            llm_client=selector_llm,
        ),
        executor=ToolExecutor(
            registry=tool_registry
        ),
    )

    agent = ResearchAgent(
        llm_client=(
            CapturingResearchLLM()
        ),
        search_tool=(
            FakeWebSearch()
        ),
        tool_runtime=tool_runtime,
    )

    state, task = build_state()

    artifact = agent.execute(
        task,
        state,
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
        is False
    )

    assert (
        artifact.metadata[
            "dynamic_tool_name"
        ]
        is None
    )

    assert (
        artifact.type
        == ArtifactType.RESEARCH
    )


def test_research_agent_preserves_old_behavior_without_tool_runtime():
    agent = ResearchAgent(
        llm_client=(
            CapturingResearchLLM()
        ),
        search_tool=(
            FakeWebSearch()
        ),
        tool_runtime=None,
    )

    state, task = build_state()

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata[
            "dynamic_tools_enabled"
        ]
        is False
    )

    assert (
        artifact.metadata[
            "dynamic_tool_used"
        ]
        is False
    )

    assert (
        artifact.metadata[
            "dynamic_tool_name"
        ]
        is None
    )

    assert (
        artifact.type
        == ArtifactType.RESEARCH
    )


def test_dynamic_tool_does_not_replace_web_sources(
    tmp_path,
):
    workspace = (
        tmp_path
        / "workspace"
    )

    workspace.mkdir()

    (
        workspace
        / "notes.txt"
    ).write_text(
        "Internal FastAPI notes.",
        encoding="utf-8",
    )

    registry = (
        build_production_tool_registry(
            workspace_root=str(
                workspace
            )
        )
    )

    selector = ToolSelector(
        registry=registry,
        llm_client=ToolSelectorLLM(
            json.dumps(
                {
                    "use_tool": True,
                    "tool_name": (
                        "read_text_file"
                    ),
                    "arguments": {
                        "path": "notes.txt"
                    },
                    "reason": (
                        "Use internal notes "
                        "as supporting context."
                    ),
                    "confidence": 0.9,
                }
            )
        ),
    )

    research_llm = (
        CapturingResearchLLM()
    )

    agent = ResearchAgent(
        llm_client=research_llm,
        search_tool=FakeWebSearch(),
        tool_runtime=ToolRuntime(
            selector=selector,
            executor=ToolExecutor(
                registry=registry
            ),
        ),
    )

    state, task = build_state()

    artifact = agent.execute(
        task,
        state,
    )

    sources = (
        artifact.content[
            "sources"
        ]
    )

    assert len(sources) == 1

    assert (
        sources[0]["url"]
        == "https://fastapi.tiangolo.com/"
    )

    assert (
        "notes.txt"
        in research_llm.last_user_prompt
    )
