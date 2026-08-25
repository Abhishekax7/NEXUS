import json

from app.agents.requirements import RequirementsAgent
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


def test_requirements_agent_returns_artifact():
    state = NexusState(
        user_request="Build a RAG application"
    )

    task = AgentTask(
        title="Analyze requirements",
        description="Analyze the user request.",
        assigned_agent=AgentRole.REQUIREMENTS,
    )

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
        artifact.content["acceptance_criteria"]
    ) > 0
