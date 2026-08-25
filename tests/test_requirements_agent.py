from app.agents.requirements import RequirementsAgent
from app.core.models import (
    AgentRole,
    AgentTask,
    ArtifactType,
)
from app.core.state import NexusState


def test_requirements_agent_returns_artifact():
    state = NexusState(
        user_request="Build a RAG application"
    )

    task = AgentTask(
        title="Analyze requirements",
        description="Analyze the user request.",
        assigned_agent=AgentRole.REQUIREMENTS,
    )

    agent = RequirementsAgent()

    artifact = agent.execute(
        task,
        state,
    )

    assert artifact.type == ArtifactType.REQUIREMENTS
    assert artifact.created_by == AgentRole.REQUIREMENTS
    assert artifact.content["original_request"] == (
        "Build a RAG application"
    )

