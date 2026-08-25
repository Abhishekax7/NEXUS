from app.agents.base import BaseAgent
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState


class RequirementsAgent(BaseAgent):
    role = AgentRole.REQUIREMENTS

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:

        content = {
            "original_request": state.user_request,
            "summary": "Structured requirement analysis placeholder.",
        }

        return Artifact(
            type=ArtifactType.REQUIREMENTS,
            name="requirements_analysis",
            content=content,
            created_by=self.role,
        )
