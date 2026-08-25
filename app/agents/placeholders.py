from app.agents.base import BaseAgent
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState


class ResearchAgent(BaseAgent):
    role = AgentRole.RESEARCH

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.RESEARCH,
            name="research_output",
            content={
                "summary": "Research placeholder",
            },
            created_by=self.role,
        )


class ArchitectAgent(BaseAgent):
    role = AgentRole.ARCHITECT

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.ARCHITECTURE,
            name="architecture_output",
            content={
                "summary": "Architecture placeholder",
            },
            created_by=self.role,
        )


class CoderAgent(BaseAgent):
    role = AgentRole.CODER

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.CODE,
            name="implementation_output",
            content={
                "summary": "Code placeholder",
            },
            created_by=self.role,
        )


class TesterAgent(BaseAgent):
    role = AgentRole.TESTER

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.TEST_RESULT,
            name="test_output",
            content={
                "passed": True,
            },
            created_by=self.role,
        )


class SecurityAgent(BaseAgent):
    role = AgentRole.SECURITY

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.SECURITY_REPORT,
            name="security_output",
            content={
                "status": "passed",
            },
            created_by=self.role,
        )


class CriticAgent(BaseAgent):
    role = AgentRole.CRITIC

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.EVALUATION,
            name="critic_output",
            content={
                "score": 100,
            },
            created_by=self.role,
        )

