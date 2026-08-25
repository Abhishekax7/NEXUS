from abc import ABC, abstractmethod

from app.core.models import AgentRole, AgentTask, Artifact
from app.core.state import NexusState


class BaseAgent(ABC):
    role: AgentRole

    @abstractmethod
    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        """
        Execute an assigned task and return an artifact.
        """
        raise NotImplementedError
