from typing import Dict, Type

from app.agents.base import BaseAgent
from app.core.models import AgentRole


class AgentNotRegistered(Exception):
    """Raised when no agent exists for a requested role."""


class AgentRegistry:
    def __init__(self):
        self._agents: Dict[AgentRole, Type[BaseAgent]] = {}

    def register(
        self,
        role: AgentRole,
        agent_class: Type[BaseAgent],
    ) -> None:
        self._agents[role] = agent_class

    def get_agent(self, role: AgentRole) -> BaseAgent:
        agent_class = self._agents.get(role)

        if agent_class is None:
            raise AgentNotRegistered(
                f"No agent registered for role: {role.value}"
            )

        return agent_class()

    def is_registered(self, role: AgentRole) -> bool:
        return role in self._agents

    def registered_roles(self):
        return list(self._agents.keys())
