from collections.abc import Callable
from typing import Union

from app.agents.base import BaseAgent
from app.core.models import AgentRole


class AgentRegistryError(Exception):
    """Raised when agent registration or resolution fails."""


AgentFactory = Callable[
    [],
    BaseAgent,
]

AgentProvider = Union[
    type[BaseAgent],
    AgentFactory,
    BaseAgent,
]


class AgentRegistry:
    """
    Registry responsible for resolving NEXUS agents.

    A role can be registered with:

    - an agent class
    - a zero-argument factory
    - a prebuilt agent instance

    Factory support allows production dependencies
    such as memory retrievers to be injected cleanly.
    """

    def __init__(self):
        self._providers: dict[
            AgentRole,
            AgentProvider,
        ] = {}

    def register(
        self,
        role: AgentRole,
        provider: AgentProvider,
    ) -> None:
        if not isinstance(
            role,
            AgentRole,
        ):
            raise AgentRegistryError(
                "role must be an AgentRole."
            )

        if not (
            isinstance(
                provider,
                BaseAgent,
            )
            or isinstance(
                provider,
                type,
            )
            or callable(
                provider
            )
        ):
            raise AgentRegistryError(
                "Agent provider must be an agent "
                "instance, agent class, or callable factory."
            )

        self._providers[
            role
        ] = provider

    def get_agent(
        self,
        role: AgentRole,
    ) -> BaseAgent:
        if role not in self._providers:
            raise AgentRegistryError(
                "No agent registered for role: "
                f"{role.value}"
            )

        provider = self._providers[
            role
        ]

        if isinstance(
            provider,
            BaseAgent,
        ):
            agent = provider

        else:
            try:
                agent = provider()

            except Exception as exc:
                raise AgentRegistryError(
                    "Failed to create agent for role "
                    f"{role.value}: {exc}"
                ) from exc

        if not isinstance(
            agent,
            BaseAgent,
        ):
            raise AgentRegistryError(
                "Registered provider for "
                f"{role.value} did not produce "
                "a BaseAgent instance."
            )

        return agent

    def registered_roles(
        self,
    ) -> list[AgentRole]:
        return list(
            self._providers.keys()
        )

    def is_registered(
        self,
        role: AgentRole,
    ) -> bool:
        return (
            role
            in self._providers
        )
