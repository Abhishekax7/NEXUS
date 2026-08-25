from app.core.models import AgentRole, AgentTask
from app.core.state import NexusState


class OrchestratorAgent:
    def create_initial_plan(self, user_request: str) -> NexusState:
        state = NexusState(user_request=user_request)

        requirements_task = AgentTask(
            title="Analyze requirements",
            description="Understand and structure the user's request.",
            assigned_agent=AgentRole.REQUIREMENTS,
        )

        research_task = AgentTask(
            title="Research solution",
            description="Research technical approaches and relevant information.",
            assigned_agent=AgentRole.RESEARCH,
            dependencies=[requirements_task.id],
        )

        architecture_task = AgentTask(
            title="Design architecture",
            description="Design the system architecture based on requirements and research.",
            assigned_agent=AgentRole.ARCHITECT,
            dependencies=[
                requirements_task.id,
                research_task.id,
            ],
        )

        implementation_task = AgentTask(
            title="Implement solution",
            description="Write the required application code.",
            assigned_agent=AgentRole.CODER,
            dependencies=[architecture_task.id],
        )

        testing_task = AgentTask(
            title="Test implementation",
            description="Execute tests and verify correctness.",
            assigned_agent=AgentRole.TESTER,
            dependencies=[implementation_task.id],
        )

        security_task = AgentTask(
            title="Security review",
            description="Review implementation for security issues.",
            assigned_agent=AgentRole.SECURITY,
            dependencies=[implementation_task.id],
        )

        critic_task = AgentTask(
            title="Critique solution",
            description="Evaluate quality, completeness, and requirement coverage.",
            assigned_agent=AgentRole.CRITIC,
            dependencies=[
                testing_task.id,
                security_task.id,
            ],
        )

        tasks = [
            requirements_task,
            research_task,
            architecture_task,
            implementation_task,
            testing_task,
            security_task,
            critic_task,
        ]

        for task in tasks:
            state.add_task(task)
            state.execution_order.append(task.id)

        return state

