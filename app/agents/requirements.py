import json
from typing import Optional

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.core.llm import LLMClient
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState


class RequirementAnalysis(BaseModel):
    objective: str

    functional_requirements: list[str] = Field(
        min_length=1
    )

    non_functional_requirements: list[str] = Field(
        min_length=1
    )

    constraints: list[str] = Field(
        min_length=1
    )

    assumptions: list[str] = Field(
        min_length=1
    )

    acceptance_criteria: list[str] = Field(
        min_length=1
    )


class RequirementsAgent(BaseAgent):
    role = AgentRole.REQUIREMENTS

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
    ):
        self.llm = llm_client or LLMClient()

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:

        system_prompt = (
            "You are a senior software requirements engineer. "
            "Return precise structured JSON only."
        )

        user_prompt = f"""
Analyze the following software request.

USER REQUEST:
{state.user_request}

Return JSON with exactly these keys:

objective
functional_requirements
non_functional_requirements
constraints
assumptions
acceptance_criteria

Rules:

- objective must be a string
- every other field must be a non-empty array of strings
- infer reasonable assumptions where information is missing
- include realistic technical constraints
- include measurable acceptance criteria
- include non-functional requirements such as security,
  performance, reliability, scalability, or usability
- do not return markdown
- do not return explanations outside the JSON
"""

        raw_output = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        parsed_json = json.loads(raw_output)

        analysis = RequirementAnalysis.model_validate(
            parsed_json
        )

        return Artifact(
            type=ArtifactType.REQUIREMENTS,
            name="requirements_analysis",
            content=analysis.model_dump(),
            created_by=self.role,
        )
