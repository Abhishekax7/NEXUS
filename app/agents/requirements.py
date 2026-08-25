import json
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

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
    objective: str = Field(
        min_length=1
    )

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


class RequirementsGenerationError(Exception):
    """Raised when requirement generation cannot be validated."""


class RequirementsAgent(BaseAgent):
    role = AgentRole.REQUIREMENTS

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_validation_retries: int = 2,
    ):
        self.llm = llm_client or LLMClient()
        self.max_validation_retries = (
            max_validation_retries
        )

    def _build_prompt(
        self,
        user_request: str,
    ) -> str:
        return f"""
Analyze this software request:

{user_request}

Return one JSON object containing exactly:

objective
functional_requirements
non_functional_requirements
constraints
assumptions
acceptance_criteria

Requirements:

- objective must be a non-empty string
- every other field must be a non-empty array
- every array item must be a string
- infer reasonable assumptions
- include realistic technical constraints
- include security and reliability requirements
- include measurable acceptance criteria
- do not omit any field
- do not return markdown
- do not return explanations outside the JSON
"""

    def _validate_output(
        self,
        raw_output: str,
    ) -> RequirementAnalysis:
        parsed = json.loads(
            raw_output
        )

        return RequirementAnalysis.model_validate(
            parsed
        )

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        system_prompt = (
            "You are the Requirements Agent inside "
            "NEXUS, an autonomous AI software "
            "engineering system. Produce complete, "
            "precise and machine-valid JSON."
        )

        prompt = self._build_prompt(
            state.user_request
        )

        last_error = None

        for attempt in range(
            self.max_validation_retries + 1
        ):
            raw_output = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=prompt,
                json_mode=True,
            )

            try:
                analysis = self._validate_output(
                    raw_output
                )

                return Artifact(
                    type=ArtifactType.REQUIREMENTS,
                    name="requirements_analysis",
                    content=analysis.model_dump(),
                    created_by=self.role,
                    metadata={
                        "validation_attempts":
                            attempt + 1,
                    },
                )

            except (
                json.JSONDecodeError,
                ValidationError,
            ) as exc:
                last_error = exc

                prompt = f"""
The previous response failed schema validation.

ERROR:
{exc}

PREVIOUS RESPONSE:
{raw_output}

Repair the response.

Return ONLY a complete JSON object containing:

objective
functional_requirements
non_functional_requirements
constraints
assumptions
acceptance_criteria

Every field is mandatory.
Every array must contain at least one item.
"""

        raise RequirementsGenerationError(
            "Requirements output could not be "
            f"validated after retries: {last_error}"
        )
