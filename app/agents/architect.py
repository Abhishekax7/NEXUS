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


class ComponentDesign(BaseModel):
    name: str = Field(min_length=1)
    responsibility: str = Field(min_length=1)
    technology: str = Field(min_length=1)


class ArchitectureDesign(BaseModel):
    architecture_style: str = Field(min_length=1)

    components: list[ComponentDesign] = Field(
        min_length=1
    )

    data_flow: list[str] = Field(
        min_length=1
    )

    technology_stack: list[str] = Field(
        min_length=1
    )

    interfaces: list[str] = Field(
        min_length=1
    )

    security_considerations: list[str] = Field(
        min_length=1
    )

    design_decisions: list[str] = Field(
        min_length=1
    )


class ArchitectureGenerationError(Exception):
    """Raised when architecture output cannot be validated."""


class ArchitectAgent(BaseAgent):
    role = AgentRole.ARCHITECT

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_validation_retries: int = 2,
    ):
        self.llm = llm_client or LLMClient()
        self.max_validation_retries = (
            max_validation_retries
        )

    def _get_requirements(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> dict:
        for artifact_id in task.input_artifact_ids:
            artifact = state.artifacts.get(
                artifact_id
            )

            if (
                artifact
                and artifact.type
                == ArtifactType.REQUIREMENTS
            ):
                return artifact.content

        for artifact in state.artifacts.values():
            if (
                artifact.type
                == ArtifactType.REQUIREMENTS
            ):
                return artifact.content

        raise ArchitectureGenerationError(
            "Requirements artifact not found."
        )

    def _validate_output(
        self,
        raw_output: str,
    ) -> ArchitectureDesign:
        parsed = json.loads(raw_output)

        return ArchitectureDesign.model_validate(
            parsed
        )

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        requirements = self._get_requirements(
            task,
            state,
        )

        system_prompt = (
            "You are the Architect Agent inside NEXUS. "
            "Design technically strong, modular software "
            "architectures from validated requirements. "
            "Return machine-valid JSON only."
        )

        prompt = f"""
Design the software architecture for these requirements:

{json.dumps(requirements, indent=2)}

Return exactly one JSON object with:

architecture_style
components
data_flow
technology_stack
interfaces
security_considerations
design_decisions

Rules:

- architecture_style must be a non-empty string
- components must be a non-empty array of objects
- every component object must contain:
  name
  responsibility
  technology
- data_flow must describe execution between components
- technology_stack must contain concrete technologies
- interfaces must describe important boundaries or APIs
- security_considerations must be non-empty
- design_decisions must explain important architectural choices
- prefer free and open-source technologies where practical
- every field is mandatory
- do not return markdown
"""

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
                design = self._validate_output(
                    raw_output
                )

                return Artifact(
                    type=ArtifactType.ARCHITECTURE,
                    name="architecture_design",
                    content=design.model_dump(),
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
The previous architecture failed validation.

ERROR:
{exc}

PREVIOUS RESPONSE:
{raw_output}

Repair it and return ONLY a complete JSON object.

Required fields:

architecture_style
components
data_flow
technology_stack
interfaces
security_considerations
design_decisions

Every field is mandatory and non-empty.
"""

        raise ArchitectureGenerationError(
            "Architecture could not be validated "
            f"after retries: {last_error}"
        )
