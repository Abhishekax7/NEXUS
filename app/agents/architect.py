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

    research_influences: list[str] = Field(
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

    def _get_artifact_content(
        self,
        task: AgentTask,
        state: NexusState,
        artifact_type: ArtifactType,
    ) -> dict:
        """
        Prefer explicit task inputs.

        Fall back to state lookup so the agent can
        still be executed independently in tests
        or manual smoke tests.
        """

        for artifact_id in task.input_artifact_ids:
            artifact = state.artifacts.get(
                artifact_id
            )

            if (
                artifact
                and artifact.type == artifact_type
            ):
                return artifact.content

        for artifact in state.artifacts.values():
            if artifact.type == artifact_type:
                return artifact.content

        raise ArchitectureGenerationError(
            f"{artifact_type.value} artifact not found."
        )

    def _validate_output(
        self,
        raw_output: str,
    ) -> ArchitectureDesign:
        parsed = json.loads(
            raw_output
        )

        return ArchitectureDesign.model_validate(
            parsed
        )

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        requirements = self._get_artifact_content(
            task,
            state,
            ArtifactType.REQUIREMENTS,
        )

        research = self._get_artifact_content(
            task,
            state,
            ArtifactType.RESEARCH,
        )

        system_prompt = (
            "You are the Architect Agent inside NEXUS, "
            "an autonomous AI software engineering system. "
            "Design technically strong architectures using "
            "both validated requirements and evidence-backed "
            "technical research. Return valid JSON only."
        )

        prompt = f"""
VALIDATED REQUIREMENTS:

{json.dumps(requirements, indent=2)}

EVIDENCE-BACKED RESEARCH:

{json.dumps(research, indent=2)}

Design the software architecture using BOTH inputs.

Return exactly one JSON object containing:

architecture_style
components
data_flow
technology_stack
interfaces
security_considerations
design_decisions
research_influences

Rules:

- architecture_style must be a non-empty string

- components must be a non-empty array of objects

- every component must contain:
  name
  responsibility
  technology

- data_flow must explain how components interact

- technology_stack must contain concrete technologies

- interfaces must describe APIs or component boundaries

- security_considerations must be non-empty

- design_decisions must explain important choices

- research_influences must explain which architectural
  choices were influenced by the supplied research

- use the supplied research as evidence

- do not invent external research sources

- respect constraints from the requirements

- prefer free/open-source technologies where required

- every field is mandatory and non-empty

- do not return markdown

- do not return text outside the JSON object
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
                        "grounded_in_requirements":
                            True,
                        "grounded_in_research":
                            True,
                    },
                )

            except (
                json.JSONDecodeError,
                ValidationError,
            ) as exc:
                last_error = exc

                prompt = f"""
The previous architecture response failed validation.

ERROR:

{exc}

PREVIOUS RESPONSE:

{raw_output}

Repair the architecture.

Return ONLY one complete JSON object containing:

architecture_style
components
data_flow
technology_stack
interfaces
security_considerations
design_decisions
research_influences

Every field is mandatory and non-empty.

The architecture must remain grounded in BOTH
the validated requirements and supplied research.
"""

        raise ArchitectureGenerationError(
            "Architecture could not be validated "
            f"after retries: {last_error}"
        )
