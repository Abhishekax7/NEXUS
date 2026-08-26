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
from app.memory.retriever import MemoryRetriever


class ComponentDesign(BaseModel):
    name: str = Field(
        min_length=1
    )

    responsibility: str = Field(
        min_length=1
    )

    technology: str = Field(
        min_length=1
    )


class ArchitectureDesign(BaseModel):
    architecture_style: str = Field(
        min_length=1
    )

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
        memory_retriever: Optional[
            MemoryRetriever
        ] = None,
        max_validation_retries: int = 2,
        memory_limit: int = 4,
    ):
        self.llm = (
            llm_client
            or LLMClient()
        )

        self.memory_retriever = (
            memory_retriever
        )

        self.max_validation_retries = (
            max_validation_retries
        )

        self.memory_limit = (
            memory_limit
        )

    def _get_artifact_content(
        self,
        task: AgentTask,
        state: NexusState,
        artifact_type: ArtifactType,
    ) -> dict:
        for artifact_id in task.input_artifact_ids:
            artifact = state.artifacts.get(
                artifact_id
            )

            if (
                artifact
                and artifact.type
                == artifact_type
            ):
                return artifact.content

        for artifact in state.artifacts.values():
            if (
                artifact.type
                == artifact_type
            ):
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

    def _build_memory_query(
        self,
        state: NexusState,
        requirements: dict,
        research: dict,
    ) -> str:
        parts = [
            state.user_request,
            str(
                requirements.get(
                    "objective",
                    "",
                )
            ),
        ]

        technologies = (
            research.get(
                "recommended_technologies",
                [],
            )
        )

        if isinstance(
            technologies,
            list,
        ):
            parts.extend(
                str(item)
                for item in technologies
            )

        constraints = (
            requirements.get(
                "constraints",
                [],
            )
        )

        if isinstance(
            constraints,
            list,
        ):
            parts.extend(
                str(item)
                for item in constraints
            )

        return " ".join(
            part
            for part in parts
            if part
        )

    def _retrieve_memory_context(
        self,
        state: NexusState,
        requirements: dict,
        research: dict,
    ) -> list[dict]:
        if self.memory_retriever is None:
            return []

        query = self._build_memory_query(
            state,
            requirements,
            research,
        )

        if not query.strip():
            return []

        results = (
            self.memory_retriever.retrieve(
                query=query,
                limit=self.memory_limit,
                memory_types=[
                    "artifact",
                    "critic",
                    "security",
                ],
                exclude_run_id=state.run_id,
            )
        )

        memories = []

        for result in results:
            memories.append(
                {
                    "score":
                        result.score,
                    "memory_type":
                        result.memory[
                            "memory_type"
                        ],
                    "run_id":
                        result.memory[
                            "run_id"
                        ],
                    "key":
                        result.memory[
                            "key"
                        ],
                    "value":
                        result.memory[
                            "value"
                        ],
                    "metadata":
                        result.memory[
                            "metadata"
                        ],
                }
            )

        return memories

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        requirements = (
            self._get_artifact_content(
                task,
                state,
                ArtifactType.REQUIREMENTS,
            )
        )

        research = (
            self._get_artifact_content(
                task,
                state,
                ArtifactType.RESEARCH,
            )
        )

        memory_context = (
            self._retrieve_memory_context(
                state,
                requirements,
                research,
            )
        )

        system_prompt = (
            "You are the Architect Agent inside NEXUS, "
            "an autonomous AI software engineering "
            "system. Design technically strong, modular "
            "architectures using validated requirements, "
            "evidence-backed technical research, and "
            "relevant experience from previous NEXUS "
            "runs when available. Past experience is "
            "advisory only. Current requirements and "
            "research are the source of truth. "
            "Return machine-valid JSON only."
        )

        prompt = f"""
CURRENT USER REQUEST:

{state.user_request}

VALIDATED REQUIREMENTS:

{json.dumps(requirements, indent=2)}

CURRENT EVIDENCE-BACKED RESEARCH:

{json.dumps(research, indent=2)}

RELEVANT PAST NEXUS EXPERIENCE:

{json.dumps(memory_context, indent=2)}

Design the software architecture using the CURRENT
requirements and research.

You may use useful lessons from past NEXUS experience,
but do not blindly copy previous architectures.

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

- research_influences must explain which choices
  were influenced by current technical research

- current requirements take priority over memory

- current research takes priority over memory

- previous critic feedback may be used to avoid
  repeating earlier quality problems

- previous security feedback may be used to avoid
  repeating earlier security weaknesses

- previous architecture artifacts may be used only
  when relevant to the current system

- never treat past memory as guaranteed truth

- do not invent external research sources

- respect all current constraints

- prefer free/open-source technologies when required

- every field is mandatory and non-empty

- do not return markdown

- return JSON only
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
                design = (
                    self._validate_output(
                        raw_output
                    )
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
                        "memory_context_count":
                            len(memory_context),
                        "memory_augmented":
                            bool(memory_context),
                    },
                )

            except (
                json.JSONDecodeError,
                ValidationError,
            ) as exc:
                last_error = exc

                prompt = f"""
The previous architecture response
failed validation.

ERROR:

{exc}

PREVIOUS RESPONSE:

{raw_output}

CURRENT REQUIREMENTS:

{json.dumps(requirements, indent=2)}

CURRENT RESEARCH:

{json.dumps(research, indent=2)}

RELEVANT PAST EXPERIENCE:

{json.dumps(memory_context, indent=2)}

Repair the architecture response.

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

- every field is mandatory
- every field must be non-empty
- current requirements are the source of truth
- current research takes priority over memory
- past experience is advisory only
- return JSON only
"""

        raise ArchitectureGenerationError(
            "Architecture could not be "
            "validated after retries: "
            f"{last_error}"
        )
