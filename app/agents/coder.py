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


class GeneratedFile(BaseModel):
    path: str = Field(
        min_length=1
    )

    content: str = Field(
        min_length=1
    )

    purpose: str = Field(
        min_length=1
    )


class CodeBundle(BaseModel):
    project_name: str = Field(
        min_length=1
    )

    summary: str = Field(
        min_length=1
    )

    files: list[GeneratedFile] = Field(
        min_length=1
    )

    dependencies: list[str] = Field(
        min_length=1
    )

    run_commands: list[str] = Field(
        min_length=1
    )

    test_commands: list[str] = Field(
        min_length=1
    )

    implementation_notes: list[str] = Field(
        min_length=1
    )


class CodeGenerationError(Exception):
    """
    Raised when generated code
    cannot be validated.
    """


class CoderAgent(BaseAgent):
    role = AgentRole.CODER

    def __init__(
        self,
        llm_client: Optional[
            LLMClient
        ] = None,
        max_validation_retries: int = 2,
        max_tokens: int = 3500,
    ):
        self.llm = (
            llm_client
            or LLMClient()
        )

        self.max_validation_retries = (
            max_validation_retries
        )

        self.max_tokens = (
            max_tokens
        )

    def _get_architecture(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> dict:
        for artifact_id in (
            task.input_artifact_ids
        ):
            artifact = (
                state.artifacts.get(
                    artifact_id
                )
            )

            if (
                artifact
                and artifact.type
                == ArtifactType.ARCHITECTURE
            ):
                return (
                    artifact.content
                )

        for artifact in (
            state.artifacts.values()
        ):
            if (
                artifact.type
                == ArtifactType.ARCHITECTURE
            ):
                return (
                    artifact.content
                )

        raise CodeGenerationError(
            "Architecture artifact "
            "not found."
        )

    def _get_optional_artifact(
        self,
        state: NexusState,
        artifact_type: ArtifactType,
    ) -> Optional[dict]:
        for artifact in (
            state.artifacts.values()
        ):
            if (
                artifact.type
                == artifact_type
            ):
                return (
                    artifact.content
                )

        return None

    def _validate_output(
        self,
        raw_output: str,
    ) -> CodeBundle:
        parsed = json.loads(
            raw_output
        )

        return (
            CodeBundle.model_validate(
                parsed
            )
        )

    def _validate_file_paths(
        self,
        bundle: CodeBundle,
    ) -> None:
        seen_paths = set()

        for generated_file in (
            bundle.files
        ):
            path = (
                generated_file.path
            )

            if path.startswith(
                "/"
            ):
                raise (
                    CodeGenerationError(
                        "Absolute file path "
                        "is not allowed: "
                        f"{path}"
                    )
                )

            if (
                ".."
                in path.split("/")
            ):
                raise (
                    CodeGenerationError(
                        "Parent directory "
                        "traversal is not "
                        "allowed: "
                        f"{path}"
                    )
                )

            if path in seen_paths:
                raise (
                    CodeGenerationError(
                        "Duplicate generated "
                        "file path: "
                        f"{path}"
                    )
                )

            seen_paths.add(
                path
            )

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        architecture = (
            self._get_architecture(
                task,
                state,
            )
        )

        requirements = (
            self._get_optional_artifact(
                state,
                ArtifactType.REQUIREMENTS,
            )
        )

        research = (
            self._get_optional_artifact(
                state,
                ArtifactType.RESEARCH,
            )
        )

        system_prompt = (
            "You are the Coder Agent "
            "inside NEXUS, an autonomous "
            "AI software engineering "
            "system. Generate coherent, "
            "runnable, modular software "
            "from validated architecture. "
            "Return machine-valid JSON "
            "only."
        )

        prompt = f"""
USER REQUEST:

{state.user_request}

VALIDATED REQUIREMENTS:

{json.dumps(
    requirements,
    indent=2,
)}

TECHNICAL RESEARCH:

{json.dumps(
    research,
    indent=2,
)}

VALIDATED ARCHITECTURE:

{json.dumps(
    architecture,
    indent=2,
)}

Generate an initial runnable implementation.

Return exactly one JSON object containing:

project_name
summary
files
dependencies
run_commands
test_commands
implementation_notes

FILES FORMAT:

files must be an array of objects.

Every object must contain:

path
content
purpose

Rules:

- generate multiple files when appropriate
- paths must be relative paths
- never use absolute paths
- never use ../ directory traversal
- do not generate duplicate file paths
- code must be internally consistent
- imports between generated files must match
- include dependency declarations
- include commands required to run the project
- include commands required to test the project
- use the architecture as the primary implementation contract
- respect requirements and technical constraints
- prefer free/open-source dependencies where required
- never include API keys, passwords, tokens, or secrets
- use environment variables for secrets
- every field is mandatory and non-empty
- do not wrap source code in markdown fences
- keep the implementation compact and demo-ready
- generate only the minimum files required for a runnable implementation
- prefer 3 to 5 generated files
- avoid verbose comments and documentation inside generated source files
- keep implementation_notes concise
- keep summary concise
- do not generate README files
- do not duplicate information between files or notes
- prioritize runnable code over explanatory text
- return JSON only
"""

        last_error = None

        for attempt in range(
            self.max_validation_retries
            + 1
        ):
            raw_output = (
                self.llm.generate(
                    system_prompt=(
                        system_prompt
                    ),
                    user_prompt=prompt,
                    json_mode=True,
                    max_tokens=(
                        self.max_tokens
                    ),
                )
            )

            try:
                bundle = (
                    self._validate_output(
                        raw_output
                    )
                )

                self._validate_file_paths(
                    bundle
                )

                return Artifact(
                    type=ArtifactType.CODE,
                    name=(
                        "generated_code_bundle"
                    ),
                    content=(
                        bundle.model_dump()
                    ),
                    created_by=self.role,
                    metadata={
                        "validation_attempts":
                            attempt + 1,

                        "file_count":
                            len(
                                bundle.files
                            ),

                        "grounded_in_architecture":
                            True,

                        "requirements_available":
                            requirements
                            is not None,

                        "research_available":
                            research
                            is not None,
                    },
                )

            except (
                json.JSONDecodeError,
                ValidationError,
                CodeGenerationError,
            ) as exc:
                last_error = exc

                prompt = f"""
The previous generated implementation
failed validation.

ERROR:

{exc}

PREVIOUS RESPONSE:

{raw_output}

CURRENT ARCHITECTURE:

{json.dumps(
    architecture,
    indent=2,
)}

Repair the implementation.

Return exactly one complete JSON object
containing:

project_name
summary
files
dependencies
run_commands
test_commands
implementation_notes

Every file must contain:

path
content
purpose

Rules:

- all fields must be non-empty
- paths must be relative
- ../ is forbidden
- duplicate file paths are forbidden
- code must remain consistent with the supplied architecture
- do not include secrets
- keep the repaired implementation compact
- preserve only files required for a runnable solution
- prefer 3 to 5 generated files
- avoid verbose comments and documentation
- keep summary and implementation_notes concise
- do not generate README files
- do not wrap source code in markdown fences
- return JSON only
"""

        raise CodeGenerationError(
            "Code generation could not "
            "be validated after retries: "
            f"{last_error}"
        )
