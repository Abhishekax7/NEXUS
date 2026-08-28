import json
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    content: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


class CodeBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    files: list[GeneratedFile] = Field(min_length=1)
    dependencies: list[str] = Field(min_length=1)
    run_commands: list[str] = Field(min_length=1)
    test_commands: list[str] = Field(min_length=1)
    implementation_notes: list[str] = Field(min_length=1)


class CodeGenerationError(Exception):
    """Raised when generated code cannot be validated."""


class CoderAgent(BaseAgent):
    role = AgentRole.CODER

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_validation_retries: int = 2,
        max_tokens: int = 3500,
    ):
        self.llm = llm_client or LLMClient()
        self.max_validation_retries = max_validation_retries
        self.max_tokens = max_tokens

    def _get_architecture(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        for artifact_id in task.input_artifact_ids:
            artifact = state.get_artifact(
                artifact_id
            )

            if (
                artifact is not None
                and artifact.type
                == ArtifactType.ARCHITECTURE
            ):
                return artifact

        for artifact in state.artifacts.values():
            if (
                artifact.type
                == ArtifactType.ARCHITECTURE
            ):
                return artifact

        raise CodeGenerationError(
            "Architecture artifact not found "
            "for coder task."
        )

    def _get_optional_artifact(
        self,
        state: NexusState,
        artifact_type: ArtifactType,
    ) -> Optional[Artifact]:
        for artifact in reversed(
           list(state.artifacts.values())
        ):
            if artifact.type == artifact_type:
                return artifact

        return None

    def _validate_output(
        self,
        raw_output: str,
    ) -> CodeBundle:
        if not raw_output:
            raise CodeGenerationError(
                "Coder received an empty "
                "LLM response."
            )

        try:
            parsed = json.loads(
                raw_output
            )

            bundle = (
                CodeBundle.model_validate(
                    parsed
                )
            )

            self._validate_file_paths(
                bundle
            )

            return bundle

        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as exc:
            raise CodeGenerationError(
                str(exc)
            ) from exc

    def _validate_file_paths(
        self,
        bundle: CodeBundle,
    ) -> None:
        seen_paths: set[str] = set()

        for generated_file in bundle.files:
            path = generated_file.path.strip()

            if path.startswith(
                ("/", "\\")
            ):
                raise ValueError(
                    "Absolute file path "
                    f"is not allowed: {path}"
                )

            normalized_parts = (
                path.replace(
                    "\\",
                    "/",
                ).split("/")
            )

            if ".." in normalized_parts:
                raise ValueError(
                    "Parent directory traversal "
                    f"is not allowed: {path}"
                )

            if path in seen_paths:
                raise ValueError(
                    "Duplicate generated "
                    f"file path: {path}"
                )

            seen_paths.add(path)

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

        architecture_json = (
            json.dumps(
                architecture.content,
                indent=2,
                default=str,
            )
        )

        requirements_json = (
            json.dumps(
                requirements.content,
                indent=2,
                default=str,
            )
            if requirements
            else "Not available"
        )

        research_json = (
            json.dumps(
                research.content,
                indent=2,
                default=str,
            )
            if research
            else "Not available"
        )

        system_prompt = """
You are the Coder Agent inside NEXUS,
an autonomous AI engineering system.

Your responsibility is to convert an
approved architecture into a compact,
runnable implementation.

Generate implementation code only from
the supplied requirements, research,
architecture, and task context.

The result must satisfy the structured
CodeBundle schema supplied by the runtime.

Prioritize correctness, security,
maintainability, and runnable code.

Do not include markdown code fences.
Do not include prose outside the
structured response.
""".strip()

        prompt = f"""
USER REQUEST:
{state.user_request}

CODER TASK:
Title: {task.title}
Description: {task.description}

REQUIREMENTS:
{requirements_json}

RESEARCH:
{research_json}

APPROVED ARCHITECTURE:
{architecture_json}

Generate a runnable implementation.

Generation rules:

- Follow the approved architecture.
- Respect the supplied requirements.
- Use research only where relevant.
- Keep the implementation compact and
  demo-ready.
- Generate only the minimum files required
  for a runnable implementation.
- Prefer 3 to 5 generated files.
- Include meaningful tests.
- Include required dependencies.
- Include commands required to run the
  implementation.
- Include commands required to run tests.
- Avoid verbose comments and documentation.
- Keep implementation_notes concise.
- Keep summary concise.
- Do not generate README files.
- Do not duplicate information.
- Prioritize runnable code over explanatory
  text.
- Every generated file must use a safe,
  relative project path.
- Never use absolute paths.
- Never use parent-directory traversal.
- Never duplicate generated file paths.
""".strip()

        repair_system_prompt = """
You are the Coder Repair Agent inside
NEXUS.

A previous structured code-generation
response failed NEXUS validation.

Return a corrected CodeBundle only.

The result must satisfy the structured
CodeBundle schema supplied by the runtime.

Do not include markdown fences.
Do not include prose outside the
structured response.
""".strip()

        validation_attempts = 0
        previous_output = ""
        previous_error = ""

        for attempt in range(
            self.max_validation_retries + 1
        ):
            validation_attempts = (
                attempt + 1
            )

            if attempt == 0:
                current_system_prompt = (
                    system_prompt
                )

                current_prompt = prompt

            else:
                current_system_prompt = (
                    repair_system_prompt
                )

                current_prompt = f"""
The previous code-generation response
failed validation.

VALIDATION ERROR:
{previous_error}

PREVIOUS RESPONSE:
{previous_output}

Return a corrected runnable implementation.

Repair rules:

- Return only the structured CodeBundle.
- Keep the repaired implementation compact.
- Preserve only files required for a
  runnable solution.
- Prefer 3 to 5 generated files.
- Include meaningful tests.
- Include required dependencies.
- Include run and test commands.
- Avoid verbose comments and documentation.
- Keep summary concise.
- Keep implementation_notes concise.
- Do not generate README files.
- Use only safe relative file paths.
- Never use parent-directory traversal.
- Never duplicate generated file paths.
- Do not use markdown code fences.
""".strip()

            try:
                raw_output = self.llm.generate(
                    system_prompt=(
                        current_system_prompt
                    ),
                    user_prompt=(
                        current_prompt
                    ),
                    json_mode=False,
                    max_tokens=(
                        self.max_tokens
                    ),
                    json_schema=(
                        CodeBundle
                        .model_json_schema()
                    ),
                    schema_name=(
                        "nexus_code_bundle"
                    ),
                    strict_schema=True,
                    reasoning_effort="low",
                )

                previous_output = raw_output

                bundle = (
                    self._validate_output(
                        raw_output
                    )
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
                            validation_attempts,
                        "file_count":
                            len(bundle.files),
                        "grounded_in_architecture":
                            True,
                        "requirements_available":
                            requirements
                            is not None,
                        "research_available":
                            research
                            is not None,
                        "structured_output":
                            True,
                        "reasoning_effort":
                            "low",
                    },
                )

            except CodeGenerationError as exc:
                previous_error = str(exc)

                if (
                    attempt
                    >= self.max_validation_retries
                ):
                    raise CodeGenerationError(
                        "Code generation could not "
                        "be validated after retries: "
                        f"{exc}"
                    ) from exc

        raise CodeGenerationError(
            "Code generation failed."
        )
