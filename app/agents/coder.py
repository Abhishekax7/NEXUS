import json
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
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
from app.tools.executor import (
    CommandExecutor,
    ExecutionError,
)


class GeneratedFile(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

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
    model_config = ConfigDict(
        extra="forbid"
    )

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

        # CommandExecutor is the single
        # authoritative runtime-policy source.
        self.command_executor = (
            CommandExecutor()
        )

        self.allowed_executables = (
            self.command_executor
            .allowed_executables
        )

    def _get_architecture(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        for artifact_id in (
            task.input_artifact_ids
        ):
            artifact = (
                state.get_artifact(
                    artifact_id
                )
            )

            if (
                artifact is not None
                and artifact.type
                == ArtifactType.ARCHITECTURE
            ):
                return artifact

        for artifact in (
            state.artifacts.values()
        ):
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
            list(
                state.artifacts.values()
            )
        ):
            if (
                artifact.type
                == artifact_type
            ):
                return artifact

        return None

    def _validate_file_paths(
        self,
        bundle: CodeBundle,
    ) -> None:
        seen_paths: set[str] = set()

        for generated_file in (
            bundle.files
        ):
            path = (
                generated_file.path.strip()
            )

            if not path:
                raise ValueError(
                    "Generated file path "
                    "cannot be empty."
                )

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

            seen_paths.add(
                path
            )

    def _validate_command(
        self,
        command: str,
        command_type: str,
    ) -> None:
        """
        Validate a generated command using
        CommandExecutor's public runtime-policy API.

        No command is executed here.
        """

        try:
            self.command_executor.validate_command(
                command
            )

        except ExecutionError as exc:
            raise ValueError(
                f"Unsupported "
                f"{command_type} command: "
                f"{exc}"
            ) from exc

    def _validate_commands(
        self,
        bundle: CodeBundle,
    ) -> None:
        for command in (
            bundle.run_commands
        ):
            self._validate_command(
                command,
                "run",
            )

        for command in (
            bundle.test_commands
        ):
            self._validate_command(
                command,
                "test",
            )

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

            self._validate_commands(
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

    def _allowed_executables_text(
        self,
    ) -> str:
        return ", ".join(
            sorted(
                self.allowed_executables
            )
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

        architecture_json = json.dumps(
            architecture.content,
            indent=2,
            default=str,
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

        allowed_executables_text = (
            self._allowed_executables_text()
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

You must obey the NEXUS runtime execution
capabilities supplied in the user prompt.

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

NEXUS EXECUTION CAPABILITIES:

The generated project will execute inside
a restricted NEXUS runtime.

Allowed command executables:

{allowed_executables_text}

Every run_commands entry and every
test_commands entry MUST begin with one
of those allowed executables.

Do NOT emit commands using unsupported
executables such as:

- mvn
- gradle
- npm
- yarn
- pnpm
- bash
- sh

Do not use shell operators, pipelines,
redirection, or chained shell commands.

If the requested architecture normally
uses an unsupported runtime command,
adapt the implementation so that it can
be executed using the allowed NEXUS
runtime capabilities instead.

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
- Every generated run command must use
  an allowed executable.
- Every generated test command must use
  an allowed executable.
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
- Never include API keys, passwords,
  tokens, or secrets.
- Use environment variables for secrets.
""".strip()

        repair_system_prompt = """
You are the Coder Repair Agent inside
NEXUS.

A previous structured code-generation
response failed NEXUS validation.

Return a corrected CodeBundle only.

The result must satisfy the structured
CodeBundle schema supplied by the runtime.

You must also obey the supplied NEXUS
execution capabilities.

Do not include markdown fences.
Do not include prose outside the
structured response.
""".strip()

        validation_attempts = 0
        previous_output = ""
        previous_error = ""

        current_system_prompt = (
            system_prompt
        )

        current_prompt = prompt

        for attempt in range(
            self.max_validation_retries + 1
        ):
            validation_attempts = (
                attempt + 1
            )

            try:
                raw_output = (
                    self.llm.generate(
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
                )

            except Exception as exc:
                if (
                    attempt
                    >= self.max_validation_retries
                ):
                    raise CodeGenerationError(
                        "Coder LLM generation failed "
                        f"after retries: {exc}"
                    ) from exc

                previous_error = str(
                    exc
                )

                current_system_prompt = (
                    repair_system_prompt
                )

                current_prompt = f"""
The previous code-generation attempt
failed before a valid CodeBundle could
be accepted.

ERROR:

{previous_error}

NEXUS EXECUTION CAPABILITIES:

Allowed command executables:

{allowed_executables_text}

Every run_commands and test_commands
entry must begin with one of the allowed
executables.

Do not use mvn, gradle, npm, yarn,
pnpm, bash, sh, or any other unsupported
executable.

Recreate the implementation from the
original task below.

ORIGINAL REQUEST:

{prompt}

Return a corrected CodeBundle only.

Repair rules:

- Satisfy the CodeBundle schema exactly.
- Preserve the approved architecture.
- Preserve only files required for a
  runnable solution.
- Prefer 3 to 5 generated files.
- Include meaningful tests.
- Include required dependencies.
- Include safe run and test commands.
- Every run command must use a supported
  NEXUS executable.
- Every test command must use a supported
  NEXUS executable.
- Avoid verbose comments and documentation.
- Keep summary concise.
- Keep implementation_notes concise.
- Do not generate README files.
- Use only safe relative file paths.
- Never use parent-directory traversal.
- Never duplicate generated file paths.
- Do not use markdown code fences.
""".strip()

                continue

            previous_output = (
                raw_output or ""
            )

            try:
                bundle = (
                    self._validate_output(
                        previous_output
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
                    created_by=(
                        AgentRole.CODER
                    ),
                    metadata={
                        "validation_attempts": (
                            validation_attempts
                        ),
                        "structured_output": True,
                        "strict_schema": True,
                        "schema_name": (
                            "nexus_code_bundle"
                        ),
                        "max_tokens": (
                            self.max_tokens
                        ),
                        "reasoning_effort": (
                            "low"
                        ),
                        "grounded_in_architecture": True,
                        "requirements_available": (
                            requirements is not None
                        ),
                        "research_available": (
                            research is not None
                        ),
                        "runtime_capability_aware": (
                            True
                        ),
                        "allowed_executables": (
                            sorted(
                                self.allowed_executables
                            )
                        ),
                    },
                )

            except CodeGenerationError as exc:
                previous_error = str(
                    exc
                )

                if (
                    attempt
                    >= self.max_validation_retries
                ):
                    raise CodeGenerationError(
                        "Generated code could not "
                        "be validated after retries: "
                        f"{previous_error}"
                    ) from exc

                current_system_prompt = (
                    repair_system_prompt
                )

                current_prompt = f"""
The previous generated implementation
failed NEXUS validation.

VALIDATION ERROR:

{previous_error}

PREVIOUS OUTPUT:

{previous_output}

NEXUS EXECUTION CAPABILITIES:

Allowed command executables:

{allowed_executables_text}

Every run_commands entry and every
test_commands entry MUST start with an
allowed executable.

Do not use unsupported commands such as
mvn, gradle, npm, yarn, pnpm, bash,
or sh.

Correct the implementation.

Repair rules:

- Return a complete CodeBundle.
- Satisfy the schema exactly.
- Fix the validation error.
- Preserve the approved architecture.
- Preserve working implementation details.
- Use only NEXUS-supported runtime
  commands.
- Do not use shell operators or chained
  shell commands.
- Use only safe relative file paths.
- Never use absolute file paths.
- Never use parent-directory traversal.
- Never duplicate generated file paths.
- Keep the implementation compact.
- Do not generate README files.
- Do not use markdown code fences.

Return only the corrected structured
CodeBundle.
""".strip()

        raise CodeGenerationError(
            "Coder exhausted validation "
            "retries without producing "
            "a valid implementation."
        )
