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
from app.memory.retriever import MemoryRetriever
from app.tools.executor import CommandExecutor


class FilePatch(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )
    path: str = Field(min_length=1)
    new_content: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class DebugReport(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )
    root_cause: str = Field(
        min_length=1
    )

    failure_summary: str = Field(
        min_length=1
    )

    patches: list[FilePatch] = Field(
        min_length=1
    )

    retry_test_commands: list[str] = Field(
        min_length=1
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    notes: list[str] = Field(
        min_length=1
    )


class DebugGenerationError(Exception):
    """Raised when a valid debug repair cannot be generated."""


class DebuggerAgent(BaseAgent):
    role = AgentRole.DEBUGGER

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        memory_retriever: Optional[
            MemoryRetriever
        ] = None,
        max_validation_retries: int = 2,
        memory_limit: int = 3,
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

        self.allowed_executables = frozenset(
            CommandExecutor.ALLOWED_EXECUTABLES
        )

    def _validate_retry_test_commands(
        self,
        report: DebugReport,
    ) -> None:
        for command in (
            report.retry_test_commands
        ):
            if not isinstance(
                command,
                str,
            ):
                raise DebugGenerationError(
                    "Retry test command must "
                    "be a string."
                )

            command = command.strip()

            if not command:
                raise DebugGenerationError(
                    "Retry test command "
                    "cannot be empty."
                )

            try:
                executor = CommandExecutor()
                parts = executor.validate_command(
                        command
                    )
                
            except Exception as exc:
                raise DebugGenerationError(
                    "Invalid retry test "
                    f"command: {exc}"
                ) from exc

            if not parts:
                raise DebugGenerationError(
                    "Retry test command "
                    "cannot be empty."
                )

            executable = parts[0]

            if executable not in (
                self.allowed_executables
            ):
                allowed = ", ".join(
                    sorted(
                        self.allowed_executables
                    )
                )

                raise DebugGenerationError(
                    "Unsupported retry test "
                    f"executable: {executable}. "
                    "NEXUS runtime allows "
                    f"only: {allowed}."
                )

    def _get_artifact(
        self,
        task: AgentTask,
        state: NexusState,
        artifact_type: ArtifactType,
    ) -> Artifact:
        for artifact_id in task.input_artifact_ids:
            artifact = state.artifacts.get(
                artifact_id
            )

            if (
                artifact
                and artifact.type
                == artifact_type
            ):
                return artifact

        for artifact in state.artifacts.values():
            if artifact.type == artifact_type:
                return artifact

        raise DebugGenerationError(
            f"{artifact_type.value} artifact not found."
        )

    def _build_debug_schema(
        self,
        code_artifact: Artifact,
    ) -> dict:
        existing_paths = sorted({
            file_data["path"]
            for file_data
            in code_artifact.content.get(
                "files",
                [],
            )
            if isinstance(
                file_data,
                dict,
            )
            and isinstance(
                file_data.get("path"),
                str,
            )
            and file_data["path"].strip()
        })

        if not existing_paths:
            raise DebugGenerationError(
                "CODE artifact contains no valid files."
            )

        schema = DebugReport.model_json_schema()

        try:
            path_schema = (
                schema["$defs"]
                ["FilePatch"]
                ["properties"]
                ["path"]
            )
        except KeyError as exc:
            raise DebugGenerationError(
                "Unable to constrain debugger "
                "patch-path schema."
            ) from exc

        path_schema["enum"] = existing_paths

        return schema

    def _validate_patch_paths(
        self,
        report: DebugReport,
        code_artifact: Artifact,
    ) -> None:
        existing_paths = {
            file_data["path"]
            for file_data
            in code_artifact.content.get(
                "files",
                [],
            )
            if isinstance(
                file_data,
                dict,
            )
            and isinstance(
                file_data.get(
                    "path"
                ),
                str,
            )
        }

        if not existing_paths:
            raise DebugGenerationError(
                "CODE artifact contains no valid files."
            )

        seen_paths = set()

        for patch in report.patches:
            path = patch.path

            if path.startswith("/"):
                raise DebugGenerationError(
                    "Absolute patch path is forbidden: "
                    f"{path}"
                )

            if ".." in path.split("/"):
                raise DebugGenerationError(
                    "Patch path traversal is forbidden: "
                    f"{path}"
                )

            if path not in existing_paths:
                raise DebugGenerationError(
                    "Debugger attempted to patch "
                    f"unknown file: {path}"
                )

            if path in seen_paths:
                raise DebugGenerationError(
                    f"Duplicate patch path: {path}"
                )

            seen_paths.add(
                path
            )

    def _validate_output(
        self,
        raw_output: str,
    ) -> DebugReport:
        parsed = json.loads(
            raw_output
        )

        return DebugReport.model_validate(
            parsed
        )

    def _build_memory_query(
        self,
        state: NexusState,
        test_artifact: Artifact,
    ) -> str:
        failure_summary = (
            test_artifact.content.get(
                "summary",
                "",
            )
        )

        failed_commands = (
            test_artifact.content.get(
                "failed_command_names",
                [],
            )
        )

        results = (
            test_artifact.content.get(
                "results",
                [],
            )
        )

        failure_text_parts = [
            state.user_request,
            str(failure_summary),
            " ".join(
                str(command)
                for command
                in failed_commands
            ),
        ]

        for result in results:
            if not isinstance(
                result,
                dict,
            ):
                continue

            failure_text_parts.extend(
                [
                    str(
                        result.get(
                            "stdout",
                            "",
                        )
                    ),
                    str(
                        result.get(
                            "stderr",
                            "",
                        )
                    ),
                ]
            )

        return " ".join(
            part
            for part
            in failure_text_parts
            if part
        )

    def _retrieve_memory_context(
        self,
        state: NexusState,
        test_artifact: Artifact,
    ) -> list[dict]:
        if self.memory_retriever is None:
            return []

        query = self._build_memory_query(
            state,
            test_artifact,
        )

        if not query.strip():
            return []

        past_repairs = (
            self.memory_retriever
            .retrieve_repairs(
                query=query,
                limit=self.memory_limit,
                exclude_run_id=state.run_id,
            )
        )

        past_failures = (
            self.memory_retriever
            .retrieve_failures(
                query=query,
                limit=self.memory_limit,
                exclude_run_id=state.run_id,
            )
        )

        memories = []

        for result in (
            past_repairs
            + past_failures
        ):
            memories.append(
                {
                    "score": result.score,
                    "memory_type": (
                        result.memory[
                            "memory_type"
                        ]
                    ),
                    "run_id": (
                        result.memory[
                            "run_id"
                        ]
                    ),
                    "key": (
                        result.memory[
                            "key"
                        ]
                    ),
                    "value": (
                        result.memory[
                            "value"
                        ]
                    ),
                    "metadata": (
                        result.memory[
                            "metadata"
                        ]
                    ),
                }
            )

        memories.sort(
            key=lambda memory: (
                memory["score"]
            ),
            reverse=True,
        )

        return memories[
            :self.memory_limit
        ]

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        code_artifact = self._get_artifact(
            task,
            state,
            ArtifactType.CODE,
        )

        test_artifact = self._get_artifact(
            task,
            state,
            ArtifactType.TEST_RESULT,
        )

        if (
            test_artifact.content.get(
                "passed"
            )
            is True
        ):
            raise DebugGenerationError(
                "Debugger should not run "
                "when tests already pass."
            )

        memory_context = (
            self._retrieve_memory_context(
                state,
                test_artifact,
            )
        )

        allowed_executables_text = ", ".join(
            sorted(
                self.allowed_executables
            )
        )

        debug_schema = self._build_debug_schema(
            code_artifact
        )

        allowed_patch_paths = sorted({
            file_data["path"]
            for file_data
            in code_artifact.content.get(
                "files",
                [],
            )
            if isinstance(
                file_data,
                dict,
            )
            and isinstance(
                file_data.get("path"),
                str,
            )
        })

        allowed_patch_paths_text = ", ".join(
            allowed_patch_paths
        )

        system_prompt = (
            "You are the Debugger Agent inside "
            "NEXUS, an autonomous AI software "
            "engineering system. Analyze failing "
            "test output and produce minimal, safe "
            "file-level repairs. You may use relevant "
            "past NEXUS experience as supporting "
            "evidence, but you must still reason from "
            "the current code and current failure. "
            "Return valid JSON only."
        )

        prompt = f"""
GENERATED CODE:

{json.dumps(code_artifact.content, indent=2)}

CURRENT TEST FAILURE REPORT:

{json.dumps(test_artifact.content, indent=2)}

RELEVANT PAST NEXUS EXPERIENCE:

{json.dumps(memory_context, indent=2)}

Diagnose the CURRENT failure and propose a repair.

Return exactly one JSON object containing:

root_cause
failure_summary
patches
retry_test_commands
confidence
notes

PATCH FORMAT:

patches must be an array of objects containing:

path
new_content
reason

Rules:

- only patch files that already exist in GENERATED CODE
- EXACT ALLOWED PATCH PATHS:
  {allowed_patch_paths_text}
- patches[].path MUST be one of those exact paths
- never infer or invent package files such as __init__.py
- patch paths must be relative
- never use absolute paths
- never use ../ traversal
- do not invent new files
- new_content must contain the complete replacement content
  for the target file
- prefer the smallest repair that fixes the current failure
- preserve working behavior
- retry_test_commands must contain commands that should
  be executed after the repair
- NEXUS EXECUTION CAPABILITIES:
  allowed executables are: {allowed_executables_text}
- every retry_test_commands entry must start with one
  of those allowed executables
- commands using mvn, gradle, npm, yarn, pnpm, bash,
  sh, or other unsupported executables are forbidden
- confidence must be between 0.0 and 1.0
- do not include secrets
- past memories are advisory evidence only
- never copy a past repair blindly
- verify that any past solution applies to the current code
- current code and current test evidence take priority
- return JSON only
"""

        last_error = None

        for attempt in range(
            self.max_validation_retries + 1
        ):
            raw_output = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=prompt,
                json_mode=False,
                json_schema=debug_schema,
                schema_name="nexus_debug_report",
                strict_schema=True,
                reasoning_effort="low",
            )

            try:
                report = self._validate_output(
                    raw_output
                )

                self._validate_patch_paths(
                    report,
                    code_artifact,
                )

                self._validate_retry_test_commands(
                    report
                )

                return Artifact(
                    type=ArtifactType.DEBUG_REPORT,
                    name="debug_repair_report",
                    content=report.model_dump(),
                    created_by=self.role,
                    metadata={
                        "validation_attempts":
                            attempt + 1,
                        "patch_count":
                            len(report.patches),
                        "memory_context_count":
                            len(memory_context),
                        "memory_augmented":
                            bool(memory_context),
                        "runtime_capability_aware":
                            True,
                        "structured_output":
                            True,
                        "strict_schema":
                            True,
                        "schema_name":
                            "nexus_debug_report",
                        "reasoning_effort":
                            "low",
                        "allowed_executables":
                            sorted(
                                self.allowed_executables
                            ),
                    },
                )

            except (
                json.JSONDecodeError,
                ValidationError,
                DebugGenerationError,
            ) as exc:
                last_error = exc

                prompt = f"""
The previous debug response failed validation.

ERROR:

{exc}

PREVIOUS RESPONSE:

{raw_output}

CURRENT CODE:

{json.dumps(code_artifact.content, indent=2)}

CURRENT TEST FAILURE:

{json.dumps(test_artifact.content, indent=2)}

RELEVANT PAST EXPERIENCE:

{json.dumps(memory_context, indent=2)}

Repair the debug response.

Return exactly one JSON object containing:

root_cause
failure_summary
patches
retry_test_commands
confidence
notes

Each patch must contain:

path
new_content
reason

Rules:

- patch only files that already exist
- EXACT ALLOWED PATCH PATHS:
  {allowed_patch_paths_text}
- patches[].path MUST be one of those exact paths
- never infer or invent package files such as __init__.py
- absolute paths are forbidden
- ../ traversal is forbidden
- duplicate patch paths are forbidden
- every field must be non-empty
- past memories are advisory only
- current evidence takes priority
- NEXUS EXECUTION CAPABILITIES:
  allowed executables are: {allowed_executables_text}
- every retry_test_commands entry must start with one
  of those allowed executables
- unsupported executables such as mvn, gradle, npm,
  yarn, pnpm, bash and sh are forbidden
- return JSON only
"""

        raise DebugGenerationError(
            "Debug repair could not be "
            "validated after retries: "
            f"{last_error}"
        )
