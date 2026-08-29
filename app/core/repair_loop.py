from app.agents.debugger import DebuggerAgent
from app.agents.tester import TesterAgent
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState
from app.tools.patcher import PatchApplicator


class RepairLoopError(Exception):
    """Raised when autonomous repair cannot complete successfully."""


class RepairLoopResult:
    def __init__(
        self,
        passed: bool,
        attempts: int,
        final_test_artifact: Artifact,
        debug_artifacts: list[Artifact],
    ):
        self.passed = passed
        self.attempts = attempts
        self.final_test_artifact = final_test_artifact
        self.debug_artifacts = debug_artifacts


class RepairLoop:
    def __init__(
        self,
        tester: TesterAgent,
        debugger: DebuggerAgent,
        patcher: PatchApplicator,
        max_repairs: int = 2,
    ):
        if max_repairs < 0:
            raise ValueError(
                "max_repairs cannot be negative."
            )

        self.tester = tester
        self.debugger = debugger
        self.patcher = patcher
        self.max_repairs = max_repairs

    def _find_code_artifact(
        self,
        state: NexusState,
    ) -> Artifact:
        for artifact in state.artifacts.values():
            if artifact.type == ArtifactType.CODE:
                return artifact

        raise RepairLoopError(
            "CODE artifact not found."
        )

    def _sync_code_artifact(
        self,
        code_artifact: Artifact,
        debug_artifact: Artifact,
    ) -> None:
        files = code_artifact.content.get(
            "files"
        )

        if not isinstance(files, list):
            raise RepairLoopError(
                "CODE artifact does not contain "
                "a valid files list."
            )

        files_by_path = {}

        for file_data in files:
            if not isinstance(file_data, dict):
                continue

            file_path = file_data.get(
                "path"
            )

            if isinstance(file_path, str):
                files_by_path[file_path] = (
                    file_data
                )

        patches = debug_artifact.content.get(
            "patches"
        )

        if not isinstance(patches, list):
            raise RepairLoopError(
                "DEBUG_REPORT does not contain "
                "a valid patches list."
            )

        for patch in patches:
            if not isinstance(patch, dict):
                raise RepairLoopError(
                    "DEBUG_REPORT contains an "
                    "invalid patch."
                )

            patch_path = patch.get(
                "path"
            )
            new_content = patch.get(
                "new_content"
            )

            if patch_path not in files_by_path:
                raise RepairLoopError(
                    "Patched file is not present "
                    "in CODE artifact: "
                    f"{patch_path}"
                )

            if not isinstance(
                new_content,
                str,
            ):
                raise RepairLoopError(
                    "Patched CODE content must "
                    "be a string."
                )

            files_by_path[
                patch_path
            ]["content"] = new_content

    def run(
        self,
        state: NexusState,
    ) -> RepairLoopResult:
        code_artifact = self._find_code_artifact(
            state
        )

        tester_task = AgentTask(
            title="Test generated implementation",
            description="Run generated test commands.",
            assigned_agent=AgentRole.TESTER,
            input_artifact_ids=[
                code_artifact.id
            ],
        )

        debug_artifacts = []

        test_artifact = self.tester.execute(
            tester_task,
            state,
        )

        state.add_artifact(
            test_artifact
        )

        if test_artifact.content.get(
            "passed"
        ) is True:
            return RepairLoopResult(
                passed=True,
                attempts=0,
                final_test_artifact=test_artifact,
                debug_artifacts=debug_artifacts,
            )

        for attempt in range(
            1,
            self.max_repairs + 1,
        ):
            debugger_task = AgentTask(
                title="Debug failing implementation",
                description="Analyze failed generated tests.",
                assigned_agent=AgentRole.DEBUGGER,
                input_artifact_ids=[
                    code_artifact.id,
                    test_artifact.id,
                ],
            )

            debug_artifact = (
                self.debugger.execute(
                    debugger_task,
                    state,
                )
            )

            state.add_artifact(
                debug_artifact
            )

            debug_artifacts.append(
                debug_artifact
            )

            self.patcher.apply_debug_artifact(
                debug_artifact,
                state,
            )

            self._sync_code_artifact(
                code_artifact,
                debug_artifact,
            )

            test_artifact = self.tester.execute(
                tester_task,
                state,
            )

            state.add_artifact(
                test_artifact
            )

            if test_artifact.content.get(
                "passed"
            ) is True:
                return RepairLoopResult(
                    passed=True,
                    attempts=attempt,
                    final_test_artifact=test_artifact,
                    debug_artifacts=debug_artifacts,
                )

        return RepairLoopResult(
            passed=False,
            attempts=self.max_repairs,
            final_test_artifact=test_artifact,
            debug_artifacts=debug_artifacts,
        )
