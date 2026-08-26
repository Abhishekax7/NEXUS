from typing import Optional

from app.core.models import (
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.state import NexusState
from app.memory.store import MemoryStore


class MemoryManager:
    def __init__(
        self,
        store: Optional[MemoryStore] = None,
    ):
        self.store = (
            store
            or MemoryStore()
        )

    def record_task_event(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> int:
        return self.store.save(
            run_id=state.run_id,
            memory_type="task_event",
            key=task.id,
            value={
                "title": task.title,
                "description": task.description,
                "agent": task.assigned_agent.value,
                "status": task.status.value,
                "retry_count": task.retry_count,
                "error": task.error,
            },
            metadata={
                "task_id": task.id,
            },
        )

    def record_artifact(
        self,
        artifact: Artifact,
        state: NexusState,
    ) -> int:
        return self.store.save(
            run_id=state.run_id,
            memory_type="artifact",
            key=artifact.type.value,
            value={
                "artifact_id": artifact.id,
                "name": artifact.name,
                "type": artifact.type.value,
                "created_by": artifact.created_by.value,
                "content": artifact.content,
            },
            metadata={
                "artifact_id": artifact.id,
                "created_by": artifact.created_by.value,
            },
        )

    def record_failure(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> int:
        return self.store.save(
            run_id=state.run_id,
            memory_type="failure",
            key=task.id,
            value={
                "task_id": task.id,
                "title": task.title,
                "agent": task.assigned_agent.value,
                "error": task.error,
                "retry_count": task.retry_count,
            },
            metadata={
                "status": task.status.value,
            },
        )

    def record_repair(
        self,
        debug_artifact: Artifact,
        state: NexusState,
    ) -> int:
        if (
            debug_artifact.type
            != ArtifactType.DEBUG_REPORT
        ):
            raise ValueError(
                "record_repair expects a DEBUG_REPORT artifact."
            )

        return self.store.save(
            run_id=state.run_id,
            memory_type="repair",
            key=debug_artifact.id,
            value={
                "artifact_id": debug_artifact.id,
                "root_cause": (
                    debug_artifact.content.get(
                        "root_cause"
                    )
                ),
                "failure_summary": (
                    debug_artifact.content.get(
                        "failure_summary"
                    )
                ),
                "patches": (
                    debug_artifact.content.get(
                        "patches",
                        [],
                    )
                ),
                "confidence": (
                    debug_artifact.content.get(
                        "confidence"
                    )
                ),
            },
            metadata={
                "patch_count": (
                    debug_artifact.metadata.get(
                        "patch_count",
                        0,
                    )
                ),
            },
        )

    def record_security_review(
        self,
        security_artifact: Artifact,
        state: NexusState,
    ) -> int:
        if (
            security_artifact.type
            != ArtifactType.SECURITY_REPORT
        ):
            raise ValueError(
                "record_security_review expects "
                "a SECURITY_REPORT artifact."
            )

        return self.store.save(
            run_id=state.run_id,
            memory_type="security",
            key=security_artifact.id,
            value={
                "passed": (
                    security_artifact.content.get(
                        "passed"
                    )
                ),
                "risk_score": (
                    security_artifact.content.get(
                        "risk_score"
                    )
                ),
                "summary": (
                    security_artifact.content.get(
                        "summary"
                    )
                ),
                "findings": (
                    security_artifact.content.get(
                        "findings",
                        [],
                    )
                ),
            },
            metadata={
                "finding_count": (
                    security_artifact.metadata.get(
                        "finding_count",
                        0,
                    )
                ),
            },
        )

    def record_critic_verdict(
        self,
        critic_artifact: Artifact,
        state: NexusState,
    ) -> int:
        if (
            critic_artifact.type
            != ArtifactType.EVALUATION
        ):
            raise ValueError(
                "record_critic_verdict expects "
                "an EVALUATION artifact."
            )

        return self.store.save(
            run_id=state.run_id,
            memory_type="critic",
            key=critic_artifact.id,
            value={
                "verdict": (
                    critic_artifact.content.get(
                        "verdict"
                    )
                ),
                "quality_score": (
                    critic_artifact.content.get(
                        "quality_score"
                    )
                ),
                "summary": (
                    critic_artifact.content.get(
                        "summary"
                    )
                ),
                "required_improvements": (
                    critic_artifact.content.get(
                        "required_improvements",
                        [],
                    )
                ),
            },
            metadata={
                "issue_count": (
                    critic_artifact.metadata.get(
                        "issue_count",
                        0,
                    )
                ),
            },
        )

    def record_important_artifact(
        self,
        artifact: Artifact,
        state: NexusState,
    ) -> int:
        if (
            artifact.type
            == ArtifactType.DEBUG_REPORT
        ):
            return self.record_repair(
                artifact,
                state,
            )

        if (
            artifact.type
            == ArtifactType.SECURITY_REPORT
        ):
            return self.record_security_review(
                artifact,
                state,
            )

        if (
            artifact.type
            == ArtifactType.EVALUATION
        ):
            return self.record_critic_verdict(
                artifact,
                state,
            )

        return self.record_artifact(
            artifact,
            state,
        )

    def remember_task(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> list[int]:
        memory_ids = [
            self.record_task_event(
                task,
                state,
            )
        ]

        if (
            task.status
            == TaskStatus.FAILED
        ):
            memory_ids.append(
                self.record_failure(
                    task,
                    state,
                )
            )

        return memory_ids

    def get_run_history(
        self,
        state: NexusState,
    ) -> list[dict]:
        return self.store.get_by_run(
            state.run_id
        )

    def get_recent_failures(
        self,
        limit: int = 20,
    ) -> list[dict]:
        return self.store.get_by_type(
            "failure",
            limit=limit,
        )

    def get_recent_repairs(
        self,
        limit: int = 20,
    ) -> list[dict]:
        return self.store.get_by_type(
            "repair",
            limit=limit,
        )

    def get_recent_critic_verdicts(
        self,
        limit: int = 20,
    ) -> list[dict]:
        return self.store.get_by_type(
            "critic",
            limit=limit,
        )
