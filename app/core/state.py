from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.models import AgentTask, Artifact


class NexusState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))

    user_request: str

    tasks: Dict[str, AgentTask] = Field(default_factory=dict)
    artifacts: Dict[str, Artifact] = Field(default_factory=dict)

    execution_order: List[str] = Field(default_factory=list)

    active_task_id: Optional[str] = None

    iteration: int = 0

    completed: bool = False
    failed: bool = False

    final_artifact_id: Optional[str] = None

    errors: List[str] = Field(default_factory=list)

    metadata: Dict = Field(default_factory=dict)

    def add_task(self, task: AgentTask) -> None:
        self.tasks[task.id] = task

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts[artifact.id] = artifact

    def get_task(self, task_id: str) -> Optional[AgentTask]:
        return self.tasks.get(task_id)

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        return self.artifacts.get(artifact_id)
