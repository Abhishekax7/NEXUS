from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    WAITING_FOR_HUMAN = "waiting_for_human"
    FAILED = "failed"
    RETRYING = "retrying"
    COMPLETED = "completed"


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    REQUIREMENTS = "requirements"
    RESEARCH = "research"
    ARCHITECT = "architect"
    PLANNER = "planner"
    CODER = "coder"
    TESTER = "tester"
    DEBUGGER = "debugger"
    SECURITY = "security"
    CRITIC = "critic"


class ArtifactType(str, Enum):
    REQUIREMENTS = "requirements"
    RESEARCH = "research"
    ARCHITECTURE = "architecture"
    PLAN = "plan"
    CODE = "code"
    TEST_RESULT = "test_result"
    DEBUG_REPORT = "debug_report"
    SECURITY_REPORT = "security_report"
    EVALUATION = "evaluation"
    FINAL_OUTPUT = "final_output"


class Artifact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: ArtifactType
    name: str
    content: Any
    created_by: AgentRole
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))

    title: str
    description: str

    assigned_agent: AgentRole
    status: TaskStatus = TaskStatus.PENDING

    dependencies: List[str] = Field(default_factory=list)

    input_artifact_ids: List[str] = Field(default_factory=list)
    output_artifact_ids: List[str] = Field(default_factory=list)

    retry_count: int = 0
    max_retries: int = 3

    error: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

