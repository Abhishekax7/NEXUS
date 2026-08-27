from enum import Enum

from pydantic import (
    BaseModel,
    Field,
)


class EvaluationStatus(
    str,
    Enum,
):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class EvaluationDimension(
    str,
    Enum,
):
    TASK_COMPLETION = "task_completion"
    ARTIFACT_QUALITY = "artifact_quality"
    GROUNDING = "grounding"
    TEST_QUALITY = "test_quality"
    SECURITY = "security"
    REPAIR_EFFICIENCY = "repair_efficiency"
    REPLANNING_EFFICIENCY = "replanning_efficiency"
    TOOL_USE = "tool_use"
    CRITIC_QUALITY = "critic_quality"
    WORKFLOW_RELIABILITY = "workflow_reliability"


class MetricScore(BaseModel):
    dimension: EvaluationDimension

    score: float = Field(
        ge=0.0,
        le=100.0,
    )

    status: EvaluationStatus

    reason: str = Field(
        min_length=1
    )

    evidence: list[str] = Field(
        min_length=1
    )


class AgentEvaluation(BaseModel):
    agent_role: str = Field(
        min_length=1
    )

    score: float = Field(
        ge=0.0,
        le=100.0,
    )

    metrics: list[
        MetricScore
    ] = Field(
        min_length=1
    )

    strengths: list[str]

    weaknesses: list[str]


class WorkflowEvaluation(BaseModel):
    run_id: str = Field(
        min_length=1
    )

    overall_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    status: EvaluationStatus

    metrics: list[
        MetricScore
    ] = Field(
        min_length=1
    )

    agent_evaluations: list[
        AgentEvaluation
    ]

    strengths: list[str]

    weaknesses: list[str]

    recommendations: list[str]

    regression_risk: float = Field(
        ge=0.0,
        le=1.0,
    )


class EvaluationError(Exception):
    """
    Raised when an evaluation cannot
    be produced safely.
    """
