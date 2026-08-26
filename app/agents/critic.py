import json
from typing import Literal, Optional

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


Verdict = Literal[
    "accept",
    "revise",
    "reject",
]


class QualityIssue(BaseModel):
    category: str = Field(
        min_length=1
    )

    severity: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]

    description: str = Field(
        min_length=1
    )

    recommendation: str = Field(
        min_length=1
    )


class QualityGateReport(BaseModel):
    verdict: Verdict

    quality_score: int = Field(
        ge=0,
        le=100,
    )

    summary: str = Field(
        min_length=1
    )

    requirements_satisfied: bool
    architecture_acceptable: bool
    implementation_acceptable: bool
    tests_acceptable: bool
    security_acceptable: bool

    issues: list[QualityIssue]

    strengths: list[str] = Field(
        min_length=1
    )

    required_improvements: list[str]

    final_recommendation: str = Field(
        min_length=1
    )


class CriticGenerationError(Exception):
    """Raised when the critic cannot produce a valid report."""


class CriticAgent(BaseAgent):
    role = AgentRole.CRITIC

    REQUIRED_TYPES = {
        ArtifactType.REQUIREMENTS,
        ArtifactType.ARCHITECTURE,
        ArtifactType.CODE,
        ArtifactType.TEST_RESULT,
        ArtifactType.SECURITY_REPORT,
    }

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_validation_retries: int = 2,
    ):
        self.llm = (
            llm_client
            or LLMClient()
        )

        self.max_validation_retries = (
            max_validation_retries
        )

    def _collect_artifacts(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> dict[ArtifactType, Artifact]:

        collected = {}

        for artifact_id in task.input_artifact_ids:
            artifact = state.artifacts.get(
                artifact_id
            )

            if (
                artifact
                and artifact.type
                in self.REQUIRED_TYPES
            ):
                collected[
                    artifact.type
                ] = artifact

        for artifact in state.artifacts.values():
            if (
                artifact.type
                in self.REQUIRED_TYPES
                and artifact.type
                not in collected
            ):
                collected[
                    artifact.type
                ] = artifact

        missing = (
            self.REQUIRED_TYPES
            - set(collected.keys())
        )

        if missing:
            names = sorted(
                artifact_type.value
                for artifact_type in missing
            )

            raise CriticGenerationError(
                "Missing required artifacts: "
                + ", ".join(names)
            )

        return collected

    def _validate_report(
        self,
        raw_output: str,
    ) -> QualityGateReport:

        parsed = json.loads(
            raw_output
        )

        report = (
            QualityGateReport.model_validate(
                parsed
            )
        )

        if (
            report.verdict == "accept"
            and report.required_improvements
        ):
            raise CriticGenerationError(
                "Accepted output cannot contain "
                "required improvements."
            )

        if (
            report.verdict == "accept"
            and not all(
                [
                    report.requirements_satisfied,
                    report.architecture_acceptable,
                    report.implementation_acceptable,
                    report.tests_acceptable,
                    report.security_acceptable,
                ]
            )
        ):
            raise CriticGenerationError(
                "Accepted output must pass "
                "all quality gates."
            )

        if (
            report.verdict
            in {
                "revise",
                "reject",
            }
            and not report.required_improvements
        ):
            raise CriticGenerationError(
                "Non-accepted output must include "
                "required improvements."
            )

        return report

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:

        artifacts = self._collect_artifacts(
            task,
            state,
        )

        evidence = {
            artifact_type.value:
                artifact.content
            for (
                artifact_type,
                artifact,
            ) in artifacts.items()
        }

        system_prompt = (
            "You are the final Critic Agent inside "
            "NEXUS, an autonomous AI software "
            "engineering system. Act as a strict "
            "senior engineering quality gate. "
            "Evaluate only supplied evidence. "
            "Return valid JSON only."
        )

        prompt = f"""
USER REQUEST:

{state.user_request}

EXECUTION EVIDENCE:

{json.dumps(evidence, indent=2)}

Evaluate the completed software workflow.

Return exactly one JSON object with:

verdict
quality_score
summary
requirements_satisfied
architecture_acceptable
implementation_acceptable
tests_acceptable
security_acceptable
issues
strengths
required_improvements
final_recommendation

Allowed verdicts:

accept
revise
reject

Each issue must contain:

category
severity
description
recommendation

Allowed severity:

low
medium
high
critical

Rules:

- quality_score must be between 0 and 100
- evaluate requirements against implementation
- evaluate architecture consistency
- evaluate implementation quality
- use the test result as evidence
- use the security report as evidence
- do not invent failures
- do not invent successful tests
- do not invent security findings
- accept only when all five quality gates pass
- accept must have an empty required_improvements list
- revise means the implementation is recoverable
- reject means major redesign or fundamental correction is needed
- revise or reject must contain required improvements
- strengths must contain at least one concrete strength
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
                report = self._validate_report(
                    raw_output
                )

                return Artifact(
                    type=ArtifactType.EVALUATION,
                    name="final_quality_gate",
                    content=report.model_dump(),
                    created_by=self.role,
                    metadata={
                        "validation_attempts":
                            attempt + 1,
                        "verdict":
                            report.verdict,
                        "quality_score":
                            report.quality_score,
                        "issue_count":
                            len(report.issues),
                    },
                )

            except (
                json.JSONDecodeError,
                ValidationError,
                CriticGenerationError,
            ) as exc:

                last_error = exc

                prompt = f"""
Your previous quality-gate response failed validation.

ERROR:

{exc}

PREVIOUS RESPONSE:

{raw_output}

Repair the response.

Return exactly one JSON object with:

verdict
quality_score
summary
requirements_satisfied
architecture_acceptable
implementation_acceptable
tests_acceptable
security_acceptable
issues
strengths
required_improvements
final_recommendation

Rules:

- verdict must be accept, revise, or reject
- quality_score must be 0 through 100
- accept requires all five gates to be true
- accept requires required_improvements to be empty
- revise or reject requires at least one required improvement
- do not invent evidence
- return JSON only
"""

        raise CriticGenerationError(
            "Critic report could not be "
            "validated after retries: "
            f"{last_error}"
        )
