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


Severity = Literal[
    "info",
    "low",
    "medium",
    "high",
    "critical",
]


class SecurityFinding(BaseModel):
    title: str = Field(
        min_length=1
    )

    severity: Severity

    category: str = Field(
        min_length=1
    )

    affected_files: list[str] = Field(
        min_length=1
    )

    evidence: str = Field(
        min_length=1
    )

    impact: str = Field(
        min_length=1
    )

    recommendation: str = Field(
        min_length=1
    )


class SecurityReport(BaseModel):
    passed: bool

    risk_score: int = Field(
        ge=0,
        le=100,
    )

    summary: str = Field(
        min_length=1
    )

    findings: list[SecurityFinding]

    reviewed_files: list[str] = Field(
        min_length=1
    )

    positive_controls: list[str] = Field(
        min_length=1
    )

    recommended_actions: list[str] = Field(
        min_length=1
    )


class SecurityGenerationError(Exception):
    """Raised when a valid security review cannot be produced."""


class SecurityAgent(BaseAgent):
    role = AgentRole.SECURITY

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

    def _get_code_artifact(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        for artifact_id in task.input_artifact_ids:
            artifact = state.artifacts.get(
                artifact_id
            )

            if (
                artifact
                and artifact.type
                == ArtifactType.CODE
            ):
                return artifact

        for artifact in state.artifacts.values():
            if artifact.type == ArtifactType.CODE:
                return artifact

        raise SecurityGenerationError(
            "CODE artifact not found."
        )

    def _validate_output(
        self,
        raw_output: str,
    ) -> SecurityReport:
        parsed = json.loads(
            raw_output
        )

        return SecurityReport.model_validate(
            parsed
        )

    def _validate_reviewed_files(
        self,
        report: SecurityReport,
        code_artifact: Artifact,
    ) -> None:
        valid_paths = {
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
        }

        if not valid_paths:
            raise SecurityGenerationError(
                "CODE artifact contains no valid files."
            )

        for path in report.reviewed_files:
            if path not in valid_paths:
                raise SecurityGenerationError(
                    "Security Agent referenced "
                    f"unknown reviewed file: {path}"
                )

        for finding in report.findings:
            for path in finding.affected_files:
                if path not in valid_paths:
                    raise SecurityGenerationError(
                        "Security finding referenced "
                        f"unknown file: {path}"
                    )

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        code_artifact = self._get_code_artifact(
            task,
            state,
        )

        system_prompt = (
            "You are the Security Agent inside NEXUS, "
            "an autonomous AI software engineering system. "
            "Perform defensive application-security review "
            "of generated source code. "
            "Return machine-valid JSON only."
        )

        prompt = f"""
GENERATED CODE BUNDLE:

{json.dumps(code_artifact.content, indent=2)}

Perform a defensive security review.

Return exactly one JSON object containing:

passed
risk_score
summary
findings
reviewed_files
positive_controls
recommended_actions

FINDING FORMAT:

Each finding must contain:

title
severity
category
affected_files
evidence
impact
recommendation

Allowed severity values:

info
low
medium
high
critical

Rules:

- risk_score must be between 0 and 100
- higher risk_score means greater security risk
- reviewed_files must contain only files present
  in GENERATED CODE BUNDLE
- affected_files must contain only files present
  in GENERATED CODE BUNDLE
- never invent file paths
- findings may be an empty array when no issues exist
- positive_controls must describe security practices
  already present in the generated implementation
- recommended_actions must contain concrete improvements
- inspect for hard-coded secrets
- inspect for unsafe subprocess or shell usage
- inspect for path traversal risks
- inspect for insecure file handling
- inspect for injection risks
- inspect for authentication/authorization weaknesses
- inspect for unsafe deserialization
- inspect for missing input validation
- inspect for insecure network or API behavior
- inspect for exposed credentials
- inspect for dangerous dependency usage
- base findings only on the supplied code
- do not invent vulnerabilities
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
                report = self._validate_output(
                    raw_output
                )

                self._validate_reviewed_files(
                    report,
                    code_artifact,
                )

                return Artifact(
                    type=ArtifactType.SECURITY_REPORT,
                    name="security_review",
                    content=report.model_dump(),
                    created_by=self.role,
                    metadata={
                        "validation_attempts":
                            attempt + 1,
                        "finding_count":
                            len(report.findings),
                        "risk_score":
                            report.risk_score,
                    },
                )

            except (
                json.JSONDecodeError,
                ValidationError,
                SecurityGenerationError,
            ) as exc:
                last_error = exc

                prompt = f"""
The previous security review failed validation.

ERROR:

{exc}

PREVIOUS RESPONSE:

{raw_output}

Repair the security review.

Return exactly one JSON object containing:

passed
risk_score
summary
findings
reviewed_files
positive_controls
recommended_actions

Every finding must contain:

title
severity
category
affected_files
evidence
impact
recommendation

Allowed severity values:

info
low
medium
high
critical

Rules:

- risk_score must be between 0 and 100
- reference only files that exist in the supplied code
- do not invent vulnerabilities
- return JSON only
"""

        raise SecurityGenerationError(
            "Security review could not be validated "
            f"after retries: {last_error}"
        )
