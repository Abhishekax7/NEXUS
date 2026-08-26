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
from app.memory.retriever import MemoryRetriever


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
        memory_retriever: Optional[
            MemoryRetriever
        ] = None,
        max_validation_retries: int = 2,
        memory_limit: int = 4,
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
                for artifact_type
                in missing
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

    def _build_memory_query(
        self,
        state: NexusState,
        artifacts: dict[
            ArtifactType,
            Artifact,
        ],
    ) -> str:
        requirements = artifacts[
            ArtifactType.REQUIREMENTS
        ].content

        architecture = artifacts[
            ArtifactType.ARCHITECTURE
        ].content

        security = artifacts[
            ArtifactType.SECURITY_REPORT
        ].content

        parts = [
            state.user_request,
            str(
                requirements.get(
                    "objective",
                    "",
                )
            ),
            str(
                architecture.get(
                    "architecture_style",
                    "",
                )
            ),
            str(
                security.get(
                    "summary",
                    "",
                )
            ),
        ]

        technology_stack = (
            architecture.get(
                "technology_stack",
                [],
            )
        )

        if isinstance(
            technology_stack,
            list,
        ):
            parts.extend(
                str(item)
                for item
                in technology_stack
            )

        return " ".join(
            part
            for part
            in parts
            if part
        )

    def _retrieve_memory_context(
        self,
        state: NexusState,
        artifacts: dict[
            ArtifactType,
            Artifact,
        ],
    ) -> list[dict]:
        if self.memory_retriever is None:
            return []

        query = self._build_memory_query(
            state,
            artifacts,
        )

        if not query.strip():
            return []

        results = (
            self.memory_retriever.retrieve(
                query=query,
                limit=self.memory_limit,
                memory_types=[
                    "critic",
                    "security",
                    "artifact",
                    "failure",
                    "repair",
                ],
                exclude_run_id=state.run_id,
            )
        )

        memories = []

        for result in results:
            memories.append(
                {
                    "score":
                        result.score,
                    "memory_type":
                        result.memory[
                            "memory_type"
                        ],
                    "run_id":
                        result.memory[
                            "run_id"
                        ],
                    "key":
                        result.memory[
                            "key"
                        ],
                    "value":
                        result.memory[
                            "value"
                        ],
                    "metadata":
                        result.memory[
                            "metadata"
                        ],
                }
            )

        return memories

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

        memory_context = (
            self._retrieve_memory_context(
                state,
                artifacts,
            )
        )

        system_prompt = (
            "You are the final Critic Agent inside "
            "NEXUS, an autonomous AI software "
            "engineering system. Act as a strict "
            "senior engineering quality gate. "
            "Evaluate only supplied current evidence. "
            "Relevant past NEXUS experience may be "
            "used to identify repeated mistakes or "
            "quality regressions, but current evidence "
            "always takes priority. "
            "Return valid JSON only."
        )

        prompt = f"""
CURRENT USER REQUEST:

{state.user_request}

CURRENT EXECUTION EVIDENCE:

{json.dumps(evidence, indent=2)}

RELEVANT PAST NEXUS EXPERIENCE:

{json.dumps(memory_context, indent=2)}

Evaluate the CURRENT completed software workflow.

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

- current requirements are the source of truth

- evaluate current requirements against
  current implementation

- evaluate current architecture consistency

- evaluate current implementation quality

- use the CURRENT test result as evidence

- use the CURRENT security report as evidence

- past memories are advisory only

- use previous critic feedback to detect
  repeated quality problems

- use previous security feedback to detect
  repeated security weaknesses

- use previous failures and repairs only when
  relevant to current evidence

- never downgrade current evidence because
  an old run succeeded

- never mark current tests as successful based
  on previous runs

- never mark current security as safe based
  on previous runs

- do not invent failures

- do not invent successful tests

- do not invent security findings

- accept only when all five current
  quality gates pass

- accept must have an empty
  required_improvements list

- revise means the implementation is recoverable

- reject means major redesign or fundamental
  correction is needed

- revise or reject must contain
  required improvements

- strengths must contain at least one
  concrete strength

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
                        "memory_context_count":
                            len(memory_context),
                        "memory_augmented":
                            bool(memory_context),
                    },
                )

            except (
                json.JSONDecodeError,
                ValidationError,
                CriticGenerationError,
            ) as exc:
                last_error = exc

                prompt = f"""
Your previous quality-gate response
failed validation.

ERROR:

{exc}

PREVIOUS RESPONSE:

{raw_output}

CURRENT EXECUTION EVIDENCE:

{json.dumps(evidence, indent=2)}

RELEVANT PAST EXPERIENCE:

{json.dumps(memory_context, indent=2)}

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

- accept requires all five current gates
  to be true

- accept requires required_improvements
  to be empty

- revise or reject requires at least one
  required improvement

- current evidence takes priority over memory

- past experience is advisory only

- do not invent evidence

- return JSON only
"""

        raise CriticGenerationError(
            "Critic report could not be "
            "validated after retries: "
            f"{last_error}"
        )

