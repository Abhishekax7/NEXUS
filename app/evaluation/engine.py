from statistics import mean

from app.core.models import (
    AgentRole,
    ArtifactType,
    TaskStatus,
)
from app.core.state import NexusState
from app.evaluation.models import (
    AgentEvaluation,
    EvaluationDimension,
    EvaluationError,
    EvaluationStatus,
    MetricScore,
    WorkflowEvaluation,
)


class EvaluationEngine:
    """
    Deterministically evaluates a completed
    NEXUS workflow from NexusState.

    No LLM is used here. Scores are derived
    only from recorded workflow evidence.
    """

    def __init__(
        self,
        pass_threshold: float = 80.0,
        warn_threshold: float = 60.0,
    ):
        if not (
            0.0
            <= warn_threshold
            <= pass_threshold
            <= 100.0
        ):
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= warn <= pass <= 100."
            )

        self.pass_threshold = (
            pass_threshold
        )

        self.warn_threshold = (
            warn_threshold
        )

    def _status_for_score(
        self,
        score: float,
    ) -> EvaluationStatus:
        if score >= self.pass_threshold:
            return EvaluationStatus.PASS

        if score >= self.warn_threshold:
            return EvaluationStatus.WARN

        return EvaluationStatus.FAIL

    def _metric(
        self,
        dimension: EvaluationDimension,
        score: float,
        reason: str,
        evidence: list[str],
    ) -> MetricScore:
        score = max(
            0.0,
            min(
                100.0,
                float(score),
            ),
        )

        return MetricScore(
            dimension=dimension,
            score=score,
            status=self._status_for_score(
                score
            ),
            reason=reason,
            evidence=evidence,
        )

    def _task_completion_metric(
        self,
        state: NexusState,
    ) -> MetricScore:
        tasks = list(
            state.tasks.values()
        )

        if not tasks:
            return self._metric(
                EvaluationDimension.TASK_COMPLETION,
                0.0,
                "No workflow tasks were recorded.",
                [
                    "Task count: 0"
                ],
            )

        completed = sum(
            1
            for task in tasks
            if task.status
            == TaskStatus.COMPLETED
        )

        score = (
            completed
            / len(tasks)
        ) * 100.0

        return self._metric(
            EvaluationDimension.TASK_COMPLETION,
            score,
            (
                f"{completed} of {len(tasks)} "
                "tasks completed."
            ),
            [
                f"Completed tasks: {completed}",
                f"Total tasks: {len(tasks)}",
            ],
        )

    def _artifact_quality_metric(
        self,
        state: NexusState,
    ) -> MetricScore:
        artifacts = list(
            state.artifacts.values()
        )

        if not artifacts:
            return self._metric(
                EvaluationDimension.ARTIFACT_QUALITY,
                0.0,
                "No artifacts were produced.",
                [
                    "Artifact count: 0"
                ],
            )

        non_empty = 0

        for artifact in artifacts:
            content = artifact.content

            if isinstance(
                content,
                dict,
            ):
                if content:
                    non_empty += 1

            elif content:
                non_empty += 1

        score = (
            non_empty
            / len(artifacts)
        ) * 100.0

        return self._metric(
            EvaluationDimension.ARTIFACT_QUALITY,
            score,
            (
                f"{non_empty} of {len(artifacts)} "
                "artifacts contain non-empty output."
            ),
            [
                f"Non-empty artifacts: {non_empty}",
                f"Total artifacts: {len(artifacts)}",
            ],
        )

    def _grounding_metric(
        self,
        state: NexusState,
    ) -> MetricScore:
        grounded = []
        observed = 0

        for artifact in (
            state.artifacts.values()
        ):
            metadata = (
                artifact.metadata
                or {}
            )

            grounding_flags = [
                key
                for key in metadata
                if key.startswith(
                    "grounded_in_"
                )
            ]

            if not grounding_flags:
                continue

            observed += 1

            if all(
                bool(
                    metadata[key]
                )
                for key in grounding_flags
            ):
                grounded.append(
                    artifact.name
                )

        if observed == 0:
            return self._metric(
                EvaluationDimension.GROUNDING,
                70.0,
                (
                    "No explicit grounding metadata "
                    "was available."
                ),
                [
                    "Grounded artifacts observed: 0"
                ],
            )

        score = (
            len(grounded)
            / observed
        ) * 100.0

        return self._metric(
            EvaluationDimension.GROUNDING,
            score,
            (
                f"{len(grounded)} of {observed} "
                "grounded artifacts passed their "
                "grounding checks."
            ),
            [
                (
                    "Grounded artifacts: "
                    + (
                        ", ".join(grounded)
                        if grounded
                        else "none"
                    )
                )
            ],
        )

    def _test_quality_metric(
        self,
        state: NexusState,
    ) -> MetricScore:
        test_artifacts = [
            artifact
            for artifact
            in state.artifacts.values()
            if artifact.type
            == ArtifactType.TEST_RESULT
        ]

        if not test_artifacts:
            return self._metric(
                EvaluationDimension.TEST_QUALITY,
                0.0,
                "No test result artifact exists.",
                [
                    "Test artifacts: 0"
                ],
            )

        latest = test_artifacts[-1]

        passed = bool(
            latest.content.get(
                "passed"
            )
        )

        score = (
            100.0
            if passed
            else 20.0
        )

        return self._metric(
            EvaluationDimension.TEST_QUALITY,
            score,
            (
                "Latest test result passed."
                if passed
                else "Latest test result failed."
            ),
            [
                str(
                    latest.content.get(
                        "summary",
                        "No test summary.",
                    )
                )
            ],
        )

    def _security_metric(
        self,
        state: NexusState,
    ) -> MetricScore:
        security_artifacts = [
            artifact
            for artifact
            in state.artifacts.values()
            if artifact.type
            == ArtifactType.SECURITY_REPORT
        ]

        if not security_artifacts:
            return self._metric(
                EvaluationDimension.SECURITY,
                0.0,
                (
                    "No security report artifact "
                    "exists."
                ),
                [
                    "Security reports: 0"
                ],
            )

        latest = security_artifacts[-1]

        content = latest.content

        passed = content.get(
            "passed"
        )

        risk_score = content.get(
            "risk_score"
        )

        if isinstance(
            risk_score,
            (int, float),
        ):
            score = (
                100.0
                - float(risk_score)
            )

        elif passed is True:
            score = 100.0

        elif passed is False:
            score = 30.0

        else:
            score = 60.0

        return self._metric(
            EvaluationDimension.SECURITY,
            score,
            str(
                content.get(
                    "summary",
                    "Security evaluation recorded.",
                )
            ),
            [
                f"Security passed: {passed}",
                f"Risk score: {risk_score}",
            ],
        )

    def _repair_efficiency_metric(
        self,
        state: NexusState,
    ) -> MetricScore:
        repair_memories = []

        if hasattr(
            state,
            "metadata"
        ):
            repair_memories = (
                state.metadata.get(
                    "repair_history",
                    [],
                )
                or []
            )

        debug_artifacts = [
            artifact
            for artifact
            in state.artifacts.values()
            if artifact.type
            == ArtifactType.DEBUG_REPORT
        ]

        repair_count = max(
            len(repair_memories),
            len(debug_artifacts),
        )

        if repair_count == 0:
            return self._metric(
                EvaluationDimension.REPAIR_EFFICIENCY,
                100.0,
                (
                    "No repair cycle was required."
                ),
                [
                    "Repair attempts: 0"
                ],
            )

        score = max(
            30.0,
            100.0
            - (
                repair_count
                * 20.0
            ),
        )

        return self._metric(
            EvaluationDimension.REPAIR_EFFICIENCY,
            score,
            (
                f"{repair_count} repair cycle(s) "
                "were required."
            ),
            [
                f"Repair cycles: {repair_count}"
            ],
        )

    def _replanning_efficiency_metric(
        self,
        state: NexusState,
    ) -> MetricScore:
        replan_count = int(
            state.metadata.get(
                "replan_count",
                0,
            )
        )

        if replan_count == 0:
            score = 100.0
            reason = (
                "Workflow completed without "
                "requiring replanning."
            )

        else:
            score = max(
                40.0,
                100.0
                - (
                    replan_count
                    * 15.0
                ),
            )

            reason = (
                f"Workflow required "
                f"{replan_count} replan(s)."
            )

        return self._metric(
            EvaluationDimension.REPLANNING_EFFICIENCY,
            score,
            reason,
            [
                f"Replan count: {replan_count}"
            ],
        )

    def _tool_use_metric(
        self,
        state: NexusState,
    ) -> MetricScore:
        tool_artifacts = []

        for artifact in (
            state.artifacts.values()
        ):
            metadata = (
                artifact.metadata
                or {}
            )

            if (
                metadata.get(
                    "dynamic_tools_enabled"
                )
                is not True
            ):
                continue

            tool_artifacts.append(
                artifact
            )

        if not tool_artifacts:
            return self._metric(
                EvaluationDimension.TOOL_USE,
                75.0,
                (
                    "No dynamic-tool evidence was "
                    "recorded for this workflow."
                ),
                [
                    "Tool-aware artifacts: 0"
                ],
            )

        successful = 0

        for artifact in tool_artifacts:
            metadata = artifact.metadata

            used = bool(
                metadata.get(
                    "dynamic_tool_used"
                )
            )

            success = metadata.get(
                "dynamic_tool_success"
            )

            if not used:
                successful += 1

            elif success is True:
                successful += 1

        score = (
            successful
            / len(tool_artifacts)
        ) * 100.0

        return self._metric(
            EvaluationDimension.TOOL_USE,
            score,
            (
                f"{successful} of "
                f"{len(tool_artifacts)} "
                "tool-routing decisions were "
                "successful."
            ),
            [
                (
                    "Tool-aware artifacts: "
                    f"{len(tool_artifacts)}"
                )
            ],
        )

    def _critic_quality_metric(
        self,
        state: NexusState,
    ) -> MetricScore:
        evaluations = [
            artifact
            for artifact
            in state.artifacts.values()
            if artifact.type
            == ArtifactType.EVALUATION
        ]

        if not evaluations:
            return self._metric(
                EvaluationDimension.CRITIC_QUALITY,
                0.0,
                "No final critic evaluation exists.",
                [
                    "Evaluation artifacts: 0"
                ],
            )

        latest = evaluations[-1]

        quality_score = (
            latest.content.get(
                "quality_score"
            )
        )

        verdict = (
            latest.content.get(
                "verdict"
            )
        )

        if isinstance(
            quality_score,
            (int, float),
        ):
            score = float(
                quality_score
            )

        elif verdict == "accept":
            score = 90.0

        elif verdict == "revise":
            score = 65.0

        elif verdict == "reject":
            score = 30.0

        else:
            score = 50.0

        return self._metric(
            EvaluationDimension.CRITIC_QUALITY,
            score,
            (
                f"Final critic verdict: "
                f"{verdict}"
            ),
            [
                f"Critic quality score: {quality_score}",
                f"Critic verdict: {verdict}",
            ],
        )

    def _workflow_reliability_metric(
        self,
        state: NexusState,
    ) -> MetricScore:
        if (
            state.completed
            and not state.failed
            and not state.errors
        ):
            score = 100.0

        elif (
            state.completed
            and not state.failed
        ):
            score = 80.0

        elif state.failed:
            score = 20.0

        else:
            score = 50.0

        return self._metric(
            EvaluationDimension.WORKFLOW_RELIABILITY,
            score,
            (
                "Workflow reliability derived "
                "from completion and failure state."
            ),
            [
                f"completed={state.completed}",
                f"failed={state.failed}",
                f"errors={len(state.errors)}",
            ],
        )

    def _evaluate_agent(
        self,
        role: AgentRole,
        state: NexusState,
    ) -> AgentEvaluation | None:
        role_tasks = [
            task
            for task
            in state.tasks.values()
            if task.assigned_agent
            == role
        ]

        role_artifacts = [
            artifact
            for artifact
            in state.artifacts.values()
            if artifact.created_by
            == role
        ]

        if (
            not role_tasks
            and not role_artifacts
        ):
            return None

        metrics = []

        if role_tasks:
            completed = sum(
                1
                for task in role_tasks
                if task.status
                == TaskStatus.COMPLETED
            )

            score = (
                completed
                / len(role_tasks)
            ) * 100.0

            metrics.append(
                self._metric(
                    EvaluationDimension.TASK_COMPLETION,
                    score,
                    (
                        f"{completed} of "
                        f"{len(role_tasks)} "
                        f"{role.value} tasks completed."
                    ),
                    [
                        (
                            f"{role.value} task "
                            f"count: {len(role_tasks)}"
                        )
                    ],
                )
            )

        if role_artifacts:
            non_empty = sum(
                1
                for artifact
                in role_artifacts
                if artifact.content
            )

            score = (
                non_empty
                / len(role_artifacts)
            ) * 100.0

            metrics.append(
                self._metric(
                    EvaluationDimension.ARTIFACT_QUALITY,
                    score,
                    (
                        f"{non_empty} of "
                        f"{len(role_artifacts)} "
                        f"{role.value} artifacts "
                        "were non-empty."
                    ),
                    [
                        (
                            f"{role.value} artifact "
                            f"count: {len(role_artifacts)}"
                        )
                    ],
                )
            )

        if not metrics:
            return None

        score = mean(
            metric.score
            for metric in metrics
        )

        strengths = [
            metric.reason
            for metric in metrics
            if metric.status
            == EvaluationStatus.PASS
        ]

        weaknesses = [
            metric.reason
            for metric in metrics
            if metric.status
            != EvaluationStatus.PASS
        ]

        return AgentEvaluation(
            agent_role=role.value,
            score=round(
                score,
                2,
            ),
            metrics=metrics,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    def evaluate(
        self,
        state: NexusState,
    ) -> WorkflowEvaluation:
        if not state.run_id:
            raise EvaluationError(
                "NexusState has no run_id."
            )

        metrics = [
            self._task_completion_metric(
                state
            ),
            self._artifact_quality_metric(
                state
            ),
            self._grounding_metric(
                state
            ),
            self._test_quality_metric(
                state
            ),
            self._security_metric(
                state
            ),
            self._repair_efficiency_metric(
                state
            ),
            self._replanning_efficiency_metric(
                state
            ),
            self._tool_use_metric(
                state
            ),
            self._critic_quality_metric(
                state
            ),
            self._workflow_reliability_metric(
                state
            ),
        ]

        overall_score = mean(
            metric.score
            for metric in metrics
        )

        agent_evaluations = []

        for role in AgentRole:
            evaluation = (
                self._evaluate_agent(
                    role,
                    state,
                )
            )

            if evaluation is not None:
                agent_evaluations.append(
                    evaluation
                )

        strengths = [
            metric.reason
            for metric in metrics
            if metric.status
            == EvaluationStatus.PASS
        ]

        weaknesses = [
            metric.reason
            for metric in metrics
            if metric.status
            == EvaluationStatus.FAIL
        ]

        recommendations = [
            (
                f"Improve {metric.dimension.value}: "
                f"{metric.reason}"
            )
            for metric in metrics
            if metric.status
            != EvaluationStatus.PASS
        ]

        failed_metrics = sum(
            1
            for metric in metrics
            if metric.status
            == EvaluationStatus.FAIL
        )

        warned_metrics = sum(
            1
            for metric in metrics
            if metric.status
            == EvaluationStatus.WARN
        )

        regression_risk = min(
            1.0,
            (
                failed_metrics
                * 0.15
                + warned_metrics
                * 0.05
            ),
        )

        return WorkflowEvaluation(
            run_id=state.run_id,
            overall_score=round(
                overall_score,
                2,
            ),
            status=self._status_for_score(
                overall_score
            ),
            metrics=metrics,
            agent_evaluations=(
                agent_evaluations
            ),
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=(
                recommendations
            ),
            regression_risk=round(
                regression_risk,
                3,
            ),
        )
