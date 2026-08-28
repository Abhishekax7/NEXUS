from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------
# Make project root importable when running:
#
#     python scripts/demo_nexus.py
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.agents.orchestrator import OrchestratorAgent
from app.core.runtime import build_nexus_engine
from app.core.state import NexusState


DEFAULT_REQUEST = (
    "Build a production-ready Python REST API for a task "
    "management service. It should support creating, listing, "
    "updating, and deleting tasks, include input validation, "
    "tests, and a security review."
)


def heading(
    title: str,
) -> None:
    line = "=" * 72

    print()
    print(line)
    print(title)
    print(line)


def subheading(
    title: str,
) -> None:
    print()
    print(f"--- {title} ---")


def enum_value(
    value: Any,
) -> str:
    raw = getattr(
        value,
        "value",
        value,
    )

    return str(raw)


def short_id(
    value: str,
) -> str:
    return value[:8]


def build_demo_state(
    user_request: str,
) -> NexusState:
    orchestrator = OrchestratorAgent()

    return orchestrator.create_initial_plan(
        user_request
    )


def print_plan(
    state: NexusState,
) -> None:
    heading(
        "NEXUS INITIAL EXECUTION PLAN"
    )

    print(
        f"Run ID: {state.run_id}"
    )

    print(
        f"Tasks: {len(state.tasks)}"
    )

    for index, task_id in enumerate(
        state.execution_order,
        start=1,
    ):
        task = state.tasks[
            task_id
        ]

        dependencies = [
            short_id(dependency)
            for dependency
            in task.dependencies
        ]

        dependency_text = (
            ", ".join(dependencies)
            if dependencies
            else "none"
        )

        print(
            f"{index}. "
            f"{task.title}"
        )

        print(
            "   Agent: "
            f"{enum_value(task.assigned_agent)}"
        )

        print(
            "   Status: "
            f"{enum_value(task.status)}"
        )

        print(
            "   Dependencies: "
            f"{dependency_text}"
        )


def print_execution_summary(
    state: NexusState,
) -> None:
    heading(
        "WORKFLOW EXECUTION SUMMARY"
    )

    print(
        f"Run ID: {state.run_id}"
    )

    print(
        f"Completed: {state.completed}"
    )

    print(
        f"Failed: {state.failed}"
    )

    print(
        f"Iterations: {state.iteration}"
    )

    print(
        f"Tasks: {len(state.tasks)}"
    )

    print(
        f"Artifacts: {len(state.artifacts)}"
    )

    print(
        "Replans: "
        f"{state.metadata.get('replan_count', 0)}"
    )

    if state.errors:
        subheading(
            "Errors"
        )

        for error in state.errors:
            print(
                f"- {error}"
            )


def print_task_results(
    state: NexusState,
) -> None:
    heading(
        "AGENT EXECUTION RESULTS"
    )

    for index, task_id in enumerate(
        state.execution_order,
        start=1,
    ):
        task = state.tasks[
            task_id
        ]

        print(
            f"{index}. {task.title}"
        )

        print(
            "   Agent: "
            f"{enum_value(task.assigned_agent)}"
        )

        print(
            "   Status: "
            f"{enum_value(task.status)}"
        )

        print(
            "   Retries: "
            f"{task.retry_count}"
        )

        print(
            "   Output artifacts: "
            f"{len(task.output_artifact_ids)}"
        )

        if task.error:
            print(
                "   Error: "
                f"{task.error}"
            )


def print_artifacts(
    state: NexusState,
) -> None:
    heading(
        "GENERATED ARTIFACTS"
    )

    if not state.artifacts:
        print(
            "No artifacts were generated."
        )

        return

    for index, artifact in enumerate(
        state.artifacts.values(),
        start=1,
    ):
        print(
            f"{index}. {artifact.name}"
        )

        print(
            "   ID: "
            f"{artifact.id}"
        )

        print(
            "   Type: "
            f"{enum_value(artifact.type)}"
        )

        print(
            "   Created by: "
            f"{enum_value(artifact.created_by)}"
        )


def print_evaluation(
    state: NexusState,
) -> None:
    heading(
        "AUTOMATED EVALUATION"
    )

    evaluation = state.metadata.get(
        "evaluation"
    )

    if evaluation is None:
        print(
            "No evaluation result available."
        )

        return

    print(
        "Overall score: "
        f"{evaluation.get('overall_score')}"
    )

    print(
        "Status: "
        f"{evaluation.get('status')}"
    )

    metrics = evaluation.get(
        "metrics",
        [],
    )

    if metrics:
        subheading(
            "Evaluation dimensions"
        )

        for metric in metrics:
            print(
                "- "
                f"{metric.get('dimension')}: "
                f"{metric.get('score')} "
                f"({metric.get('status')})"
            )

    strengths = evaluation.get(
        "strengths",
        [],
    )

    if strengths:
        subheading(
            "Strengths"
        )

        for strength in strengths:
            print(
                f"- {strength}"
            )

    weaknesses = evaluation.get(
        "weaknesses",
        [],
    )

    if weaknesses:
        subheading(
            "Weaknesses"
        )

        for weakness in weaknesses:
            print(
                f"- {weakness}"
            )

    recommendations = evaluation.get(
        "recommendations",
        [],
    )

    if recommendations:
        subheading(
            "Recommendations"
        )

        for recommendation in recommendations:
            print(
                f"- {recommendation}"
            )

    benchmark = state.metadata.get(
        "evaluation_benchmark"
    )

    if benchmark is not None:
        subheading(
            "Regression benchmark"
        )

        print(
            json.dumps(
                benchmark,
                indent=2,
                default=str,
            )
        )


def print_observability(
    state: NexusState,
) -> None:
    heading(
        "OBSERVABILITY"
    )

    observability = (
        state.metadata.get(
            "observability"
        )
    )

    if observability is None:
        print(
            "No observability summary available."
        )

        return

    for key, value in (
        observability.items()
    ):
        print(
            f"{key}: {value}"
        )


def print_replanning(
    state: NexusState,
) -> None:
    heading(
        "AUTONOMOUS REPLANNING"
    )

    count = state.metadata.get(
        "replan_count",
        0,
    )

    print(
        f"Replan count: {count}"
    )

    history = state.metadata.get(
        "replan_history",
        [],
    )

    if not history:
        print(
            "No plan mutation was required."
        )

        return

    for index, mutation in enumerate(
        history,
        start=1,
    ):
        print(
            f"{index}. "
            f"{mutation.get('action')}"
        )

        print(
            "   Added task: "
            f"{mutation.get('added_task_id')}"
        )

        print(
            "   Removed task: "
            f"{mutation.get('removed_task_id')}"
        )

        print(
            "   Replaced task: "
            f"{mutation.get('replaced_task_id')}"
        )


def print_checkpointing(
    engine,
    state: NexusState,
) -> None:
    heading(
        "CHECKPOINTING & RECOVERY"
    )

    service = (
        engine.checkpoint_service
    )

    if service is None:
        print(
            "Checkpointing is disabled."
        )

        return

    info = service.recovery_info(
        state.run_id
    )

    print(
        "Recovery status: "
        f"{enum_value(info.status)}"
    )

    print(
        "Latest checkpoint ID: "
        f"{info.latest_checkpoint_id}"
    )

    print(
        "Sequence: "
        f"{info.sequence}"
    )

    checkpoint_type = (
        enum_value(info.checkpoint_type)
        if info.checkpoint_type
        is not None
        else None
    )

    print(
        "Checkpoint type: "
        f"{checkpoint_type}"
    )

    print(
        "Reason: "
        f"{info.reason}"
    )


def print_memory(
    engine,
) -> None:
    heading(
        "PERSISTENT MEMORY"
    )

    if engine.memory_manager is None:
        print(
            "Memory is disabled."
        )

        return

    print(
        "Persistent memory manager: enabled"
    )

    print(
        "Memory store: "
        f"{type(engine.memory_manager.store).__name__}"
    )


def print_tools(
    engine,
) -> None:
    heading(
        "TOOL RUNTIME"
    )

    if engine.tool_runtime is None:
        print(
            "Tool runtime is disabled."
        )

        return

    print(
        "Tool runtime: enabled"
    )

    print(
        "Tool registry: "
        f"{type(engine.tool_registry).__name__}"
    )

    print(
        "Runtime: "
        f"{type(engine.tool_runtime).__name__}"
    )


def print_governance(
    engine,
) -> None:
    heading(
        "GOVERNANCE"
    )

    governance = getattr(
        engine,
        "governance_service",
        None,
    )

    if governance is None:
        print(
            "Governance is disabled."
        )

        return

    print(
        "Governance service: enabled"
    )

    print(
        "Policy engine: "
        f"{type(governance.policy_engine).__name__}"
    )

    print(
        "Resource budget guard: "
        f"{type(governance.budget_guard).__name__}"
    )

    print(
        "Rate limiter: "
        f"{type(governance.rate_limiter).__name__}"
    )

    print(
        "Concurrency guard: "
        f"{type(governance.concurrency_guard).__name__}"
    )


def run_demo(
    user_request: str,
    resume_run_id: str | None = None,
) -> int:
    heading(
        "NEXUS — AUTONOMOUS AI ENGINEERING SYSTEM"
    )

    if resume_run_id is None:
        print(
            "User request:"
        )

        print(
            user_request
        )

    else:
        print(
            "Resume requested for persisted workflow:"
        )

        print(
            resume_run_id
        )

    heading(
        "BUILDING PRODUCTION RUNTIME"
    )

    engine = build_nexus_engine()

    print(
        "Production runtime initialized."
    )

    print_memory(
        engine
    )

    print_tools(
        engine
    )

    print_governance(
        engine
    )

    if resume_run_id is not None:
        heading(
            "RECOVERING PERSISTED WORKFLOW"
        )

        print(
            "Run ID:"
        )

        print(
            resume_run_id
        )

        try:
            state = engine.restore_run(
                resume_run_id,
                allow_failed=True,
            )

        except Exception as exc:
            print()
            print(
                "NEXUS recovery failed."
            )

            print(
                f"Error: {exc}"
            )

            return 1

        print(
            "Checkpoint restored successfully."
        )

        recovery_metadata = state.metadata.get(
            "recovered_from_checkpoint",
            {},
        )

        print(
            "Checkpoint ID: "
            f"{recovery_metadata.get('checkpoint_id')}"
        )

        print(
            "Checkpoint sequence: "
            f"{recovery_metadata.get('sequence')}"
        )

        print(
            "Checkpoint type: "
            f"{recovery_metadata.get('checkpoint_type')}"
        )

    else:
        state = build_demo_state(
            user_request
        )

    print_plan(
        state
    )

    heading(
        "EXECUTING MULTI-AGENT WORKFLOW"
    )

    try:
        result = engine.run(
            state
        )

    except Exception as exc:
        print()
        print(
            "NEXUS execution interrupted."
        )

        print(
            f"Error: {exc}"
        )

        print_execution_summary(
            state
        )

        print_task_results(
            state
        )

        print_checkpointing(
            engine,
            state,
        )

        print()
        print(
            "This workflow can be resumed with:"
        )

        print(
            "python scripts/demo_nexus.py "
            f"--resume {state.run_id}"
        )

        return 1

    print_execution_summary(
        result
    )

    print_task_results(
        result
    )

    print_artifacts(
        result
    )

    print_replanning(
        result
    )

    print_evaluation(
        result
    )

    print_observability(
        result
    )

    print_checkpointing(
        engine,
        result,
    )

    heading(
        "NEXUS DEMO COMPLETE"
    )

    print(
        "The autonomous workflow completed "
        "successfully."
    )

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the NEXUS autonomous "
            "AI engineering demo."
        )
    )

    parser.add_argument(
        "--request",
        type=str,
        default=DEFAULT_REQUEST,
        help=(
            "Engineering request for NEXUS "
            "to execute."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help=(
            "Resume a persisted NEXUS workflow "
            "from its latest checkpoint."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    return run_demo(
        args.request,
        resume_run_id=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
