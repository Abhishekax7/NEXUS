import json
from typing import Optional

from pydantic import ValidationError

from app.core.llm import LLMClient
from app.core.models import (
    AgentRole,
    TaskStatus,
)
from app.core.replanning import (
    ReplanningDecision,
    ReplanningError,
    validate_replanning_decision,
)
from app.core.state import NexusState


class ReplannerGenerationError(Exception):
    """Raised when the replanner cannot produce a valid decision."""


class ReplannerAgent:
    """
    Evaluates the current NEXUS execution state and
    decides whether the remaining plan should change.

    The Replanner does not mutate NexusState itself.
    It only returns a validated ReplanningDecision.
    """

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

    def _build_state_snapshot(
        self,
        state: NexusState,
    ) -> dict:
        tasks = []

        for task in state.tasks.values():
            tasks.append(
                {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "assigned_agent":
                        task.assigned_agent.value,
                    "status": task.status.value,
                    "dependencies":
                        list(task.dependencies),
                    "input_artifact_ids":
                        list(
                            task.input_artifact_ids
                        ),
                    "output_artifact_ids":
                        list(
                            task.output_artifact_ids
                        ),
                }
            )

        artifacts = []

        for artifact in state.artifacts.values():
            artifacts.append(
                {
                    "id": artifact.id,
                    "name": artifact.name,
                    "type": artifact.type.value,
                    "created_by":
                        artifact.created_by.value,
                    "content": artifact.content,
                    "metadata": artifact.metadata,
                }
            )

        return {
            "run_id": state.run_id,
            "user_request": state.user_request,
            "iteration": state.iteration,
            "completed": state.completed,
            "failed": state.failed,
            "errors": list(state.errors),
            "tasks": tasks,
            "artifacts": artifacts,
        }

    def _validate_semantics(
        self,
        decision: ReplanningDecision,
        state: NexusState,
    ) -> None:
        validate_replanning_decision(
            decision
        )

        if decision.target_task_id:
            if (
                decision.target_task_id
                not in state.tasks
            ):
                raise ReplanningError(
                    "target_task_id does not "
                    "exist in current state."
                )

            target = state.tasks[
                decision.target_task_id
            ]

            if (
                target.status
                == TaskStatus.RUNNING
            ):
                raise ReplanningError(
                    "Cannot remove or replace "
                    "a running task."
                )

            if (
                target.status
                == TaskStatus.COMPLETED
            ):
                raise ReplanningError(
                    "Cannot remove or replace "
                    "a completed task."
                )

        if (
            decision.proposed_task
            is not None
        ):
            proposed = (
                decision.proposed_task
            )

            if (
                proposed.assigned_agent
                == AgentRole.CRITIC
                and not proposed.depends_on_roles
            ):
                raise ReplanningError(
                    "A proposed critic task must "
                    "depend on prior work."
                )

    def decide(
        self,
        state: NexusState,
    ) -> ReplanningDecision:
        snapshot = (
            self._build_state_snapshot(
                state
            )
        )

        system_prompt = (
            "You are the Replanner inside NEXUS, "
            "an autonomous multi-agent software "
            "engineering system. "
            "Inspect the CURRENT execution state "
            "and determine whether the remaining "
            "workflow plan must change. "
            "Prefer KEEP_PLAN unless current "
            "evidence demonstrates a concrete "
            "reason to modify the plan. "
            "Never invent failures or requirements. "
            "Return valid JSON only."
        )

        prompt = f"""
USER REQUEST:

{state.user_request}

CURRENT NEXUS STATE:

{json.dumps(snapshot, indent=2)}

Decide whether the remaining execution plan
should change.

Return exactly one JSON object with:

should_replan
reason
action
proposed_task
target_task_id
confidence
evidence

Allowed action values:

keep_plan
add_task
remove_task
replace_task

If proposed_task is present, it must contain:

title
description
assigned_agent
depends_on_roles
metadata

Rules:

- use only CURRENT state evidence

- KEEP_PLAN should be preferred when the
  existing workflow remains sufficient

- should_replan=false requires action=keep_plan

- should_replan=true cannot use keep_plan

- add_task requires proposed_task

- remove_task requires target_task_id

- replace_task requires both target_task_id
  and proposed_task

- target_task_id must refer to an existing
  task in CURRENT NEXUS STATE

- never remove or replace a RUNNING task

- never remove or replace a COMPLETED task

- do not add duplicate work

- do not re-add work already represented by
  an unfinished task

- add a task only when new evidence makes
  additional work necessary

- replace a task when its planned work is
  fundamentally unsuitable before execution

- remove a task only when it has become
  unnecessary before execution

- security findings may justify additional
  implementation or verification work

- test failures may justify additional
  implementation/debugging work

- critic feedback may justify additional
  implementation, architecture, testing,
  or security work

- confidence must be between 0 and 1

- evidence must contain concrete facts from
  the current state

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
                parsed = json.loads(
                    raw_output
                )

                decision = (
                    ReplanningDecision
                    .model_validate(
                        parsed
                    )
                )

                self._validate_semantics(
                    decision,
                    state,
                )

                return decision

            except (
                json.JSONDecodeError,
                ValidationError,
                ReplanningError,
            ) as exc:
                last_error = exc

                prompt = f"""
Your previous replanning decision
failed validation.

ERROR:

{exc}

PREVIOUS RESPONSE:

{raw_output}

CURRENT NEXUS STATE:

{json.dumps(snapshot, indent=2)}

Repair the decision.

Return exactly one valid JSON object
with:

should_replan
reason
action
proposed_task
target_task_id
confidence
evidence

Remember:

- false => keep_plan
- true => add_task, remove_task,
  or replace_task
- add_task requires proposed_task
- remove_task requires target_task_id
- replace_task requires both
- target task must exist
- RUNNING tasks cannot be removed
  or replaced
- COMPLETED tasks cannot be removed
  or replaced
- do not invent evidence
- return JSON only
"""

        raise ReplannerGenerationError(
            "Replanning decision could not "
            "be validated after retries: "
            f"{last_error}"
        )
