from fastapi.testclient import (
    TestClient,
)

from app.agents.base import (
    BaseAgent,
)
from app.agents.registry import (
    AgentRegistry,
)

from app.api.app import (
    create_app,
)
from app.api.control_plane import (
    NexusControlPlane,
)

from app.core.engine import (
    NexusEngine,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)

from app.governance.budget import (
    ResourceBudgetGuard,
)
from app.governance.limits import (
    ConcurrencyGuard,
    RateLimitConfig,
    SlidingWindowRateLimiter,
)
from app.governance.models import (
    PolicyEffect,
    PolicyRule,
    ResourceBudget,
)
from app.governance.policy import (
    PolicyEngine,
)
from app.governance.service import (
    GovernanceService,
)

from app.jobs.manager import (
    JobManager,
)
from app.jobs.queue import (
    PriorityJobQueue,
)
from app.jobs.worker import (
    WorkflowWorker,
)


class CountingAgent(
    BaseAgent
):
    role = AgentRole.CODER

    def __init__(
        self,
        counter,
    ):
        self.counter = counter

    def execute(
        self,
        task,
        state,
    ):
        self.counter[
            "executions"
        ] = (
            self.counter.get(
                "executions",
                0,
            )
            + 1
        )

        return Artifact(
            type=ArtifactType.CODE,
            name="governed_code",
            content={
                "files": [],
            },
            created_by=self.role,
        )


def build_governance(
    *,
    rules=None,
    max_tasks=None,
    rate_requests=None,
    max_concurrent=None,
):
    policy_engine = PolicyEngine(
        rules=rules or [],
        default_effect=(
            PolicyEffect.ALLOW
        ),
    )

    budget_guard = None

    if max_tasks is not None:
        budget_guard = (
            ResourceBudgetGuard(
                ResourceBudget(
                    max_tasks=max_tasks
                )
            )
        )

    rate_limiter = None

    if rate_requests is not None:
        rate_limiter = (
            SlidingWindowRateLimiter(
                RateLimitConfig(
                    max_requests=(
                        rate_requests
                    ),
                    window_seconds=60,
                )
            )
        )

    concurrency_guard = None

    if max_concurrent is not None:
        concurrency_guard = (
            ConcurrencyGuard(
                max_concurrent=(
                    max_concurrent
                ),
                max_per_subject=1,
            )
        )

    return GovernanceService(
        policy_engine=policy_engine,
        budget_guard=budget_guard,
        rate_limiter=rate_limiter,
        concurrency_guard=(
            concurrency_guard
        ),
    )


def build_stack(
    governance_service,
):
    counter = {}

    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        lambda: CountingAgent(
            counter
        ),
    )

    engine = NexusEngine(
        registry=registry
    )

    engine.governance_service = (
        governance_service
    )

    worker = WorkflowWorker(
        engine=engine
    )

    manager = JobManager(
        queue=PriorityJobQueue(),
        worker=worker,
        governance_service=(
            governance_service
        ),
    )

    plane = NexusControlPlane(
        engine,
        job_manager=manager,
    )

    app = create_app(
        plane
    )

    return (
        TestClient(app),
        plane,
        manager,
        governance_service,
        counter,
    )


def create_run_with_task(
    client,
    plane,
):
    response = client.post(
        "/runs",
        json={
            "user_request":
                "Execute governed workflow."
        },
    )

    assert (
        response.status_code
        == 201
    )

    run_id = response.json()[
        "run_id"
    ]

    state = plane.get_state(
        run_id
    )

    task = AgentTask(
        title="Generate code",
        description=(
            "Generate governed code."
        ),
        assigned_agent=(
            AgentRole.CODER
        ),
    )

    state.add_task(
        task
    )

    return run_id


def submit_job(
    client,
    run_id,
):
    response = client.post(
        "/jobs",
        json={
            "run_id": run_id,
            "priority": "normal",
            "max_attempts": 1,
            "metadata": {
                "source":
                    "phase-18-e2e"
            },
        },
    )

    assert (
        response.status_code
        == 201
    )

    return response.json()[
        "job_id"
    ]


def test_allowed_workflow_executes():
    governance = build_governance()

    (
        client,
        plane,
        _,
        _,
        counter,
    ) = build_stack(
        governance
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    response = client.post(
        f"/jobs/{job_id}/execute"
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "success"
        ]
        is True
    )

    assert (
        response.json()[
            "status"
        ]
        == "completed"
    )

    assert (
        counter["executions"]
        == 1
    )


def test_denied_workflow_never_reaches_agent():
    governance = build_governance(
        rules=[
            PolicyRule(
                id="deny-workflows",
                action="workflow.run",
                effect=(
                    PolicyEffect.DENY
                ),
                reason=(
                    "Workflow execution "
                    "is disabled."
                ),
            )
        ]
    )

    (
        client,
        plane,
        _,
        _,
        counter,
    ) = build_stack(
        governance
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    response = client.post(
        f"/jobs/{job_id}/execute"
    )

    assert (
        response.status_code
        == 400
    )

    assert (
        "disabled"
        in response.json()[
            "detail"
        ]
    )

    assert (
        counter.get(
            "executions",
            0,
        )
        == 0
    )


def test_resource_budget_blocks_large_workflow():
    governance = build_governance(
        max_tasks=0
    )

    (
        client,
        plane,
        _,
        _,
        counter,
    ) = build_stack(
        governance
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    response = client.post(
        f"/jobs/{job_id}/execute"
    )

    assert (
        response.status_code
        == 400
    )

    assert (
        "Resource budget exceeded"
        in response.json()[
            "detail"
        ]
    )

    assert (
        counter.get(
            "executions",
            0,
        )
        == 0
    )


def test_rate_limit_blocks_repeated_run():
    governance = build_governance(
        rate_requests=1
    )

    (
        client,
        plane,
        _,
        _,
        counter,
    ) = build_stack(
        governance
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    first_job = submit_job(
        client,
        run_id,
    )

    first = client.post(
        f"/jobs/{first_job}/execute"
    )

    assert (
        first.status_code
        == 200
    )

    second_job = submit_job(
        client,
        run_id,
    )

    second = client.post(
        f"/jobs/{second_job}/execute"
    )

    assert (
        second.status_code
        == 400
    )

    assert (
        "Rate limit exceeded"
        in second.json()[
            "detail"
        ]
    )

    assert (
        counter["executions"]
        == 1
    )


def test_concurrency_limit_blocks_execution():
    governance = build_governance(
        max_concurrent=1
    )

    (
        client,
        plane,
        _,
        service,
        counter,
    ) = build_stack(
        governance
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    service.concurrency_guard.acquire(
        run_id
    )

    try:
        response = client.post(
            f"/jobs/{job_id}/execute"
        )

        assert (
            response.status_code
            == 400
        )

        assert (
            "concurrency"
            in response.json()[
                "detail"
            ].lower()
        )

        assert (
            counter.get(
                "executions",
                0,
            )
            == 0
        )

    finally:
        service.concurrency_guard.release(
            run_id
        )


def test_approval_policy_is_detected():
    governance = build_governance(
        rules=[
            PolicyRule(
                id="approval-rule",
                action="tool.sensitive",
                effect=(
                    PolicyEffect
                    .REQUIRE_APPROVAL
                ),
                reason=(
                    "Sensitive tool requires "
                    "human approval."
                ),
            )
        ]
    )

    decision = governance.evaluate(
        action="tool.sensitive",
        subject="run-approval",
    )

    assert (
        decision.requires_approval
        is True
    )

    assert (
        decision.policy.effect
        == PolicyEffect
        .REQUIRE_APPROVAL
    )


def test_governance_releases_concurrency_after_success():
    governance = build_governance(
        max_concurrent=1
    )

    (
        client,
        plane,
        _,
        service,
        _,
    ) = build_stack(
        governance
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    response = client.post(
        f"/jobs/{job_id}/execute"
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        service
        .concurrency_guard
        .active_total()
        == 0
    )


def test_production_runtime_shares_governance():
    from app.api.server import (
        build_job_manager,
    )
    from app.core.runtime import (
        build_nexus_engine,
    )

    engine = build_nexus_engine(
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_evaluation=False,
        enable_observability=False,
        enable_checkpointing=False,
    )

    manager = build_job_manager(
        engine
    )

    assert (
        engine.governance_service
        is not None
    )

    assert (
        engine.tool_runtime
        .governance_service
        is engine.governance_service
    )

    assert (
        manager.governance_service
        is engine.governance_service
    )
