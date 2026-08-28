import json

from app.agents.architect import ArchitectAgent
from app.agents.critic import CriticAgent
from app.agents.debugger import DebuggerAgent
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.runtime import (
    build_memory_manager,
    build_memory_retriever,
)
from app.core.state import NexusState


class CapturingLLM:
    def __init__(
        self,
        response,
    ):
        self.response = response
        self.last_user_prompt = None

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
        max_tokens=None,
        json_schema=None,
        Schema_name="nexus_structured_output",
        strict_schema=True,
        reasoning_effort=None,
    ):
        self.last_user_prompt = user_prompt
        return self.response


def build_architect_response():
    return json.dumps(
        {
            "architecture_style": (
                "Modular FastAPI architecture"
            ),
            "components": [
                {
                    "name": "API",
                    "responsibility": (
                        "Expose HTTP endpoints"
                    ),
                    "technology": "FastAPI",
                }
            ],
            "data_flow": [
                "Request enters API",
            ],
            "technology_stack": [
                "Python",
                "FastAPI",
            ],
            "interfaces": [
                "REST API",
            ],
            "security_considerations": [
                "Validate all input.",
            ],
            "design_decisions": [
                "Use modular components.",
            ],
            "research_influences": [
                "FastAPI selected from current research.",
            ],
        }
    )


def build_critic_response():
    return json.dumps(
        {
            "verdict": "accept",
            "quality_score": 92,
            "summary": (
                "Current implementation passes "
                "the quality gate."
            ),
            "requirements_satisfied": True,
            "architecture_acceptable": True,
            "implementation_acceptable": True,
            "tests_acceptable": True,
            "security_acceptable": True,
            "issues": [],
            "strengths": [
                "Tests pass.",
            ],
            "required_improvements": [],
            "final_recommendation": (
                "Accept implementation."
            ),
        }
    )


def build_debugger_response():
    return json.dumps(
        {
            "root_cause": (
                "Incorrect return value"
            ),
            "failure_summary": (
                "Assertion expected True"
            ),
            "patches": [
                {
                    "path": "app.py",
                    "new_content": (
                        "def run():\n"
                        "    return True\n"
                    ),
                    "reason": (
                        "Fix incorrect return value."
                    ),
                }
            ],
            "retry_test_commands": [
                "pytest -v",
            ],
            "confidence": 0.95,
            "notes": [
                "Minimal repair.",
            ],
        }
    )


def create_architect_state():
    state = NexusState(
        user_request=(
            "Build a secure FastAPI API "
            "using free tools."
        )
    )

    requirements = Artifact(
        type=ArtifactType.REQUIREMENTS,
        name="requirements",
        content={
            "objective": (
                "Build a secure FastAPI API."
            ),
            "constraints": [
                "Use free tools.",
            ],
        },
        created_by=AgentRole.REQUIREMENTS,
    )

    research = Artifact(
        type=ArtifactType.RESEARCH,
        name="research",
        content={
            "recommended_technologies": [
                "FastAPI",
            ],
        },
        created_by=AgentRole.RESEARCH,
    )

    state.add_artifact(
        requirements
    )

    state.add_artifact(
        research
    )

    task = AgentTask(
        title="Design architecture",
        description="Design architecture.",
        assigned_agent=AgentRole.ARCHITECT,
        input_artifact_ids=[
            requirements.id,
            research.id,
        ],
    )

    return state, task


def create_critic_state():
    state = NexusState(
        user_request=(
            "Build a secure FastAPI service."
        )
    )

    artifacts = [
        Artifact(
            type=ArtifactType.REQUIREMENTS,
            name="requirements",
            content={
                "objective": (
                    "Build secure FastAPI service."
                ),
            },
            created_by=AgentRole.REQUIREMENTS,
        ),
        Artifact(
            type=ArtifactType.ARCHITECTURE,
            name="architecture",
            content={
                "architecture_style": (
                    "Modular FastAPI architecture"
                ),
                "technology_stack": [
                    "FastAPI",
                ],
            },
            created_by=AgentRole.ARCHITECT,
        ),
        Artifact(
            type=ArtifactType.CODE,
            name="code",
            content={
                "files": [
                    {
                        "path": "app.py",
                        "content": "print('ok')",
                    }
                ],
            },
            created_by=AgentRole.CODER,
        ),
        Artifact(
            type=ArtifactType.TEST_RESULT,
            name="tests",
            content={
                "passed": True,
            },
            created_by=AgentRole.TESTER,
        ),
        Artifact(
            type=ArtifactType.SECURITY_REPORT,
            name="security",
            content={
                "passed": True,
                "risk_score": 10,
                "summary": (
                    "FastAPI security review passed."
                ),
                "findings": [],
            },
            created_by=AgentRole.SECURITY,
        ),
    ]

    for artifact in artifacts:
        state.add_artifact(
            artifact
        )

    task = AgentTask(
        title="Final quality gate",
        description="Evaluate quality.",
        assigned_agent=AgentRole.CRITIC,
        input_artifact_ids=[
            artifact.id
            for artifact in artifacts
        ],
    )

    return state, task


def create_debugger_state():
    state = NexusState(
        user_request=(
            "Build a FastAPI application."
        )
    )

    code = Artifact(
        type=ArtifactType.CODE,
        name="code",
        content={
            "files": [
                {
                    "path": "app.py",
                    "content": (
                        "def run():\n"
                        "    return False\n"
                    ),
                }
            ],
        },
        created_by=AgentRole.CODER,
    )

    tests = Artifact(
        type=ArtifactType.TEST_RESULT,
        name="tests",
        content={
            "passed": False,
            "summary": (
                "AssertionError expected True"
            ),
            "failed_command_names": [
                "pytest -v",
            ],
            "results": [
                {
                    "stderr": (
                        "AssertionError expected True"
                    ),
                    "stdout": "",
                }
            ],
        },
        created_by=AgentRole.TESTER,
    )

    state.add_artifact(
        code
    )

    state.add_artifact(
        tests
    )

    task = AgentTask(
        title="Debug implementation",
        description="Repair implementation.",
        assigned_agent=AgentRole.DEBUGGER,
        input_artifact_ids=[
            code.id,
            tests.id,
        ],
    )

    return state, task


def test_run_b_architect_uses_run_a_memory(
    tmp_path,
):
    memory_path = (
        tmp_path
        / "shared_memory.db"
    )

    manager_a = build_memory_manager(
        memory_db_path=str(
            memory_path
        )
    )

    manager_a.store.save(
        run_id="run-a",
        memory_type="critic",
        key="fastapi_quality_feedback",
        value={
            "summary": (
                "FastAPI design needed "
                "strict input validation."
            ),
            "required_improvements": [
                "Add strict FastAPI "
                "request validation."
            ],
        },
    )

    manager_b = build_memory_manager(
        memory_db_path=str(
            memory_path
        )
    )

    retriever_b = (
        build_memory_retriever(
            manager_b
        )
    )

    state_b, task_b = (
        create_architect_state()
    )

    fake_llm = CapturingLLM(
        build_architect_response()
    )

    architect = ArchitectAgent(
        llm_client=fake_llm,
        memory_retriever=retriever_b,
    )

    artifact = architect.execute(
        task_b,
        state_b,
    )

    assert (
        artifact.metadata[
            "memory_augmented"
        ]
        is True
    )

    assert (
        "strict input validation"
        in fake_llm.last_user_prompt
    )


def test_run_b_critic_uses_run_a_memory(
    tmp_path,
):
    memory_path = (
        tmp_path
        / "shared_memory.db"
    )

    manager_a = build_memory_manager(
        memory_db_path=str(
            memory_path
        )
    )

    manager_a.store.save(
        run_id="run-a",
        memory_type="security",
        key="fastapi_security_history",
        value={
            "summary": (
                "FastAPI input validation "
                "was previously weak."
            ),
            "risk_score": 70,
        },
    )

    manager_b = build_memory_manager(
        memory_db_path=str(
            memory_path
        )
    )

    retriever_b = (
        build_memory_retriever(
            manager_b
        )
    )

    state_b, task_b = (
        create_critic_state()
    )

    fake_llm = CapturingLLM(
        build_critic_response()
    )

    critic = CriticAgent(
        llm_client=fake_llm,
        memory_retriever=retriever_b,
    )

    artifact = critic.execute(
        task_b,
        state_b,
    )

    assert (
        artifact.metadata[
            "memory_augmented"
        ]
        is True
    )

    assert (
        "previously weak"
        in fake_llm.last_user_prompt
    )


def test_run_b_debugger_uses_run_a_repair(
    tmp_path,
):
    memory_path = (
        tmp_path
        / "shared_memory.db"
    )

    manager_a = build_memory_manager(
        memory_db_path=str(
            memory_path
        )
    )

    manager_a.store.save(
        run_id="run-a",
        memory_type="repair",
        key="boolean_assertion_fix",
        value={
            "root_cause": (
                "Function returned False "
                "instead of True."
            ),
            "failure_summary": (
                "AssertionError expected True"
            ),
            "patches": [
                {
                    "path": "app.py",
                    "reason": (
                        "Return True."
                    ),
                }
            ],
        },
    )

    manager_b = build_memory_manager(
        memory_db_path=str(
            memory_path
        )
    )

    retriever_b = (
        build_memory_retriever(
            manager_b
        )
    )

    state_b, task_b = (
        create_debugger_state()
    )

    fake_llm = CapturingLLM(
        build_debugger_response()
    )

    debugger = DebuggerAgent(
        llm_client=fake_llm,
        memory_retriever=retriever_b,
    )

    artifact = debugger.execute(
        task_b,
        state_b,
    )

    assert (
        artifact.metadata[
            "memory_augmented"
        ]
        is True
    )

    assert (
        "boolean_assertion_fix"
        in fake_llm.last_user_prompt
    )


def test_memory_persists_across_manager_instances(
    tmp_path,
):
    memory_path = (
        tmp_path
        / "shared_memory.db"
    )

    manager_a = build_memory_manager(
        memory_db_path=str(
            memory_path
        )
    )

    manager_a.store.save(
        run_id="run-a",
        memory_type="repair",
        key="persistent_fix",
        value={
            "summary": (
                "FastAPI failure fixed."
            )
        },
    )

    manager_b = build_memory_manager(
        memory_db_path=str(
            memory_path
        )
    )

    memories = (
        manager_b.store.get_by_type(
            "repair"
        )
    )

    assert len(memories) == 1

    assert (
        memories[0]["key"]
        == "persistent_fix"
    )


def test_run_b_does_not_retrieve_its_own_memory(
    tmp_path,
):
    memory_path = (
        tmp_path
        / "shared_memory.db"
    )

    manager = build_memory_manager(
        memory_db_path=str(
            memory_path
        )
    )

    state_b, task_b = (
        create_debugger_state()
    )

    manager.store.save(
        run_id=state_b.run_id,
        memory_type="repair",
        key="same_run_fix",
        value={
            "failure_summary": (
                "AssertionError expected True"
            ),
            "root_cause": (
                "Function returned False."
            ),
        },
    )

    retriever = (
        build_memory_retriever(
            manager
        )
    )

    fake_llm = CapturingLLM(
        build_debugger_response()
    )

    debugger = DebuggerAgent(
        llm_client=fake_llm,
        memory_retriever=retriever,
    )

    artifact = debugger.execute(
        task_b,
        state_b,
    )

    assert (
        artifact.metadata[
            "memory_augmented"
        ]
        is False
    )

    assert (
        "same_run_fix"
        not in fake_llm.last_user_prompt
    )
