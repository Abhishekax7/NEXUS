# NEXUS — Autonomous AI Engineering System

NEXUS is a multi-agent AI software engineering system that transforms a software request into a planned, implemented, tested, security-reviewed, and critically evaluated solution.

Instead of relying on a single LLM call to generate code, NEXUS coordinates specialized AI agents through a dependency-aware workflow and combines structured LLM outputs with deterministic execution, autonomous repair, checkpoint recovery, persistent memory, governance controls, evaluation, and observability.

The project explores production-oriented agentic AI architecture: how autonomous agents can plan work, exchange structured artifacts, execute tools safely, recover from failures, and evaluate their own results.

---

## Core Capabilities

- Multi-agent software engineering workflow
- Dependency-aware DAG scheduling
- Structured LLM outputs with schema validation
- Autonomous test → debug → patch → retest loop
- Dynamic workflow replanning
- Persistent cross-run memory
- Checkpointing and failed-run recovery
- Runtime capability enforcement
- Isolated generated-project workspaces
- Human approval gates
- Tool registry, selection, and execution runtime
- Security review
- Automated quality evaluation
- Historical regression benchmarking
- Governance and resource controls
- Event-driven execution
- Observability and tracing
- Background job execution
- FastAPI control plane
- Resumable end-to-end demo workflow

---

## System Architecture

```text
User Request
     |
     v
Orchestrator
     |
     v
Dependency-Aware Execution Plan
     |
     +--> Requirements Agent
     |
     +--> Research Agent
     |
     +--> Architect Agent
                |
                v
            Coder Agent
                |
                v
        Generated Workspace
                |
                v
            Tester Agent
                |
          Tests passing?
           /         \
         yes          no
          |            |
          |            v
          |      Debugger Agent
          |            |
          |            v
          |      Patch Applicator
          |            |
          |            v
          |          Retest
          |            |
          +<-----------+
          |
          +--> Security Agent
          |
          +--> Critic Agent
                  |
                  v
             Quality Gate
                  |
                  v
        Evaluation & Observability
```

NEXUS represents workflow communication through typed tasks and artifacts rather than relying only on free-form agent-to-agent messages.

This makes dependencies, provenance, persistence, validation, recovery, and evaluation explicit parts of the execution model.

---

## Specialized Agents

### Requirements Agent

Transforms the original request into structured engineering requirements including objectives, functional requirements, non-functional requirements, risks, and success criteria.

### Research Agent

Collects relevant technical context that can inform architecture and implementation decisions.

### Architect Agent

Transforms requirements and research into a structured software architecture and technology plan.

### Coder Agent

Generates a structured code bundle containing project files, dependencies, execution commands, test commands, and implementation notes.

Critical generated output is schema validated before entering the execution pipeline.

### Tester Agent

Materializes generated code into an isolated run workspace and executes validated test commands through the controlled command executor.

### Debugger Agent

Analyzes failing tests and generates structured file patches.

Debugger output uses strict schema validation, and patch paths are constrained to files that actually exist in the generated CODE artifact.

### Security Agent

Reviews implementation evidence for security issues and produces a structured security report.

### Critic Agent

Acts as the final engineering quality gate by evaluating the current requirements, architecture, implementation, test results, and security evidence.

It produces a verdict, quality score, issues, strengths, required improvements, and final recommendation.

---

## Autonomous Repair Loop

One of the central NEXUS capabilities is automatic recovery from generated-code failures.

```text
Generated Code
     |
     v
Metalize Workspace
     |
     v
Run Tests
     |
     v
Test Failure
     |
     v
Debugger
     |
     v
Structured Patch
     |
     v
Apply Patch
     |
     v
Synchronize CODE Artifact
     |
     v
Retest
     |
     +---- pass ----> Continue Workflow
     |
     +---- fail ----> Next Repair Attempt
```

Repair attempts operate on the latest patched implementation rather than repeatedly reasoning from the original source.

The persisted CODE artifact is synchronized with the actual generated workspace after each patch.

Repair attempts are bounded to prevent uncontrolled autonomous loops.

---

## Runtime Safety

LLM-generated commands are not executed directly.

The command executor acts as the final runtime security boundary.

It provides:

- executable allowlisting
- forbidden-token validation
- shell-free subprocess execution
- execution timeouts
- workspace validation
- generated-project isolation
- controlled Python/pytest execution

Unsupported commands are rejected before execution.

Generated pytest runs are isolated from NEXUS's own parent pytest configuration, preventing the host repository from accidentally changing module resolution or test behavior inside generated projects.

---

## Structured LLM Outputs

NEXUS uses structured outputs for critical agent boundaries rather than relying only on free-form text parsing.

Pydantic models define contracts for agent artifacts.

For supported Groq models, NEXUS can use strict JSON Schema structured output with constrained decoding.

The LLM layer also provides:

- bounded retry behavior
- rate-limit backoff
- completion-token controls
- reasoning-effort configuration
- structured-output fallback
- application-side Pydantic validation

This creates two validation layers:

```text
LLM Provider Schema Enforcement
              +
    NEXUS Application Validation
```

---

## Dynamic Replanning

NEXUS can modify an execution plan when workflow conditions require additional work.

The replanning system can reason about failed or incomplete execution state and mutate the remaining plan while preserving task dependencies.

This is separate from the local repair loop:

- **Repair** fixes a failing implementation.
- **Replanning** changes the workflow itself.

---

## Checkpointing & Recovery

Execution state can be persisted to SQLite checkpoints throughout the workflow.

Checkpoints are created for events including:

- workflow start
- task completion
- iteration completion
- repair completion
- replanning
- approval waiting
- workflow completion
- workflow failure

A failed workflow can later be restored with its tasks, artifacts, metadata, and execution state.

```bash
python scripts/demo_nexus.py --resume <RUN_ID>
```

Recovery logic can reopen the appropriate failed execution path instead of blindly restarting the entire workflow.

It also validates recovered CODE artifacts against the current runtime capabilities so stale checkpoints cannot bypass newer execution restrictions.

---

## Persistent Cross-Run Memory

NEXUS includes persistent memory so later workflows can retrieve relevant experience from previous runs.

Memory can capture information such as:

- previous artifacts
- failures
- repairs
- critic feedback
- security findings

Current execution evidence remains authoritative; memory is context, Not proof of current correctness.

---

## Tool Runtime

NEXUS separates agent reasoning from deterministic tool execution through a tool abstraction layer.

The tool system includes:

- contracts
- registry
- selector
- runtime
- executor integration
- production tools
- web search integration

This allows tools to be discovered, selected, governed, approved, executed, and observed independently of the LLM agents.

---

## Human Approval

The approval subsystem provides explicit control points for operations that should not run autonomously.

It includes approval policies, gates, state models, and management so the runtime can distinguish automatic operations from operations requiring human authorization.

---

## Governance

Autonomous execution is constrained by a governance layer including:

- policy enforcement
- resource budgets
- rate limiting
- concurrency controls

These mechanisms place deterministic boundaries around agent behavior.

---

## Evaluation & Regression Benchmarking

NEXUS includes an evaluation subsystem for scoring completed workflows and comparing results across runs.

Evaluation can consider engineering dimensions such as:

- task completion
- artifact quality
- grounding
- test quality
- security
- repair efficiency
- replanning efficiency
- tool use
- critic quality
- workflow reliability

Historical evaluations can be compared to identify improvements or potential regressions.

---

## Observability & Events

NEXUS records workflow events and execution traces so run behavior can be inspected after execution.

The system can track run status, execution events, duration, repair activity, replanning, and artifact creation.

An internal event bus and typed event models allow runtime events to be consumed by other components and exposed through the server layer.

---

## Background Jobs & API

Long-running workflows can be managed through the job subsystem, which includes job models, a queue, worker, manager, event integration, and API integration.

NEXUS also exposes its runtime through a FastAPI-based control plane.

The server can be started with:

```bash
uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```text
nexus/
├─ app/
│   ├─ agents/          # Specialized AI agents
│   ├─ api/             # FastAPI control plane
▂   ├─ approval/        # Human approval system
▂   ├─ checkpointing/   # Persistent workflow recovery
│   ├─ core/            # Engine, scheduler, runner, state
▂   ├─ evaluation/      # Evaluation and benchmarking
▂   ├─ events/          # Runtime event system
│   ├─ governance/       # Policies and resource controls
▂   ├─ jobs/             # Queue and background workers
│   ├─ memory/           # Persistent cross-run memory
▂   ├─ observability/    # Tracing and metrics
│   ├─ tools/            # Tool and execution runtime
│   └─ ui/               # UI package
├─ scripts/
│   └─ demo_nexus.py     # End-to-end demonstration
├─ tests/              # Unit, integration and E2E tests
├─ requirements.txt
├─ Procfile
┗ ─ README.md
```

---

## Technology Stack

- Python 3.12
- Groq API
- Pydantic
- FastAPI
- Uvicorn
- SQLite
- Pytest
- DDGS
- Git / GitHub

The default LLM configuration uses `openai/gpt-oss-20b` through Groq.

---

## Running NEXUS

### 1. Clone the repository

```bash
git clone <repository-url>
cd nexus
```

### 2. Create a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

``bash
cp .env.example .env
```

Add your Groq API key to `.env`:

```text
GROQ_API_KEY=your_key_here
```

Never commit your real `.env` file.

### 5. Run the test suite

```bash
pytest -q
```

Current verified result:

```text
964 passed
```

### 6. Run the flagship demo

```bash
python scripts/demo_nexus.py
```

Persisted failed or interrupted workflows can be resumed with:

```bash
python scripts/demo_nexus.py --resume <RUN_ID>
```

---

## Recovery Validation

NEXUS has been validated against a persisted failed workflow in which generated Python code initially failed its tests.

During recovery, the system restored the failed checkpoint, reopened the failed testing path, invoked autonomous repair, applied a schema-constrained patch, reran the generated tests in an isolated workspace, and continued through the final Critic quality gate.

The validation run completed with:

```text
Workflow: completed
Latest tests: passed
Critic verdict: accept
Evaluation score: 90.0
Repair cycles: 1
```

---

## Testing

The test suite covers both isolated components and cross-system behavior, including:

- agent validation
- orchestration
- scheduling
- execution
- autonomous repair
- dynamic replanning
- checkpoint recovery
- persistent memory
- runtime security
- workspace isolation
- tool intelligence
- approval flows
- governance
- evaluation
- observability
- event streaming
- background jobs
- API integration
- production smoke tests
- end-to-end workflows

Current verified suite:

```text
964 passed
```

---

## Engineering Principles

**LLMs propose; deterministic systems enforce.**

Agent outputs are validated before they influence execution.

**Execution state is explicit.**

Tasks, dependencies, artifacts, checkpoints, and workflow status are represented as structured state.

**Failures are recoverable.**

Testing, debugging, patching, replanning, and checkpoint recovery are first-class workflow concepts.

**Autonomy requires boundaries.**

Runtime allowlists, governance, approval gates, resource limits, and bounded repair loops constrain autonomous behavior.

---

## Project Status

NEXUS is an advanced AI engineering portfolio project focused on production-oriented agentic system design.

The core multi-agent engine, autonomous repair, dynamic replanning, memory, tool runtime, checkpoint recovery, governance, evaluation, observability, event system, background jobs, and API layer are implemented and covered by automated tests.

Future work can extend the system with additional tools, model providers, distributed workers, richer user interfaces, and larger software-engineering benchmarks.

---

## Author

**Abhishek Soni**

AI / Software Engineering
