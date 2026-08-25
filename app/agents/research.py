import json
from typing import Optional

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
from app.tools.web_search import WebSearchTool


class ResearchSource(BaseModel):
    title: str
    url: str
    summary: str


class ResearchReport(BaseModel):
    research_question: str = Field(min_length=1)

    findings: list[str] = Field(
        min_length=1
    )

    recommended_technologies: list[str] = Field(
        min_length=1
    )

    tradeoffs: list[str] = Field(
        min_length=1
    )

    risks: list[str] = Field(
        min_length=1
    )

    sources: list[ResearchSource] = Field(
        min_length=1
    )


class ResearchGenerationError(Exception):
    """Raised when research output cannot be validated."""


class ResearchAgent(BaseAgent):
    role = AgentRole.RESEARCH

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        search_tool: Optional[WebSearchTool] = None,
        max_validation_retries: int = 2,
    ):
        self.llm = llm_client or LLMClient()
        self.search_tool = (
            search_tool or WebSearchTool(
                max_results=5
            )
        )

        self.max_validation_retries = (
            max_validation_retries
        )

    def _get_requirements(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> dict:
        for artifact_id in task.input_artifact_ids:
            artifact = state.artifacts.get(
                artifact_id
            )

            if (
                artifact
                and artifact.type
                == ArtifactType.REQUIREMENTS
            ):
                return artifact.content

        for artifact in state.artifacts.values():
            if (
                artifact.type
                == ArtifactType.REQUIREMENTS
            ):
                return artifact.content

        raise ResearchGenerationError(
            "Requirements artifact not found."
        )

    def _validate_output(
        self,
        raw_output: str,
    ) -> ResearchReport:
        parsed = json.loads(
            raw_output
        )

        return ResearchReport.model_validate(
            parsed
        )

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        requirements = self._get_requirements(
            task,
            state,
        )

        objective = requirements[
            "objective"
        ]

        search_query = (
            f"{objective} architecture technologies "
            "best practices open source"
        )

        search_results = self.search_tool.search(
            search_query
        )

        if not search_results:
            raise ResearchGenerationError(
                "Web research returned no results."
            )

        source_context = [
            {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
            }
            for result in search_results
        ]

        system_prompt = (
            "You are the Research Agent inside NEXUS. "
            "Analyze technical sources and produce "
            "evidence-grounded engineering research. "
            "Return valid JSON only."
        )

        prompt = f"""
USER REQUIREMENTS:

{json.dumps(requirements, indent=2)}

WEB SEARCH RESULTS:

{json.dumps(source_context, indent=2)}

Produce a technical research report.

Return one JSON object with exactly:

research_question
findings
recommended_technologies
tradeoffs
risks
sources

Rules:

- research_question must be a string
- findings must be a non-empty array
- recommended_technologies must be a non-empty array
- tradeoffs must be a non-empty array
- risks must be a non-empty array

- sources must be a non-empty array of objects
- each source object must contain:
  title
  url
  summary

- use only URLs present in WEB SEARCH RESULTS
- do not invent sources
- prefer official documentation when available
- explain technical tradeoffs
- identify implementation risks
- prefer free/open-source technologies where practical
- do not return markdown
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

                allowed_urls = {
                    item["url"]
                    for item in source_context
                }

                for source in report.sources:
                    if source.url not in allowed_urls:
                        raise ResearchGenerationError(
                            "Research Agent invented "
                            f"an unsupported URL: {source.url}"
                        )

                return Artifact(
                    type=ArtifactType.RESEARCH,
                    name="technical_research",
                    content=report.model_dump(),
                    created_by=self.role,
                    metadata={
                        "validation_attempts":
                            attempt + 1,
                        "search_query":
                            search_query,
                        "search_result_count":
                            len(search_results),
                    },
                )

            except (
                json.JSONDecodeError,
                ValidationError,
                ResearchGenerationError,
            ) as exc:
                last_error = exc

                prompt = f"""
The previous research response failed validation.

ERROR:
{exc}

PREVIOUS RESPONSE:
{raw_output}

Repair the response.

You must return all required fields.

Every cited URL MUST come from this list:

{json.dumps(
    list(
        {
            item["url"]
            for item in source_context
        }
    ),
    indent=2,
)}

Return only valid JSON.
"""

        raise ResearchGenerationError(
            "Research output could not be validated "
            f"after retries: {last_error}"
        )
