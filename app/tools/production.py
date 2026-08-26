from pathlib import Path
from typing import Optional

from app.memory.retriever import MemoryRetriever
from app.tools.contracts import (
    ToolCapability,
    ToolCategory,
    ToolParameter,
    ToolRiskLevel,
)
from app.tools.registry import ToolRegistry


class ProductionToolError(Exception):
    """
    Raised when a production NEXUS tool
    cannot complete its operation.
    """


def _read_text_file(
    path: str,
    workspace_root: str,
) -> dict:
    root = Path(
        workspace_root
    ).resolve()

    target = (
        root
        / path
    ).resolve()

    try:
        target.relative_to(
            root
        )
    except ValueError as exc:
        raise ProductionToolError(
            "Path escapes workspace."
        ) from exc

    if not target.exists():
        raise ProductionToolError(
            f"File does not exist: {path}"
        )

    if not target.is_file():
        raise ProductionToolError(
            f"Path is not a file: {path}"
        )

    try:
        content = target.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError as exc:
        raise ProductionToolError(
            "File is not valid UTF-8 text."
        ) from exc

    return {
        "path": path,
        "content": content,
        "size": len(
            content
        ),
    }


def _list_workspace_files(
    workspace_root: str,
) -> dict:
    root = Path(
        workspace_root
    ).resolve()

    if not root.exists():
        return {
            "files": []
        }

    files = []

    for path in sorted(
        root.rglob("*")
    ):
        if not path.is_file():
            continue

        files.append(
            str(
                path.relative_to(
                    root
                )
            )
        )

    return {
        "files": files
    }


def build_production_tool_registry(
    workspace_root: str,
    memory_retriever: Optional[
        MemoryRetriever
    ] = None,
) -> ToolRegistry:
    """
    Build the allow-listed production
    tool registry available to NEXUS.

    Only explicitly registered capabilities
    can be selected by the AI.
    """

    registry = ToolRegistry()

    registry.register(
        ToolCapability(
            name="list_workspace_files",
            description=(
                "List files currently available "
                "inside the NEXUS workspace."
            ),
            category=(
                ToolCategory.FILESYSTEM
            ),
            risk_level=(
                ToolRiskLevel.LOW
            ),
            parameters=[],
            tags=[
                "workspace",
                "filesystem",
                "inspection",
            ],
            metadata={
                "read_only": True,
            },
        ),
        lambda: _list_workspace_files(
            workspace_root
        ),
    )

    registry.register(
        ToolCapability(
            name="read_text_file",
            description=(
                "Read a UTF-8 text file from "
                "inside the NEXUS workspace."
            ),
            category=(
                ToolCategory.FILESYSTEM
            ),
            risk_level=(
                ToolRiskLevel.LOW
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    description=(
                        "Path relative to the "
                        "workspace root."
                    ),
                    required=True,
                    parameter_type="string",
                )
            ],
            tags=[
                "workspace",
                "filesystem",
                "read",
            ],
            metadata={
                "read_only": True,
                "workspace_scoped": True,
            },
        ),
        lambda path: _read_text_file(
            path=path,
            workspace_root=(
                workspace_root
            ),
        ),
    )

    if memory_retriever is not None:

        def search_memory(
            query: str,
            limit: int = 5,
        ):
            return (
                memory_retriever.retrieve(
                    query=query,
                    limit=limit,
                )
            )

        registry.register(
            ToolCapability(
                name="search_memory",
                description=(
                    "Retrieve relevant knowledge "
                    "from persistent NEXUS memory."
                ),
                category=(
                    ToolCategory.MEMORY
                ),
                risk_level=(
                    ToolRiskLevel.LOW
                ),
                parameters=[
                    ToolParameter(
                        name="query",
                        description=(
                            "Memory retrieval query."
                        ),
                        required=True,
                        parameter_type="string",
                    ),
                    ToolParameter(
                        name="limit",
                        description=(
                            "Maximum memories "
                            "to retrieve."
                        ),
                        required=False,
                        parameter_type="integer",
                        default=5,
                    ),
                ],
                tags=[
                    "memory",
                    "retrieval",
                    "reasoning",
                ],
                metadata={
                    "read_only": True,
                },
            ),
            search_memory,
        )

    return registry
