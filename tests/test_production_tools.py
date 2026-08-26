import pytest

from app.memory.retriever import MemoryRetriever
from app.memory.store import MemoryStore
from app.tools.production import (
    ProductionToolError,
    build_production_tool_registry,
)


def build_memory_retriever(
    tmp_path,
):
    store = MemoryStore(
        db_path=str(
            tmp_path
            / "memory.db"
        )
    )

    return store, MemoryRetriever(
        store
    )


def test_production_registry_contains_workspace_tools(
    tmp_path,
):
    registry = build_production_tool_registry(
        workspace_root=str(
            tmp_path
        )
    )

    names = set(
        registry.names()
    )

    assert (
        "list_workspace_files"
        in names
    )

    assert (
        "read_text_file"
        in names
    )


def test_memory_tool_is_not_registered_without_retriever(
    tmp_path,
):
    registry = build_production_tool_registry(
        workspace_root=str(
            tmp_path
        ),
        memory_retriever=None,
    )

    assert (
        "search_memory"
        not in registry.names()
    )


def test_memory_tool_is_registered_with_retriever(
    tmp_path,
):
    _, retriever = build_memory_retriever(
        tmp_path
    )

    registry = build_production_tool_registry(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_retriever=retriever,
    )

    assert (
        "search_memory"
        in registry.names()
    )


def test_list_workspace_files_returns_relative_paths(
    tmp_path,
):
    workspace = (
        tmp_path
        / "workspace"
    )

    workspace.mkdir()

    (
        workspace
        / "app.py"
    ).write_text(
        "print('hello')",
        encoding="utf-8",
    )

    nested = (
        workspace
        / "tests"
    )

    nested.mkdir()

    (
        nested
        / "test_app.py"
    ).write_text(
        "def test_app(): pass",
        encoding="utf-8",
    )

    registry = build_production_tool_registry(
        workspace_root=str(
            workspace
        )
    )

    handler = registry.get_handler(
        "list_workspace_files"
    )

    result = handler()

    assert (
        result["files"]
        == [
            "app.py",
            "tests/test_app.py",
        ]
    )


def test_list_workspace_files_handles_missing_workspace(
    tmp_path,
):
    workspace = (
        tmp_path
        / "missing"
    )

    registry = build_production_tool_registry(
        workspace_root=str(
            workspace
        )
    )

    result = registry.get_handler(
        "list_workspace_files"
    )()

    assert (
        result
        == {
            "files": []
        }
    )


def test_read_text_file_reads_workspace_file(
    tmp_path,
):
    workspace = (
        tmp_path
        / "workspace"
    )

    workspace.mkdir()

    target = (
        workspace
        / "app.py"
    )

    target.write_text(
        "print('hello')",
        encoding="utf-8",
    )

    registry = build_production_tool_registry(
        workspace_root=str(
            workspace
        )
    )

    handler = registry.get_handler(
        "read_text_file"
    )

    result = handler(
        path="app.py"
    )

    assert (
        result["path"]
        == "app.py"
    )

    assert (
        result["content"]
        == "print('hello')"
    )

    assert (
        result["size"]
        == len(
            "print('hello')"
        )
    )


def test_read_text_file_supports_nested_file(
    tmp_path,
):
    workspace = (
        tmp_path
        / "workspace"
    )

    nested = (
        workspace
        / "src"
    )

    nested.mkdir(
        parents=True
    )

    (
        nested
        / "service.py"
    ).write_text(
        "VALUE = 42",
        encoding="utf-8",
    )

    registry = build_production_tool_registry(
        workspace_root=str(
            workspace
        )
    )

    result = registry.get_handler(
        "read_text_file"
    )(
        path="src/service.py"
    )

    assert (
        result["content"]
        == "VALUE = 42"
    )


def test_read_text_file_blocks_parent_traversal(
    tmp_path,
):
    workspace = (
        tmp_path
        / "workspace"
    )

    workspace.mkdir()

    outside = (
        tmp_path
        / "secret.txt"
    )

    outside.write_text(
        "secret",
        encoding="utf-8",
    )

    registry = build_production_tool_registry(
        workspace_root=str(
            workspace
        )
    )

    handler = registry.get_handler(
        "read_text_file"
    )

    with pytest.raises(
        ProductionToolError,
        match="escapes workspace",
    ):
        handler(
            path="../secret.txt"
        )


def test_read_text_file_blocks_absolute_outside_path(
    tmp_path,
):
    workspace = (
        tmp_path
        / "workspace"
    )

    workspace.mkdir()

    outside = (
        tmp_path
        / "outside.txt"
    )

    outside.write_text(
        "outside",
        encoding="utf-8",
    )

    registry = build_production_tool_registry(
        workspace_root=str(
            workspace
        )
    )

    handler = registry.get_handler(
        "read_text_file"
    )

    with pytest.raises(
        ProductionToolError,
        match="escapes workspace",
    ):
        handler(
            path=str(
                outside
            )
        )


def test_read_text_file_rejects_missing_file(
    tmp_path,
):
    workspace = (
        tmp_path
        / "workspace"
    )

    workspace.mkdir()

    registry = build_production_tool_registry(
        workspace_root=str(
            workspace
        )
    )

    handler = registry.get_handler(
        "read_text_file"
    )

    with pytest.raises(
        ProductionToolError,
        match="does not exist",
    ):
        handler(
            path="missing.py"
        )


def test_read_text_file_rejects_directory(
    tmp_path,
):
    workspace = (
        tmp_path
        / "workspace"
    )

    directory = (
        workspace
        / "folder"
    )

    directory.mkdir(
        parents=True
    )

    registry = build_production_tool_registry(
        workspace_root=str(
            workspace
        )
    )

    handler = registry.get_handler(
        "read_text_file"
    )

    with pytest.raises(
        ProductionToolError,
        match="not a file",
    ):
        handler(
            path="folder"
        )


def test_memory_search_returns_relevant_memory(
    tmp_path,
):
    store, retriever = build_memory_retriever(
        tmp_path
    )

    store.save(
        run_id="run-1",
        memory_type="repair",
        key="fastapi_dependency_fix",
        value={
            "root_cause": (
                "FastAPI dependency "
                "was missing."
            ),
            "patches": [
                {
                    "path": (
                        "requirements.txt"
                    ),
                }
            ],
        },
    )

    registry = build_production_tool_registry(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_retriever=retriever,
    )

    handler = registry.get_handler(
        "search_memory"
    )

    results = handler(
        query="FastAPI dependency",
        limit=5,
    )

    assert len(results) > 0

    assert (
        results[0]
        .memory[
            "memory_type"
        ]
        == "repair"
    )


def test_memory_search_respects_limit(
    tmp_path,
):
    store, retriever = build_memory_retriever(
        tmp_path
    )

    for index in range(5):
        store.save(
            run_id=f"run-{index}",
            memory_type="repair",
            key=f"fastapi_fix_{index}",
            value={
                "summary": (
                    "FastAPI dependency "
                    "repair."
                )
            },
        )

    registry = build_production_tool_registry(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_retriever=retriever,
    )

    results = registry.get_handler(
        "search_memory"
    )(
        query="FastAPI dependency",
        limit=2,
    )

    assert (
        len(results)
        <= 2
    )


def test_workspace_tools_are_read_only(
    tmp_path,
):
    registry = build_production_tool_registry(
        workspace_root=str(
            tmp_path
        )
    )

    list_capability = (
        registry.get_capability(
            "list_workspace_files"
        )
    )

    read_capability = (
        registry.get_capability(
            "read_text_file"
        )
    )

    assert (
        list_capability.metadata[
            "read_only"
        ]
        is True
    )

    assert (
        read_capability.metadata[
            "read_only"
        ]
        is True
    )


def test_read_text_file_is_workspace_scoped(
    tmp_path,
):
    registry = build_production_tool_registry(
        workspace_root=str(
            tmp_path
        )
    )

    capability = (
        registry.get_capability(
            "read_text_file"
        )
    )

    assert (
        capability.metadata[
            "workspace_scoped"
        ]
        is True
    )
