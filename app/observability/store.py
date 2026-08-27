import sqlite3

from pathlib import Path
from typing import Optional

from app.observability.models import (
    TraceSummary,
    WorkflowTrace,
)


class TraceStoreError(Exception):
    """
    Raised when workflow traces cannot
    be persisted or retrieved safely.
    """


class TraceStore:
    """
    SQLite-backed persistent storage
    for NEXUS workflow execution traces.
    """

    def __init__(
        self,
        db_path: str = (
            "data/nexus_traces.db"
        ),
    ):
        self.db_path = Path(
            db_path
        )

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    def _connect(
        self,
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(
                self.db_path
            )
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection

    def _initialize(
        self,
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS
                    workflow_traces (
                        run_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        completed_at TEXT,
                        total_duration_ms REAL,
                        event_count INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        created_at TEXT NOT NULL
                            DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL
                            DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise TraceStoreError(
                "Could not initialize trace "
                f"store: {exc}"
            ) from exc

    def save(
        self,
        trace: WorkflowTrace,
    ) -> None:
        payload = (
            trace.model_dump_json()
        )

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO workflow_traces (
                        run_id,
                        status,
                        started_at,
                        completed_at,
                        total_duration_ms,
                        event_count,
                        payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id)
                    DO UPDATE SET
                        status = excluded.status,
                        started_at = excluded.started_at,
                        completed_at = excluded.completed_at,
                        total_duration_ms =
                            excluded.total_duration_ms,
                        event_count =
                            excluded.event_count,
                        payload = excluded.payload,
                        updated_at =
                            CURRENT_TIMESTAMP
                    """,
                    (
                        trace.run_id,
                        trace.status.value,
                        trace.started_at.isoformat(),
                        (
                            trace.completed_at
                            .isoformat()
                            if trace.completed_at
                            is not None
                            else None
                        ),
                        trace.total_duration_ms,
                        len(
                            trace.events
                        ),
                        payload,
                    ),
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise TraceStoreError(
                "Could not save workflow "
                f"trace: {exc}"
            ) from exc

    def get(
        self,
        run_id: str,
    ) -> Optional[
        WorkflowTrace
    ]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT payload
                    FROM workflow_traces
                    WHERE run_id = ?
                    """,
                    (
                        run_id,
                    ),
                ).fetchone()

        except sqlite3.Error as exc:
            raise TraceStoreError(
                "Could not retrieve workflow "
                f"trace: {exc}"
            ) from exc

        if row is None:
            return None

        try:
            return (
                WorkflowTrace
                .model_validate_json(
                    row["payload"]
                )
            )

        except Exception as exc:
            raise TraceStoreError(
                "Stored workflow trace "
                "payload is invalid."
            ) from exc

    def exists(
        self,
        run_id: str,
    ) -> bool:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT 1
                    FROM workflow_traces
                    WHERE run_id = ?
                    LIMIT 1
                    """,
                    (
                        run_id,
                    ),
                ).fetchone()

        except sqlite3.Error as exc:
            raise TraceStoreError(
                "Could not check workflow "
                f"trace existence: {exc}"
            ) from exc

        return row is not None

    def count(
        self,
    ) -> int:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM workflow_traces
                    """
                ).fetchone()

        except sqlite3.Error as exc:
            raise TraceStoreError(
                "Could not count workflow "
                f"traces: {exc}"
            ) from exc

        return int(
            row["count"]
        )

    def list_recent(
        self,
        limit: int = 10,
    ) -> list[
        WorkflowTrace
    ]:
        if limit <= 0:
            raise ValueError(
                "limit must be greater "
                "than zero."
            )

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT payload
                    FROM workflow_traces
                    ORDER BY updated_at DESC,
                             rowid DESC
                    LIMIT ?
                    """,
                    (
                        limit,
                    ),
                ).fetchall()

        except sqlite3.Error as exc:
            raise TraceStoreError(
                "Could not list workflow "
                f"traces: {exc}"
            ) from exc

        traces = []

        for row in rows:
            try:
                trace = (
                    WorkflowTrace
                    .model_validate_json(
                        row["payload"]
                    )
                )

            except Exception as exc:
                raise TraceStoreError(
                    "Stored workflow trace "
                    "payload is invalid."
                ) from exc

            traces.append(
                trace
            )

        return traces

    def delete(
        self,
        run_id: str,
    ) -> bool:
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM workflow_traces
                    WHERE run_id = ?
                    """,
                    (
                        run_id,
                    ),
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise TraceStoreError(
                "Could not delete workflow "
                f"trace: {exc}"
            ) from exc

        return (
            cursor.rowcount > 0
        )

    def clear(
        self,
    ) -> int:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM workflow_traces
                    """
                ).fetchone()

                count = int(
                    row["count"]
                )

                connection.execute(
                    """
                    DELETE FROM workflow_traces
                    """
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise TraceStoreError(
                "Could not clear workflow "
                f"traces: {exc}"
            ) from exc

        return count

    def summary(
        self,
        run_id: str,
    ) -> Optional[
        TraceSummary
    ]:
        trace = self.get(
            run_id
        )

        if trace is None:
            return None

        agents_used = sorted(
            {
                event.agent_role
                for event in trace.events
                if event.agent_role
            }
        )

        return TraceSummary(
            run_id=trace.run_id,
            status=trace.status,
            total_events=len(
                trace.events
            ),
            total_duration_ms=(
                trace.total_duration_ms
            ),
            task_count=(
                trace.task_count
            ),
            completed_task_count=(
                trace.completed_task_count
            ),
            failed_task_count=(
                trace.failed_task_count
            ),
            repair_count=(
                trace.repair_count
            ),
            replan_count=(
                trace.replan_count
            ),
            artifact_count=(
                trace.artifact_count
            ),
            agents_used=agents_used,
        )
