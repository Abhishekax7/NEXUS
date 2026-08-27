import sqlite3

from pathlib import Path
from typing import Optional

from app.checkpointing.models import (
    CheckpointError,
    CheckpointStatus,
    WorkflowCheckpoint,
)


class CheckpointStore:
    """
    SQLite-backed persistent storage for
    ordered NEXUS workflow checkpoints.
    """

    def __init__(
        self,
        db_path: str = (
            "data/nexus_checkpoints.db"
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
                    workflow_checkpoints (
                        id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        checkpoint_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        task_id TEXT,
                        reason TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        UNIQUE(
                            run_id,
                            sequence
                        )
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_checkpoints_run_sequence
                    ON workflow_checkpoints (
                        run_id,
                        sequence DESC
                    )
                    """
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise CheckpointError(
                "Could not initialize "
                f"checkpoint store: {exc}"
            ) from exc

    def save(
        self,
        checkpoint: WorkflowCheckpoint,
    ) -> None:
        payload = (
            checkpoint.model_dump_json()
        )

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO workflow_checkpoints (
                        id,
                        run_id,
                        checkpoint_type,
                        status,
                        sequence,
                        created_at,
                        task_id,
                        reason,
                        payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checkpoint.id,
                        checkpoint.run_id,
                        checkpoint
                        .checkpoint_type
                        .value,
                        checkpoint.status.value,
                        checkpoint.sequence,
                        checkpoint
                        .created_at
                        .isoformat(),
                        checkpoint.task_id,
                        checkpoint.reason,
                        payload,
                    ),
                )

                connection.commit()

        except sqlite3.IntegrityError as exc:
            raise CheckpointError(
                "Checkpoint sequence already "
                f"exists for run "
                f"'{checkpoint.run_id}': "
                f"{checkpoint.sequence}"
            ) from exc

        except sqlite3.Error as exc:
            raise CheckpointError(
                "Could not save checkpoint: "
                f"{exc}"
            ) from exc

    def get(
        self,
        checkpoint_id: str,
    ) -> Optional[
        WorkflowCheckpoint
    ]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT payload
                    FROM workflow_checkpoints
                    WHERE id = ?
                    """,
                    (
                        checkpoint_id,
                    ),
                ).fetchone()

        except sqlite3.Error as exc:
            raise CheckpointError(
                "Could not retrieve checkpoint: "
                f"{exc}"
            ) from exc

        if row is None:
            return None

        try:
            return (
                WorkflowCheckpoint
                .model_validate_json(
                    row["payload"]
                )
            )

        except Exception as exc:
            raise CheckpointError(
                "Stored checkpoint payload "
                "is invalid."
            ) from exc

    def latest(
        self,
        run_id: str,
    ) -> Optional[
        WorkflowCheckpoint
    ]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT payload
                    FROM workflow_checkpoints
                    WHERE run_id = ?
                    ORDER BY sequence DESC
                    LIMIT 1
                    """,
                    (
                        run_id,
                    ),
                ).fetchone()

        except sqlite3.Error as exc:
            raise CheckpointError(
                "Could not retrieve latest "
                f"checkpoint: {exc}"
            ) from exc

        if row is None:
            return None

        try:
            return (
                WorkflowCheckpoint
                .model_validate_json(
                    row["payload"]
                )
            )

        except Exception as exc:
            raise CheckpointError(
                "Stored checkpoint payload "
                "is invalid."
            ) from exc

    def list_run(
        self,
        run_id: str,
    ) -> list[
        WorkflowCheckpoint
    ]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT payload
                    FROM workflow_checkpoints
                    WHERE run_id = ?
                    ORDER BY sequence ASC
                    """,
                    (
                        run_id,
                    ),
                ).fetchall()

        except sqlite3.Error as exc:
            raise CheckpointError(
                "Could not list checkpoints "
                f"for run: {exc}"
            ) from exc

        checkpoints = []

        for row in rows:
            try:
                checkpoint = (
                    WorkflowCheckpoint
                    .model_validate_json(
                        row["payload"]
                    )
                )

            except Exception as exc:
                raise CheckpointError(
                    "Stored checkpoint payload "
                    "is invalid."
                ) from exc

            checkpoints.append(
                checkpoint
            )

        return checkpoints

    def count(
        self,
        run_id: Optional[
            str
        ] = None,
    ) -> int:
        try:
            with self._connect() as connection:
                if run_id is None:
                    row = connection.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM workflow_checkpoints
                        """
                    ).fetchone()

                else:
                    row = connection.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM workflow_checkpoints
                        WHERE run_id = ?
                        """,
                        (
                            run_id,
                        ),
                    ).fetchone()

        except sqlite3.Error as exc:
            raise CheckpointError(
                "Could not count checkpoints: "
                f"{exc}"
            ) from exc

        return int(
            row["count"]
        )

    def next_sequence(
        self,
        run_id: str,
    ) -> int:
        latest = self.latest(
            run_id
        )

        if latest is None:
            return 0

        return (
            latest.sequence
            + 1
        )

    def mark_status(
        self,
        checkpoint_id: str,
        status: CheckpointStatus,
    ) -> None:
        checkpoint = self.get(
            checkpoint_id
        )

        if checkpoint is None:
            raise CheckpointError(
                "Checkpoint not found: "
                f"{checkpoint_id}"
            )

        checkpoint.status = (
            status
        )

        payload = (
            checkpoint.model_dump_json()
        )

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE workflow_checkpoints
                    SET status = ?,
                        payload = ?
                    WHERE id = ?
                    """,
                    (
                        status.value,
                        payload,
                        checkpoint_id,
                    ),
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise CheckpointError(
                "Could not update checkpoint "
                f"status: {exc}"
            ) from exc

    def delete_run(
        self,
        run_id: str,
    ) -> int:
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM workflow_checkpoints
                    WHERE run_id = ?
                    """,
                    (
                        run_id,
                    ),
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise CheckpointError(
                "Could not delete workflow "
                f"checkpoints: {exc}"
            ) from exc

        return int(
            cursor.rowcount
        )
