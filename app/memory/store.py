import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class MemoryStoreError(Exception):
    """Raised when persistent memory operations fail."""


class MemoryStore:
    def __init__(
        self,
        db_path: str = "data/nexus_memory.db",
    ):
        self.db_path = Path(
            db_path
        ).resolve()

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    def _connect(
        self,
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path
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
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_memories_run_id
                    ON memories(run_id)
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_memories_type
                    ON memories(memory_type)
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_memories_key
                    ON memories(key)
                    """
                )

        except sqlite3.Error as exc:
            raise MemoryStoreError(
                f"Failed to initialize memory store: {exc}"
            ) from exc

    def save(
        self,
        run_id: str,
        memory_type: str,
        key: str,
        value: Any,
        metadata: Optional[dict] = None,
    ) -> int:
        if not run_id.strip():
            raise MemoryStoreError(
                "run_id cannot be empty."
            )

        if not memory_type.strip():
            raise MemoryStoreError(
                "memory_type cannot be empty."
            )

        if not key.strip():
            raise MemoryStoreError(
                "key cannot be empty."
            )

        metadata = (
            metadata
            or {}
        )

        try:
            value_json = json.dumps(
                value
            )

            metadata_json = json.dumps(
                metadata
            )

        except TypeError as exc:
            raise MemoryStoreError(
                "Memory value or metadata is not JSON serializable."
            ) from exc

        created_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO memories (
                        run_id,
                        memory_type,
                        key,
                        value_json,
                        created_at,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        memory_type,
                        key,
                        value_json,
                        created_at,
                        metadata_json,
                    ),
                )

                return int(
                    cursor.lastrowid
                )

        except sqlite3.Error as exc:
            raise MemoryStoreError(
                f"Failed to save memory: {exc}"
            ) from exc

    def get_by_run(
        self,
        run_id: str,
    ) -> list[dict]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE run_id = ?
                    ORDER BY id ASC
                    """,
                    (
                        run_id,
                    ),
                ).fetchall()

        except sqlite3.Error as exc:
            raise MemoryStoreError(
                f"Failed to load run memories: {exc}"
            ) from exc

        return [
            self._deserialize_row(
                row
            )
            for row in rows
        ]

    def get_by_type(
        self,
        memory_type: str,
        limit: int = 100,
    ) -> list[dict]:
        if limit <= 0:
            raise MemoryStoreError(
                "limit must be greater than zero."
            )

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE memory_type = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (
                        memory_type,
                        limit,
                    ),
                ).fetchall()

        except sqlite3.Error as exc:
            raise MemoryStoreError(
                f"Failed to load memories by type: {exc}"
            ) from exc

        return [
            self._deserialize_row(
                row
            )
            for row in rows
        ]

    def get_latest(
        self,
        key: str,
        memory_type: Optional[str] = None,
    ) -> Optional[dict]:
        try:
            with self._connect() as connection:
                if memory_type is None:
                    row = connection.execute(
                        """
                        SELECT *
                        FROM memories
                        WHERE key = ?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (
                            key,
                        ),
                    ).fetchone()

                else:
                    row = connection.execute(
                        """
                        SELECT *
                        FROM memories
                        WHERE key = ?
                        AND memory_type = ?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (
                            key,
                            memory_type,
                        ),
                    ).fetchone()

        except sqlite3.Error as exc:
            raise MemoryStoreError(
                f"Failed to load latest memory: {exc}"
            ) from exc

        if row is None:
            return None

        return self._deserialize_row(
            row
        )

    def count(
        self,
    ) -> int:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM memories
                    """
                ).fetchone()

        except sqlite3.Error as exc:
            raise MemoryStoreError(
                f"Failed to count memories: {exc}"
            ) from exc

        return int(
            row["total"]
        )

    def _deserialize_row(
        self,
        row: sqlite3.Row,
    ) -> dict:
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "memory_type": row["memory_type"],
            "key": row["key"],
            "value": json.loads(
                row["value_json"]
            ),
            "created_at": row[
                "created_at"
            ],
            "metadata": json.loads(
                row["metadata_json"]
            ),
        }
