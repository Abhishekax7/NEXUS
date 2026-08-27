import json
import sqlite3

from pathlib import Path
from typing import Optional

from app.evaluation.models import (
    WorkflowEvaluation,
)


class EvaluationHistoryError(Exception):
    """
    Raised when evaluation history cannot
    be stored or retrieved safely.
    """


class EvaluationHistoryStore:
    """
    Persistent SQLite-backed storage for
    NEXUS workflow evaluations and baseline
    management.
    """

    def __init__(
        self,
        db_path: str = (
            "data/nexus_evaluations.db"
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
    ):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                evaluations (
                    run_id TEXT PRIMARY KEY,
                    overall_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    regression_risk REAL NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                evaluation_baseline (
                    id INTEGER PRIMARY KEY
                    CHECK (id = 1),
                    run_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id)
                    REFERENCES evaluations(run_id)
                )
                """
            )

            connection.commit()

    def save(
        self,
        evaluation: WorkflowEvaluation,
    ) -> None:
        payload = (
            evaluation.model_dump_json()
        )

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO
                    evaluations (
                        run_id,
                        overall_score,
                        status,
                        regression_risk,
                        payload
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        evaluation.run_id,
                        evaluation.overall_score,
                        evaluation.status.value,
                        evaluation.regression_risk,
                        payload,
                    ),
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise EvaluationHistoryError(
                "Could not save evaluation: "
                f"{exc}"
            ) from exc

    def get(
        self,
        run_id: str,
    ) -> Optional[
        WorkflowEvaluation
    ]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload
                FROM evaluations
                WHERE run_id = ?
                """,
                (
                    run_id,
                ),
            ).fetchone()

        if row is None:
            return None

        try:
            return (
                WorkflowEvaluation
                .model_validate_json(
                    row["payload"]
                )
            )

        except Exception as exc:
            raise EvaluationHistoryError(
                "Stored evaluation payload "
                "is invalid."
            ) from exc

    def exists(
        self,
        run_id: str,
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM evaluations
                WHERE run_id = ?
                LIMIT 1
                """,
                (
                    run_id,
                ),
            ).fetchone()

        return row is not None

    def count(
        self,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM evaluations
                """
            ).fetchone()

        return int(
            row["count"]
        )

    def list_recent(
        self,
        limit: int = 10,
    ) -> list[
        WorkflowEvaluation
    ]:
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM evaluations
                ORDER BY created_at DESC,
                         rowid DESC
                LIMIT ?
                """,
                (
                    limit,
                ),
            ).fetchall()

        evaluations = []

        for row in rows:
            try:
                evaluation = (
                    WorkflowEvaluation
                    .model_validate_json(
                        row["payload"]
                    )
                )

            except Exception as exc:
                raise EvaluationHistoryError(
                    "Stored evaluation payload "
                    "is invalid."
                ) from exc

            evaluations.append(
                evaluation
            )

        return evaluations

    def set_baseline(
        self,
        run_id: str,
    ) -> None:
        if not self.exists(
            run_id
        ):
            raise EvaluationHistoryError(
                "Cannot set baseline because "
                f"run '{run_id}' does not exist."
            )

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO
                    evaluation_baseline (
                        id,
                        run_id
                    )
                    VALUES (1, ?)
                    ON CONFLICT(id)
                    DO UPDATE SET
                        run_id = excluded.run_id,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        run_id,
                    ),
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise EvaluationHistoryError(
                "Could not set evaluation "
                f"baseline: {exc}"
            ) from exc

    def get_baseline(
        self,
    ) -> Optional[
        WorkflowEvaluation
    ]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id
                FROM evaluation_baseline
                WHERE id = 1
                """
            ).fetchone()

        if row is None:
            return None

        evaluation = self.get(
            row["run_id"]
        )

        if evaluation is None:
            raise EvaluationHistoryError(
                "Baseline references a missing "
                "evaluation."
            )

        return evaluation

    def clear_baseline(
        self,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM
                evaluation_baseline
                WHERE id = 1
                """
            )

            connection.commit()

    def delete(
        self,
        run_id: str,
    ) -> bool:
        baseline = self.get_baseline()

        if (
            baseline is not None
            and baseline.run_id
            == run_id
        ):
            raise EvaluationHistoryError(
                "Cannot delete the active "
                "baseline evaluation."
            )

        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM evaluations
                WHERE run_id = ?
                """,
                (
                    run_id,
                ),
            )

            connection.commit()

        return (
            cursor.rowcount > 0
        )

    def export_json(
        self,
    ) -> str:
        evaluations = (
            self.list_recent(
                limit=max(
                    self.count(),
                    1,
                )
            )
        )

        baseline = (
            self.get_baseline()
        )

        payload = {
            "baseline_run_id": (
                baseline.run_id
                if baseline is not None
                else None
            ),
            "evaluations": [
                evaluation.model_dump(
                    mode="json"
                )
                for evaluation
                in evaluations
            ],
        }

        return json.dumps(
            payload,
            indent=2,
        )
