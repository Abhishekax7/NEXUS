import re
from dataclasses import dataclass
from typing import Iterable, Optional

from app.memory.store import MemoryStore


@dataclass
class RetrievedMemory:
    memory: dict
    score: float


class MemoryRetriever:
    """
    Lightweight local memory retrieval.

    Uses token overlap, memory-type weighting,
    and recency to rank past memories.
    """

    DEFAULT_TYPE_WEIGHTS = {
        "repair": 1.5,
        "failure": 1.4,
        "critic": 1.3,
        "security": 1.25,
        "artifact": 1.0,
        "task_event": 0.8,
    }

    def __init__(
        self,
        store: MemoryStore,
        type_weights: Optional[dict[str, float]] = None,
    ):
        self.store = store

        self.type_weights = {
            **self.DEFAULT_TYPE_WEIGHTS,
            **(type_weights or {}),
        }

    def _tokenize(
        self,
        text: str,
    ) -> set[str]:
        if not text:
            return set()

        tokens = re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower(),
        )

        return {
            token
            for token in tokens
            if len(token) > 2
        }

    def _flatten_value(
        self,
        value,
    ) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(
            value,
            (int, float, bool),
        ):
            return str(value)

        if isinstance(value, dict):
            parts = []

            for key, item in value.items():
                parts.append(
                    str(key)
                )

                parts.append(
                    self._flatten_value(
                        item
                    )
                )

            return " ".join(parts)

        if isinstance(value, list):
            return " ".join(
                self._flatten_value(
                    item
                )
                for item in value
            )

        return str(value)

    def _memory_text(
        self,
        memory: dict,
    ) -> str:
        return " ".join(
            [
                str(
                    memory.get(
                        "memory_type",
                        "",
                    )
                ),
                str(
                    memory.get(
                        "key",
                        "",
                    )
                ),
                self._flatten_value(
                    memory.get(
                        "value"
                    )
                ),
                self._flatten_value(
                    memory.get(
                        "metadata"
                    )
                ),
            ]
        )

    def _calculate_score(
        self,
        query_tokens: set[str],
        memory: dict,
        newest_id: int,
    ) -> float:
        memory_tokens = self._tokenize(
            self._memory_text(
                memory
            )
        )

        if not memory_tokens:
            return 0.0

        overlap = (
            query_tokens
            & memory_tokens
        )

        if not overlap:
            return 0.0

        overlap_score = (
            len(overlap)
            / max(
                len(query_tokens),
                1,
            )
        )

        type_weight = (
            self.type_weights.get(
                memory.get(
                    "memory_type",
                    ""
                ),
                1.0,
            )
        )

        memory_id = int(
            memory.get(
                "id",
                0,
            )
        )

        if newest_id > 0:
            recency_score = (
                memory_id
                / newest_id
            )
        else:
            recency_score = 0.0

        score = (
            overlap_score
            * type_weight
            + (
                recency_score
                * 0.10
            )
        )

        return round(
            score,
            6,
        )

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[
            Iterable[str]
        ] = None,
        exclude_run_id: Optional[str] = None,
    ) -> list[RetrievedMemory]:
        if not isinstance(
            query,
            str,
        ):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        allowed_types = None

        if memory_types is not None:
            allowed_types = set(
                memory_types
            )

        all_memories = []

        for memory_type in (
            self.type_weights.keys()
        ):
            memories = (
                self.store.get_by_type(
                    memory_type,
                    limit=500,
                )
            )

            all_memories.extend(
                memories
            )

        if not all_memories:
            return []

        newest_id = max(
            memory["id"]
            for memory in all_memories
        )

        ranked = []

        seen_ids = set()

        for memory in all_memories:
            memory_id = memory["id"]

            if memory_id in seen_ids:
                continue

            seen_ids.add(
                memory_id
            )

            if (
                exclude_run_id is not None
                and memory.get(
                    "run_id"
                )
                == exclude_run_id
            ):
                continue

            if (
                allowed_types is not None
                and memory.get(
                    "memory_type"
                )
                not in allowed_types
            ):
                continue

            score = self._calculate_score(
                query_tokens,
                memory,
                newest_id,
            )

            if score <= 0:
                continue

            ranked.append(
                RetrievedMemory(
                    memory=memory,
                    score=score,
                )
            )

        ranked.sort(
            key=lambda item: (
                item.score,
                item.memory["id"],
            ),
            reverse=True,
        )

        return ranked[
            :limit
        ]

    def retrieve_failures(
        self,
        query: str,
        limit: int = 5,
        exclude_run_id: Optional[str] = None,
    ) -> list[RetrievedMemory]:
        return self.retrieve(
            query=query,
            limit=limit,
            memory_types=[
                "failure",
            ],
            exclude_run_id=exclude_run_id,
        )

    def retrieve_repairs(
        self,
        query: str,
        limit: int = 5,
        exclude_run_id: Optional[str] = None,
    ) -> list[RetrievedMemory]:
        return self.retrieve(
            query=query,
            limit=limit,
            memory_types=[
                "repair",
            ],
            exclude_run_id=exclude_run_id,
        )

    def retrieve_critic_feedback(
        self,
        query: str,
        limit: int = 5,
        exclude_run_id: Optional[str] = None,
    ) -> list[RetrievedMemory]:
        return self.retrieve(
            query=query,
            limit=limit,
            memory_types=[
                "critic",
            ],
            exclude_run_id=exclude_run_id,
        )
