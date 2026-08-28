import re
import time
from typing import Optional

from groq import Groq

from app.core.config import settings


DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_RETRY_DELAY = 2.0
DEFAULT_MAX_RETRY_DELAY = 60.0
RATE_LIMIT_BUFFER_SECONDS = 1.0


class LLMRetryExhaustedError(
    RuntimeError
):
    """
    Raised when a transient LLM request
    continues failing after the configured
    retry budget has been exhausted.
    """


class LLMClient:
    """
    Shared Groq LLM client for NEXUS.

    Features:
    - text generation
    - optional provider JSON mode
    - application-side structured validation
    - automatic rate-limit recovery
    - exponential retry backoff
    - provider JSON-mode fallback
    - finite retry budget
    - optional completion-token control
    """

    def __init__(
        self,
        *,
        max_retries: int = (
            DEFAULT_MAX_RETRIES
        ),
        base_retry_delay: float = (
            DEFAULT_BASE_RETRY_DELAY
        ),
        max_retry_delay: float = (
            DEFAULT_MAX_RETRY_DELAY
        ),
    ):
        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

        if base_retry_delay < 0:
            raise ValueError(
                "base_retry_delay cannot "
                "be negative."
            )

        if max_retry_delay <= 0:
            raise ValueError(
                "max_retry_delay must be "
                "greater than zero."
            )

        self.client = Groq(
            api_key=settings.groq_api_key
        )

        self.max_retries = (
            max_retries
        )

        self.base_retry_delay = (
            base_retry_delay
        )

        self.max_retry_delay = (
            max_retry_delay
        )

    def _status_code_for(
        self,
        exc: Exception,
    ) -> Optional[int]:
        status_code = getattr(
            exc,
            "status_code",
            None,
        )

        if isinstance(
            status_code,
            int,
        ):
            return status_code

        response = getattr(
            exc,
            "response",
            None,
        )

        if response is not None:
            response_status = getattr(
                response,
                "status_code",
                None,
            )

            if isinstance(
                response_status,
                int,
            ):
                return response_status

        return None

    def _is_rate_limit_error(
        self,
        exc: Exception,
    ) -> bool:
        status_code = (
            self._status_code_for(
                exc
            )
        )

        if status_code == 429:
            return True

        message = str(
            exc
        ).lower()

        return (
            "rate limit"
            in message
            or "rate_limit_exceeded"
            in message
            or "too many requests"
            in message
        )

    def _is_json_validation_error(
        self,
        exc: Exception,
    ) -> bool:
        """
        Detect provider-side failures where
        Groq cannot satisfy JSON response
        formatting before returning content.

        NEXUS can safely fall back to plain
        completion because agents perform
        their own JSON + Pydantic validation.
        """

        status_code = (
            self._status_code_for(
                exc
            )
        )

        message = str(
            exc
        ).lower()

        if (
            "json_validate_failed"
            in message
        ):
            return True

        if (
            status_code == 400
            and
            "failed to validate json"
            in message
        ):
            return True

        if (
            status_code == 400
            and
            "failed to generate json"
            in message
        ):
            return True

        return False

    def _retry_after_from_headers(
        self,
        exc: Exception,
    ) -> Optional[float]:
        response = getattr(
            exc,
            "response",
            None,
        )

        if response is None:
            return None

        headers = getattr(
            response,
            "headers",
            None,
        )

        if not headers:
            return None

        value = (
            headers.get(
                "retry-after"
            )
            or headers.get(
                "Retry-After"
            )
        )

        if value is None:
            return None

        try:
            delay = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

        if delay < 0:
            return None

        return delay

    def _retry_after_from_message(
        self,
        exc: Exception,
    ) -> Optional[float]:
        message = str(
            exc
        )

        patterns = [
            (
                r"try again in\s+"
                r"([0-9]+(?:\.[0-9]+)?)s"
            ),
            (
                r"retry after\s+"
                r"([0-9]+(?:\.[0-9]+)?)"
                r"\s*seconds?"
            ),
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                message,
                flags=re.IGNORECASE,
            )

            if match is None:
                continue

            try:
                return float(
                    match.group(1)
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        return None

    def _retry_delay(
        self,
        exc: Exception,
        attempt: int,
    ) -> float:
        provider_delay = (
            self._retry_after_from_headers(
                exc
            )
        )

        if provider_delay is None:
            provider_delay = (
                self
                ._retry_after_from_message(
                    exc
                )
            )

        if provider_delay is not None:
            delay = (
                provider_delay
                + RATE_LIMIT_BUFFER_SECONDS
            )

        else:
            delay = (
                self.base_retry_delay
                * (
                    2 ** attempt
                )
            )

        return min(
            delay,
            self.max_retry_delay,
        )

    def _create_completion(
        self,
        kwargs: dict,
    ):
        """
        Execute a completion request.

        Recovery strategies:

        1. Retry provider rate limits.

        2. If Groq provider-side JSON
           formatting fails, retry once
           without response_format.

           Agent-level JSON/Pydantic
           validation remains active.
        """

        attempt = 0
        json_fallback_used = False

        while True:
            try:
                return (
                    self.client
                    .chat
                    .completions
                    .create(
                        **kwargs
                    )
                )

            except Exception as exc:
                if (
                    self._is_json_validation_error(
                        exc
                    )
                    and
                    "response_format"
                    in kwargs
                    and
                    not json_fallback_used
                ):
                    kwargs = dict(
                        kwargs
                    )

                    kwargs.pop(
                        "response_format",
                        None,
                    )

                    json_fallback_used = True

                    print(
                        "\n"
                        "[NEXUS LLM] "
                        "Provider JSON validation "
                        "failed. Retrying with "
                        "NEXUS-side structured "
                        "validation..."
                    )

                    continue

                if not self._is_rate_limit_error(
                    exc
                ):
                    raise

                if (
                    attempt
                    >= self.max_retries
                ):
                    raise (
                        LLMRetryExhaustedError(
                            "Groq rate limit "
                            "persisted after "
                            f"{self.max_retries} "
                            "retries."
                        )
                    ) from exc

                delay = self._retry_delay(
                    exc,
                    attempt,
                )

                print(
                    "\n"
                    "[NEXUS LLM] "
                    "Groq rate limit reached. "
                    f"Retrying in "
                    f"{delay:.1f}s "
                    f"(attempt "
                    f"{attempt + 1}/"
                    f"{self.max_retries})..."
                )

                time.sleep(
                    delay
                )

                attempt += 1

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        max_tokens: Optional[
            int
        ] = None,
    ) -> str:
        kwargs = {
            "model":
                settings.groq_model,

            "messages": [
                {
                    "role":
                        "system",

                    "content":
                        system_prompt,
                },
                {
                    "role":
                        "user",

                    "content":
                        user_prompt,
                },
            ],

            "temperature":
                0.1,
        }

        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError(
                    "max_tokens must be "
                    "greater than zero."
                )

            kwargs[
                "max_completion_tokens"
            ] = max_tokens

        if json_mode:
            kwargs[
                "response_format"
            ] = {
                "type":
                    "json_object"
            }

        response = (
            self._create_completion(
                kwargs
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if content is None:
            return ""

        return content
