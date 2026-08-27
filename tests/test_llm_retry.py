from types import SimpleNamespace

import pytest

from app.core.llm import (
    LLMClient,
    LLMRetryExhaustedError,
)


class FakeRateLimitError(
    Exception
):
    def __init__(
        self,
        message=(
            "Rate limit reached. "
            "Please try again in 0.1s."
        ),
    ):
        super().__init__(
            message
        )

        self.status_code = 429


def fake_response(
    content="hello",
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=(
                    SimpleNamespace(
                        content=content
                    )
                )
            )
        ]
    )


def build_client(
    monkeypatch,
    responses,
    *,
    max_retries=3,
):
    class FakeCompletions:
        def __init__(
            self,
        ):
            self.calls = 0

        def create(
            self,
            **kwargs,
        ):
            result = responses[
                self.calls
            ]

            self.calls += 1

            if isinstance(
                result,
                Exception,
            ):
                raise result

            return result

    completions = (
        FakeCompletions()
    )

    fake_groq = (
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=(
                    completions
                )
            )
        )
    )

    monkeypatch.setattr(
        "app.core.llm.Groq",
        lambda **kwargs:
            fake_groq,
    )

    client = LLMClient(
        max_retries=max_retries,
        base_retry_delay=0,
    )

    return (
        client,
        completions,
    )


def test_generate_returns_content(
    monkeypatch,
):
    client, completions = (
        build_client(
            monkeypatch,
            [
                fake_response(
                    "success"
                )
            ],
        )
    )

    result = client.generate(
        "system",
        "user",
    )

    assert result == "success"

    assert (
        completions.calls
        == 1
    )


def test_rate_limit_is_retried(
    monkeypatch,
):
    sleeps = []

    monkeypatch.setattr(
        "app.core.llm.time.sleep",
        lambda delay:
            sleeps.append(
                delay
            ),
    )

    client, completions = (
        build_client(
            monkeypatch,
            [
                FakeRateLimitError(),
                fake_response(
                    "recovered"
                ),
            ],
        )
    )

    result = client.generate(
        "system",
        "user",
    )

    assert (
        result
        == "recovered"
    )

    assert (
        completions.calls
        == 2
    )

    assert len(
        sleeps
    ) == 1


def test_provider_retry_delay_is_parsed(
    monkeypatch,
):
    client, _ = build_client(
        monkeypatch,
        [
            fake_response()
        ],
    )

    exc = FakeRateLimitError(
        "Please try again in 7.1925s."
    )

    delay = client._retry_delay(
        exc,
        0,
    )

    assert delay >= 8.0


def test_non_rate_limit_error_is_not_retried(
    monkeypatch,
):
    client, completions = (
        build_client(
            monkeypatch,
            [
                RuntimeError(
                    "normal failure"
                )
            ],
        )
    )

    with pytest.raises(
        RuntimeError,
        match="normal failure",
    ):
        client.generate(
            "system",
            "user",
        )

    assert (
        completions.calls
        == 1
    )


def test_retry_budget_is_finite(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.core.llm.time.sleep",
        lambda delay: None,
    )

    client, completions = (
        build_client(
            monkeypatch,
            [
                FakeRateLimitError(),
                FakeRateLimitError(),
                FakeRateLimitError(),
            ],
            max_retries=2,
        )
    )

    with pytest.raises(
        LLMRetryExhaustedError,
        match="persisted",
    ):
        client.generate(
            "system",
            "user",
        )

    assert (
        completions.calls
        == 3
    )


def test_json_mode_is_forwarded(
    monkeypatch,
):
    captured = {}

    class FakeCompletions:
        def create(
            self,
            **kwargs,
        ):
            captured.update(
                kwargs
            )

            return fake_response(
                "{}"
            )

    fake_groq = (
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=(
                    FakeCompletions()
                )
            )
        )
    )

    monkeypatch.setattr(
        "app.core.llm.Groq",
        lambda **kwargs:
            fake_groq,
    )

    client = LLMClient()

    client.generate(
        "system",
        "user",
        json_mode=True,
    )

    assert (
        captured[
            "response_format"
        ][
            "type"
        ]
        == "json_object"
    )
