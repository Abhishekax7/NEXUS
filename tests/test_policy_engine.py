import pytest

from app.governance.models import (
    PolicyDeniedError,
    PolicyEffect,
    PolicyRule,
)
from app.governance.policy import (
    PolicyEngine,
)


def make_rule(
    *,
    rule_id,
    action,
    effect,
    reason="Test policy rule.",
):
    return PolicyRule(
        id=rule_id,
        action=action,
        effect=effect,
        reason=reason,
    )


def test_default_policy_allows_action():
    engine = PolicyEngine()

    decision = engine.evaluate(
        "tool.read"
    )

    assert (
        decision.effect
        == PolicyEffect.ALLOW
    )

    assert decision.allowed is True

    assert (
        decision.requires_approval
        is False
    )


def test_default_policy_can_be_deny():
    engine = PolicyEngine(
        default_effect=(
            PolicyEffect.DENY
        )
    )

    decision = engine.evaluate(
        "unknown.action"
    )

    assert (
        decision.effect
        == PolicyEffect.DENY
    )

    assert decision.allowed is False


def test_exact_allow_rule_matches():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="allow-read",
                action="tool.read",
                effect=(
                    PolicyEffect.ALLOW
                ),
            )
        ]
    )

    decision = engine.evaluate(
        "tool.read"
    )

    assert decision.allowed is True

    assert (
        decision.matched_rule_id
        == "allow-read"
    )


def test_exact_deny_rule_matches():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="deny-shell",
                action="tool.shell",
                effect=(
                    PolicyEffect.DENY
                ),
            )
        ]
    )

    decision = engine.evaluate(
        "tool.shell"
    )

    assert decision.allowed is False

    assert (
        decision.effect
        == PolicyEffect.DENY
    )


def test_approval_rule_matches():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="approve-write",
                action="tool.write",
                effect=(
                    PolicyEffect
                    .REQUIRE_APPROVAL
                ),
            )
        ]
    )

    decision = engine.evaluate(
        "tool.write"
    )

    assert decision.allowed is False

    assert (
        decision.requires_approval
        is True
    )


def test_namespace_wildcard_matches():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="deny-shells",
                action="tool.shell.*",
                effect=(
                    PolicyEffect.DENY
                ),
            )
        ]
    )

    decision = engine.evaluate(
        "tool.shell.execute"
    )

    assert (
        decision.effect
        == PolicyEffect.DENY
    )


def test_namespace_wildcard_does_not_overmatch():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="deny-shells",
                action="tool.shell.*",
                effect=(
                    PolicyEffect.DENY
                ),
            )
        ]
    )

    decision = engine.evaluate(
        "tool.files.read"
    )

    assert (
        decision.effect
        == PolicyEffect.ALLOW
    )


def test_global_wildcard_matches_everything():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="deny-all",
                action="*",
                effect=(
                    PolicyEffect.DENY
                ),
            )
        ]
    )

    decision = engine.evaluate(
        "anything.at.all"
    )

    assert (
        decision.matched_rule_id
        == "deny-all"
    )


def test_first_matching_rule_wins():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="first",
                action="tool.*",
                effect=(
                    PolicyEffect.DENY
                ),
            ),
            make_rule(
                rule_id="second",
                action="tool.read",
                effect=(
                    PolicyEffect.ALLOW
                ),
            ),
        ]
    )

    decision = engine.evaluate(
        "tool.read"
    )

    assert (
        decision.matched_rule_id
        == "first"
    )

    assert (
        decision.effect
        == PolicyEffect.DENY
    )


def test_enforce_raises_for_denied_action():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="deny-delete",
                action="tool.delete",
                effect=(
                    PolicyEffect.DENY
                ),
                reason=(
                    "Deletion prohibited."
                ),
            )
        ]
    )

    with pytest.raises(
        PolicyDeniedError,
        match="Deletion prohibited",
    ):
        engine.enforce(
            "tool.delete"
        )


def test_enforce_returns_allow_decision():
    engine = PolicyEngine()

    decision = engine.enforce(
        "tool.read"
    )

    assert decision.allowed is True


def test_enforce_returns_approval_decision():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="approval",
                action="tool.write",
                effect=(
                    PolicyEffect
                    .REQUIRE_APPROVAL
                ),
            )
        ]
    )

    decision = engine.enforce(
        "tool.write"
    )

    assert (
        decision.requires_approval
        is True
    )


def test_context_is_preserved():
    engine = PolicyEngine()

    decision = engine.evaluate(
        "tool.read",
        context={
            "run_id": "run-1",
            "agent": "research",
        },
    )

    assert (
        decision.metadata[
            "context"
        ][
            "run_id"
        ]
        == "run-1"
    )


def test_duplicate_rule_is_rejected():
    engine = PolicyEngine()

    rule = make_rule(
        rule_id="unique",
        action="tool.read",
        effect=PolicyEffect.ALLOW,
    )

    engine.add_rule(
        rule
    )

    with pytest.raises(
        ValueError,
        match="already exists",
    ):
        engine.add_rule(
            rule
        )


def test_rule_can_be_removed():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="temporary",
                action="tool.read",
                effect=(
                    PolicyEffect.DENY
                ),
            )
        ]
    )

    engine.remove_rule(
        "temporary"
    )

    decision = engine.evaluate(
        "tool.read"
    )

    assert (
        decision.effect
        == PolicyEffect.ALLOW
    )


def test_missing_rule_removal_fails():
    engine = PolicyEngine()

    with pytest.raises(
        KeyError,
        match="not found",
    ):
        engine.remove_rule(
            "missing"
        )


def test_empty_action_is_rejected():
    engine = PolicyEngine()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        engine.evaluate(
            "   "
        )


def test_matching_is_case_insensitive():
    engine = PolicyEngine(
        rules=[
            make_rule(
                rule_id="case-rule",
                action="TOOL.READ",
                effect=(
                    PolicyEffect.DENY
                ),
            )
        ]
    )

    decision = engine.evaluate(
        "tool.read"
    )

    assert (
        decision.effect
        == PolicyEffect.DENY
    )
