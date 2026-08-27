from typing import Optional

from app.governance.models import (
    PolicyDecision,
    PolicyDeniedError,
    PolicyEffect,
    PolicyRule,
)


class PolicyEngine:
    """
    Deterministic policy evaluator for
    NEXUS actions and tool execution.

    Rules are evaluated in registration
    order. The first matching rule wins.

    If no rule matches, the configured
    default effect is applied.
    """

    def __init__(
        self,
        rules: Optional[
            list[PolicyRule]
        ] = None,
        default_effect: PolicyEffect = (
            PolicyEffect.ALLOW
        ),
    ):
        self._rules = list(
            rules or []
        )

        self.default_effect = (
            default_effect
        )

    def add_rule(
        self,
        rule: PolicyRule,
    ) -> None:
        if any(
            existing.id == rule.id
            for existing
            in self._rules
        ):
            raise ValueError(
                "Policy rule already "
                f"exists: {rule.id}"
            )

        self._rules.append(
            rule
        )

    def remove_rule(
        self,
        rule_id: str,
    ) -> None:
        for index, rule in enumerate(
            self._rules
        ):
            if rule.id == rule_id:
                del self._rules[
                    index
                ]
                return

        raise KeyError(
            "Policy rule not found: "
            f"{rule_id}"
        )

    def rules(
        self,
    ) -> list[
        PolicyRule
    ]:
        return list(
            self._rules
        )

    def _matches(
        self,
        rule: PolicyRule,
        action: str,
    ) -> bool:
        """
        Match exact actions and simple
        namespace wildcards.

        Examples:

        tool.file.read
        tool.file.*

        A global '*' rule matches every
        action.
        """

        rule_action = (
            rule.action
            .strip()
            .lower()
        )

        candidate = (
            action
            .strip()
            .lower()
        )

        if rule_action == "*":
            return True

        if rule_action == candidate:
            return True

        if rule_action.endswith(
            ".*"
        ):
            prefix = (
                rule_action[:-1]
            )

            return candidate.startswith(
                prefix
            )

        return False

    def _decision(
        self,
        *,
        action: str,
        effect: PolicyEffect,
        reason: str,
        matched_rule_id: Optional[
            str
        ] = None,
        metadata: Optional[
            dict
        ] = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            action=action,
            effect=effect,
            allowed=(
                effect
                == PolicyEffect.ALLOW
            ),
            requires_approval=(
                effect
                == PolicyEffect
                .REQUIRE_APPROVAL
            ),
            matched_rule_id=(
                matched_rule_id
            ),
            reason=reason,
            metadata=dict(
                metadata or {}
            ),
        )

    def evaluate(
        self,
        action: str,
        *,
        context: Optional[
            dict
        ] = None,
    ) -> PolicyDecision:
        normalized = (
            action.strip()
        )

        if not normalized:
            raise ValueError(
                "Policy action cannot "
                "be empty."
            )

        for rule in self._rules:
            if not self._matches(
                rule,
                normalized,
            ):
                continue

            metadata = dict(
                rule.metadata
            )

            if context:
                metadata[
                    "context"
                ] = dict(
                    context
                )

            return self._decision(
                action=normalized,
                effect=rule.effect,
                reason=rule.reason,
                matched_rule_id=(
                    rule.id
                ),
                metadata=metadata,
            )

        return self._decision(
            action=normalized,
            effect=(
                self.default_effect
            ),
            reason=(
                "No policy rule matched; "
                "default policy applied."
            ),
            metadata={
                "context": dict(
                    context or {}
                )
            },
        )

    def enforce(
        self,
        action: str,
        *,
        context: Optional[
            dict
        ] = None,
    ) -> PolicyDecision:
        decision = self.evaluate(
            action,
            context=context,
        )

        if (
            decision.effect
            == PolicyEffect.DENY
        ):
            raise PolicyDeniedError(
                decision.reason
            )

        return decision

    def is_allowed(
        self,
        action: str,
        *,
        context: Optional[
            dict
        ] = None,
    ) -> bool:
        decision = self.evaluate(
            action,
            context=context,
        )

        return decision.allowed

    def requires_approval(
        self,
        action: str,
        *,
        context: Optional[
            dict
        ] = None,
    ) -> bool:
        decision = self.evaluate(
            action,
            context=context,
        )

        return (
            decision.requires_approval
        )
