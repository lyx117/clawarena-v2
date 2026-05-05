from __future__ import annotations

from openclaw_env.backend.base import CommandResult
from openclaw_env.skills.policies.openclaw_fallback import should_fallback_to_mock


def test_fallback_disabled_returns_false():
    result = CommandResult(stdout="", stderr="gateway connect failed", exit_code=1)
    assert should_fallback_to_mock(enabled=False, command="openclaw status", result=result) is False


def test_fallback_on_unknown_channel_tag():
    result = CommandResult(
        stdout="",
        stderr="Error: Unknown channel: discord",
        exit_code=1,
        meta={"error_tags": ["unknown_channel"]},
    )
    assert should_fallback_to_mock(
        enabled=True,
        command="openclaw message send --channel discord --target +123 --message hi",
        result=result,
    ) is True


def test_fallback_on_gateway_token_mismatch_marker():
    result = CommandResult(
        stdout="",
        stderr="gateway connect failed: Error: unauthorized: gateway token mismatch",
        exit_code=1,
    )
    assert should_fallback_to_mock(enabled=True, command="openclaw cron list", result=result) is True


def test_no_fallback_on_success_exit_code():
    result = CommandResult(stdout="ok", stderr="", exit_code=0)
    assert should_fallback_to_mock(enabled=True, command="openclaw status", result=result) is False
