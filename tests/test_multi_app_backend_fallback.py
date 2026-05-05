from __future__ import annotations

from openclaw_env.backend.base import CommandResult
from openclaw_env.backend.multi_app_backend import MultiAppBackend


def test_openclaw_network_error_falls_back_to_mock(tmp_path):
    backend = MultiAppBackend(
        real_openclaw=True,
        fallback_openclaw_network_to_mock=True,
    )
    backend.initialize(str(tmp_path), {})
    try:
        backend._openclaw_backend.execute_cli = lambda _cmd: CommandResult(  # type: ignore[attr-defined]
            stdout="",
            stderr=(
                "gateway connect failed: Error: unauthorized: gateway token mismatch "
                "(set gateway.remote.token to match gateway.auth.token)"
            ),
            exit_code=1,
            meta={"error_tags": []},
        )
        result = backend.execute_cli("openclaw cron list")
        assert result.exit_code == 0
        assert "No cron jobs" in result.stdout
        assert result.meta is not None
        assert result.meta.get("fallback_to_mock") is True
    finally:
        backend.cleanup()


def test_openclaw_unknown_channel_falls_back_to_mock(tmp_path):
    backend = MultiAppBackend(
        real_openclaw=True,
        fallback_openclaw_network_to_mock=True,
    )
    backend.initialize(str(tmp_path), {})
    try:
        backend._openclaw_backend.execute_cli = lambda _cmd: CommandResult(  # type: ignore[attr-defined]
            stdout="",
            stderr="Error: Unknown channel: discord\n",
            exit_code=1,
            meta={"error_tags": ["unknown_channel"]},
        )
        result = backend.execute_cli(
            "openclaw message send --channel discord --target +123 --message hi"
        )
        assert result.exit_code == 0
        assert result.meta is not None
        assert result.meta.get("fallback_to_mock") is True
    finally:
        backend.cleanup()
