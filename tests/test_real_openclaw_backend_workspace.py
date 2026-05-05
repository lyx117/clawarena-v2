from __future__ import annotations

from types import SimpleNamespace

from openclaw_env.backend.real_openclaw_backend import RealOpenClawBackend


def test_agents_add_injects_workspace(monkeypatch, tmp_path):
    backend = RealOpenClawBackend()
    backend.initialize(str(tmp_path), {})

    seen: dict[str, list[str]] = {}

    def _fake_run(args, **_kwargs):
        seen["args"] = list(args)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "openclaw_env.backend.real_openclaw_backend.subprocess.run",
        _fake_run,
    )

    result = backend.execute_cli("openclaw agents add --name researcher --model openai/gpt-4o")

    assert result.exit_code == 0
    assert "--workspace" in seen["args"]
    ws_idx = seen["args"].index("--workspace")
    assert seen["args"][ws_idx + 1] == str(tmp_path / "workspace")
    assert result.meta is not None
    tags = result.meta.get("error_tags", [])
    assert "rewritten_agents_add_workspace" in tags


def test_agents_set_identity_uses_workspace_cwd_and_rewrite(monkeypatch, tmp_path):
    backend = RealOpenClawBackend()
    backend.initialize(str(tmp_path), {})
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)

    seen: dict[str, object] = {}

    def _fake_run(args, **kwargs):
        seen["args"] = list(args)
        seen["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "openclaw_env.backend.real_openclaw_backend.subprocess.run",
        _fake_run,
    )

    result = backend.execute_cli("openclaw agents set-identity --name helper --emoji '🔍'")

    assert result.exit_code == 0
    assert seen["cwd"] == str(tmp_path / "workspace")
    args = seen["args"]
    assert isinstance(args, list)
    assert "--agent" in args
    assert "--name" not in args


def test_agents_set_identity_infers_emoji_state_change(monkeypatch, tmp_path):
    backend = RealOpenClawBackend()
    backend.initialize(str(tmp_path), {})
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)

    def _fake_run(_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "openclaw_env.backend.real_openclaw_backend.subprocess.run",
        _fake_run,
    )

    result = backend.execute_cli("openclaw agents set-identity --name helper --emoji '🔍'")

    assert result.exit_code == 0
    assert result.state_changes is not None
    assert result.state_changes.get("agents", {}).get("helper", {}).get("emoji") == "🔍"
