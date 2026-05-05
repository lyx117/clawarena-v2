"""Unit tests for the MockBackend."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from openclaw_env.backend.mock_backend import MockBackend


@pytest.fixture
def backend(tmp_path):
    b = MockBackend()
    config = {
        "agent": {"model": "anthropic/claude-sonnet-4-5-20250929"},
        "gateway": {"port": 18789, "auth": {"mode": "token", "token": "test"}},
        "channels": {},
    }
    (tmp_path / "openclaw.json").write_text(json.dumps(config))
    b.initialize(str(tmp_path), {})
    return b


# ---- Status / Health ----

def test_status_text(backend):
    result = backend.execute_cli("openclaw status")
    assert result.exit_code == 0
    assert "Gateway" in result.stdout


def test_status_json(backend):
    result = backend.execute_cli("openclaw status --json")
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "gateway" in data


def test_health_no_gateway(backend):
    result = backend.execute_cli("openclaw health")
    assert result.exit_code == 1
    assert "not running" in result.stderr


def test_health_after_start(backend):
    backend.execute_cli("openclaw gateway start")
    result = backend.execute_cli("openclaw health")
    assert result.exit_code == 0
    assert "OK" in result.stdout


# ---- Gateway ----

def test_gateway_start_stop(backend):
    r = backend.execute_cli("openclaw gateway start")
    assert r.exit_code == 0
    assert backend._gateway_running is True

    r = backend.execute_cli("openclaw gateway stop")
    assert r.exit_code == 0
    assert backend._gateway_running is False


def test_gateway_start_idempotent(backend):
    backend.execute_cli("openclaw gateway start")
    r = backend.execute_cli("openclaw gateway start")
    assert r.exit_code == 0
    assert "already running" in r.stdout


# ---- Agents ----

def test_agent_create(backend):
    result = backend.execute_cli("openclaw agents add --name alice --model openai/gpt-4o")
    assert result.exit_code == 0
    assert "alice" in result.stdout
    assert "alice" in backend._agents


def test_agent_create_duplicate(backend):
    backend.execute_cli("openclaw agents add --name bob --model openai/gpt-4o")
    r = backend.execute_cli("openclaw agents add --name bob --model openai/gpt-4o")
    assert r.exit_code == 1
    assert "already exists" in r.stderr


def test_agent_list_empty(backend):
    r = backend.execute_cli("openclaw agents list")
    assert r.exit_code == 0
    assert "No agents" in r.stdout


def test_agent_list(backend):
    backend.execute_cli("openclaw agents add --name researcher --model openai/gpt-4o")
    r = backend.execute_cli("openclaw agents list")
    assert "researcher" in r.stdout


def test_agent_delete(backend):
    backend.execute_cli("openclaw agents add --name temp --model openai/gpt-4o")
    r = backend.execute_cli("openclaw agents delete --name temp")
    assert r.exit_code == 0
    assert "temp" not in backend._agents


def test_agent_set_identity(backend):
    backend.execute_cli("openclaw agents add --name helper --model openai/gpt-4o")
    r = backend.execute_cli("openclaw agents set-identity --name helper --emoji 🤖")
    assert r.exit_code == 0
    assert backend._agents["helper"]["emoji"] == "🤖"


def test_agent_online_weather_pattern(backend):
    r = backend.execute_cli(
        "openclaw agent --local --agent main "
        "--message \"Use weather for location 'London'. "
        "Reply with WEATHER_RESULT:.\""
    )
    assert r.exit_code == 0
    assert "WEATHER_RESULT:" in r.stdout
    assert "London" in r.stdout


def test_agent_online_calendar_pattern(backend):
    r = backend.execute_cli(
        "openclaw agent --local --agent main "
        "--message \"Summarize events for day 'today'. "
        "Reply with CALENDAR_RESULT:.\""
    )
    assert r.exit_code == 0
    assert "CALENDAR_RESULT:" in r.stdout
    assert "today" in r.stdout.lower()


# ---- Messaging ----

def test_message_send(backend):
    r = backend.execute_cli(
        "openclaw message send --channel telegram --target @alice --message 'Hello world'"
    )
    assert r.exit_code == 0
    assert len(backend._messages) == 1
    assert backend._messages[0]["target"] == "@alice"


def test_message_send_with_channel(backend):
    r = backend.execute_cli(
        "openclaw message send --channel telegram --target @bob --message 'Hi'"
    )
    assert r.exit_code == 0
    assert backend._messages[0]["channel"] == "telegram"


def test_message_send_missing_target(backend):
    r = backend.execute_cli("openclaw message send --message 'oops'")
    assert r.exit_code == 1


def test_message_send_missing_channel(backend):
    r = backend.execute_cli("openclaw message send --target @alice --message 'oops'")
    assert r.exit_code == 1


def test_message_broadcast(backend):
    r = backend.execute_cli(
        "openclaw message broadcast --targets @a,@b,@c --message 'Broadcast'"
    )
    assert r.exit_code == 0
    assert len(backend._messages) == 3


def test_message_poll(backend):
    r = backend.execute_cli(
        "openclaw message poll --target #general --question 'Vote?' --options 'yes,no'"
    )
    assert r.exit_code == 0
    assert backend._messages[0]["type"] == "poll"


def test_message_search(backend):
    backend.execute_cli("openclaw message send --channel slack --target @x --message 'hello team'")
    backend.execute_cli("openclaw message send --channel slack --target @y --message 'random stuff'")
    r = backend.execute_cli("openclaw message search --query hello")
    assert r.exit_code == 0
    assert "Found 1" in r.stdout


def test_message_react(backend):
    r = backend.execute_cli(
        "openclaw message react --target @alice --emoji 👍 --channel telegram"
    )
    assert r.exit_code == 0
    assert "👍" in r.stdout


# ---- Channels ----

def test_channels_login(backend):
    r = backend.execute_cli("openclaw channels login --channel telegram")
    assert r.exit_code == 0
    assert "telegram" in backend._channels
    assert backend._channels["telegram"]["status"] == "connected"


def test_channels_login_requires_flag(backend):
    r = backend.execute_cli("openclaw channels login telegram")
    assert r.exit_code == 1


def test_channels_list(backend):
    r = backend.execute_cli("openclaw channels list")
    assert r.exit_code == 0


def test_channels_config_read(backend):
    backend.execute_cli("openclaw channels login --channel slack")
    r = backend.execute_cli("openclaw channels config --channel slack")
    assert r.exit_code == 0


def test_channels_config_set(backend):
    backend.execute_cli("openclaw channels login --channel discord")
    r = backend.execute_cli(
        "openclaw channels config --channel discord --key notification_level --value all"
    )
    assert r.exit_code == 0
    assert backend._channels["discord"]["config"]["notification_level"] == "all"


def test_channels_config_not_found(backend):
    r = backend.execute_cli("openclaw channels config --channel nonexistent")
    assert r.exit_code == 1


# ---- Plugins ----

def test_plugin_install(backend):
    r = backend.execute_cli("openclaw plugins install slack-integration")
    assert r.exit_code == 0
    assert "installed" in r.stdout.lower()
    assert len(backend._plugins) == 1
    assert backend._plugins[0]["name"] == "slack-integration"
    assert r.state_changes is not None
    assert "plugins" in r.state_changes


def test_plugin_install_duplicate(backend):
    backend.execute_cli("openclaw plugins install weather-plugin")
    r = backend.execute_cli("openclaw plugins install weather-plugin")
    assert r.exit_code == 0
    assert "already installed" in r.stdout.lower()
    assert len(backend._plugins) == 1  # no duplicate


def test_plugin_list(backend):
    r = backend.execute_cli("openclaw plugins list")
    assert r.exit_code == 0
    assert "No plugins" in r.stdout


def test_plugin_remove(backend):
    backend.execute_cli("openclaw plugins install github-notifier")
    r = backend.execute_cli("openclaw plugins remove github-notifier")
    assert r.exit_code == 0
    assert len(backend._plugins) == 0


# ---- Cron ----

def test_cron_add(backend):
    r = backend.execute_cli(
        "openclaw cron add --schedule '0 9 * * *' --command 'openclaw status'"
    )
    assert r.exit_code == 0
    assert len(backend._cron_jobs) == 1
    assert backend._cron_jobs[0]["schedule"] == "0 9 * * *"
    assert r.state_changes is not None
    assert "cron_jobs" in r.state_changes


def test_cron_add_missing_required_fields_fails(backend):
    r = backend.execute_cli("openclaw cron add")
    assert r.exit_code == 1
    assert len(backend._cron_jobs) == 0


def test_cron_list(backend):
    r = backend.execute_cli("openclaw cron list")
    assert r.exit_code == 0
    assert "No cron" in r.stdout


def test_cron_delete(backend):
    backend.execute_cli("openclaw cron add --schedule '* * * * *' --command 'echo hi'")
    job_id = backend._cron_jobs[0]["id"]
    r = backend.execute_cli(f"openclaw cron delete {job_id}")
    assert r.exit_code == 0
    assert len(backend._cron_jobs) == 0


# ---- Security ----

def test_security_status(backend):
    r = backend.execute_cli("openclaw security")
    assert r.exit_code == 0
    assert "Auth" in r.stdout


def test_security_set_token(backend):
    r = backend.execute_cli("openclaw security set-token mysecrettoken")
    assert r.exit_code == 0
    assert backend._config["gateway"]["auth"]["token"] == "mysecrettoken"
    assert backend._config["gateway"]["auth"]["mode"] == "token"


# ---- Models ----

def test_models_list(backend):
    r = backend.execute_cli("openclaw models")
    assert r.exit_code == 0
    assert "claude" in r.stdout.lower()


def test_models_set(backend):
    r = backend.execute_cli("openclaw models set anthropic/claude-opus-4-6")
    assert r.exit_code == 0
    assert backend._config["agent"]["model"] == "anthropic/claude-opus-4-6"


# ---- Configure ----

def test_configure_set_value(backend):
    r = backend.execute_cli("openclaw configure gateway.port=9090")
    assert r.exit_code == 0
    assert backend._config["gateway"]["port"] == "9090"


# ---- Doctor ----

def test_doctor(backend):
    r = backend.execute_cli("openclaw doctor")
    assert r.exit_code == 0
    assert "Doctor" in r.stdout


# ---- Devices ----

def test_devices_pair(backend):
    r = backend.execute_cli("openclaw devices pair")
    assert r.exit_code == 0
    assert "pairing" in r.stdout.lower()


def test_devices_list(backend):
    r = backend.execute_cli("openclaw devices list")
    assert r.exit_code == 0


# ---- Unknown commands ----

def test_unknown_subcommand(backend):
    r = backend.execute_cli("openclaw nonexistent")
    assert r.exit_code == 1


def test_non_openclaw_command(backend):
    r = backend.execute_cli("ls -la")
    assert r.exit_code == 127
