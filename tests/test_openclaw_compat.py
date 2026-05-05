"""Tests for OpenClaw runtime compatibility rewrite helpers."""

from __future__ import annotations

from openclaw_env.backend.openclaw_compat import classify_error, rewrite_command


def test_rewrite_message_poll_question_flag():
    cmd = "openclaw message poll --target #general --question 'Vote?' --options yes,no"
    d = rewrite_command(cmd)
    assert d.compat_status == "rewritten"
    assert "--poll-question" in d.executed_action
    assert "--question" not in d.executed_action


def test_rewrite_cron_add_injects_name():
    cmd = "openclaw cron add --schedule '0 9 * * *' --command 'openclaw status'"
    d = rewrite_command(cmd)
    assert d.compat_status == "rewritten"
    assert "--name" in d.executed_action


def test_rewrite_cron_nested_command_quotes():
    cmd = (
        "openclaw cron add --schedule '0 8 * * 1' "
        "--command 'openclaw message send --channel whatsapp --target @alice "
        "--message 'Daily report''"
    )
    d = rewrite_command(cmd)
    assert d.compat_status == "rewritten"
    assert "rewritten_cron_command_quoting" in d.error_tags
    # Ensure there is no trailing token outside --command due broken quoting.
    assert not d.executed_action.endswith(" report")
    assert "--name" in d.executed_action


def test_skip_incomplete_cron_add():
    d = rewrite_command("openclaw cron add", skip_incompatible=True)
    assert d.compat_status == "skipped_incompatible"
    assert "incomplete_cron_add" in d.error_tags


def test_skip_incompatible_message_react_target_only():
    cmd = "openclaw message react --channel slack --target @bob --emoji '✅'"
    d = rewrite_command(cmd, skip_incompatible=True)
    assert d.compat_status == "skipped_incompatible"
    assert "incompatible_message_react_target_only" in d.error_tags


def test_rewrite_agents_set_identity_name_to_agent():
    cmd = "openclaw agents set-identity --name helper --emoji '🔍'"
    d = rewrite_command(cmd)
    assert d.compat_status == "rewritten"
    assert "--agent" in d.executed_action
    assert "--name" not in d.executed_action
    assert "rewritten_agents_set_identity_name" in d.error_tags


def test_rewrite_configure_key_value_to_config_set():
    cmd = "openclaw configure gateway.port=18789"
    d = rewrite_command(cmd)
    assert d.compat_status == "rewritten"
    assert d.executed_action == "openclaw config set gateway.port 18789"
    assert "rewritten_configure_key_value" in d.error_tags


def test_classify_error_tags():
    text = "Invalid config at /tmp/x/openclaw.json\\nerror: unknown option '--name'"
    tags = classify_error(text)
    assert "invalid_config" in tags
    assert "unknown_option" in tags


def test_classify_error_channel_and_plugin_tags():
    text = (
        "Error: Unknown channel: discord\n"
        "Channel login failed: Error: Unsupported channel: discord\n"
        "Error: Channel is required (no configured channels detected).\n"
        "Error: Broadcast requires at least one configured channel.\n"
        "Plugin not found: weather-plugin\n"
    )
    tags = classify_error(text)
    assert "unknown_channel" in tags
    assert "unsupported_channel" in tags
    assert "channel_required" in tags
    assert "channel_not_configured" in tags
    assert "plugin_not_found" in tags
