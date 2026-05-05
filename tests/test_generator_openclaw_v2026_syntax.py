from __future__ import annotations

import openclaw_env.tasks.generators.agent_mgmt  # noqa: F401
import openclaw_env.tasks.generators.calendar_tasks  # noqa: F401
import openclaw_env.tasks.generators.channel_mgmt  # noqa: F401
import openclaw_env.tasks.generators.composite  # noqa: F401
import openclaw_env.tasks.generators.complex_workflows  # noqa: F401
import openclaw_env.tasks.generators.cron_webhook  # noqa: F401
import openclaw_env.tasks.generators.device_node  # noqa: F401
import openclaw_env.tasks.generators.email_tasks  # noqa: F401
import openclaw_env.tasks.generators.file_tasks  # noqa: F401
import openclaw_env.tasks.generators.hard_decision_workflows  # noqa: F401
import openclaw_env.tasks.generators.branch_sensitive_workflows  # noqa: F401
import openclaw_env.tasks.generators.messaging  # noqa: F401
import openclaw_env.tasks.generators.monitoring  # noqa: F401
import openclaw_env.tasks.generators.online_reads  # noqa: F401
import openclaw_env.tasks.generators.plugin_skill  # noqa: F401
import openclaw_env.tasks.generators.security  # noqa: F401
import openclaw_env.tasks.generators.setup_config  # noqa: F401
import openclaw_env.tasks.generators.tasks_tasks  # noqa: F401
import openclaw_env.tasks.generators.weather_tasks  # noqa: F401
from openclaw_env.tasks.generation_options import set_generation_options
from openclaw_env.tasks.registry import generate_all_tasks


BANNED_PATTERNS = [
    "openclaw configure ",
    "openclaw security set-token",
    "openclaw channels config",
    "openclaw plugins remove",
    "openclaw webhooks add",
    "openclaw agents add --name",
    "openclaw agents set-identity --name",
    "openclaw message poll --question",
    "openclaw cron add --schedule",
]


def _all_solution_commands(tasks) -> list[str]:
    cmds: list[str] = []
    for task in tasks:
        if not task.ground_truth:
            continue
        cmds.extend(task.ground_truth.solution_commands)
    return cmds


def test_generated_tasks_do_not_use_legacy_openclaw_syntax():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    tasks = generate_all_tasks()
    commands = _all_solution_commands(tasks)
    lowered = [c.lower() for c in commands]

    for banned in BANNED_PATTERNS:
        assert all(banned not in cmd for cmd in lowered), banned


def test_generated_tasks_include_local_skill_profile_markers():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    tasks = generate_all_tasks()
    commands = _all_solution_commands(tasks)

    assert any(cmd.startswith("openclaw config set gateway.port") for cmd in commands)
    assert any("openclaw agents add " in cmd and "--name" not in cmd for cmd in commands)
    assert any("openclaw plugins enable " in cmd for cmd in commands)
    assert any("openclaw plugins disable " in cmd for cmd in commands)
    assert any("openclaw plugins install @openclaw/" in cmd for cmd in commands)
    assert any("openclaw cron add " in cmd and "--cron" in cmd and "--name" in cmd for cmd in commands)
    assert any(cmd.startswith("weather get --location ") for cmd in commands)
    assert any(cmd.startswith("calendar add-event --title ") for cmd in commands)
    assert all(not cmd.startswith("gcalcli ") for cmd in commands)
    assert all(not cmd.startswith("curl -sG ") for cmd in commands)


def test_email_review_wording_requires_email_read_command():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    tasks = generate_all_tasks(generator_ids=["email_search_and_read", "hard_decision_workflow"])

    for task in tasks:
        instruction = (task.instruction or "").lower()
        commands = list(task.ground_truth.solution_commands) if task.ground_truth else []
        if "email" in instruction and ("review " in instruction or " read the matching email" in instruction):
            assert any(cmd.startswith("email read --id ") for cmd in commands), task.task_id


def test_universal_profile_is_not_supported():
    try:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="universal",
        )
    except ValueError as exc:
        assert "Only 'local_skill' is supported" in str(exc)
    else:
        raise AssertionError("expected ValueError for command_profile=universal")


def test_message_dry_run_generation_option_applies_to_message_generators():
    set_generation_options(
        message_dry_run=True,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    tasks = generate_all_tasks(
        generator_ids=[
            "msg_send_text",
            "msg_broadcast",
            "msg_create_poll",
            "msg_search",
            "msg_react",
        ]
    )
    try:
        commands = _all_solution_commands(tasks)
        assert commands
        assert all("--dry-run" in cmd for cmd in commands)
        assert all(
            task.ground_truth.metadata["generation_options"]["message_dry_run"] is True
            for task in tasks
            if task.ground_truth
        )
    finally:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="local_skill",
        )


def test_plugin_install_mode_stable_has_no_real_install_commands():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="stable",
        command_profile="local_skill",
    )
    try:
        tasks = generate_all_tasks(
            generator_ids=["install_plugin", "install_and_verify_plugin"]
        )
        commands = _all_solution_commands(tasks)
        assert commands
        assert all("openclaw plugins install " not in c for c in commands)
        assert any("openclaw plugins enable " in c for c in commands)
    finally:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="local_skill",
        )


def test_generated_tasks_do_not_include_legacy_plugin_names():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    tasks = generate_all_tasks()
    lowered = [c.lower() for c in _all_solution_commands(tasks)]
    banned_legacy_names = ("weather-plugin", "slack-integration", "github-notifier")
    for token in banned_legacy_names:
        assert all(token not in cmd for cmd in lowered), token
