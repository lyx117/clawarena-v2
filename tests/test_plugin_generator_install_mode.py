from __future__ import annotations

from openclaw_env.tasks.generation_options import set_generation_options
from openclaw_env.tasks.registry import generate_all_tasks


ONLINE_PACKAGE_BY_ID = {
    "discord": "@openclaw/discord",
    "slack": "@openclaw/slack",
    "telegram": "@openclaw/telegram",
    "whatsapp": "@openclaw/whatsapp",
}


def _tasks_by_name(tasks):
    out = {}
    for task in tasks:
        if not task.ground_truth:
            continue
        params = task.ground_truth.metadata.get("params", {})
        name = params.get("name")
        if name is not None:
            out[name] = task
    return out


def test_install_plugin_uses_real_install_for_online_subset_in_mixed_mode():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    try:
        tasks = generate_all_tasks(generator_ids=["install_plugin"])
        by_name = _tasks_by_name(tasks)

        for plugin_id, package in ONLINE_PACKAGE_BY_ID.items():
            task = by_name[plugin_id]
            commands = task.ground_truth.solution_commands
            assert commands == [f"openclaw plugins install {package}"]
            check = task.ground_truth.evaluation_checks[0]
            assert check["match_type"] == "regex"
            assert "install|already" in check["expected"]
    finally:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="local_skill",
        )


def test_install_and_verify_uses_install_and_list_checks_in_mixed_mode():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    try:
        tasks = generate_all_tasks(generator_ids=["install_and_verify_plugin"])
        by_name = _tasks_by_name(tasks)
        task = by_name["discord"]
        commands = task.ground_truth.solution_commands
        assert commands[0] == "openclaw plugins install @openclaw/discord"
        assert commands[1] == "openclaw plugins list"

        checks = task.ground_truth.evaluation_checks
        assert checks[0]["output_field"] == "command_history"
        assert "openclaw\\ plugins\\ install\\ @openclaw/discord" in checks[0]["expected"]
        assert checks[1]["type"] == "effect"
        assert checks[1]["expected"]["value"] == "@openclaw/discord"
        assert checks[2]["type"] == "output"
        assert checks[2]["match_type"] == "regex"
    finally:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="local_skill",
        )


def test_plugin_install_mode_stable_keeps_enable_commands():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="stable",
        command_profile="local_skill",
    )
    try:
        tasks = generate_all_tasks(
            generator_ids=["install_plugin", "install_and_verify_plugin"]
        )
        for task in tasks:
            commands = task.ground_truth.solution_commands
            assert all("openclaw plugins install " not in c for c in commands)
            assert "openclaw plugins enable " in commands[0]
    finally:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="local_skill",
        )
