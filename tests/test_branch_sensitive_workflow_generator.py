from __future__ import annotations

import openclaw_env.tasks.generators.agent_mgmt  # noqa: F401
import openclaw_env.tasks.generators.branch_sensitive_workflows  # noqa: F401
import openclaw_env.tasks.generators.calendar_tasks  # noqa: F401
import openclaw_env.tasks.generators.channel_mgmt  # noqa: F401
import openclaw_env.tasks.generators.complex_workflows  # noqa: F401
import openclaw_env.tasks.generators.composite  # noqa: F401
import openclaw_env.tasks.generators.cron_webhook  # noqa: F401
import openclaw_env.tasks.generators.device_node  # noqa: F401
import openclaw_env.tasks.generators.email_tasks  # noqa: F401
import openclaw_env.tasks.generators.file_tasks  # noqa: F401
import openclaw_env.tasks.generators.hard_decision_workflows  # noqa: F401
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


def _branch_tasks(variants_per_scenario: int = 12):
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="standard",
        complex_scenario_profile="life_work",
        complex_min_steps=3,
        complex_max_steps=5,
        hard_decision_variants_per_scenario=16,
        include_branch_sensitive=True,
        branch_sensitive_variants_per_scenario=variants_per_scenario,
    )
    return generate_all_tasks(generator_ids=["branch_sensitive_workflow"])


def test_branch_sensitive_workflow_is_disabled_by_default():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="standard",
        complex_scenario_profile="life_work",
        complex_min_steps=3,
        complex_max_steps=5,
        hard_decision_variants_per_scenario=16,
        include_branch_sensitive=False,
        branch_sensitive_variants_per_scenario=0,
    )
    assert generate_all_tasks(generator_ids=["branch_sensitive_workflow"]) == []


def test_branch_sensitive_workflow_generator_produces_expected_default_pack():
    tasks = _branch_tasks()
    assert len(tasks) == 48

    by_scenario: dict[str, int] = {}
    signatures: set[tuple[str, tuple[str, ...]]] = set()
    for task in tasks:
        assert task.task_id.startswith("branch_sensitive_workflow_")
        scenario = str(task.data.public.get("branch_sensitive_scenario", ""))
        assert scenario
        by_scenario[scenario] = by_scenario.get(scenario, 0) + 1
        assert task.data.public.get("branch_basis")
        assert task.data.public.get("step_count") == len(task.ground_truth.solution_commands)
        assert "challenge" in task.realism_tags
        assert "branch_sensitive" in task.realism_tags
        assert "state_branching" in task.realism_tags
        sig = (task.instruction, tuple(task.ground_truth.solution_commands))
        assert sig not in signatures
        signatures.add(sig)

    assert by_scenario == {
        "inbox_stateful_triage": 12,
        "channel_delivery_recovery": 12,
        "release_gate_branch": 12,
        "ops_commitment_branch": 12,
    }


def test_branch_sensitive_workflow_generation_is_stable_for_same_options():
    tasks_a = _branch_tasks()
    tasks_b = _branch_tasks()
    assert [t.task_id for t in tasks_a] == [t.task_id for t in tasks_b]
    assert [t.instruction for t in tasks_a] == [t.instruction for t in tasks_b]
    assert [
        tuple(t.ground_truth.solution_commands) for t in tasks_a
    ] == [
        tuple(t.ground_truth.solution_commands) for t in tasks_b
    ]


def test_branch_sensitive_workflow_variant_count_is_configurable():
    tasks = _branch_tasks(variants_per_scenario=4)
    assert len(tasks) == 16
    counts: dict[str, int] = {}
    for task in tasks:
        scenario = str(task.data.public.get("branch_sensitive_scenario", ""))
        counts[scenario] = counts.get(scenario, 0) + 1
    assert all(v == 4 for v in counts.values())


def test_branch_sensitive_workflow_alignment_and_branch_probe_invariants_hold():
    tasks = _branch_tasks()
    for task in tasks:
        instruction = task.instruction.lower()
        commands = list(task.ground_truth.solution_commands)
        scenario = str(task.data.public.get("branch_sensitive_scenario", ""))
        if scenario == "inbox_stateful_triage":
            assert "review" in instruction
            assert any(cmd.startswith("email read --id ") for cmd in commands)
            if "live follow-up" in instruction:
                pass
        elif scenario == "channel_delivery_recovery":
            assert "recover" in instruction
            assert commands[0] == "openclaw channels list --json"
            assert any(cmd.startswith("openclaw channels login --channel ") for cmd in commands)
        elif scenario == "release_gate_branch":
            assert "check the current release model" in instruction or "current release gate setup" in instruction
            assert commands[0] == "openclaw config get agent.model"
            assert any(cmd.startswith("file create --path ") for cmd in commands)
        elif scenario == "ops_commitment_branch":
            assert "forecast" in instruction
            assert commands[0].startswith("weather forecast --location ")
            assert "openclaw cron list" in commands
        else:
            raise AssertionError(f"unexpected scenario: {scenario}")


def test_branch_sensitive_workflow_checks_are_result_first():
    tasks = _branch_tasks()
    for task in tasks:
        scenario = str(task.data.public.get("branch_sensitive_scenario", ""))
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        if scenario == "inbox_stateful_triage":
            assert "task created" in names
            assert not any(name.startswith("task title references ") for name in names)
            assert not any(name.startswith("calendar title references ") for name in names)
        elif scenario == "channel_delivery_recovery":
            assert "task created" in names
            assert not any(name.startswith("task title references ") for name in names)
        elif scenario == "release_gate_branch":
            assert "task created" in names
            assert "calendar event created" in names
            assert not any(name.startswith("task title references ") for name in names)
        elif scenario == "ops_commitment_branch":
            assert "task created" in names
            assert "calendar event created" in names
            assert not any(name.startswith("calendar title references ") for name in names)
