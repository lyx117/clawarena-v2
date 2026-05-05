from __future__ import annotations

import pytest

from openclaw_env.tasks.generation_options import (
    get_generation_options,
    set_generation_options,
)


def test_generation_options_defaults_include_local_skill_profile():
    set_generation_options()
    opts = get_generation_options()
    assert opts.message_dry_run is False
    assert opts.plugin_install_mode == "mixed"
    assert opts.command_profile == "local_skill"
    assert opts.complex_task_pack == "standard"
    assert opts.complex_scenario_profile == "life_work"
    assert opts.complex_min_steps == 3
    assert opts.complex_max_steps == 5
    assert opts.hard_decision_variants_per_scenario == 16
    assert opts.hard_decision_scenario_counts == {}
    assert opts.include_branch_sensitive is False
    assert opts.branch_sensitive_variants_per_scenario == 0
    assert opts.to_dict()["command_profile"] == "local_skill"
    assert opts.to_dict()["complex_task_pack"] == "standard"
    assert opts.to_dict()["complex_scenario_profile"] == "life_work"
    assert opts.to_dict()["hard_decision_variants_per_scenario"] == 16
    assert opts.to_dict()["hard_decision_scenario_counts"] == {}
    assert opts.to_dict()["include_branch_sensitive"] is False
    assert opts.to_dict()["branch_sensitive_variants_per_scenario"] == 0


def test_generation_options_reject_universal_profile():
    with pytest.raises(ValueError, match="Only 'local_skill' is supported"):
        set_generation_options(command_profile="universal")


def test_generation_options_reject_invalid_profile():
    with pytest.raises(ValueError, match="Invalid command_profile"):
        set_generation_options(command_profile="invalid")


def test_generation_options_reject_invalid_complex_pack():
    with pytest.raises(ValueError, match="Invalid complex_task_pack"):
        set_generation_options(complex_task_pack="invalid")


def test_generation_options_reject_invalid_complex_scenario_profile():
    with pytest.raises(ValueError, match="Invalid complex_scenario_profile"):
        set_generation_options(complex_scenario_profile="invalid")


def test_generation_options_reject_invalid_step_bounds():
    with pytest.raises(ValueError, match="complex_min_steps"):
        set_generation_options(complex_min_steps=0)
    with pytest.raises(ValueError, match="complex_max_steps must be >="):
        set_generation_options(complex_min_steps=4, complex_max_steps=3)
    with pytest.raises(ValueError, match="<= 5"):
        set_generation_options(complex_max_steps=6)


def test_generation_options_reject_invalid_hard_decision_variant_count():
    with pytest.raises(ValueError, match="hard_decision_variants_per_scenario"):
        set_generation_options(hard_decision_variants_per_scenario=0)


def test_generation_options_reject_invalid_branch_sensitive_variant_count():
    with pytest.raises(ValueError, match="branch_sensitive_variants_per_scenario"):
        set_generation_options(branch_sensitive_variants_per_scenario=-1)


def test_generation_options_require_positive_branch_variant_count_when_enabled():
    with pytest.raises(ValueError, match="include_branch_sensitive"):
        set_generation_options(include_branch_sensitive=True, branch_sensitive_variants_per_scenario=0)


def test_generation_options_accept_hard_decision_scenario_counts():
    set_generation_options(
        hard_decision_scenario_counts={
            "existing_state_followthrough": 24,
            "state_repair_followthrough": 10,
        }
    )
    opts = get_generation_options()
    assert opts.hard_decision_scenario_counts == {
        "existing_state_followthrough": 24,
        "state_repair_followthrough": 10,
    }


def test_generation_options_reject_invalid_hard_decision_scenario_count_values():
    with pytest.raises(ValueError, match="hard_decision_scenario_counts values"):
        set_generation_options(hard_decision_scenario_counts={"existing_state_followthrough": 0})


def test_generation_options_reject_invalid_hard_decision_scenario_count_keys():
    with pytest.raises(ValueError, match="hard_decision_scenario_counts keys"):
        set_generation_options(hard_decision_scenario_counts={"": 3})
