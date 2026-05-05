"""Unit tests for the evaluation checkers."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from openclaw_env.core.observation import CheckResult, EvaluationResult
from openclaw_env.evaluation.checkers.config_checker import ConfigChecker
from openclaw_env.evaluation.checkers.effect_checker import EffectChecker
from openclaw_env.evaluation.checkers.output_checker import OutputChecker
from openclaw_env.evaluation.checkers.state_checker import StateChecker
from openclaw_env.evaluation.evaluator import EvaluatorComb, build_evaluator
from examples.train_and_eval import ExpertAgent


# ---- StateChecker ----

class TestStateChecker:
    def make_state(self):
        return {
            "agents": {
                "alice": {"model": "gpt-4o", "emoji": "🤖"},
            },
            "channels": {
                "telegram": {"status": "connected"},
            },
            "gateway_status": {"running": True},
        }

    def test_exists_pass(self):
        checker = StateChecker(field="agents.alice", condition="exists", expected=None)
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_exists_fail(self):
        checker = StateChecker(field="agents.bob", condition="exists", expected=None)
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_not_exists(self):
        checker = StateChecker(field="agents.unknown", condition="not_exists", expected=None)
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_equals_pass(self):
        checker = StateChecker(
            field="agents.alice.model", condition="equals", expected="gpt-4o"
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_equals_fail(self):
        checker = StateChecker(
            field="agents.alice.model", condition="equals", expected="claude"
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_count_gte_pass(self):
        state = {"cron_jobs": [{"id": "1"}, {"id": "2"}]}
        checker = StateChecker(field="cron_jobs", condition="count_gte", expected=2)
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_count_gte_fail(self):
        state = {"cron_jobs": [{"id": "1"}]}
        checker = StateChecker(field="cron_jobs", condition="count_gte", expected=3)
        result = checker.evaluate(state, {})
        assert result.passed is False

    def test_gateway_running(self):
        checker = StateChecker(
            field="gateway_status.running", condition="equals", expected=True
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True


# ---- OutputChecker ----

class TestOutputChecker:
    def make_state(self, stdout="Hello World"):
        return {"last_stdout": stdout, "last_stderr": ""}

    def test_contains_pass(self):
        checker = OutputChecker(match_type="contains", expected="Hello")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_contains_fail(self):
        checker = OutputChecker(match_type="contains", expected="Goodbye")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_contains_case_insensitive(self):
        checker = OutputChecker(match_type="contains", expected="hello", ignore_case=True)
        result = checker.evaluate(self.make_state("Hello World"), {})
        assert result.passed is True

    def test_exact_pass(self):
        checker = OutputChecker(match_type="exact", expected="Hello World")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_exact_fail(self):
        checker = OutputChecker(match_type="exact", expected="Hello")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_regex_pass(self):
        checker = OutputChecker(match_type="regex", expected=r"Hello\s+\w+")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_regex_fail(self):
        checker = OutputChecker(match_type="regex", expected=r"^\d+$")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_not_contains_pass(self):
        checker = OutputChecker(match_type="not_contains", expected="Error")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_not_contains_fail(self):
        checker = OutputChecker(match_type="not_contains", expected="World")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_stderr_field(self):
        checker = OutputChecker(
            match_type="contains", expected="error", output_field="last_stderr"
        )
        result = checker.evaluate({"last_stdout": "ok", "last_stderr": "some error"}, {})
        assert result.passed is True

    def test_from_spec_condition_alias(self):
        checker = OutputChecker.from_spec({
            "condition": "contains",
            "expected": "hello",
        })
        result = checker.evaluate(self.make_state("Hello World"), {})
        assert result.passed is True

    def test_exit_code_zero_with_null_expected(self):
        checker = OutputChecker.from_spec({
            "condition": "exit_code_zero",
            "expected": None,
        })
        ok = checker.evaluate({"last_stdout": "", "last_exit_code": 0}, {})
        bad = checker.evaluate({"last_stdout": "", "last_exit_code": 1}, {})
        assert ok.passed is True
        assert bad.passed is False


def test_expert_agent_should_stop_after_commands():
    agent = ExpertAgent()
    task = SimpleNamespace(
        ground_truth=SimpleNamespace(solution_commands=["tasks list"])
    )
    agent.reset("dummy", task=task)
    assert agent.should_stop() is False
    _ = agent.act(SimpleNamespace())
    assert agent.should_stop() is True


# ---- EffectChecker ----

class TestEffectChecker:
    def make_state(self, effects=None):
        return {
            "effects": effects or {},
        }

    def test_exists_pass(self):
        checker = EffectChecker(
            effect_type="messages_sent", condition="exists", expected=None
        )
        state = self.make_state({"messages_sent": [{"target": "@alice"}]})
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_exists_fail(self):
        checker = EffectChecker(
            effect_type="messages_sent", condition="exists", expected=None
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_not_exists(self):
        checker = EffectChecker(
            effect_type="messages_sent", condition="not_exists", expected=None
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_count(self):
        checker = EffectChecker(
            effect_type="messages_sent", condition="count", expected=2
        )
        state = self.make_state({
            "messages_sent": [{"id": "1"}, {"id": "2"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_count_gte(self):
        checker = EffectChecker(
            effect_type="cron_jobs_created", condition="count_gte", expected=1
        )
        state = self.make_state({"cron_jobs_created": [{"id": "abc"}]})
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_count_lte(self):
        checker = EffectChecker(
            effect_type="tasks_created", condition="count_lte", expected=0
        )
        result = checker.evaluate(self.make_state({"tasks_created": []}), {})
        assert result.passed is True

    def test_field_equals_pass(self):
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_equals",
            expected={"field": "target", "value": "@alice"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@alice", "message": "hi"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_field_equals_fail(self):
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_equals",
            expected={"field": "target", "value": "@bob"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@alice", "message": "hi"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is False

    def test_field_contains_pass(self):
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_contains",
            expected={"field": "message", "value": "hello"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@x", "message": "say hello world"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_field_contains_is_case_insensitive(self):
        checker = EffectChecker(
            effect_type="tasks_created",
            condition="field_contains",
            expected={"field": "title", "value": "Outage"},
        )
        state = self.make_state({
            "tasks_created": [{"title": "outage follow-up"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_plugins_installed(self):
        checker = EffectChecker(
            effect_type="plugins_installed",
            condition="field_equals",
            expected={"field": "name", "value": "slack-integration"},
        )
        state = self.make_state({
            "plugins_installed": [{"name": "slack-integration", "version": "1.0.0"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True


# ---- ConfigChecker ----



def test_merge_backend_state_does_not_backfill_created_effects_from_initial_state():
    from openclaw_env.skills.state_merge import merge_backend_state_into_eval

    state = {"effects": {
        "tasks_created": [],
        "calendar_events_created": [],
        "cron_jobs_created": [],
        "emails_sent": [],
    }}
    effects = state["effects"]
    backend_state = {
        "tasks": [{"id": "task_existing", "title": "Existing ops next step", "status": "pending"}],
        "calendar_events": [{"id": "evt_existing", "title": "Existing ops review block"}],
        "cron_jobs": [{"id": "cron_existing", "name": "existing-ops-check"}],
    }

    merge_backend_state_into_eval(state, effects, backend_state)

    assert state["tasks_list"][0]["id"] == "task_existing"
    assert state["calendar_events"][0]["id"] == "evt_existing"
    assert state["cron_jobs"][0]["id"] == "cron_existing"
    assert effects["tasks_created"] == []
    assert effects["calendar_events_created"] == []
    assert effects["cron_jobs_created"] == []


def test_duplicate_avoidance_effect_checks_ignore_existing_state_resources():
    checker = EffectChecker(
        effect_type="tasks_created",
        condition="not_exists",
        expected=None,
    )
    result = checker.evaluate({"effects": {"tasks_created": []}}, {})
    assert result.passed is True

class TestConfigChecker:
    def make_state(self):
        return {
            "config": {
                "agent": {"model": "anthropic/claude-opus-4-6"},
                "gateway": {
                    "port": "18789",
                    "auth": {"mode": "token"},
                },
            }
        }

    def test_equals_pass(self):
        checker = ConfigChecker(
            config_path="agent.model",
            condition="equals",
            expected="anthropic/claude-opus-4-6",
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_equals_fail(self):
        checker = ConfigChecker(
            config_path="agent.model",
            condition="equals",
            expected="wrong-model",
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_equals_numeric_string_and_int_pass(self):
        checker = ConfigChecker(
            config_path="gateway.port",
            condition="equals",
            expected=18789,
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_nested_path(self):
        checker = ConfigChecker(
            config_path="gateway.auth.mode",
            condition="equals",
            expected="token",
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_missing_path(self):
        checker = ConfigChecker(
            config_path="nonexistent.key",
            condition="equals",
            expected="value",
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_alias_agent_model_path(self):
        checker = ConfigChecker(
            config_path="agent.model",
            condition="equals",
            expected="anthropic/claude-opus-4-6",
        )
        state = {
            "config": {
                "agents": {
                    "defaults": {
                        "model": {"primary": "anthropic/claude-opus-4-6"}
                    }
                }
            }
        }
        result = checker.evaluate(state, {})
        assert result.passed is True


# ---- EvaluatorComb ----

class TestEvaluatorComb:
    def test_all_pass(self):
        checkers = [
            StateChecker(field="agents.alice", condition="exists", expected=None, weight=1.0),
            OutputChecker(match_type="contains", expected="hello", weight=1.0),
        ]
        comb = EvaluatorComb(checkers)
        state = {
            "agents": {"alice": {}},
            "last_stdout": "hello world",
        }
        result = comb(state, {})
        assert result.success is True
        assert result.score == pytest.approx(1.0)

    def test_partial_fail(self):
        checkers = [
            StateChecker(field="agents.alice", condition="exists", expected=None, weight=1.0),
            StateChecker(field="agents.bob", condition="exists", expected=None, weight=1.0),
        ]
        comb = EvaluatorComb(checkers)
        state = {"agents": {"alice": {}}}
        result = comb(state, {})
        assert result.success is False
        assert result.score == pytest.approx(0.5)

    def test_weighted_score(self):
        checkers = [
            StateChecker(field="agents.alice", condition="exists", expected=None, weight=3.0),
            StateChecker(field="agents.bob", condition="exists", expected=None, weight=1.0),
        ]
        comb = EvaluatorComb(checkers)
        state = {"agents": {"alice": {}}}
        result = comb(state, {})
        assert result.success is False
        assert result.score == pytest.approx(0.75)  # 3/4 weight passes

    def test_build_from_spec(self):
        checks = [
            {"type": "state", "field": "agents.alice", "condition": "exists"},
            {"type": "output", "match_type": "contains", "expected": "hello"},
            {"type": "effect", "effect_type": "messages_sent", "condition": "exists"},
            {"type": "config", "config_path": "agent.model", "condition": "equals",
             "expected": "gpt-4o"},
        ]
        comb = build_evaluator(checks)
        assert len(comb.evaluators) == 4

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown check type"):
            build_evaluator([{"type": "bogus", "field": "x"}])
