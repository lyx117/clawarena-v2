"""Tests for examples/train_and_eval.py CLI mode handling and preflight."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

import examples.train_and_eval as train_and_eval


def test_preflight_requires_openclaw_for_real_and_hybrid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(train_and_eval.shutil, "which", lambda _: None)

    for mode in ("real", "hybrid"):
        with pytest.raises(RuntimeError, match="openclaw"):
            train_and_eval._preflight_mode(mode)


def test_preflight_skips_openclaw_check_for_mock_and_multi(monkeypatch: pytest.MonkeyPatch):
    def _unexpected(_: str):
        raise AssertionError("shutil.which should not be called for mock/multi")

    monkeypatch.setattr(train_and_eval.shutil, "which", _unexpected)

    train_and_eval._preflight_mode("mock")
    train_and_eval._preflight_mode("multi")


def test_preflight_reports_status_failures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(train_and_eval.shutil, "which", lambda _: "/usr/bin/openclaw")
    monkeypatch.setattr(
        train_and_eval.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(returncode=1, stderr="boom", stdout=""),
    )

    with pytest.raises(RuntimeError, match="status --json"):
        train_and_eval._preflight_mode("real")


def test_cli_accepts_real_mode(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (calls.setdefault("eval_mode", kwargs["mode"]), _DummySummary())[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_and_eval.py", "--agent", "expert", "--split", "dev", "--mode", "real", "--limit", "1"],
    )

    train_and_eval.main()

    assert calls.get("mode") == "real"
    assert calls.get("eval_mode") == "real"
    assert calls.get("printed") is True


def test_cli_accepts_online_flags(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("online_clean", kwargs["online_clean"]),
            calls.setdefault("skip_incompatible_openclaw", kwargs["skip_incompatible_openclaw"]),
            calls.setdefault("online_openclaw_only", kwargs["online_openclaw_only"]),
            _DummySummary(),
        )[3],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--mode",
            "hybrid",
            "--online-clean",
            "--online-openclaw-only",
            "--skip-incompatible-openclaw",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "hybrid"
    assert calls.get("online_clean") is True
    assert calls.get("skip_incompatible_openclaw") is True
    assert calls.get("online_openclaw_only") is True
    assert calls.get("printed") is True


def test_cli_defaults_max_steps_to_fifteen(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(
        train_and_eval,
        "_build_agent",
        lambda *_, **kwargs: (calls.setdefault("max_steps_hint", kwargs["max_steps_hint"]), object())[1],
    )
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (calls.setdefault("max_steps", kwargs["max_steps"]), _DummySummary())[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_and_eval.py", "--agent", "expert", "--split", "dev", "--limit", "1"],
    )

    train_and_eval.main()

    assert calls.get("max_steps") == 15
    assert calls.get("max_steps_hint") == 15
    assert calls.get("printed") is True


def test_cli_total_split_keeps_default_max_steps_to_fifteen(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(
        train_and_eval,
        "_build_agent",
        lambda *_, **kwargs: (calls.setdefault("max_steps_hint", kwargs["max_steps_hint"]), object())[1],
    )
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (calls.setdefault("max_steps", kwargs["max_steps"]), _DummySummary())[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_and_eval.py", "--agent", "expert", "--split", "total", "--limit", "1"],
    )

    train_and_eval.main()

    assert calls.get("max_steps") == 15
    assert calls.get("max_steps_hint") == 15
    assert calls.get("printed") is True


def test_cli_accepts_task_data_dir(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("task_data_dir", kwargs["task_data_dir"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--mode",
            "hybrid",
            "--task-data-dir",
            "/tmp/profile_local",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "hybrid"
    assert calls.get("task_data_dir") == "/tmp/profile_local"
    assert calls.get("printed") is True


def test_cli_accepts_task_prefix(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("task_prefix", kwargs["task_prefix"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--task-prefix",
            "hard_decision_workflow_",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "multi"
    assert calls.get("task_prefix") == "hard_decision_workflow_"
    assert calls.get("printed") is True


def test_cli_accepts_custom_dataset_split_names(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("split", kwargs["split"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "duplicate_avoidance_followthrough",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "multi"
    assert calls.get("split") == "duplicate_avoidance_followthrough"
    assert calls.get("printed") is True


def test_cli_verbose_log_tees_stdout_to_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    class _DummySummary:
        def print_report(self) -> None:
            print("REPORT LINE")

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary())

    log_path = tmp_path / "verbose.log"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--limit",
            "1",
            "--verbose-log",
            str(log_path),
        ],
    )

    train_and_eval.main()

    captured = capsys.readouterr()
    assert "REPORT LINE" in captured.out
    assert log_path.exists()
    assert "REPORT LINE" in log_path.read_text(encoding="utf-8")




def test_cli_accepts_inter_task_sleep(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("inter_task_sleep_s", kwargs["inter_task_sleep_s"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--inter-task-sleep",
            "2.5",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "multi"
    assert calls.get("inter_task_sleep_s") == 2.5
    assert calls.get("printed") is True


def test_run_evaluation_sleeps_between_tasks(monkeypatch: pytest.MonkeyPatch):
    task_ids = ["agent_create_1", "tasks_add_1", "file_create_1"]
    sleeps: list[float] = []

    monkeypatch.setattr(train_and_eval, "load_task_ids", lambda *_, **__: list(task_ids))
    monkeypatch.setattr(
        train_and_eval,
        "run_episode",
        lambda task_id, *_, **__: train_and_eval.EpisodeResult(
            task_id=task_id,
            agent_name="expert",
            success=True,
            score=1.0,
            steps=1,
            duration_s=0.0,
        ),
    )
    monkeypatch.setattr(train_and_eval.time, "sleep", lambda seconds: sleeps.append(seconds))

    summary = train_and_eval.run_evaluation(
        agent=train_and_eval.ExpertAgent(),
        split="dev",
        inter_task_sleep_s=1.25,
        verbose=False,
    )

    assert summary.total == 3
    assert sleeps == [1.25, 1.25]

def test_cli_accepts_openai_llm_options(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    def _build_agent(*args, **kwargs):
        calls["build_agent_args"] = args
        calls["build_agent_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", _build_agent)
    monkeypatch.setattr(train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "llm",
            "--model",
            "gpt-5.2",
            "--llm-provider",
            "openai",
            "--llm-base-url",
            "https://example.test/v1",
            "--llm-api-key-env",
            "CUSTOM_API_KEY",
            "--llm-history-mode",
            "summary",
            "--llm-temperature",
            "0.3",
            "--llm-max-tokens",
            "384",
            "--llm-timeout-s",
            "17",
            "--llm-request-retries",
            "2",
            "--llm-retry-backoff-s",
            "2.5",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "multi"
    assert calls["build_agent_args"] == ("llm", "gpt-5.2")
    assert calls["build_agent_kwargs"] == {
        "llm_provider": "openai",
        "llm_base_url": "https://example.test/v1",
        "llm_api_key_env": "CUSTOM_API_KEY",
        "llm_history_mode": "summary",
        "llm_temperature": 0.3,
        "llm_max_tokens": 384,
        "llm_timeout_s": 17,
        "llm_request_retries": 2,
        "llm_retry_backoff_s": 2.5,
        "max_steps_hint": 15,
    }
    assert calls.get("printed") is True


def test_llm_agent_full_history_mode_keeps_messages_for_openai():
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2", history_mode="full")
    agent._history = [
        {"action": "tasks list --status pending", "stdout": "Tasks...", "stderr": "", "exit_code": 0},
    ]
    messages = [
        {"role": "user", "content": "Task: first"},
        {"role": "assistant", "content": "cmd-1"},
        {"role": "user", "content": "obs-1"},
        {"role": "assistant", "content": "cmd-2"},
        {"role": "user", "content": "obs-2"},
    ]

    assert agent._request_messages(messages) == messages


def test_llm_agent_summary_history_mode_trims_messages_for_anthropic():
    agent = train_and_eval.LLMAgent(provider="anthropic", model="claude-haiku-4-5-20251001", history_mode="summary")
    agent._history = [
        {"action": "weather forecast --location Berlin --days 1", "stdout": "Rain tomorrow", "stderr": "", "exit_code": 0},
        {"action": "calendar today --timezone Europe/Berlin", "stdout": "Review at 14:00", "stderr": "", "exit_code": 0},
        {"action": "tasks list --status pending", "stdout": "1. Berlin follow-up", "stderr": "", "exit_code": 0},
    ]
    messages = [
        {"role": "user", "content": "Task: first"},
        {"role": "assistant", "content": "cmd-1"},
        {"role": "user", "content": "obs-1"},
        {"role": "assistant", "content": "cmd-2"},
        {"role": "user", "content": "obs-2"},
        {"role": "assistant", "content": "cmd-3"},
        {"role": "user", "content": "obs-3"},
        {"role": "assistant", "content": "cmd-4"},
        {"role": "user", "content": "obs-4"},
        {"role": "assistant", "content": "cmd-5"},
        {"role": "user", "content": "obs-5"},
    ]

    trimmed = agent._request_messages(messages)

    assert trimmed[0] == messages[0]
    assert trimmed[1]["role"] == "user"
    assert trimmed[1]["content"].startswith("Known context:\n")
    assert len(trimmed) < len(messages)


def test_llm_agent_auto_history_mode_is_provider_agnostic():
    history = [
        {"action": "weather forecast --location Berlin --days 1", "stdout": "Rain tomorrow", "stderr": "", "exit_code": 0},
        {"action": "calendar today --timezone Europe/Berlin", "stdout": "Review at 14:00", "stderr": "", "exit_code": 0},
        {"action": "tasks list --status pending", "stdout": "1. Berlin follow-up", "stderr": "", "exit_code": 0},
    ]
    messages = [
        {"role": "user", "content": "Task: first"},
        {"role": "assistant", "content": "cmd-1"},
        {"role": "user", "content": "obs-1"},
        {"role": "assistant", "content": "cmd-2"},
        {"role": "user", "content": "obs-2"},
        {"role": "assistant", "content": "cmd-3"},
        {"role": "user", "content": "obs-3"},
        {"role": "assistant", "content": "cmd-4"},
        {"role": "user", "content": "obs-4"},
        {"role": "assistant", "content": "cmd-5"},
        {"role": "user", "content": "obs-5"},
    ]
    openai_agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2", history_mode="auto")
    anthropic_agent = train_and_eval.LLMAgent(provider="anthropic", model="claude-haiku-4-5-20251001", history_mode="auto")
    openai_agent._history = list(history)
    anthropic_agent._history = list(history)

    assert openai_agent._request_messages(messages) == anthropic_agent._request_messages(messages)


def test_cli_accepts_fallback_openclaw_network_to_mock_flag(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault(
                "fallback_openclaw_network_to_mock",
                kwargs["fallback_openclaw_network_to_mock"],
            ),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--mode",
            "hybrid",
            "--fallback-openclaw-network-to-mock",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "hybrid"
    assert calls.get("fallback_openclaw_network_to_mock") is True
    assert calls.get("printed") is True


def test_cli_accepts_strict_online_data_flag(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("strict_online_data", kwargs["strict_online_data"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--mode",
            "hybrid",
            "--strict-online-data",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "hybrid"
    assert calls.get("strict_online_data") is True
    assert calls.get("printed") is True


def test_run_episode_records_extended_trajectory_fields():
    result = train_and_eval.run_episode(
        task_id="agent_create_1",
        agent=train_and_eval.ExpertAgent(),
        mode="mock",
        max_steps=3,
        record_trajectory=True,
        online_clean=True,
        verbose=False,
    )

    assert result.trajectory


def test_openai_llm_agent_calls_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "weather get --location 'Paris'"}}]}
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
    )
    agent.reset("Check the weather in Paris.")
    observation = SimpleNamespace(
        step_number=0,
        command_output="",
        error_output="",
        exit_code=0,
    )

    action = agent.act(observation)

    assert action == "weather get --location 'Paris'"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["body"]["model"] == "legacy-openai-chat-model"
    assert captured["body"]["max_tokens"] == 256
    assert "max_completion_tokens" not in captured["body"]
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["messages"][1]["role"] == "user"
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_openai_gpt_models_use_responses_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "weather get --location 'Paris'"}
                            ],
                        }
                    ],
                }
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
    )
    agent.reset("Check the weather in Paris.")

    action = agent.act(SimpleNamespace(step_number=0, command_output="", error_output="", exit_code=0))

    assert action == "weather get --location 'Paris'"
    assert captured["url"] == "https://example.test/v1/responses"
    assert captured["body"]["model"] == "gpt-5.2"
    assert captured["body"]["max_output_tokens"] == 256
    assert captured["body"]["instructions"] == agent._system_prompt
    assert captured["body"]["input"][0]["role"] == "user"


def test_openai_responses_api_deduplicates_adjacent_command_parts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "email search --query budget"},
                                {"type": "output_text", "text": "email search --query budget"},
                            ],
                        }
                    ],
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        train_and_eval.urllib.request,
        "urlopen",
        lambda req, timeout=0: _Response(),
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5.4",
        base_url="https://example.test/v1",
    )
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset(task.instruction, task=task)

    assert agent._openai_response_text() == "email search --query budget"
    assert any("deduped 1 adjacent duplicate" in item for item in agent.debug_reply_attempts())


def test_openai_responses_api_preserves_split_command_chunks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "email search --query "},
                                {"type": "output_text", "text": "budget"},
                            ],
                        }
                    ],
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        train_and_eval.urllib.request,
        "urlopen",
        lambda req, timeout=0: _Response(),
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5.4",
        base_url="https://example.test/v1",
    )
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset(task.instruction, task=task)

    assert agent._openai_response_text() == "email search --query budget"



def test_openai_request_falls_back_to_responses_when_chat_is_unsupported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[str] = []

    class _Body:
        def __init__(self, payload: str) -> None:
            self._payload = payload.encode("utf-8")

        def read(self) -> bytes:
            return self._payload

        def close(self) -> None:
            return None

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "tasks list --status pending"}
                            ],
                        }
                    ],
                }
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        calls.append(req.full_url)
        if req.full_url.endswith("/chat/completions"):
            raise train_and_eval.urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=_Body('{"error":{"message":"The requested operation is unsupported."}}'),
            )
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=0,
    )
    agent.reset("List pending tasks.")

    assert agent._openai_response_text() == "tasks list --status pending"
    assert calls == [
        "https://example.test/v1/chat/completions",
        "https://example.test/v1/responses",
    ]
    assert any("unsupported; retrying with responses API" in item for item in agent.debug_reply_attempts())


def test_openai_response_text_rejects_null_content(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": None, "role": "assistant"}}]}
            ).encode("utf-8")

    monkeypatch.setattr(
        train_and_eval.urllib.request,
        "urlopen",
        lambda req, timeout=0: _Response(),
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
    )
    agent.reset("Check the weather in Paris.")

    with pytest.raises(RuntimeError, match="empty message content"):
        agent._openai_response_text()


def test_openai_request_backs_off_before_retry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = {"count": 0}
    sleeps: list[float] = []

    class _Body:
        def __init__(self, payload: str) -> None:
            self._payload = payload.encode("utf-8")

        def read(self) -> bytes:
            return self._payload

        def close(self) -> None:
            return None

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "weather get --location 'Paris'", "role": "assistant"}}]}
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        calls["count"] += 1
        if calls["count"] == 1:
            raise train_and_eval.urllib.error.HTTPError(
                req.full_url,
                429,
                "Too Many Requests",
                hdrs=None,
                fp=_Body('{"error":{"message":"queue is full"}}'),
            )
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(train_and_eval.time, "sleep", lambda seconds: sleeps.append(seconds))

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=1,
        retry_backoff_s=2.0,
    )
    agent.reset("Check the weather in Paris.")

    assert agent._openai_response_text() == "weather get --location 'Paris'"
    assert sleeps == [2.0]
    assert any("api-backoff" in attempt for attempt in agent.debug_reply_attempts())


def test_openai_response_text_extracts_command_from_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "reasoning_content": (
                                    "I should check the inbox first.\n"
                                    "Next command: email search --query budget"
                                ),
                                "role": "assistant",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        train_and_eval.urllib.request,
        "urlopen",
        lambda req, timeout=0: _Response(),
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
    )
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("Review budget-related email.", task=task)

    assert agent._openai_response_text() == "email search --query budget"


def test_openai_response_text_reprompts_when_reasoning_has_no_command(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured_bodies: list[dict[str, object]] = []

    class _Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "I should inspect the calendar context before deciding.",
                            "role": "assistant",
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "calendar today --timezone America/Chicago",
                            "role": "assistant",
                        }
                    }
                ]
            },
        ]
    )

    def _urlopen(req, timeout=0):
        captured_bodies.append(json.loads(req.data.decode("utf-8")))
        return _Response(next(responses))

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="Kimi-K2.5",
        base_url="https://example.test/v1",
    )
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("Set up a daily ops check for Austin.", task=task)

    assert agent._openai_response_text() == "calendar today --timezone America/Chicago"
    assert len(captured_bodies) == 2
    assert captured_bodies[0]["max_tokens"] == 256
    assert captured_bodies[1]["max_tokens"] == 64
    assert "previous reply included reasoning but no final command" in captured_bodies[1]["messages"][-1]["content"]


def test_openai_request_uses_max_tokens_for_non_gpt5_models(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "weather get --location 'Paris'"}}]}
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="Kimi-K2.5",
        base_url="https://example.test/v1",
    )
    agent.reset("Check the weather in Paris.")
    observation = SimpleNamespace(
        step_number=0,
        command_output="",
        error_output="",
        exit_code=0,
    )

    assert agent.act(observation) == "weather get --location 'Paris'"
    assert captured["body"]["max_tokens"] == 256
    assert "max_completion_tokens" not in captured["body"]


def test_openai_response_text_rejects_content_without_text_parts(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [{"type": "tool_use", "id": "abc"}],
                                "role": "assistant",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        train_and_eval.urllib.request,
        "urlopen",
        lambda req, timeout=0: _Response(),
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
    )
    agent.reset("Check the weather in Paris.")

    with pytest.raises(RuntimeError, match="no text parts"):
        agent._openai_response_text()


def test_command_validator_rejects_unknown_cli_shape():
    task = SimpleNamespace(domains=["email", "tasks", "calendar"])

    ok, reason = train_and_eval._validate_command("email read --path email_seed_3", task)

    assert ok is False
    assert "invalid command" in reason
    assert "email read" in reason


def test_command_validator_rejects_unclosed_quotes():
    task = SimpleNamespace(domains=["tasks"])

    ok, reason = train_and_eval._validate_command('tasks add --title "', task)

    assert ok is False
    assert "shell parse error" in reason


def test_command_validator_rejects_template_placeholder_values():
    data_dir = Path(train_and_eval.DEFAULT_TASK_DATA_DIR)
    task = train_and_eval.load_task("hard_decision_workflow_1", data_dir=data_dir)

    ok, reason = train_and_eval._validate_command(
        "calendar today --timezone TIMEZONE",
        task,
    )

    assert ok is False
    assert "template placeholder" in reason


def test_command_validator_allows_brackets_inside_quoted_free_text():
    task = SimpleNamespace(domains=["email"])

    ok, reason = train_and_eval._validate_command(
        'email send --to finance@example.com --subject "Toronto Daily Ops Setup Confirmed" '
        '--body "Confirmed: cron job [cron_existing_ops] is already scheduled."',
        task,
    )

    assert ok is True
    assert reason == ""


def test_command_validator_rejects_incomplete_openclaw_side_effect_commands():
    task = SimpleNamespace(domains=["cron_webhook", "messaging", "channel_mgmt", "setup_config"])

    ok, reason = train_and_eval._validate_command("openclaw cron add", task)
    assert ok is False
    assert "--name" in reason or "missing required" in reason.lower()

    ok, reason = train_and_eval._validate_command("openclaw message send --target @alice --message hi", task)
    assert ok is False
    assert "--channel" in reason

    ok, reason = train_and_eval._validate_command("openclaw channels login", task)
    assert ok is False
    assert "--channel" in reason


def test_llm_agent_retries_after_invalid_command(monkeypatch: pytest.MonkeyPatch):
    responses = iter(["email read --path email_seed_3", "email search --query budget"])
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    agent.reset(
        "Review budget-related email.",
        task=SimpleNamespace(domains=["email", "tasks", "calendar"]),
    )
    monkeypatch.setattr(agent, "_openai_response_text", lambda: next(responses))

    action = agent.act(
        SimpleNamespace(
            step_number=0,
            command_output="",
            error_output="",
            exit_code=0,
        )
    )

    assert action == "email search --query budget"
    assert any(
        msg["role"] == "user" and "invalid command" in msg["content"]
        for msg in agent._messages
    )
    attempts = agent.debug_reply_attempts()
    assert any("parsed: raw='email read --path email_seed_3'" in attempt for attempt in attempts)
    assert any("validator: invalid command" in attempt for attempt in attempts)
    assert any("parsed: raw='email search --query budget'" in attempt for attempt in attempts)


def test_llm_agent_exhausted_invalid_replies_stops_instead_of_tasks_list(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    agent.reset(
        "Check the forecast in New York and reschedule the event if needed.",
        task=train_and_eval.load_task(
            "calendar_weather_check_19",
            data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
        ),
    )
    monkeypatch.setattr(agent, "_openai_response_text", lambda: "tasks list")

    action = agent.act(
        SimpleNamespace(
            step_number=0,
            command_output="",
            error_output="",
            exit_code=0,
        )
    )

    assert action == "DONE"
    assert agent.should_stop() is True


def test_llm_agent_warns_on_repeated_no_progress(monkeypatch: pytest.MonkeyPatch):
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    agent.reset(
        "Review budget-related email.",
        task=SimpleNamespace(domains=["email", "tasks", "calendar"]),
    )
    agent._pending_action = "tasks list"
    agent._history = [
        {"action": "tasks list", "stdout": "Tasks:\n...", "stderr": "", "exit_code": 0},
        {"action": "tasks list", "stdout": "Tasks:\n...", "stderr": "", "exit_code": 0},
    ]
    monkeypatch.setattr(agent, "_openai_response_text", lambda: "email search --query budget")

    action = agent.act(
        SimpleNamespace(
            step_number=3,
            command_output="Tasks:\n...",
            error_output="",
            exit_code=0,
        )
    )

    assert action == "email search --query budget"
    assert any(
        msg["role"] == "user" and "Pick a different command from the catalog." in msg["content"]
        for msg in agent._messages
    )


def test_normalize_model_command_strips_code_fence_and_prompt_marker():
    raw = "```bash\n$ tasks add --title 'Write report' --due 2026-03-05\n```"
    assert train_and_eval._normalize_model_command(raw) == "tasks add --title 'Write report' --due 2026-03-05"


def test_normalize_model_command_strips_trailing_explanation():
    raw = "calendar today --timezone TIMEZONE - for reviewing calendar context"
    assert train_and_eval._normalize_model_command(raw) == "calendar today --timezone TIMEZONE"


def test_normalize_model_command_strips_catalog_optional_suffix():
    raw = "openclaw status [--json]"
    assert train_and_eval._normalize_model_command(raw) == "openclaw status"


def test_normalize_model_command_preserves_quoted_dash_content():
    raw = (
        'email send --to alice@example.com --subject "Budget Report Follow-up - Berlin" '
        '--body "Follow-up on the budget report for Berlin."'
    )
    assert train_and_eval._normalize_model_command(raw) == raw


def test_normalize_model_command_strips_provider_json_suffix_noise():
    raw = "tasks list】【。json"
    assert train_and_eval._normalize_model_command(raw) == "tasks list"


def test_normalize_model_command_strips_provider_analysis_suffix_noise():
    raw = "tasks list】【。analysis to=final code  omitted"
    assert train_and_eval._normalize_model_command(raw) == "tasks list"


def test_normalize_model_command_preserves_valid_json_path_argument():
    raw = 'file create --path notes.json --content "hello"'
    assert train_and_eval._normalize_model_command(raw) == raw


def test_command_catalog_is_scoped_by_relevant_domains_with_gt_backfill():
    data_dir = Path(train_and_eval.DEFAULT_TASK_DATA_DIR)
    calendar_task = train_and_eval.load_task("calendar_weather_check_19", data_dir=data_dir)
    hard_task = train_and_eval.load_task("hard_decision_workflow_1", data_dir=data_dir)
    import openclaw_env.tasks.generators.branch_sensitive_workflows  # noqa: F401
    from openclaw_env.tasks.generation_options import set_generation_options
    from openclaw_env.tasks.registry import generate_all_tasks

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
        branch_sensitive_variants_per_scenario=12,
    )
    branch_task = next(
        task
        for task in generate_all_tasks(generator_ids=["branch_sensitive_workflow"])
        if task.task_id == "branch_sensitive_workflow_27"
    )

    calendar_catalog = train_and_eval._command_catalog_text(calendar_task)
    hard_catalog = train_and_eval._command_catalog_text(hard_task)
    branch_catalog = train_and_eval._command_catalog_text(branch_task)

    assert "weather get --location LOCATION" in calendar_catalog
    assert "calendar add-event --title TITLE --start DATETIME" in calendar_catalog
    assert "tasks list" not in calendar_catalog
    assert "openclaw cron list" not in calendar_catalog

    assert "email read --id ID" in hard_catalog
    assert "email send --to EMAIL --subject SUBJECT --body BODY" in hard_catalog
    assert "tasks add --title TITLE" in hard_catalog
    assert "calendar today --timezone TIMEZONE" in hard_catalog
    assert "openclaw config get PATH" in hard_catalog
    assert "openclaw channels list [--json]" in hard_catalog
    assert "openclaw channels login --channel NAME" in hard_catalog
    assert "openclaw message send --channel NAME --target TARGET --message TEXT" in hard_catalog

    assert "openclaw config get PATH" in branch_catalog
    assert "openclaw models set MODEL" in branch_catalog


def test_run_evaluation_uses_task_data_dir_for_task_loading(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        train_and_eval,
        "load_task_ids",
        lambda split, data_dir=None, difficulty=None, domain=None: (
            calls.setdefault("ids_data_dir", data_dir),
            ["agent_create_1"],
        )[1],
    )
    monkeypatch.setattr(
        train_and_eval,
        "load_task",
        lambda task_id, data_dir=None: (
            calls.setdefault("task_data_dir", data_dir),
            SimpleNamespace(domains=["agent_mgmt"], difficulty=1),
        )[1],
    )
    monkeypatch.setattr(
        train_and_eval,
        "run_episode",
        lambda *_, **kwargs: train_and_eval.EpisodeResult(
            task_id="agent_create_1",
            agent_name="expert",
            success=True,
            score=1.0,
            steps=1,
            duration_s=0.01,
        ),
    )

    summary = train_and_eval.run_evaluation(
        agent=SimpleNamespace(name="expert"),
        split="dev",
        task_data_dir=tmp_path,
        limit=1,
    )

    assert summary.total == 1
    assert calls.get("ids_data_dir") == tmp_path
    assert calls.get("task_data_dir") == tmp_path


def test_run_evaluation_filters_task_ids_by_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        train_and_eval,
        "load_task_ids",
        lambda split, data_dir=None, difficulty=None, domain=None: [
            "hard_decision_workflow_1",
            "calendar_add_event_1",
            "hard_decision_workflow_2",
        ],
    )
    monkeypatch.setattr(
        train_and_eval,
        "load_task",
        lambda task_id, data_dir=None: SimpleNamespace(domains=["calendar"], difficulty=3),
    )
    monkeypatch.setattr(
        train_and_eval,
        "run_episode",
        lambda task_id, *_, **__: train_and_eval.EpisodeResult(
            task_id=task_id,
            agent_name="expert",
            success=True,
            score=1.0,
            steps=1,
            duration_s=0.01,
        ),
    )

    summary = train_and_eval.run_evaluation(
        agent=SimpleNamespace(name="expert"),
        split="dev",
        task_prefix="hard_decision_workflow_",
    )

    assert summary.total == 2
    assert [r.task_id for r in summary.results] == [
        "hard_decision_workflow_1",
        "hard_decision_workflow_2",
    ]


def test_run_evaluation_tracks_provider_failures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        train_and_eval,
        "load_task_ids",
        lambda split, data_dir=None, difficulty=None, domain=None: [
            "hard_decision_workflow_1",
            "hard_decision_workflow_2",
        ],
    )
    monkeypatch.setattr(
        train_and_eval,
        "load_task",
        lambda task_id, data_dir=None: SimpleNamespace(domains=["calendar"], difficulty=3),
    )

    def _run_episode(task_id, *_, **__):
        if task_id.endswith("_1"):
            return train_and_eval.EpisodeResult(
                task_id=task_id,
                agent_name="llm",
                success=False,
                score=0.0,
                steps=0,
                duration_s=0.01,
                error="HTTP 429: queue is full",
                error_type="provider_failure",
            )
        return train_and_eval.EpisodeResult(
            task_id=task_id,
            agent_name="llm",
            success=True,
            score=1.0,
            steps=1,
            duration_s=0.01,
        )

    monkeypatch.setattr(train_and_eval, "run_episode", _run_episode)

    summary = train_and_eval.run_evaluation(
        agent=SimpleNamespace(name="llm"),
        split="dev",
    )

    assert summary.total == 2
    assert summary.passed == 1
    assert summary.provider_failures == 1
    assert summary.completed_tasks == 1
    assert summary.provider_adjusted_tgc == 1.0


def test_run_evaluation_tracks_provider_impacted_tasks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        train_and_eval,
        "load_task_ids",
        lambda split, data_dir=None, difficulty=None, domain=None: [
            "hard_decision_workflow_1",
            "hard_decision_workflow_2",
        ],
    )
    monkeypatch.setattr(
        train_and_eval,
        "load_task",
        lambda task_id, data_dir=None: SimpleNamespace(domains=["calendar"], difficulty=3),
    )

    def _run_episode(task_id, *_, **__):
        return train_and_eval.EpisodeResult(
            task_id=task_id,
            agent_name="llm",
            success=task_id.endswith("_2"),
            score=1.0 if task_id.endswith("_2") else 0.5,
            steps=1,
            duration_s=0.01,
            provider_impacted=task_id.endswith("_1"),
        )

    monkeypatch.setattr(train_and_eval, "run_episode", _run_episode)

    summary = train_and_eval.run_evaluation(
        agent=SimpleNamespace(name="llm"),
        split="dev",
    )

    assert summary.total == 2
    assert summary.provider_failures == 0
    assert summary.provider_impacted_tasks == 1
    assert summary.clean_completed_tasks == 1


def test_run_episode_stops_early_on_success_by_default():
    class _StickyAgent(train_and_eval.BaseAgent):
        name = "sticky"

        def __init__(self):
            self._step = 0

        def reset(self, task_instruction: str, task=None) -> None:
            self._step = 0

        def act(self, observation):
            self._step += 1
            if self._step == 1:
                return "openclaw agents add researcher --model anthropic/claude-opus-4-6"
            return "openclaw agents list"

    result = train_and_eval.run_episode(
        task_id="agent_create_1",
        agent=_StickyAgent(),
        mode="mock",
        max_steps=10,
        record_trajectory=True,
        verbose=False,
    )

    assert result.success is True
    assert result.steps == 1
    assert len(result.trajectory) == 1


def test_run_episode_does_not_execute_done_command():
    calls = {"step": 0}

    class _DoneAgent(train_and_eval.BaseAgent):
        name = "done-agent"

        def reset(self, task_instruction: str, task=None) -> None:
            return None

        def act(self, observation):
            return "DONE"

    class _FakeEnv:
        def __init__(self):
            self.task = SimpleNamespace(domains=["tasks"], ground_truth=None)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def reset(self):
            return SimpleNamespace(
                task_instruction="Add a task.",
                command_output="",
                error_output="",
                exit_code=0,
                step_number=0,
            )

        def step(self, action):
            calls["step"] += 1
            raise AssertionError(f"env.step should not be called for {action!r}")

        def evaluate(self):
            return train_and_eval.EvaluationResult(success=False, score=0.0, details=[])

    original_make_env = train_and_eval.make_env
    try:
        train_and_eval.make_env = lambda *_, **__: _FakeEnv()
        result = train_and_eval.run_episode(
            task_id="tasks_add_1",
            agent=_DoneAgent(),
            mode="mock",
            max_steps=3,
            verbose=False,
        )
    finally:
        train_and_eval.make_env = original_make_env

    assert calls["step"] == 0
    assert result.steps == 0


def test_run_episode_stops_after_repeated_stagnant_transitions():
    class _LoopAgent(train_and_eval.BaseAgent):
        name = "loop-agent"

        def reset(self, task_instruction: str, task=None) -> None:
            return None

        def act(self, observation):
            return "tasks list"

    class _FakeEnv:
        def __init__(self):
            self.task = SimpleNamespace(domains=["tasks"], ground_truth=None)
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def reset(self):
            return SimpleNamespace(
                task_instruction="List tasks.",
                command_output="",
                error_output="",
                exit_code=0,
                step_number=0,
            )

        def step(self, action):
            self.calls += 1
            return (
                SimpleNamespace(
                    task_instruction="List tasks.",
                    command_output="Tasks:\n[item_1] Demo",
                    error_output="",
                    exit_code=0,
                    step_number=self.calls,
                ),
                0.0,
                False,
                {},
            )

        def evaluate(self):
            return train_and_eval.EvaluationResult(success=False, score=0.0, details=[])

    original_make_env = train_and_eval.make_env
    fake_env = _FakeEnv()
    try:
        train_and_eval.make_env = lambda *_, **__: fake_env
        result = train_and_eval.run_episode(
            task_id="tasks_add_1",
            agent=_LoopAgent(),
            mode="mock",
            max_steps=10,
            max_stagnant_steps=3,
            record_trajectory=True,
            verbose=False,
        )
    finally:
        train_and_eval.make_env = original_make_env

    assert fake_env.calls == 3
    assert result.steps == 3
    assert len(result.trajectory) == 3


def test_run_episode_verbose_failure_prints_reason_and_failed_checks(capsys):
    class _DoneAgent(train_and_eval.BaseAgent):
        name = "done-agent"

        def reset(self, task_instruction: str, task=None) -> None:
            return None

        def act(self, observation):
            return "DONE"

    result = train_and_eval.run_episode(
        task_id="tasks_add_1",
        agent=_DoneAgent(),
        mode="mock",
        max_steps=3,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert result.success is False
    assert "reason:" in out
    assert "failed checks:" in out


def test_run_episode_marks_provider_impacted_from_reply_debug():
    class _NoisyAgent(train_and_eval.BaseAgent):
        name = "noisy-agent"

        def reset(self, task_instruction: str, task=None) -> None:
            self._calls = 0

        def act(self, observation):
            self._calls += 1
            if self._calls == 1:
                return "tasks list"
            return "DONE"

        def debug_last_reply(self):
            return None

        def debug_reply_attempts(self):
            if self._calls == 1:
                return [
                    "api-fallback: content filter triggered; retrying with compact prompt",
                    "api: content='tasks list'",
                ]
            return ["api: content='DONE'"]

    class _FakeEnv:
        def __init__(self):
            self.task = SimpleNamespace(domains=["tasks"], ground_truth=None)
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def reset(self):
            return SimpleNamespace(
                task_instruction="List tasks.",
                command_output="",
                error_output="",
                exit_code=0,
                step_number=0,
            )

        def step(self, action):
            self.calls += 1
            return (
                SimpleNamespace(
                    task_instruction="List tasks.",
                    command_output="Tasks:\n[item_1] Demo",
                    error_output="",
                    exit_code=0,
                    step_number=self.calls,
                ),
                0.0,
                False,
                {},
            )

        def evaluate(self):
            return train_and_eval.EvaluationResult(success=False, score=0.0, details=[])

    original_make_env = train_and_eval.make_env
    try:
        train_and_eval.make_env = lambda *_, **__: _FakeEnv()
        result = train_and_eval.run_episode(
            task_id="tasks_add_1",
            agent=_NoisyAgent(),
            mode="mock",
            max_steps=2,
            record_trajectory=True,
            verbose=False,
        )
    finally:
        train_and_eval.make_env = original_make_env

    assert result.provider_impacted is True
    assert result.trajectory
    assert any(
        "api-fallback: content filter triggered" in item
        for item in result.trajectory[0]["model_debug"]
    )


def test_run_episode_verbose_prints_last_model_reply_on_done(capsys):
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("Review budget-related email.", task=task)
    agent._openai_response_text = lambda: "DONE"

    result = train_and_eval.run_episode(
        task_id="hard_decision_workflow_1",
        agent=agent,
        mode="mock",
        max_steps=3,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert result.success is False
    assert "model: parsed: raw='DONE' normalized='DONE'" in out


def test_run_episode_marks_date_anchor_mismatch_for_host_date_leak(monkeypatch: pytest.MonkeyPatch):
    class _DateLeakAgent(train_and_eval.BaseAgent):
        name = "date-leak"

        def reset(self, task_instruction: str, task=None) -> None:
            self._calls = 0

        def act(self, observation):
            self._calls += 1
            if self._calls == 1:
                return "calendar list --from 2026-04-24 --to 2026-04-25"
            return "DONE"

    class _FakeEnv:
        def __init__(self):
            self.task = train_and_eval.load_task(
                "hard_decision_workflow_1",
                data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
            )
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def reset(self):
            return SimpleNamespace(
                task_instruction=self.task.instruction,
                command_output="",
                error_output="",
                exit_code=0,
                step_number=0,
            )

        def step(self, action):
            self.calls += 1
            return (
                SimpleNamespace(
                    task_instruction=self.task.instruction,
                    command_output="No events found.",
                    error_output="",
                    exit_code=0,
                    step_number=self.calls,
                ),
                0.0,
                False,
                {},
            )

        def evaluate(self):
            return train_and_eval.EvaluationResult(success=False, score=0.5, details=[])

    class _FixedDate(train_and_eval.date):
        @classmethod
        def today(cls):
            return cls(2026, 4, 25)

    original_make_env = train_and_eval.make_env
    try:
        monkeypatch.setattr(train_and_eval, "date", _FixedDate)
        train_and_eval.make_env = lambda *_, **__: _FakeEnv()
        result = train_and_eval.run_episode(
            task_id="hard_decision_workflow_1",
            agent=_DateLeakAgent(),
            mode="mock",
            max_steps=2,
            verbose=False,
        )
    finally:
        train_and_eval.make_env = original_make_env

    assert result.date_anchor_mismatch is True


def test_llm_agent_rejects_done_after_only_read_only_steps(monkeypatch: pytest.MonkeyPatch):
    responses = iter(["DONE", "tasks add --title 'Berlin budget follow-up' --priority high"])
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("Review budget-related email, turn it into a follow-up for Berlin.", task=task)
    agent._history = [
        {"action": "email search --query budget", "stdout": "Emails matching...", "stderr": "", "exit_code": 0},
        {"action": "email read --id email_seed_3", "stdout": "Email [email_seed_3]", "stderr": "", "exit_code": 0},
    ]
    monkeypatch.setattr(agent, "_openai_response_text", lambda: next(responses))

    action = agent.act(
        SimpleNamespace(
            step_number=2,
            command_output="Email [email_seed_3]",
            error_output="",
            exit_code=0,
        )
    )

    assert action == "tasks add --title 'Berlin budget follow-up' --priority high"
    attempts = agent.debug_reply_attempts()
    assert any(
        "validator: the task likely still requires a state-changing command before DONE" in attempt
        for attempt in attempts
    )


def test_llm_agent_treats_empty_openai_content_as_invalid_reply(monkeypatch: pytest.MonkeyPatch):
    responses = iter(
        [
            RuntimeError("OpenAI-compatible response contained empty message content: {...}"),
            "email send --to alice@example.com --subject 'Budget update' --body 'Following up on Berlin.'",
        ]
    )
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("Review budget-related email and send Alice a short update.", task=task)

    def _next_response():
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(agent, "_openai_response_text", _next_response)

    action = agent.act(
        SimpleNamespace(
            step_number=2,
            command_output="Email [email_seed_3]",
            error_output="",
            exit_code=0,
        )
    )

    assert action == "email send --to alice@example.com --subject 'Budget update' --body 'Following up on Berlin.'"
    attempts = agent.debug_reply_attempts()
    assert any("api-error: OpenAI-compatible response contained empty message content" in attempt for attempt in attempts)


def test_llm_agent_builds_memory_aware_openai_request_messages():
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    messages = [
        {"role": "user", "content": "Task: first"},
        {"role": "assistant", "content": "cmd-1"},
        {"role": "user", "content": "obs-1"},
        {"role": "assistant", "content": "cmd-2"},
        {"role": "user", "content": "obs-2"},
        {"role": "assistant", "content": "cmd-3"},
        {"role": "user", "content": "obs-3"},
        {"role": "assistant", "content": "cmd-4"},
        {"role": "user", "content": "obs-4"},
    ]
    agent._history = [
        {"action": "weather forecast --location Berlin --days 1", "stdout": "Rain tomorrow", "stderr": "", "exit_code": 0},
        {"action": "calendar today --timezone Europe/Berlin", "stdout": "Review at 14:00", "stderr": "", "exit_code": 0},
        {"action": "tasks list --status pending", "stdout": "1. Berlin follow-up", "stderr": "", "exit_code": 0},
        {"action": "openclaw cron list", "stdout": "existing-daily-check", "stderr": "", "exit_code": 0},
        {"action": "openclaw config get agent.model", "stdout": "agent.model = openai/gpt-4o", "stderr": "", "exit_code": 0},
        {"action": "weather forecast --location Berlin --days 1", "stdout": "Clear later", "stderr": "", "exit_code": 0},
        {"action": "tasks add --title 'Berlin follow-up'", "stdout": "Created task", "stderr": "", "exit_code": 0},
        {"action": "calendar add-event --title 'Berlin sync' --start 2026-03-27T14:00", "stdout": "", "stderr": "slot conflict", "exit_code": 1},
    ]

    trimmed = agent._trim_request_messages(messages)

    assert trimmed[0] == messages[0]
    assert trimmed[1]["role"] == "user"
    assert trimmed[1]["content"].startswith("Known context:\n")
    assert "weather forecast --location Berlin --days 1 => stdout=Clear later" in trimmed[1]["content"]
    assert "calendar today --timezone Europe/Berlin => stdout=Review at 14:00" in trimmed[1]["content"]
    assert "tasks list --status pending => stdout=1. Berlin follow-up" in trimmed[1]["content"]
    assert "openclaw cron list => stdout=existing-daily-check" in trimmed[1]["content"]
    assert "openclaw config get agent.model => stdout=agent.model = openai/gpt-4o" in trimmed[1]["content"]
    assert "Rain tomorrow" not in trimmed[1]["content"]
    assert "last state-changing action: tasks add --title 'Berlin follow-up'" in trimmed[1]["content"]
    assert "recent state-changing action: calendar add-event --title 'Berlin sync' --start 2026-03-27T14:00" in trimmed[1]["content"]
    assert "last failed action: calendar add-event --title 'Berlin sync' --start 2026-03-27T14:00" in trimmed[1]["content"]
    assert trimmed[2:] == messages[-8:]


def test_llm_agent_skips_memory_summary_for_short_history():
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    messages = [
        {"role": "user", "content": "Task: first"},
        {"role": "assistant", "content": "cmd-1"},
        {"role": "user", "content": "obs-1"},
    ]

    trimmed = agent._trim_request_messages(messages)

    assert trimmed == messages


def test_llm_agent_uses_litellm_proxy_defaults_for_claude_openai_models(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BEDROCK_LITELLM_BASE_URL", raising=False)
    monkeypatch.setenv("LITELLM_PROXY_KEY", "proxy-key")
    monkeypatch.setenv("LITELLM_PROXY_BASE_URL", "http://localhost:4000/v1")

    agent = train_and_eval.LLMAgent(provider="openai", model="claude-sonnet-4.6")

    assert agent._api_key_name() == "LITELLM_PROXY_KEY"
    assert agent._api_key() == "proxy-key"
    assert agent._openai_base_url() == "http://localhost:4000/v1"


def test_llm_agent_hard_prompt_uses_raw_task_instruction_without_benchmark_notes():
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )

    agent.reset(task.instruction, task=task)

    assert "Benchmark time anchor" not in agent._system_prompt
    assert "board' means the task board managed with `tasks *` commands" not in agent._system_prompt
    assert "calendar today --timezone Europe/Berlin" not in agent._system_prompt
    assert agent._messages[0]["content"].startswith(f"Task: {task.instruction}")
    assert "Complete the task using the available CLI commands." in agent._messages[0]["content"]


def test_llm_agent_explicit_openai_base_url_and_key_env_override_litellm_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LITELLM_PROXY_KEY", "proxy-key")
    monkeypatch.setenv("CUSTOM_KEY_ENV", "custom-key")

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="claude-sonnet-4.6",
        base_url="https://example.test/v1",
        api_key_env="CUSTOM_KEY_ENV",
    )

    assert agent._api_key_name() == "CUSTOM_KEY_ENV"
    assert agent._api_key() == "custom-key"
    assert agent._openai_base_url() == "https://example.test/v1"


def test_openai_content_filter_retries_with_compact_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured_bodies: list[dict[str, object]] = []

    class _Body:
        def __init__(self, text: str) -> None:
            self._text = text

        def read(self) -> bytes:
            return self._text.encode("utf-8")

        def close(self) -> None:
            return None

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "tasks add --title 'Berlin follow-up'", "role": "assistant"}}]}
            ).encode("utf-8")

    calls = {"count": 0}

    def _urlopen(req, timeout=0):
        captured_bodies.append(json.loads(req.data.decode("utf-8")))
        calls["count"] += 1
        if calls["count"] == 1:
            raise train_and_eval.urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=_Body(
                    '{"error":{"message":"The response was filtered due to the prompt triggering policy.","param":"prompt","code":"content_filter","innererror":{"code":"ResponsibleAIPolicyViolation","content_filter_result":{"jailbreak":{"filtered":true,"detected":true}}}}}'
                ),
            )
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
    )
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset(task.instruction, task=task)
    agent._history = [
        {"action": "openclaw config get agent.model", "stdout": "agent.model = openai/gpt-4o", "stderr": "", "exit_code": 0},
        {"action": "calendar list", "stdout": "Events: 1", "stderr": "", "exit_code": 0},
        {"action": "tasks list --status pending", "stdout": "Tasks: 1", "stderr": "", "exit_code": 0},
    ]

    text = agent._openai_response_text()

    assert text == "tasks add --title 'Berlin follow-up'"
    assert len(captured_bodies) == 2
    assert "content filter triggered" in " ".join(agent.debug_reply_attempts())
    first_payload = captured_bodies[0]
    second_payload = captured_bodies[1]
    assert first_payload["instructions"] == agent._system_prompt
    assert len(first_payload["input"]) >= 1
    assert second_payload["instructions"] == agent._compact_system_prompt()
    compact_user = second_payload["input"][0]["content"]
    assert "Known context:" in compact_user
    assert "calendar list" in compact_user
    assert "tasks list --status pending" in compact_user
    assert "openclaw config get agent.model" in compact_user
    assert "last state-changing action" not in compact_user


def test_eval_summary_prints_hard_scenario_and_ability_sections(capsys: pytest.CaptureFixture[str]) -> None:
    summary = train_and_eval.EvalSummary(
        agent_name="llm",
        split="dev",
        total=4,
        passed=2,
        total_score=2.5,
        exec_mode="multi",
        llm_provider="openai",
        llm_model="claude-sonnet-4.6",
        llm_history_mode="full",
        by_hard_scenario={
            "existing_state_followthrough": train_and_eval.DomainMetrics(passed=1, total=2, total_score=1.1),
        },
        by_hard_ability={
            "gap_completion": train_and_eval.DomainMetrics(passed=1, total=2, total_score=1.1),
        },
        by_hard_ability_tag={
            "duplicate_avoidance": train_and_eval.DomainMetrics(passed=1, total=3, total_score=2.0),
        },
        results=[
            train_and_eval.EpisodeResult(
                task_id="hard_decision_workflow_1",
                agent_name="llm",
                success=False,
                score=0.8,
                steps=8,
                duration_s=0.1,
                stop_reason="agent returned DONE before executing another command",
                date_anchor_mismatch=True,
            ),
            train_and_eval.EpisodeResult(
                task_id="hard_decision_workflow_2",
                agent_name="llm",
                success=False,
                score=0.2,
                steps=30,
                duration_s=0.1,
                stop_reason="reached max_steps=30",
            ),
        ],
    )

    summary.print_report()
    out = capsys.readouterr().out
    assert "Mode : multi" in out
    assert "claude-sonnet-4.6" in out
    assert "History : full" in out
    assert "pass@budget=50.0%" in out
    assert "near_miss_rate=25.0%" in out
    assert "done_early_tasks=1" in out
    assert "step_capped_tasks=1" in out
    assert "date_anchor_mismatch_tasks=1" in out
    assert "Compare avg_score first and raw_accuracy second" in out
    assert "By hard scenario:" in out
    assert "existing_state_followthrough" in out
    assert "By hard primary ability:" in out
    assert "gap_completion" in out
    assert "By hard ability tag (overlapping):" in out
    assert "duplicate_avoidance" in out


def test_cli_save_report_includes_llm_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _DummySummary:
        agent_name = "llm"
        split = "test"
        exec_mode = "multi"
        llm_provider = "openai"
        llm_model = "claude-sonnet-4.6"
        llm_history_mode = "summary"
        tgc = 0.5
        pass_at_budget = 0.5
        avg_score = 0.75
        near_miss_rate = 0.25
        near_miss_tasks = 1
        done_early_tasks = 1
        step_capped_tasks = 0
        date_anchor_mismatch_tasks = 1
        score_rank_note = "Compare avg_score first and raw_accuracy second."
        passed = 1
        total = 2
        completed_tasks = 2
        clean_completed_tasks = 2
        provider_failures = 0
        provider_impacted_tasks = 0
        provider_adjusted_tgc = 0.5
        by_domain: dict[str, object] = {}
        by_difficulty: dict[str, object] = {}
        by_hard_scenario: dict[str, object] = {}
        by_hard_ability: dict[str, object] = {}
        by_hard_ability_tag: dict[str, object] = {}
        results: list[object] = []

        def print_report(self) -> None:
            return None

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary())

    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "llm",
            "--split",
            "test",
            "--save-report",
            str(report_path),
        ],
    )

    train_and_eval.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "multi"
    assert report["max_steps_budget"] == 15
    assert report["llm_provider"] == "openai"
    assert report["llm_model"] == "claude-sonnet-4.6"
    assert report["llm_history_mode"] == "summary"
    assert report["pass_at_budget"] == 0.5
    assert report["near_miss_rate"] == 0.25
    assert report["done_early_tasks"] == 1
    assert report["date_anchor_mismatch_tasks"] == 1
    assert "avg_score first" in report["score_rank_note"]
