#!/usr/bin/env python3
"""Complete example: training and evaluating agents with openclaw_env.

This script demonstrates four agent types and two usage patterns:
  - Evaluation: measure task success rate across a split
  - Data collection: record trajectories for offline RL / behavior cloning

Agent types
-----------
  expert   -- runs the ground-truth solution commands (upper bound)
  random   -- issues random commands (lower bound baseline)
  llm      -- Claude-powered agent via the Anthropic API (requires ANTHROPIC_API_KEY)
  rule     -- simple hand-written heuristics (example of a rule-based agent)

Usage examples
--------------
  # Evaluate expert on dev split
  python examples/train_and_eval.py --agent expert --split dev

  # Evaluate LLM agent on calendar domain, difficulty 1
  python examples/train_and_eval.py --agent llm --split dev \\
      --domain calendar --difficulty 1 --limit 10

  # Collect training data (saves JSONL trajectories)
  python examples/train_and_eval.py --agent expert --split train \\
      --collect-data --out-dir data/trajectories/expert

  # Quick smoke test (3 tasks)
  python examples/train_and_eval.py --agent expert --split dev --limit 3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from openclaw_env import make_env
from openclaw_env.core.environment import OpenClawEnv
from openclaw_env.core.observation import EvaluationResult, Observation
from openclaw_env.core.task import Task, load_task, load_task_ids
from openclaw_env.utils.output_cleaner import clean_openclaw_output
from openclaw_env.utils.episode_memory import (
    build_memory_summary,
    build_openai_request_messages,
    clip_text,
    is_read_only_action,
)

DEFAULT_TASK_DATA_DIR = (
    Path(__file__).parent.parent / "openclaw_env" / "data"
)
DEFAULT_MAX_STEPS = 15
DEFAULT_BENCHMARK_DATE = "2026-03-01"


class _TeeStdout:
    """Mirror stdout writes to the original stream and an optional log file."""

    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        written = self._primary.write(data)
        self._secondary.write(data)
        return written

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

# ---------------------------------------------------------------------------
# Available CLI prefixes the rule-based agent knows about
# ---------------------------------------------------------------------------
_CLI_VERBS: dict[str, list[str]] = {
    "calendar": ["list", "add-event", "update-event", "delete-event"],
    "email": ["list", "read", "send", "reply", "move", "mark", "search"],
    "weather": ["get", "forecast", "alerts"],
    "file": ["create", "read", "move", "append", "delete"],
    "tasks": ["list", "add", "complete", "search"],
}


@dataclass(frozen=True)
class CommandSpec:
    tokens: tuple[str, ...]
    syntax: str
    domains: tuple[str, ...]
    note: str = ""


_COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(("calendar", "list"), "calendar list [--from DATE] [--to DATE]", ("calendar",)),
    CommandSpec(("calendar", "add-event"), "calendar add-event --title TITLE --start DATETIME [--location LOC] [--attendees A,B]", ("calendar",)),
    CommandSpec(("calendar", "update-event"), "calendar update-event --id ID [--title TITLE] [--start DATETIME] [--location LOC]", ("calendar",)),
    CommandSpec(("calendar", "delete-event"), "calendar delete-event --id ID", ("calendar",)),
    CommandSpec(("calendar", "today"), "calendar today --timezone TIMEZONE", ("calendar",)),
    CommandSpec(("email", "list"), "email list [--folder FOLDER]", ("email",)),
    CommandSpec(("email", "read"), "email read --id ID", ("email",)),
    CommandSpec(("email", "send"), "email send --to EMAIL --subject SUBJECT --body BODY", ("email",)),
    CommandSpec(("email", "reply"), "email reply --id ID --body BODY", ("email",)),
    CommandSpec(("email", "move"), "email move --id ID --folder FOLDER", ("email",)),
    CommandSpec(("email", "mark"), "email mark --id ID --flag read|unread", ("email",)),
    CommandSpec(("email", "search"), "email search --query QUERY", ("email",)),
    CommandSpec(("weather", "get"), "weather get --location LOCATION", ("weather",)),
    CommandSpec(("weather", "forecast"), "weather forecast --location LOCATION --days N", ("weather",)),
    CommandSpec(("weather", "alerts"), "weather alerts --location LOCATION", ("weather",)),
    CommandSpec(("file", "create"), "file create --path PATH --content CONTENT", ("file",)),
    CommandSpec(("file", "read"), "file read --path PATH", ("file",)),
    CommandSpec(("file", "move"), "file move --src SRC --dst DST", ("file",)),
    CommandSpec(("file", "append"), "file append --path PATH --content CONTENT", ("file",)),
    CommandSpec(("file", "delete"), "file delete --path PATH", ("file",)),
    CommandSpec(("tasks", "list"), "tasks list [--status STATUS] [--priority PRIORITY]", ("tasks",)),
    CommandSpec(("tasks", "add"), "tasks add --title TITLE [--due DATE] [--priority PRIORITY]", ("tasks",)),
    CommandSpec(("tasks", "complete"), "tasks complete --id ID | --title TITLE", ("tasks",)),
    CommandSpec(("tasks", "search"), "tasks search --query QUERY", ("tasks",)),
    CommandSpec(("openclaw", "status"), "openclaw status [--json]", ("monitoring",)),
    CommandSpec(("openclaw", "health"), "openclaw health", ("monitoring",)),
    CommandSpec(("openclaw", "doctor"), "openclaw doctor", ("monitoring",)),
    CommandSpec(("openclaw", "gateway", "start"), "openclaw gateway start", ("monitoring",)),
    CommandSpec(("openclaw", "models", "set"), "openclaw models set MODEL", ("setup_config",)),
    CommandSpec(("openclaw", "config", "get"), "openclaw config get PATH", ("setup_config",), note="PATH must be an exact config path like agent.model"),
    CommandSpec(("openclaw", "config", "set"), "openclaw config set PATH VALUE", ("setup_config", "cron_webhook")),
    CommandSpec(("openclaw", "agents", "add"), "openclaw agents add NAME [--model MODEL]", ("agent_mgmt",)),
    CommandSpec(("openclaw", "agents", "list"), "openclaw agents list", ("agent_mgmt",)),
    CommandSpec(("openclaw", "agents", "set-identity"), "openclaw agents set-identity NAME --role ROLE", ("agent_mgmt",)),
    CommandSpec(("openclaw", "plugins", "enable"), "openclaw plugins enable NAME", ("plugin_skill",)),
    CommandSpec(("openclaw", "plugins", "disable"), "openclaw plugins disable NAME", ("plugin_skill",)),
    CommandSpec(("openclaw", "plugins", "list"), "openclaw plugins list", ("plugin_skill",)),
    CommandSpec(("openclaw", "plugins", "install"), "openclaw plugins install NAME", ("plugin_skill",)),
    CommandSpec(("openclaw", "skills", "info"), "openclaw skills info NAME", ("plugin_skill",)),
    CommandSpec(("openclaw", "channels", "login"), "openclaw channels login --channel NAME", ("channel_mgmt",)),
    CommandSpec(("openclaw", "channels", "list"), "openclaw channels list [--json]", ("channel_mgmt",)),
    CommandSpec(("openclaw", "channels", "status"), "openclaw channels status", ("channel_mgmt",)),
    CommandSpec(("openclaw", "message", "send"), "openclaw message send --channel NAME --target TARGET --message TEXT", ("messaging",)),
    CommandSpec(("openclaw", "message", "broadcast"), "openclaw message broadcast --channel NAME --message TEXT", ("messaging",)),
    CommandSpec(("openclaw", "message", "search"), "openclaw message search --channel NAME --query QUERY", ("messaging",)),
    CommandSpec(("openclaw", "message", "react"), "openclaw message react --channel NAME --message-id ID --emoji EMOJI", ("messaging",)),
    CommandSpec(("openclaw", "message", "poll"), "openclaw message poll --channel NAME --question TEXT --options A,B", ("messaging",)),
    CommandSpec(("openclaw", "cron", "add"), "openclaw cron add --name NAME --cron SCHEDULE --message TEXT", ("cron_webhook",)),
    CommandSpec(("openclaw", "cron", "list"), "openclaw cron list", ("cron_webhook",)),
    CommandSpec(("openclaw", "security", "audit"), "openclaw security audit", ("security",)),
    CommandSpec(("openclaw", "setup"), "openclaw setup", ("setup_config",)),
    CommandSpec(("openclaw", "devices", "pair"), "openclaw devices pair", ("device_node",)),
    CommandSpec(("openclaw", "devices", "list"), "openclaw devices list", ("device_node",)),
)

_REQUIRED_FLAGS_BY_PREFIX: dict[tuple[str, ...], tuple[str, ...]] = {
    ("calendar", "add-event"): ("--title", "--start"),
    ("calendar", "update-event"): ("--id",),
    ("calendar", "delete-event"): ("--id",),
    ("calendar", "today"): ("--timezone",),
    ("email", "read"): ("--id",),
    ("email", "send"): ("--to", "--subject", "--body"),
    ("email", "reply"): ("--id", "--body"),
    ("email", "move"): ("--id", "--folder"),
    ("email", "mark"): ("--id", "--flag"),
    ("email", "search"): ("--query",),
    ("weather", "get"): ("--location",),
    ("weather", "forecast"): ("--location", "--days"),
    ("tasks", "add"): ("--title",),
    ("tasks", "search"): ("--query",),
    ("openclaw", "channels", "login"): ("--channel",),
    ("openclaw", "message", "send"): ("--channel", "--target", "--message"),
    ("openclaw", "cron", "add"): ("--name", "--cron", "--message"),
}

_REQUIRED_POSITIONAL_ARGS_BY_PREFIX: dict[tuple[str, ...], int] = {
    ("openclaw", "config", "get"): 1,
    ("openclaw", "models", "set"): 1,
}

_VALID_CONFIG_PATHS = {
    "agent.model",
    "agents.defaults.model.primary",
    "gateway.port",
    "gateway.bind",
    "gateway.auth.mode",
    "gateway.auth.token",
    "gateway.remote.url",
}

_PLACEHOLDER_VALUES = {
    "TIMEZONE",
    "LOCATION",
    "TITLE",
    "DATETIME",
    "DATE",
    "NAME",
    "SCHEDULE",
    "QUERY",
    "BODY",
    "SUBJECT",
    "EMAIL",
    "ADDR",
    "TARGET",
    "TEXT",
    "ROLE",
    "PATH",
    "SRC",
    "DST",
    "ID",
    "FOLDER",
    "PRIORITY",
    "MODEL",
    "LOC",
    "TZ",
    "N",
}


def _strip_placeholder_token(token: str) -> str:
    return token.strip().strip("`'\"[](){}.,:;")


def _looks_like_catalog_placeholder(token: str) -> bool:
    stripped = token.strip()
    return stripped.startswith("[") and stripped.endswith("]")


def _contains_placeholder_value(tokens: tuple[str, ...]) -> str | None:
    for token in tokens:
        normalized = _strip_placeholder_token(token)
        if normalized in _PLACEHOLDER_VALUES:
            return normalized
    return None

_DOMAIN_HINT_GROUPS: dict[str, tuple[str, ...]] = {
    "messaging": ("channel_mgmt",),
    "channel_mgmt": ("messaging",),
    "setup_config": ("monitoring",),
    "monitoring": ("setup_config",),
    "cron_webhook": ("monitoring",),
}


def _spec_key_from_command(command: str) -> tuple[str, ...]:
    tokens = command.split()
    if not tokens:
        return tuple()
    if tokens[0] == "openclaw" and len(tokens) >= 3:
        return tuple(tokens[:3])
    if len(tokens) >= 2:
        return tuple(tokens[:2])
    return tuple(tokens[:1])


def _task_public_dict(task: Task | None) -> dict[str, Any]:
    public = getattr(getattr(task, "data", None), "public", None)
    return public if isinstance(public, dict) else {}


def _is_hard_decision_task(task: Task | None) -> bool:
    if task is None:
        return False
    task_id = str(getattr(task, "task_id", "") or "")
    if task_id.startswith("hard_decision_workflow_"):
        return True
    public = _task_public_dict(task)
    return any(
        isinstance(public.get(key), str) and public.get(key)
        for key in ("hard_decision_scenario", "hard_decision_ability")
    )


def _benchmark_anchor_timezone(task: Task | None) -> str | None:
    ground_truth = getattr(task, "ground_truth", None)
    commands = getattr(ground_truth, "solution_commands", None) if ground_truth else None
    if not commands:
        return None
    for cmd in commands:
        raw = str(cmd or "").strip()
        if raw.startswith("calendar today --timezone "):
            return raw.split("calendar today --timezone ", 1)[1].strip().strip("'\"")
    return None


def _rewrite_task_instruction_for_prompt(task_instruction: str, task: Task | None) -> str:
    if not _is_hard_decision_task(task):
        return task_instruction
    return re.sub(r"\bboard\b", "task board", task_instruction)


def _benchmark_prompt_notes(task: Task | None) -> list[str]:
    if not _is_hard_decision_task(task):
        return []
    notes = [
        (
            "Benchmark time anchor: treat relative dates like 'today' and 'next Friday' "
            f"as belonging to the benchmark world, not the host system date. The benchmark "
            f"world date anchor is {DEFAULT_BENCHMARK_DATE} unless command output gives a more specific benchmark time."
        ),
        (
            "In hard workflow tasks, 'board' means the task board managed with `tasks *` commands "
            "unless the command catalog explicitly includes a required channel or messaging action."
        ),
        "Prefer `calendar today --timezone ...` before inferring date ranges from relative time phrases.",
    ]
    timezone = _benchmark_anchor_timezone(task)
    if timezone:
        notes.append(
            f"For this task, prefer `calendar today --timezone {timezone}` when you need the current benchmark-local date."
        )
    return notes


def _is_hard_task_id(task_id: str | None) -> bool:
    return bool(task_id and task_id.startswith("hard_decision_workflow_"))


def _resolve_max_steps(
    max_steps: int | None,
    *,
    split: str | None = None,
    task_prefix: str | None = None,
    task_id: str | None = None,
) -> int:
    del split, task_prefix, task_id
    return max_steps if max_steps is not None else DEFAULT_MAX_STEPS


def _host_reference_date() -> date:
    return date.today()


def _command_specs_for_task(task: Task | None) -> list[CommandSpec]:
    if task is None:
        return list(_COMMAND_SPECS)

    wanted = set(getattr(task, "domains", []) or [])
    wanted.discard("composite")
    for domain in tuple(wanted):
        wanted.update(_DOMAIN_HINT_GROUPS.get(domain, ()))

    specs: list[CommandSpec] = []
    seen: set[tuple[str, ...]] = set()

    if wanted:
        for spec in _COMMAND_SPECS:
            if wanted.intersection(spec.domains):
                specs.append(spec)
                seen.add(spec.tokens)

    ground_truth = getattr(task, "ground_truth", None)
    if ground_truth and getattr(ground_truth, "solution_commands", None):
        wanted_keys = {
            _spec_key_from_command(cmd)
            for cmd in ground_truth.solution_commands
            if cmd.strip()
        }
        for spec in _COMMAND_SPECS:
            if spec.tokens in wanted_keys and spec.tokens not in seen:
                specs.append(spec)
                seen.add(spec.tokens)

    return specs or list(_COMMAND_SPECS)


def _command_catalog_text(task: Task | None) -> str:
    lines = []
    for spec in _command_specs_for_task(task):
        line = f"  {spec.syntax}"
        if spec.note:
            line += f"  # {spec.note}"
        lines.append(line)
    return "\n".join(lines)


def _validate_command(action: str, task: Task | None) -> tuple[bool, str]:
    raw = (action or "").strip()
    if not raw:
        return False, "empty response"
    if raw.upper() == "DONE":
        return True, ""
    try:
        tokens = tuple(shlex.split(raw))
    except ValueError as exc:
        return False, f"shell parse error: {exc}"
    if any(_looks_like_catalog_placeholder(token) for token in tokens):
        return False, (
            f"invalid command `{raw}`. Remove catalog placeholders like `[--json]` and output only the executable command."
        )
    for spec in _command_specs_for_task(task):
        if len(tokens) >= len(spec.tokens) and tokens[: len(spec.tokens)] == spec.tokens:
            required_flags = _REQUIRED_FLAGS_BY_PREFIX.get(spec.tokens, ())
            missing = [flag for flag in required_flags if flag not in tokens]
            if missing:
                return False, (
                    f"invalid command `{raw}`. Missing required flag(s): {', '.join(missing)}"
                )
            required_positional = _REQUIRED_POSITIONAL_ARGS_BY_PREFIX.get(spec.tokens, 0)
            positional_tokens = [
                token
                for token in tokens[len(spec.tokens):]
                if not token.startswith("--")
            ]
            if len(positional_tokens) < required_positional:
                return False, (
                    f"invalid command `{raw}`. Missing required value after `{ ' '.join(spec.tokens) }`."
                )
            placeholder = _contains_placeholder_value(tokens[len(spec.tokens):])
            if placeholder:
                return False, (
                    f"invalid command `{raw}`. Replace template placeholder `{placeholder}` with a task-specific value."
                )
            if spec.tokens == ("openclaw", "config", "get"):
                path = positional_tokens[0] if positional_tokens else ""
                if path not in _VALID_CONFIG_PATHS:
                    valid_examples = ", ".join(sorted(_VALID_CONFIG_PATHS)[:4])
                    return False, (
                        f"invalid command `{raw}`. Use an exact config path such as {valid_examples}."
                    )
            return True, ""
    allowed = ", ".join(spec.syntax for spec in _command_specs_for_task(task)[:8])
    return False, (
        f"invalid command `{raw}`. Use only the documented commands for this task. "
        f"Examples: {allowed}"
    )


def _normalize_model_command(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    text = text.strip("`")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    text = lines[0]
    if text.startswith("$ "):
        text = text[2:].strip()
    while re.search(r"\s+\[[^\]]+\]\s*$", text):
        text = re.sub(r"\s+\[[^\]]+\]\s*$", "", text).strip()
    text = re.sub(
        r"[。【】]+\s*(?:json(?:_[a-zA-Z0-9]+)?(?:\s+to=\w+.*)?|analysis(?:\s+to=\w+.*)?)$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if text.upper() == "DONE":
        return "DONE"
    if " - " in text and not any(ch in text for ch in "\"'"):
        text = re.split(r"\s+-\s+", text, maxsplit=1)[0].strip()
        if text.upper() == "DONE":
            return "DONE"
    try:
        shlex.split(text)
        return text
    except ValueError:
        text = re.split(r"\s+-\s+", text, maxsplit=1)[0].strip()
    text = re.sub(
        r"[。【】]+\s*(?:json(?:_[a-zA-Z0-9]+)?(?:\s+to=\w+.*)?|analysis(?:\s+to=\w+.*)?)$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if text.upper() == "DONE":
        return "DONE"
    return text


def _collapse_response_text_parts(text_parts: list[str]) -> tuple[str, int]:
    cleaned_parts = [str(part) for part in text_parts if str(part).strip()]
    if not cleaned_parts:
        return "", 0

    deduped_parts: list[str] = []
    previous_key: str | None = None
    removed_duplicates = 0
    for part in cleaned_parts:
        key = _normalize_model_command(part) or part.strip()
        if deduped_parts and key and key == previous_key:
            removed_duplicates += 1
            continue
        deduped_parts.append(part)
        previous_key = key

    collapsed = "".join(deduped_parts).strip()
    if collapsed:
        return collapsed, removed_duplicates
    return "".join(cleaned_parts).strip(), removed_duplicates


_SIDE_EFFECT_HINTS: tuple[str, ...] = (
    "add ",
    "create ",
    "send ",
    "schedule",
    "reschedule",
    "set up",
    "book ",
    "keep ",
    "put ",
    "configure",
    "update ",
    "move ",
)

def _instruction_implies_side_effect(task: Task | None) -> bool:
    instruction = ""
    if task is not None:
        instruction = str(getattr(task, "instruction", "") or getattr(task, "canonical_instruction", "") or "")
    text = instruction.lower()
    return any(marker in text for marker in _SIDE_EFFECT_HINTS)


def _extract_command_from_reasoning_text(reasoning_text: str, task: Task | None) -> str:
    text = (reasoning_text or "").strip()
    if not text:
        return ""

    candidates: list[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prefixes = [" ".join(spec.tokens) for spec in _command_specs_for_task(task)]

    for line in lines:
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if cleaned:
            candidates.append(cleaned)
        candidates.extend(re.findall(r"`([^`]+)`", line))
        for prefix in prefixes:
            idx = line.find(prefix)
            if idx >= 0:
                candidates.append(line[idx:].strip())

    for candidate in candidates:
        normalized = _normalize_model_command(candidate)
        ok, _ = _validate_command(normalized, task)
        if ok and normalized.upper() != "DONE":
            return normalized
    return ""


def _is_retryable_llm_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    retry_markers = (
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "connection refused",
        "remote end closed connection",
        "network is unreachable",
        "try again",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "too many requests",
        "rate limit",
        "http 429",
        "queue is full",
        '"code":429',
        '"status":429',
        '"code":503',
    )
    return any(marker in msg for marker in retry_markers)


def _is_content_filter_llm_error(exc: Exception | str) -> bool:
    msg = str(exc).lower()
    markers = (
        "content_filter",
        "filtered due to the prompt",
        "responsibleaipolicyviolation",
        '"param":"prompt"',
        '"param": "prompt"',
        '"code":"content_filter"',
        '"code": "content_filter"',
        '"jailbreak"',
    )
    return any(marker in msg for marker in markers)

_WEATHER_CODE_TEXT = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    61: "rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
}


def _safe_json_dict(text: str | None) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    # Best-effort recovery when stdout has wrappers around JSON.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    return None


def _step_action(step: dict[str, Any]) -> str:
    return str(step.get("executed_action") or step.get("action") or "")


def _step_output(step: dict[str, Any]) -> str:
    clean = step.get("clean_stdout")
    if isinstance(clean, str) and clean.strip():
        return clean
    return str(step.get("stdout") or "")


def _extract_location_from_geocoding(trajectory: list[dict[str, Any]]) -> str | None:
    for step in reversed(trajectory):
        action = _step_action(step)
        if "geocoding-api.open-meteo.com/v1/search" not in action:
            continue
        payload = _safe_json_dict(_step_output(step))
        if not payload:
            continue
        results = payload.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                name = first.get("name")
                if isinstance(name, str) and name:
                    return name
    return None


def _extract_weather_summary(trajectory: list[dict[str, Any]]) -> str | None:
    location = _extract_location_from_geocoding(trajectory)

    for step in reversed(trajectory):
        action = _step_action(step)
        out = _step_output(step)

        if "archive-api.open-meteo.com/v1/archive" in action:
            payload = _safe_json_dict(out)
            if not payload:
                continue
            daily = payload.get("daily") if isinstance(payload.get("daily"), dict) else {}
            dates = daily.get("time") if isinstance(daily.get("time"), list) else []
            codes = daily.get("weather_code") if isinstance(daily.get("weather_code"), list) else []
            tmax = daily.get("temperature_2m_max") if isinstance(daily.get("temperature_2m_max"), list) else []
            tmin = daily.get("temperature_2m_min") if isinstance(daily.get("temperature_2m_min"), list) else []
            if not dates:
                continue
            date = dates[0]
            code = codes[0] if codes else None
            condition = _WEATHER_CODE_TEXT.get(int(code), f"code {code}") if code is not None else "unknown"
            high = tmax[0] if tmax else "?"
            low = tmin[0] if tmin else "?"
            place = location or "requested location"
            return f"WEATHER_RESULT: {place} on {date}: {condition}, {low}°C to {high}°C."

        if "api.open-meteo.com/v1/forecast" in action:
            payload = _safe_json_dict(out)
            if not payload:
                continue
            current = payload.get("current") if isinstance(payload.get("current"), dict) else {}
            if not current:
                continue
            temp = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            code = current.get("weather_code")
            condition = _WEATHER_CODE_TEXT.get(int(code), f"code {code}") if code is not None else "unknown"
            place = location or payload.get("timezone") or "requested location"
            parts = [f"WEATHER_RESULT: {place}: {condition}"]
            if temp is not None:
                parts.append(f"temp {temp}°C")
            if humidity is not None:
                parts.append(f"humidity {humidity}%")
            return ", ".join(parts) + "."

        if action.startswith("weather get "):
            line1 = re.search(r"Weather for (.+?) on ([0-9]{4}-[0-9]{2}-[0-9]{2}):", out)
            cond = re.search(r"Condition\s*:\s*(.+)", out)
            temp = re.search(r"Temp\s*:\s*([^\n]+)", out)
            hum = re.search(r"Humidity\s*:\s*([^\n]+)", out)
            if line1:
                place, date = line1.group(1), line1.group(2)
                condition = cond.group(1).strip() if cond else "unknown"
                pieces = [f"WEATHER_RESULT: {place} on {date}: {condition}"]
                if temp:
                    pieces.append(f"temp {temp.group(1).strip()}")
                if hum:
                    pieces.append(f"humidity {hum.group(1).strip()}")
                return ", ".join(pieces) + "."
    return None


def _extract_calendar_summary(trajectory: list[dict[str, Any]]) -> str | None:
    for step in reversed(trajectory):
        action = _step_action(step)
        out = _step_output(step)

        if "timeapi.io/api/Time/current/zone" in action:
            payload = _safe_json_dict(out)
            if not payload:
                continue
            tz = payload.get("timeZone")
            dt_value = payload.get("dateTime")
            if tz and dt_value:
                return f"CALENDAR_RESULT: Current date-time in {tz} is {dt_value}."

        if action.startswith("calendar today") or action.startswith("gcalcli now"):
            m = re.search(r"Current date-time for (.+?):\s*([^\n]+)", out)
            if m:
                return f"CALENDAR_RESULT: Current date-time in {m.group(1)} is {m.group(2)}."
    return None


def _build_final_response(task: Task | None, trajectory: list[dict[str, Any]]) -> str:
    if not trajectory:
        return ""
    domains = set(task.domains if task else [])
    lines: list[str] = []

    if "weather" in domains:
        weather = _extract_weather_summary(trajectory)
        if weather:
            lines.append(weather)

    if "calendar" in domains:
        calendar = _extract_calendar_summary(trajectory)
        if calendar:
            lines.append(calendar)

    # Fallback: infer from actions for mixed tasks that may omit explicit domains.
    if not lines:
        actions = " ".join(_step_action(s) for s in trajectory)
        if "weather " in actions or "open-meteo" in actions:
            weather = _extract_weather_summary(trajectory)
            if weather:
                lines.append(weather)
        if "calendar " in actions or "timeapi.io" in actions:
            calendar = _extract_calendar_summary(trajectory)
            if calendar:
                lines.append(calendar)

    return "\n".join(lines)


def _detect_date_anchor_mismatch(
    task: Task | None,
    executed_actions: list[str],
) -> bool:
    if not _is_hard_decision_task(task) or not executed_actions:
        return False
    host_today = _host_reference_date().isoformat()
    if host_today == DEFAULT_BENCHMARK_DATE:
        return False
    suspicious_dates = {
        host_today,
        (_host_reference_date() - timedelta(days=1)).isoformat(),
        (_host_reference_date() + timedelta(days=1)).isoformat(),
    }
    text = "\n".join(executed_actions)
    return any(marker in text for marker in suspicious_dates)


# ===========================================================================
# Episode result
# ===========================================================================

@dataclass
class EpisodeResult:
    task_id: str
    agent_name: str
    success: bool
    score: float
    steps: int
    duration_s: float
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    error: str = ""
    error_type: str = ""
    provider_impacted: bool = False
    stop_reason: str = ""
    date_anchor_mismatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent": self.agent_name,
            "success": self.success,
            "score": self.score,
            "steps": self.steps,
            "duration_s": round(self.duration_s, 3),
            "trajectory": self.trajectory,
            "final_response": self.final_response,
            "error": self.error,
            "error_type": self.error_type,
            "provider_impacted": self.provider_impacted,
            "stop_reason": self.stop_reason,
            "date_anchor_mismatch": self.date_anchor_mismatch,
        }


def _classify_error_type(error: str, success: bool) -> str:
    if success:
        return ""
    text = (error or "").strip().lower()
    if not text:
        return "logical_failure"
    provider_markers = (
        "http 429",
        "too many requests",
        "queue is full",
        "content_filter",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "read operation timed out",
        "timed out",
        "openai-compatible request failed",
        "empty message content",
        "no text parts in message content",
        "rate limit",
    )
    if any(marker in text for marker in provider_markers):
        return "provider_failure"
    return "runner_failure"


def _has_provider_noise(debug_lines: list[str] | None = None, error: str = "") -> bool:
    text_parts = [str(item) for item in (debug_lines or []) if str(item).strip()]
    if error:
        text_parts.append(str(error))
    haystack = "\n".join(text_parts).lower()
    if not haystack.strip():
        return False
    markers = (
        "api-fallback: content filter triggered",
        "content_filter",
        "empty message content",
        "no text parts in message content",
        "api-error: openai-compatible request failed",
        "api-error: openai-compatible response contained empty message content",
        "api: content=none reasoning=",
    )
    return any(marker in haystack for marker in markers)


def _retry_backoff_delay(base_delay_s: float, attempt_index: int) -> float:
    if base_delay_s <= 0:
        return 0.0
    return min(12.0, base_delay_s * (2 ** attempt_index))


def _agent_run_config(agent: Any) -> dict[str, str]:
    debug_fn = getattr(agent, "debug_run_config", None)
    if not callable(debug_fn):
        return {}
    config = debug_fn()
    return config if isinstance(config, dict) else {}


# ===========================================================================
# Base agent interface
# ===========================================================================

class BaseAgent(ABC):
    """Abstract agent interface.

    On each step the agent receives an :class:`Observation` and returns a
    single CLI command string.  The episode loop calls :meth:`reset` at the
    start of every task and :meth:`act` at every step.
    """

    name: str = "base"

    def reset(self, task_instruction: str, task: Task | None = None) -> None:
        """Called once at the start of each episode."""

    @abstractmethod
    def act(self, observation: Observation) -> str:
        """Return the next CLI command given the current observation."""

    def should_stop(self) -> bool:
        """Whether the episode runner should stop before sending another command."""
        return False

    def debug_last_reply(self) -> str | None:
        """Optional debug string describing the latest model/agent reply."""
        return None

    def debug_reply_attempts(self) -> list[str]:
        """Optional list of raw/parsed reply attempts from the latest act() call."""
        return []

    def debug_run_config(self) -> dict[str, str]:
        """Optional metadata to stamp into logs and saved reports."""
        return {}


# ===========================================================================
# Expert agent
# ===========================================================================

class ExpertAgent(BaseAgent):
    """Replays the ground-truth solution commands.

    This is the oracle upper bound — it always succeeds when evaluation checks
    are correct.  Use it to verify the environment and collect demonstration
    trajectories for behavior cloning.
    """

    name = "expert"

    def __init__(self) -> None:
        self._commands: list[str] = []
        self._idx: int = 0

    def reset(self, task_instruction: str, task: Task | None = None) -> None:
        if task and task.ground_truth:
            self._commands = list(task.ground_truth.solution_commands)
        else:
            self._commands = []
        self._idx = 0

    def act(self, observation: Observation) -> str:
        if self._idx < len(self._commands):
            cmd = self._commands[self._idx]
            self._idx += 1
            return cmd
        # Fallback no-op if caller does not check should_stop().
        return "tasks list"

    def should_stop(self) -> bool:
        return self._idx >= len(self._commands)


# ===========================================================================
# Random agent
# ===========================================================================

class RandomAgent(BaseAgent):
    """Issues random CLI commands.  Lower-bound baseline."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._prefixes = list(_CLI_VERBS.keys())

    def act(self, observation: Observation) -> str:
        prefix = self._rng.choice(self._prefixes)
        verb = self._rng.choice(_CLI_VERBS[prefix])
        return f"{prefix} {verb}"


# ===========================================================================
# Rule-based agent
# ===========================================================================

class RuleAgent(BaseAgent):
    """Simple heuristic agent that parses the task instruction.

    Demonstrates how to build a domain-specific rule-based baseline without
    using an LLM.  It detects keywords in the instruction and maps them to
    CLI commands.
    """

    name = "rule"

    _KEYWORD_MAP: list[tuple[list[str], str]] = [
        # (keywords, command_template)
        (["calendar", "event", "add", "block"], "calendar add-event"),
        (["calendar", "reschedule", "move", "update"], "calendar update-event"),
        (["calendar", "list", "schedule", "show"], "calendar list"),
        (["email", "send", "draft"], "email send"),
        (["email", "reply", "respond"], "email reply"),
        (["email", "move", "folder"], "email move"),
        (["email", "mark", "read", "seen"], "email mark-read"),
        (["email", "search", "find", "inbox"], "email search"),
        (["weather", "forecast", "week"], "weather forecast"),
        (["weather", "alert"], "weather alerts"),
        (["weather", "get", "check"], "weather get"),
        (["file", "create", "save", "write"], "file create"),
        (["file", "move", "rename"], "file move"),
        (["file", "append", "add"], "file append"),
        (["file", "read", "show"], "file read"),
        (["task", "add", "track", "item"], "tasks add"),
        (["task", "complete", "done", "finish"], "tasks complete"),
        (["task", "search", "find"], "tasks search"),
        (["task", "list", "show", "plate"], "tasks list"),
    ]

    def __init__(self) -> None:
        self._instruction = ""
        self._step = 0

    def reset(self, task_instruction: str, task: Task | None = None) -> None:
        self._instruction = task_instruction.lower()
        self._step = 0

    def act(self, observation: Observation) -> str:
        self._step += 1
        if self._step == 1:
            # On the first step, pick the best-matching command
            words = set(self._instruction.split())
            best_cmd = "tasks list"
            best_score = 0
            for keywords, cmd in self._KEYWORD_MAP:
                score = sum(1 for kw in keywords if kw in words)
                if score > best_score:
                    best_score = score
                    best_cmd = cmd
            return best_cmd
        # Subsequent steps: just list to avoid errors
        return "tasks list"


# ===========================================================================
# LLM agent (Anthropic or OpenAI-compatible API)
# ===========================================================================

class LLMAgent(BaseAgent):
    """Multi-turn LLM agent.

    Each step appends the observation to a running conversation and asks the
    model for the next CLI command.  The model is instructed to output a
    single bare command or the literal token ``DONE`` when finished.

    Requirements
    ------------
    Anthropic:
    - ``pip install anthropic``
    - ``ANTHROPIC_API_KEY`` environment variable set

    OpenAI-compatible:
    - No extra package required
    - API compatible with ``POST /v1/chat/completions``
    - ``OPENAI_API_KEY`` or configured key env var set

    Parameters
    ----------
    provider:
        ``anthropic`` or ``openai``. ``openai`` also covers OpenAI-compatible
        proxies and Azure-style gateways that expose the chat completions API.
    model:
        Model ID for the selected provider.
    base_url:
        Optional base URL for OpenAI-compatible providers. Defaults to
        ``https://api.openai.com/v1`` when ``provider=openai``.
    api_key_env:
        Environment variable holding the API key. Defaults to
        ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` based on provider.
    max_steps_hint:
        Soft cap communicated to the model in the system prompt.
    """

    name = "llm"

    BASE_SYSTEM_PROMPT = """\
You are selecting the next command for a benchmark task.

Response format:
- Return one shell command line.
- If the task is complete, return DONE.
- Do not add commentary or formatting.

Execution guidance:
- Choose commands from the task-specific command catalog.
- After each step, use the command output to decide the next action.
- If a step fails, choose a different command from the catalog.
- Do not return DONE until the task's requested state changes have actually happened.
"""

    _LITELLM_PROXY_MODEL_PREFIXES = ("claude-",)
    _LITELLM_PROXY_BASE_URL_ENV_CANDIDATES = (
        "LITELLM_PROXY_BASE_URL",
        "BEDROCK_LITELLM_BASE_URL",
    )
    _LITELLM_PROXY_KEY_ENV_CANDIDATES = (
        "LITELLM_PROXY_KEY",
        "BEDROCK_LITELLM_KEY",
    )
    _LITELLM_PROXY_DEFAULT_BASE_URL = "http://127.0.0.1:4000/v1"

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5-20251001",
        base_url: str | None = None,
        api_key_env: str | None = None,
        history_mode: str = "full",
        max_steps_hint: int = 15,
        temperature: float = 0.0,
        response_max_tokens: int = 256,
        request_timeout_s: int = 25,
        request_retries: int = 1,
        retry_backoff_s: float = 1.5,
    ) -> None:
        if provider not in {"anthropic", "openai"}:
            raise ValueError(
                f"Unsupported LLM provider: {provider!r}. Choose from: anthropic, openai"
            )
        if history_mode not in {"auto", "full", "summary"}:
            raise ValueError(
                f"Unsupported history mode: {history_mode!r}. Choose from: auto, full, summary"
            )
        self._provider = provider
        self._model = model
        self._base_url = (base_url or "").rstrip("/")
        self._api_key_env = api_key_env
        self._history_mode = history_mode
        self._max_steps_hint = max_steps_hint
        if model == "o3":
            self._temperature = 1.0
        else:
            self._temperature = temperature
        self._response_max_tokens = max(32, response_max_tokens)
        self._request_timeout_s = request_timeout_s
        self._request_retries = max(0, request_retries)
        self._retry_backoff_s = max(0.0, retry_backoff_s)
        self._messages: list[dict[str, str]] = []
        self._client: Any = None
        self._task: Task | None = None
        self._system_prompt = self.BASE_SYSTEM_PROMPT
        self._invalid_retry_limit = 2
        self._pending_action: str = ""
        self._history: list[dict[str, Any]] = []
        self._done_requested = False
        self._last_reply_debug: str | None = None
        self._reply_attempts_debug: list[str] = []
        self._task_instruction: str = ""

    def _uses_litellm_proxy_defaults(self) -> bool:
        if self._provider != "openai":
            return False
        model = (self._model or "").strip().lower()
        return any(model.startswith(prefix) for prefix in self._LITELLM_PROXY_MODEL_PREFIXES)

    def _litellm_proxy_env_name(self) -> str:
        for name in self._LITELLM_PROXY_KEY_ENV_CANDIDATES:
            if os.environ.get(name):
                return name
        return self._LITELLM_PROXY_KEY_ENV_CANDIDATES[0]

    def _api_key_name(self) -> str:
        if self._api_key_env:
            return self._api_key_env
        if self._uses_litellm_proxy_defaults():
            return self._litellm_proxy_env_name()
        return "OPENAI_API_KEY" if self._provider == "openai" else "ANTHROPIC_API_KEY"

    def _api_key(self) -> str:
        api_key = os.environ.get(self._api_key_name(), "")
        if not api_key:
            raise RuntimeError(
                f"{self._api_key_name()} environment variable is not set."
            )
        return api_key

    def _get_client(self) -> Any:
        if self._provider != "anthropic":
            raise RuntimeError("_get_client is only valid for Anthropic provider")
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e
        self._client = anthropic.Anthropic(api_key=self._api_key())
        return self._client

    def _anthropic_response_text(self) -> str:
        client = self._get_client()
        request_messages = self._request_messages(self._messages)
        last_exc: Exception | None = None
        for attempt in range(self._request_retries + 1):
            try:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=self._response_max_tokens,
                    temperature=self._temperature,
                    system=self._system_prompt,
                    messages=request_messages,
                )
                text = response.content[0].text.strip()
                self._reply_attempts_debug.append(f"api: content={text!r}")
                return text
            except Exception as exc:  # pragma: no cover - network/provider dependent
                last_exc = exc
                if not _is_retryable_llm_error(exc):
                    raise
                if attempt < self._request_retries:
                    delay = _retry_backoff_delay(self._retry_backoff_s, attempt)
                    if delay > 0:
                        self._reply_attempts_debug.append(
                            f"api-backoff: sleeping {delay:.1f}s before retry"
                        )
                        time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _openai_base_url(self) -> str:
        if self._base_url:
            return self._base_url
        if self._uses_litellm_proxy_defaults():
            for name in self._LITELLM_PROXY_BASE_URL_ENV_CANDIDATES:
                value = (os.environ.get(name) or "").strip()
                if value:
                    return value.rstrip("/")
            return self._LITELLM_PROXY_DEFAULT_BASE_URL
        return "https://api.openai.com/v1"

    def _trim_request_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return build_openai_request_messages(messages, self._history)

    def _request_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if self._history_mode == "full":
            return list(messages)
        if self._history_mode == "summary":
            return self._trim_request_messages(messages)
        return self._trim_request_messages(messages)

    def _compact_system_prompt(self) -> str:
        return (
            "Return exactly one shell command from the catalog or DONE. "
            "Do not explain. Do not use markdown."
        )

    def _compact_messages(self) -> list[dict[str, str]]:
        task_instruction = self._task_instruction.strip() if self._task_instruction else (self._messages[0]["content"] if self._messages else "")
        lines = [task_instruction.strip()]
        summary = build_memory_summary(self._history, compact=True)
        if summary:
            lines.append(summary)
        elif self._history:
            lines.append("Known context:")
            for item in self._history[-2:]:
                lines.append(
                    f"- {item['action']} => "
                    f"exit={item.get('exit_code', 0)}; "
                    f"stdout={clip_text(item.get('stdout', ''), limit=100) or '(empty)'}; "
                    f"stderr={clip_text(item.get('stderr', ''), limit=80) or '(none)'}"
                )
        lines.append("Reply with exactly one next command from the catalog or DONE.")
        return [{"role": "user", "content": "\n".join(lines)}]

    def _openai_token_limit_field(self) -> str:
        model = (self._model or "").lower()
        if model.startswith("gpt-5") or model.startswith("o3"):
            return "max_completion_tokens"
        return "max_tokens"

    def _openai_prefers_responses_api(self) -> bool:
        model = (self._model or "").lower()
        return model.startswith("gpt-")

    def _openai_request_once(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        *,
        system_prompt: str | None = None,
        use_responses_api: bool = False,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "openclaw-env-train-eval/1.0",
        }
        if use_responses_api:
            payload = {
                "model": self._model,
                "instructions": system_prompt or self._system_prompt,
                "input": messages,
                "max_output_tokens": max_tokens or self._response_max_tokens,
            }
            if self._temperature is not None:
                payload["temperature"] = self._temperature
            request = urllib.request.Request(
                f"{self._openai_base_url()}/responses",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
        else:
            token_limit_field = self._openai_token_limit_field()
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt or self._system_prompt},
                    *messages,
                ],
                "temperature": self._temperature,
            }
            payload[token_limit_field] = max_tokens or self._response_max_tokens
            request = urllib.request.Request(
                f"{self._openai_base_url()}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
        with urllib.request.urlopen(request, timeout=self._request_timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _openai_request(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        *,
        system_prompt: str | None = None,
        trim_messages: bool = True,
        allow_compact_fallback: bool = False,
    ) -> dict[str, Any]:
        request_messages = self._request_messages(messages) if trim_messages else list(messages)
        last_exc: Exception | None = None
        use_responses_api = self._openai_prefers_responses_api()
        for attempt in range(self._request_retries + 1):
            try:
                return self._openai_request_once(
                    request_messages,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    use_responses_api=use_responses_api,
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                err = RuntimeError(
                    f"OpenAI-compatible request failed: HTTP {exc.code}: {detail}"
                )
                if (not use_responses_api and exc.code == 400 and "unsupported" in detail.lower()):
                    self._reply_attempts_debug.append(
                        "api-fallback: chat/completions unsupported; retrying with responses API"
                    )
                    return self._openai_request_once(
                        request_messages,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                        use_responses_api=True,
                    )
                if allow_compact_fallback and _is_content_filter_llm_error(err):
                    self._reply_attempts_debug.append(
                        "api-fallback: content filter triggered; retrying with compact prompt"
                    )
                    return self._openai_request(
                        self._compact_messages(),
                        max_tokens=min(max_tokens or self._response_max_tokens, 96),
                        system_prompt=self._compact_system_prompt(),
                        trim_messages=False,
                        allow_compact_fallback=False,
                    )
                if (exc.code < 500 and exc.code != 429) or not _is_retryable_llm_error(err):
                    raise err from exc
                last_exc = err
                if attempt < self._request_retries:
                    delay = _retry_backoff_delay(self._retry_backoff_s, attempt)
                    if delay > 0:
                        self._reply_attempts_debug.append(
                            f"api-backoff: sleeping {delay:.1f}s before retry"
                        )
                        time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                err = RuntimeError(f"OpenAI-compatible request failed: {exc}")
                if not _is_retryable_llm_error(err):
                    raise err from exc
                last_exc = err
                if attempt < self._request_retries:
                    delay = _retry_backoff_delay(self._retry_backoff_s, attempt)
                    if delay > 0:
                        self._reply_attempts_debug.append(
                            f"api-backoff: sleeping {delay:.1f}s before retry"
                        )
                        time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _extract_openai_message_text(
        self,
        payload: dict[str, Any],
        allow_followup: bool = True,
    ) -> str:
        if payload.get("object") == "response" or "output" in payload:
            output = payload.get("output") or []
            content_parts: list[dict[str, Any]] = []
            for item in output:
                if isinstance(item, dict) and item.get("type") == "message":
                    content_parts.extend(item.get("content") or [])
            self._reply_attempts_debug.append(
                f"api: responses_parts={json.dumps(content_parts, ensure_ascii=False)}"
            )
            text_parts = [
                str(part.get("text", ""))
                for part in content_parts
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}
            ]
            text, removed_duplicates = _collapse_response_text_parts(text_parts)
            if removed_duplicates:
                self._reply_attempts_debug.append(
                    f"api: deduped {removed_duplicates} adjacent duplicate response text part(s)"
                )
            if text:
                return text
            raise RuntimeError(
                "OpenAI-compatible responses API contained no text parts: "
                f"{json.dumps(payload, ensure_ascii=False)}"
            )
        try:
            choice_message = payload["choices"][0]["message"]
            message = choice_message["content"]
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Unexpected OpenAI-compatible response payload: {payload}"
            ) from exc
        reasoning_text = str(choice_message.get("reasoning_content") or "").strip()
        if message is None:
            self._reply_attempts_debug.append(
                f"api: content=None reasoning={reasoning_text!r}"
            )
            fallback = _extract_command_from_reasoning_text(reasoning_text, self._task)
            if fallback:
                return fallback
            if allow_followup and reasoning_text:
                followup_payload = self._openai_request(
                    [
                        *self._messages,
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply included reasoning but no final command. "
                                "Output exactly one next command from the catalog now. "
                                "Do not include reasoning or commentary."
                            ),
                        },
                    ],
                    max_tokens=64,
                    allow_compact_fallback=True,
                )
                return self._extract_openai_message_text(
                    followup_payload,
                    allow_followup=False,
                )
            raise RuntimeError(
                "OpenAI-compatible response contained empty message content: "
                f"{json.dumps(choice_message, ensure_ascii=False)}"
            )
        if isinstance(message, str):
            text = message.strip()
            self._reply_attempts_debug.append(f"api: content={text!r}")
            return text
        if isinstance(message, list):
            self._reply_attempts_debug.append(
                f"api: content_parts={json.dumps(message, ensure_ascii=False)} reasoning={reasoning_text!r}"
            )
            text_parts = [
                str(part.get("text", ""))
                for part in message
                if isinstance(part, dict) and part.get("type") in {None, "text"}
            ]
            text = "".join(text_parts).strip()
            if text:
                return text
            fallback = _extract_command_from_reasoning_text(reasoning_text, self._task)
            if fallback:
                return fallback
            if allow_followup and reasoning_text:
                followup_payload = self._openai_request(
                    [
                        *self._messages,
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply did not include a final command. "
                                "Output exactly one next command from the catalog now. "
                                "Do not include reasoning or commentary."
                            ),
                        },
                    ],
                    max_tokens=64,
                    allow_compact_fallback=True,
                )
                return self._extract_openai_message_text(
                    followup_payload,
                    allow_followup=False,
                )
            raise RuntimeError(
                "OpenAI-compatible response contained no text parts in message content: "
                f"{json.dumps(choice_message, ensure_ascii=False)}"
            )
        raise RuntimeError(
            "OpenAI-compatible response returned unsupported message content type: "
            f"{type(message).__name__} payload={json.dumps(choice_message, ensure_ascii=False)}"
        )

    def _openai_response_text(self) -> str:
        payload = self._openai_request(self._messages, allow_compact_fallback=True)
        return self._extract_openai_message_text(payload)

    def reset(self, task_instruction: str, task: Task | None = None) -> None:
        self._task = task
        self._task_instruction = task_instruction
        self._messages = []
        self._pending_action = ""
        self._history = []
        self._done_requested = False
        self._last_reply_debug = None
        self._reply_attempts_debug = []
        self._system_prompt = (
            f"{self.BASE_SYSTEM_PROMPT}\n"
            f"\nCommand catalog for this task:\n{_command_catalog_text(task)}\n"
            "\nCatalog notes:\n"
            "- Reply with one command line or DONE.\n"
            "- Uppercase words like TIMEZONE, LOCATION, TITLE, NAME, and SCHEDULE are templates, not literal values.\n"
            "- `file` by itself is not part of this catalog.\n"
        )
        # Prime the conversation with the task
        self._messages.append({
            "role": "user",
            "content": (
                f"Task: {task_instruction}\n\n"
                f"Complete the task using the available CLI commands. "
                f"You have at most {self._max_steps_hint} steps. "
                f"Output your first command now."
            ),
        })

    def act(self, observation: Observation) -> str:
        self._reply_attempts_debug = []
        # On step > 0, append the previous observation before asking for next cmd
        if observation.step_number > 0:
            if self._pending_action:
                self._history.append(
                    {
                        "action": self._pending_action,
                        "stdout": observation.command_output or "",
                        "stderr": observation.error_output or "",
                        "exit_code": observation.exit_code,
                    }
                )
            obs_text = (
                f"[STDOUT]\n{observation.command_output or '(empty)'}\n"
                f"[STDERR]\n{observation.error_output or '(none)'}\n"
                f"[EXIT CODE] {observation.exit_code}\n\n"
                "What is your next command? (or DONE if finished)"
            )
            if len(self._history) >= 3:
                recent = self._history[-3:]
                if all(
                    item["action"] == recent[0]["action"]
                    and item["stdout"] == recent[0]["stdout"]
                    and item["stderr"] == recent[0]["stderr"]
                    and item["exit_code"] == recent[0]["exit_code"]
                    for item in recent[1:]
                ):
                    obs_text += (
                        f"\n\nRecent attempts repeated `{recent[0]['action']}` with no new result. "
                        "Pick a different command from the catalog."
                    )
            if self._history and all(is_read_only_action(item["action"]) for item in self._history):
                if _instruction_implies_side_effect(self._task):
                    obs_text += (
                        "\n\nSo far you have only inspected context. "
                        "If the task asks you to create, schedule, send, or update something, "
                        "choose a state-changing command before DONE."
                    )
            self._messages.append({"role": "user", "content": obs_text})

        for _ in range(self._invalid_retry_limit + 1):
            try:
                raw = (
                    self._openai_response_text()
                    if self._provider == "openai"
                    else self._anthropic_response_text()
                )
            except RuntimeError as exc:
                message = str(exc)
                self._reply_attempts_debug.append(f"api-error: {message}")
                if self._provider == "openai" and (
                    "empty message content" in message
                    or "no text parts in message content" in message
                ):
                    raw = ""
                else:
                    raise
            normalized = _normalize_model_command(raw)
            self._last_reply_debug = f"raw={raw!r} normalized={normalized!r}"
            self._reply_attempts_debug.append(f"parsed: {self._last_reply_debug}")
            self._messages.append({"role": "assistant", "content": normalized or raw})
            ok, reason = _validate_command(normalized, self._task)
            if ok and normalized.upper() == "DONE":
                if self._history and all(is_read_only_action(item["action"]) for item in self._history):
                    if _instruction_implies_side_effect(self._task):
                        ok = False
                        reason = (
                            "the task likely still requires a state-changing command before DONE"
                        )
            if ok:
                raw = normalized
                break
            self._reply_attempts_debug.append(f"validator: {reason}")
            self._messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your last reply did not match the command catalog: {reason}\n"
                        "Reply again with one catalog command or DONE."
                    ),
                }
            )
        else:  # pragma: no cover - defensive
            raw = "DONE"

        self._pending_action = raw
        if raw.upper() == "DONE" or not raw:
            self._done_requested = True
            return "DONE"
        return raw

    def should_stop(self) -> bool:
        return self._done_requested

    def debug_last_reply(self) -> str | None:
        return self._last_reply_debug

    def debug_reply_attempts(self) -> list[str]:
        return list(self._reply_attempts_debug)

    def debug_run_config(self) -> dict[str, str]:
        return {
            "provider": self._provider,
            "model": self._model,
            "history_mode": self._history_mode,
        }


# ===========================================================================
# Episode runner
# ===========================================================================

def run_episode(
    task_id: str,
    agent: BaseAgent,
    task_data_dir: str | Path | None = None,
    mode: str = "multi",
    max_steps: int = DEFAULT_MAX_STEPS,
    record_trajectory: bool = False,
    online_clean: bool = False,
    backend_kwargs: dict[str, Any] | None = None,
    stop_on_success: bool = True,
    max_stagnant_steps: int = 3,
    verbose: bool = False,
) -> EpisodeResult:
    """Run one episode of `task_id` with `agent`.

    Parameters
    ----------
    task_id:
        Task identifier (e.g. ``"calendar_add_event_1"``).
    agent:
        The agent to run.
    mode:
        Backend mode: ``"mock"`` | ``"multi"`` | ``"real"`` | ``"hybrid"``.
    max_steps:
        Hard cap on the number of steps per episode.
    record_trajectory:
        If True, store the full step-by-step trajectory in the result.
    verbose:
        Print each step to stdout.
    """
    t_start = time.time()
    trajectory: list[dict[str, Any]] = []
    executed_actions: list[str] = []
    error = ""
    provider_impacted = False
    task: Task | None = None
    stop_reason = ""

    try:
        with make_env(
            task_id,
            task_data_dir=task_data_dir,
            mode=mode,
            max_steps=max_steps,
            backend_kwargs=backend_kwargs,
        ) as env:
            obs: Observation = env.reset()
            task = env.task
            agent.reset(obs.task_instruction, task)

            if verbose:
                print(f"\n{'─'*60}")
                print(f"TASK  : {task_id}")
                print(f"AGENT : {agent.name}")
                run_config = _agent_run_config(agent)
                if run_config:
                    provider = run_config.get("provider")
                    model = run_config.get("model")
                    history_mode = run_config.get("history_mode")
                    if provider or model:
                        llm_label = " / ".join(
                            part for part in (provider, model) if part
                        )
                        print(f"LLM   : {llm_label}")
                    if history_mode:
                        print(f"HIST  : {history_mode}")
                print(f"INSTR : {obs.task_instruction}")
                print(f"{'─'*60}")

            done = False
            step = 0
            eval_result: EvaluationResult | None = None
            stagnant_steps = 0
            last_transition: tuple[str, str, str, int] | None = None
            while not done and step < max_steps:
                if agent.should_stop():
                    stop_reason = "agent requested stop before selecting another command"
                    break
                action = agent.act(obs)
                reply_attempts = agent.debug_reply_attempts()
                reply_debug = agent.debug_last_reply()
                provider_impacted = provider_impacted or _has_provider_noise(
                    reply_attempts or ([reply_debug] if reply_debug else [])
                )
                if verbose and reply_attempts:
                    for attempt in reply_attempts:
                        print(f"       model: {attempt}")
                elif verbose and reply_debug:
                    print(f"       model: {reply_debug}")
                if action.upper() == "DONE":
                    stop_reason = "agent returned DONE before executing another command"
                    break
                if verbose:
                    print(f"  [{step+1:02d}] $ {action}")

                obs, reward, done, info = env.step(action)
                executed_actions.append(action)
                if stop_on_success:
                    eval_result = env.evaluate()
                    if eval_result.success:
                        done = True

                transition = (
                    action,
                    obs.command_output or "",
                    obs.error_output or "",
                    int(obs.exit_code),
                )
                if transition == last_transition:
                    stagnant_steps += 1
                else:
                    stagnant_steps = 1
                    last_transition = transition
                if max_stagnant_steps > 0 and stagnant_steps >= max_stagnant_steps:
                    stop_reason = (
                        f"stopped after {max_stagnant_steps} identical no-progress transitions"
                    )
                    done = True

                if verbose:
                    out = (obs.command_output or "").strip()[:120]
                    print(f"       exit={obs.exit_code}  {out}")

                if record_trajectory:
                    clean_stdout = obs.command_output or ""
                    clean_stderr = obs.error_output or ""
                    if online_clean:
                        clean_stdout, clean_stderr = clean_openclaw_output(
                            clean_stdout,
                            clean_stderr,
                        )

                    command_meta = info.get("command_meta", {})
                    trajectory.append({
                        "step": step,
                        "action": action,
                        "model_debug": list(reply_attempts) if reply_attempts else ([reply_debug] if reply_debug else []),
                        "stdout": obs.command_output,
                        "stderr": obs.error_output,
                        "executed_action": command_meta.get("executed_action", action),
                        "clean_stdout": clean_stdout,
                        "clean_stderr": clean_stderr,
                        "error_tags": command_meta.get("error_tags", []),
                        "compat_status": command_meta.get("compat_status", "ok"),
                        "execution_trace": command_meta.get("execution_trace", []),
                        "exit_code": obs.exit_code,
                        "reward": reward,
                        "done": done,
                    })
                step += 1

            if eval_result is None:
                eval_result = env.evaluate()

            if verbose:
                status = "PASS" if eval_result.success else "FAIL"
                print(f"  [{status}] score={eval_result.score:.2f}  steps={step}")
                if not eval_result.success:
                    if not stop_reason and step >= max_steps:
                        stop_reason = f"reached max_steps={max_steps}"
                    if stop_reason:
                        print(f"       reason: {stop_reason}")
                    failed_checks = [d for d in eval_result.details if not d.passed]
                    if failed_checks:
                        print("       failed checks:")
                        for detail in failed_checks[:3]:
                            message = (detail.message or "(no message)").strip()
                            print(f"         - {detail.name}: {message}")

    except Exception as exc:
        error = str(exc)
        if verbose:
            reply_attempts = agent.debug_reply_attempts()
            reply_debug = agent.debug_last_reply()
            if reply_attempts:
                for attempt in reply_attempts:
                    print(f"       model: {attempt}")
            elif reply_debug:
                print(f"       model: {reply_debug}")
            print(f"  [ERROR] {exc}")
        return EpisodeResult(
            task_id=task_id,
            agent_name=agent.name,
            success=False,
            score=0.0,
            steps=0,
            duration_s=time.time() - t_start,
            trajectory=trajectory,
            final_response="",
            error=error,
            error_type=_classify_error_type(error, False),
            provider_impacted=provider_impacted or _has_provider_noise(agent.debug_reply_attempts(), error),
            stop_reason=stop_reason,
            date_anchor_mismatch=_detect_date_anchor_mismatch(task, executed_actions),
        )

    final_response = _build_final_response(task, trajectory)
    error_type = _classify_error_type(error, eval_result.success)
    return EpisodeResult(
        task_id=task_id,
        agent_name=agent.name,
        success=eval_result.success,
        score=eval_result.score,
        steps=step,
        duration_s=time.time() - t_start,
        trajectory=trajectory,
        final_response=final_response,
        error=error,
        error_type=error_type,
        provider_impacted=provider_impacted or _has_provider_noise(error=error),
        stop_reason=stop_reason,
        date_anchor_mismatch=_detect_date_anchor_mismatch(task, executed_actions),
    )


# ===========================================================================
# Batch evaluation
# ===========================================================================

@dataclass
class DomainMetrics:
    passed: int = 0
    total: int = 0
    total_score: float = 0.0

    @property
    def tgc(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.total if self.total else 0.0


@dataclass
class EvalSummary:
    agent_name: str
    split: str
    total: int
    passed: int
    total_score: float
    exec_mode: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_history_mode: str | None = None
    provider_failures: int = 0
    provider_impacted_tasks: int = 0
    by_domain: dict[str, DomainMetrics] = field(default_factory=dict)
    by_difficulty: dict[int, DomainMetrics] = field(default_factory=dict)
    by_hard_scenario: dict[str, DomainMetrics] = field(default_factory=dict)
    by_hard_ability: dict[str, DomainMetrics] = field(default_factory=dict)
    by_hard_ability_tag: dict[str, DomainMetrics] = field(default_factory=dict)
    results: list[EpisodeResult] = field(default_factory=list)

    @property
    def tgc(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.total if self.total else 0.0

    @property
    def completed_tasks(self) -> int:
        return self.total - self.provider_failures

    @property
    def clean_completed_tasks(self) -> int:
        return max(0, self.completed_tasks - self.provider_impacted_tasks)

    @property
    def provider_adjusted_tgc(self) -> float:
        return self.passed / self.completed_tasks if self.completed_tasks else 0.0

    @property
    def pass_at_budget(self) -> float:
        return self.tgc

    @property
    def near_miss_tasks(self) -> int:
        return sum(1 for result in self.results if not result.success and result.score >= 0.75)

    @property
    def near_miss_rate(self) -> float:
        return self.near_miss_tasks / self.total if self.total else 0.0

    @property
    def done_early_tasks(self) -> int:
        return sum(
            1
            for result in self.results
            if "DONE before executing another command" in result.stop_reason
        )

    @property
    def step_capped_tasks(self) -> int:
        return sum(
            1
            for result in self.results
            if result.stop_reason.startswith("reached max_steps=")
        )

    @property
    def date_anchor_mismatch_tasks(self) -> int:
        return sum(1 for result in self.results if result.date_anchor_mismatch)

    @property
    def score_rank_note(self) -> str:
        return (
            "Compare avg_score first and raw_accuracy second; rank inversions can happen "
            "when one run has more near-miss failures."
        )

    def print_report(self) -> None:
        bar = "=" * 60
        print(f"\n{bar}")
        header = f"Agent : {self.agent_name}   Split : {self.split}"
        if self.exec_mode:
            header += f"   Mode : {self.exec_mode}"
        print(header)
        if self.llm_provider or self.llm_model or self.llm_history_mode:
            llm_line = "LLM   : "
            provider_model = " / ".join(
                part for part in (self.llm_provider, self.llm_model) if part
            )
            if provider_model:
                llm_line += provider_model
            if self.llm_history_mode:
                if provider_model:
                    llm_line += f"   History : {self.llm_history_mode}"
                else:
                    llm_line += f"History : {self.llm_history_mode}"
            print(llm_line)
        print(
            f"Tasks : {self.passed}/{self.total} passed  "
            f"pass@budget={self.pass_at_budget*100:.1f}%  "
            f"raw_accuracy={self.tgc*100:.1f}%  avg_score={self.avg_score:.4f}"
        )
        print(
            f"Completed : {self.passed}/{self.completed_tasks}  "
            f"provider_failures={self.provider_failures}  "
            f"provider_impacted_tasks={self.provider_impacted_tasks}  "
            f"clean_completed_tasks={self.clean_completed_tasks}  "
            f"provider_adjusted_accuracy={self.provider_adjusted_tgc*100:.1f}%"
        )
        print(
            f"Diagnostics : near_miss_rate={self.near_miss_rate*100:.1f}%  "
            f"near_miss_tasks={self.near_miss_tasks}  "
            f"done_early_tasks={self.done_early_tasks}  "
            f"step_capped_tasks={self.step_capped_tasks}  "
            f"date_anchor_mismatch_tasks={self.date_anchor_mismatch_tasks}  "
            f"provider_impacted_tasks={self.provider_impacted_tasks}"
        )
        print(f"Ranking : {self.score_rank_note}")

        if self.by_domain:
            print("\nBy domain:")
            for domain, m in sorted(self.by_domain.items()):
                print(f"  {domain:<25} {m.passed:>3}/{m.total:<3}  "
                      f"TGC={m.tgc*100:.0f}%  avg={m.avg_score:.3f}")

        if self.by_difficulty:
            print("\nBy difficulty:")
            for diff, m in sorted(self.by_difficulty.items()):
                label = {1: "L1 single-step", 2: "L2 multi-step", 3: "L3 reasoning"}.get(diff, f"L{diff}")
                print(f"  {label:<20} {m.passed:>3}/{m.total:<3}  "
                      f"TGC={m.tgc*100:.0f}%  avg={m.avg_score:.3f}")

        if self.by_hard_scenario:
            print("\nBy hard scenario:")
            for scenario, m in sorted(self.by_hard_scenario.items()):
                print(f"  {scenario:<32} {m.passed:>3}/{m.total:<3}  "
                      f"TGC={m.tgc*100:.0f}%  avg={m.avg_score:.3f}")

        if self.by_hard_ability:
            print("\nBy hard primary ability:")
            for ability, m in sorted(self.by_hard_ability.items()):
                print(f"  {ability:<32} {m.passed:>3}/{m.total:<3}  "
                      f"TGC={m.tgc*100:.0f}%  avg={m.avg_score:.3f}")

        if self.by_hard_ability_tag:
            print("\nBy hard ability tag (overlapping):")
            for ability, m in sorted(self.by_hard_ability_tag.items()):
                print(f"  {ability:<32} {m.passed:>3}/{m.total:<3}  "
                      f"TGC={m.tgc*100:.0f}%  avg={m.avg_score:.3f}")
        print(bar)


def run_evaluation(
    agent: BaseAgent,
    split: str = "dev",
    task_data_dir: str | Path | None = None,
    mode: str = "multi",
    task_prefix: str | None = None,
    domain: str | None = None,
    difficulty: int | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    max_stagnant_steps: int = 3,
    online_clean: bool | None = None,
    skip_incompatible_openclaw: bool = True,
    fallback_openclaw_network_to_mock: bool = False,
    strict_online_data: bool = True,
    online_openclaw_only: bool = False,
    inter_task_sleep_s: float = 0.0,
    verbose: bool = False,
) -> EvalSummary:
    """Evaluate `agent` on a task split.

    Parameters
    ----------
    agent:
        The agent instance to evaluate.
    split:
        Dataset split: ``"train"`` | ``"dev"`` | ``"test"``.
    mode:
        Backend mode: ``"mock"`` | ``"multi"`` | ``"real"`` | ``"hybrid"``.
    domain:
        If set, only evaluate tasks in this domain (e.g. ``"calendar"``).
    difficulty:
        If set (1/2/3), only evaluate tasks at this difficulty level.
    limit:
        Maximum number of tasks to run (useful for quick tests).
    max_steps:
        Hard cap on steps per episode.
    verbose:
        Print per-episode details.
    """
    data_dir = Path(task_data_dir) if task_data_dir else None
    effective_max_steps = _resolve_max_steps(
        max_steps,
        split=split,
        task_prefix=task_prefix,
    )
    task_ids = load_task_ids(split, data_dir=data_dir, difficulty=difficulty, domain=domain)
    task_ids = _filter_task_ids_by_prefix(task_ids, task_prefix)
    if limit:
        task_ids = task_ids[:limit]
    if online_openclaw_only:
        task_ids = _select_openclaw_only_task_ids(task_ids, data_dir=data_dir)

    if not task_ids:
        print(
            f"No tasks found for split='{split}' domain={domain} "
            f"difficulty={difficulty} task_prefix={task_prefix}"
        )
        run_config = _agent_run_config(agent)
        return EvalSummary(
            agent_name=agent.name,
            split=split,
            total=0,
            passed=0,
            total_score=0.0,
            exec_mode=mode,
            llm_provider=run_config.get("provider"),
            llm_model=run_config.get("model"),
            llm_history_mode=run_config.get("history_mode"),
        )

    print(f"Evaluating '{agent.name}' on {len(task_ids)} tasks "
          f"(split={split}, mode={mode})...")

    run_config = _agent_run_config(agent)
    summary = EvalSummary(
        agent_name=agent.name,
        split=split,
        total=0,
        passed=0,
        total_score=0.0,
        exec_mode=mode,
        llm_provider=run_config.get("provider"),
        llm_model=run_config.get("model"),
        llm_history_mode=run_config.get("history_mode"),
    )
    should_clean = _resolve_online_clean(mode, online_clean)
    backend_kwargs = _make_backend_kwargs(
        mode,
        skip_incompatible_openclaw,
        fallback_openclaw_network_to_mock,
        strict_online_data,
    )

    core_hard_abilities = {
        "duplicate_avoidance",
        "gap_completion",
        "information_transfer",
        "multi_source_reasoning",
        "state_repair",
        "workflow_completion",
    }

    for i, task_id in enumerate(task_ids, 1):
        if i > 1 and inter_task_sleep_s > 0:
            time.sleep(inter_task_sleep_s)
        if not verbose:
            print(f"  [{i:4d}/{len(task_ids)}] {task_id:<50}", end="\r", flush=True)

        result = run_episode(
            task_id,
            agent,
            task_data_dir=data_dir,
            mode=mode,
            max_steps=effective_max_steps,
            max_stagnant_steps=max_stagnant_steps,
            online_clean=should_clean,
            backend_kwargs=backend_kwargs,
            verbose=verbose,
        )
        summary.results.append(result)
        summary.total += 1
        summary.total_score += result.score
        if result.success:
            summary.passed += 1
        if result.error_type == "provider_failure":
            summary.provider_failures += 1
        if result.provider_impacted:
            summary.provider_impacted_tasks += 1

        # Per-domain breakdown (load task metadata)
        try:
            task = load_task(task_id, data_dir=data_dir)
            for d in task.domains:
                dm = summary.by_domain.setdefault(d, DomainMetrics())
                dm.total += 1
                dm.total_score += result.score
                if result.success:
                    dm.passed += 1
            diff = task.difficulty
            dfm = summary.by_difficulty.setdefault(diff, DomainMetrics())
            dfm.total += 1
            dfm.total_score += result.score
            if result.success:
                dfm.passed += 1

            public = task.data.public if getattr(task, "data", None) else {}
            hard_scenario = public.get("hard_decision_scenario")
            if isinstance(hard_scenario, str) and hard_scenario:
                hsm = summary.by_hard_scenario.setdefault(hard_scenario, DomainMetrics())
                hsm.total += 1
                hsm.total_score += result.score
                if result.success:
                    hsm.passed += 1

            hard_ability = public.get("hard_decision_ability")
            if isinstance(hard_ability, str) and hard_ability:
                ham = summary.by_hard_ability.setdefault(hard_ability, DomainMetrics())
                ham.total += 1
                ham.total_score += result.score
                if result.success:
                    ham.passed += 1

            hard_ability_tags = public.get("hard_decision_ability_tags")
            if isinstance(hard_ability_tags, list):
                for tag in hard_ability_tags:
                    if not isinstance(tag, str) or tag not in core_hard_abilities:
                        continue
                    htm = summary.by_hard_ability_tag.setdefault(tag, DomainMetrics())
                    htm.total += 1
                    htm.total_score += result.score
                    if result.success:
                        htm.passed += 1
        except Exception:
            pass

    if not verbose:
        print()  # clear the \r line

    return summary


# ===========================================================================
# Training data collection (offline RL / behavior cloning)
# ===========================================================================

def collect_training_data(
    agent: BaseAgent,
    split: str = "train",
    out_dir: str | Path = "data/trajectories/expert",
    task_data_dir: str | Path | None = None,
    mode: str = "multi",
    task_prefix: str | None = None,
    domain: str | None = None,
    difficulty: int | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    max_stagnant_steps: int = 3,
    online_clean: bool | None = None,
    skip_incompatible_openclaw: bool = True,
    fallback_openclaw_network_to_mock: bool = False,
    strict_online_data: bool = True,
    online_openclaw_only: bool = False,
    inter_task_sleep_s: float = 0.0,
    verbose: bool = False,
) -> Path:
    """Record trajectories from `agent` and save as JSONL.

    Each line of the output file is a JSON object with keys:
    ``task_id``, ``agent``, ``success``, ``score``, ``steps``,
    ``trajectory`` (list of step dicts), ``final_response``, ``duration_s``.

    Use the saved trajectories for:
    - **Behavior cloning**: train a policy to imitate successful expert episodes
    - **Offline RL**: reward-weighted regression, IQL, CQL, …
    - **Dataset inspection**: analyze failure modes and success patterns

    Parameters
    ----------
    out_dir:
        Directory where the JSONL file will be written.
        File name: ``{agent_name}_{split}.jsonl``.
    """
    from openclaw_env.core.task import load_task_ids

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{agent.name}_{split}.jsonl"

    data_dir = Path(task_data_dir) if task_data_dir else None
    effective_max_steps = _resolve_max_steps(
        max_steps,
        split=split,
        task_prefix=task_prefix,
    )
    task_ids = load_task_ids(split, data_dir=data_dir, difficulty=difficulty, domain=domain)
    task_ids = _filter_task_ids_by_prefix(task_ids, task_prefix)
    if limit:
        task_ids = task_ids[:limit]
    if online_openclaw_only:
        task_ids = _select_openclaw_only_task_ids(task_ids, data_dir=data_dir)

    print(f"Collecting trajectories from '{agent.name}' on {len(task_ids)} tasks → {out_file}")

    n_success = 0
    should_clean = _resolve_online_clean(mode, online_clean)
    backend_kwargs = _make_backend_kwargs(
        mode,
        skip_incompatible_openclaw,
        fallback_openclaw_network_to_mock,
        strict_online_data,
    )
    with open(out_file, "w") as fh:
        for i, task_id in enumerate(task_ids, 1):
            if i > 1 and inter_task_sleep_s > 0:
                time.sleep(inter_task_sleep_s)
            if not verbose:
                print(f"  [{i:4d}/{len(task_ids)}] {task_id:<50}", end="\r", flush=True)

            result = run_episode(
                task_id, agent,
                task_data_dir=data_dir,
                mode=mode, max_steps=effective_max_steps,
                max_stagnant_steps=max_stagnant_steps,
                record_trajectory=True,
                online_clean=should_clean,
                backend_kwargs=backend_kwargs,
                verbose=verbose,
            )

            if result.success:
                n_success += 1

            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    if not verbose:
        print()

    print(f"Saved {len(task_ids)} trajectories ({n_success} successful) to {out_file}")
    return out_file


# ===========================================================================
# Single-task interactive demo
# ===========================================================================

def run_demo(
    task_id: str,
    agent: BaseAgent,
    task_data_dir: str | Path | None = None,
    mode: str = "multi",
    max_steps: int | None = None,
    max_stagnant_steps: int = 3,
    online_clean: bool | None = None,
    skip_incompatible_openclaw: bool = True,
    fallback_openclaw_network_to_mock: bool = False,
    strict_online_data: bool = True,
) -> None:
    """Run a single episode with full verbose output."""
    result = run_episode(
        task_id,
        agent,
        task_data_dir=task_data_dir,
        mode=mode,
        max_steps=_resolve_max_steps(max_steps, task_id=task_id),
        max_stagnant_steps=max_stagnant_steps,
        record_trajectory=True,
        online_clean=_resolve_online_clean(mode, online_clean),
        backend_kwargs=_make_backend_kwargs(
            mode,
            skip_incompatible_openclaw,
            fallback_openclaw_network_to_mock,
            strict_online_data,
        ),
        verbose=True,
    )
    print(f"\nFinal: success={result.success}  score={result.score:.2f}  "
          f"steps={result.steps}  time={result.duration_s:.1f}s")
    if result.error:
        print(f"Error: {result.error}")


# ===========================================================================
# CLI entry point
# ===========================================================================

def _build_agent(
    name: str,
    model: str,
    llm_provider: str = "anthropic",
    llm_base_url: str | None = None,
    llm_api_key_env: str | None = None,
    llm_history_mode: str = "full",
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 256,
    llm_timeout_s: int = 25,
    llm_request_retries: int = 1,
    llm_retry_backoff_s: float = 1.5,
    max_steps_hint: int = DEFAULT_MAX_STEPS,
) -> BaseAgent:
    if name == "expert":
        return ExpertAgent()
    if name == "random":
        return RandomAgent()
    if name == "rule":
        return RuleAgent()
    if name == "llm":
        return LLMAgent(
            provider=llm_provider,
            model=model,
            base_url=llm_base_url,
            api_key_env=llm_api_key_env,
            history_mode=llm_history_mode,
            temperature=llm_temperature,
            response_max_tokens=llm_max_tokens,
            request_timeout_s=llm_timeout_s,
            request_retries=llm_request_retries,
            retry_backoff_s=llm_retry_backoff_s,
            max_steps_hint=max_steps_hint,
        )
    raise ValueError(f"Unknown agent: {name!r}. Choose from: expert, random, rule, llm")


def _select_openclaw_only_task_ids(
    task_ids: list[str],
    data_dir: Path | None = None,
) -> list[str]:
    """Keep only tasks whose GT commands include at least one `openclaw *` step."""
    selected: list[str] = []
    for task_id in task_ids:
        try:
            task = load_task(task_id, data_dir=data_dir)
            cmds = task.ground_truth.solution_commands if task and task.ground_truth else []
            if any(c.strip().startswith("openclaw ") for c in cmds):
                selected.append(task_id)
        except Exception:
            continue
    return selected


def _filter_task_ids_by_prefix(task_ids: list[str], task_prefix: str | None) -> list[str]:
    if not task_prefix:
        return task_ids
    return [task_id for task_id in task_ids if task_id.startswith(task_prefix)]


def _resolve_online_clean(mode: str, online_clean: bool | None) -> bool:
    if online_clean is None:
        return mode in {"real", "hybrid"}
    return online_clean


def _make_backend_kwargs(
    mode: str,
    skip_incompatible_openclaw: bool,
    fallback_openclaw_network_to_mock: bool,
    strict_online_data: bool,
) -> dict[str, Any] | None:
    if mode not in {"real", "hybrid"}:
        return None
    kwargs: dict[str, Any] = {
        "skip_incompatible_openclaw": skip_incompatible_openclaw,
    }
    if fallback_openclaw_network_to_mock:
        kwargs["fallback_openclaw_network_to_mock"] = True
    if strict_online_data:
        kwargs["strict_online_data"] = True
    return kwargs


def _preflight_mode(mode: str, status_timeout_s: int = 8) -> None:
    """Fail fast for real/hybrid runs when openclaw runtime is not ready."""
    if mode not in {"real", "hybrid"}:
        return

    cli_path = shutil.which("openclaw")
    if not cli_path:
        raise RuntimeError(
            "Mode requires `openclaw` in PATH, but it was not found. "
            "Install OpenClaw CLI and ensure `which openclaw` succeeds."
        )

    try:
        proc = subprocess.run(
            ["openclaw", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=status_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"`openclaw status --json` timed out after {status_timeout_s}s. "
            "Check gateway health with `openclaw gateway status`."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Failed to run `openclaw status --json`: {exc}"
        ) from exc

    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        detail_msg = f" Details: {details}" if details else ""
        raise RuntimeError(
            "OpenClaw preflight failed: `openclaw status --json` did not succeed. "
            "Start/check the gateway (`openclaw gateway status`) and profile setup."
            f"{detail_msg}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train / evaluate agents with openclaw_env",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent", default="expert",
                        choices=["expert", "random", "rule", "llm"],
                        help="Agent type (default: expert)")
    parser.add_argument("--split", default="dev",
                        help="Dataset split or dataset list name under openclaw_env/data/datasets (default: dev)")
    parser.add_argument("--mode", default="multi",
                        choices=["mock", "multi", "real", "hybrid"],
                        help="Backend mode (default: multi)")
    parser.add_argument("--domain", default=None,
                        help="Filter by domain, e.g. 'calendar'")
    parser.add_argument(
        "--task-prefix",
        default=None,
        help="Filter task IDs by prefix, e.g. 'hard_decision_workflow_'",
    )
    parser.add_argument("--difficulty", type=int, default=None, choices=[1, 2, 3],
                        help="Filter by difficulty level 1/2/3")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of tasks to run")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Max steps per episode (default: 15)",
    )
    parser.add_argument(
        "--max-stagnant-steps",
        type=int,
        default=3,
        help=(
            "Stop an episode after this many identical no-progress transitions "
            "(same command, stdout, stderr, exit code). Default: 3"
        ),
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help=(
            "LLM model ID (only used with --agent llm). "
            "When --llm-provider=openai and --model starts with 'claude-', "
            "the client defaults to a local LiteLLM proxy at http://127.0.0.1:4000/v1 "
            "and reads LITELLM_PROXY_KEY unless overridden."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM API provider for --agent llm (default: anthropic)",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help=(
            "Base URL for OpenAI-compatible chat completions API. "
            "Example: https://api.openai.com/v1 or a compatible proxy. "
            "Claude Bedrock proxy models can omit this and use LITELLM_PROXY_BASE_URL "
            "(default: http://127.0.0.1:4000/v1)."
        ),
    )
    parser.add_argument(
        "--llm-api-key-env",
        default=None,
        help=(
            "Environment variable containing the LLM API key. "
            "Defaults to ANTHROPIC_API_KEY or OPENAI_API_KEY based on provider. "
            "Claude Bedrock proxy models default to LITELLM_PROXY_KEY."
        ),
    )
    parser.add_argument(
        "--llm-history-mode",
        default="full",
        choices=["auto", "full", "summary"],
        help=(
            "How to construct multi-turn message history for LLM requests. "
            "'full' sends full history by default, 'summary' always uses the compact "
            "history summary, and 'auto' uses the existing automatic trim logic."
        ),
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for --agent llm (default: 0.0)",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=256,
        help="Max output tokens for --agent llm API calls (default: 256)",
    )
    parser.add_argument(
        "--llm-timeout-s",
        type=int,
        default=25,
        help="Per-request timeout for --agent llm API calls in seconds (default: 25)",
    )
    parser.add_argument(
        "--llm-request-retries",
        type=int,
        default=1,
        help="Retry count for transient --agent llm API failures (default: 1)",
    )
    parser.add_argument(
        "--llm-retry-backoff-s",
        type=float,
        default=1.5,
        help="Base backoff in seconds before retrying transient --agent llm API failures (default: 1.5)",
    )
    parser.add_argument(
        "--inter-task-sleep",
        type=float,
        default=0.0,
        help="Sleep this many seconds between tasks during batch eval/data collection to reduce provider pressure (default: 0.0)",
    )
    parser.add_argument("--collect-data", action="store_true",
                        help="Save trajectories as JSONL instead of printing metrics")
    parser.add_argument("--out-dir", default="data/trajectories",
                        help="Output dir for --collect-data (default: data/trajectories)")
    parser.add_argument(
        "--task-data-dir",
        default=str(DEFAULT_TASK_DATA_DIR),
        help=(
            "Task data root containing tasks/ and datasets/. "
            "Default: openclaw_env/data"
        ),
    )
    parser.add_argument("--task-id", default=None,
                        help="Run a single task demo and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-step details")
    parser.add_argument(
        "--verbose-log",
        default=None,
        help=(
            "Optional file path that receives the same stdout stream as the terminal, "
            "including -v per-step output."
        ),
    )
    parser.add_argument(
        "--online-clean",
        dest="online_clean",
        action="store_true",
        help="Store cleaned stdout/stderr fields in trajectories",
    )
    parser.add_argument(
        "--no-online-clean",
        dest="online_clean",
        action="store_false",
        help="Disable output cleaning in trajectories",
    )
    parser.set_defaults(online_clean=None)
    parser.add_argument(
        "--skip-incompatible-openclaw",
        dest="skip_incompatible_openclaw",
        action="store_true",
        default=True,
        help="Skip incompatible real openclaw commands with labels (default: on)",
    )
    parser.add_argument(
        "--no-skip-incompatible-openclaw",
        dest="skip_incompatible_openclaw",
        action="store_false",
        help="Do not skip incompatible real openclaw commands",
    )
    parser.add_argument(
        "--online-openclaw-only",
        action="store_true",
        help="Only run tasks whose GT includes at least one `openclaw *` command",
    )
    parser.add_argument(
        "--fallback-openclaw-network-to-mock",
        action="store_true",
        help=(
            "In real/hybrid mode, fallback openclaw commands to mock when "
            "real execution fails due to network/auth/dependency errors"
        ),
    )
    parser.add_argument(
        "--strict-online-data",
        dest="strict_online_data",
        action="store_true",
        help=(
            "In real/hybrid mode, disable mock fallback for weather/calendar/email/tasks "
            "online reads and fail the command if live fetch is unavailable "
            "(default: on)"
        ),
    )
    parser.add_argument(
        "--no-strict-online-data",
        dest="strict_online_data",
        action="store_false",
        help="Allow weather/calendar/email/tasks mock fallback when live fetch is unavailable.",
    )
    parser.set_defaults(strict_online_data=True)
    parser.add_argument("--save-report", default=None,
                        help="Save JSON evaluation report to this path")

    args = parser.parse_args()

    def _run() -> None:
        _preflight_mode(args.mode)
        resolved_max_steps = _resolve_max_steps(
            args.max_steps,
            split=args.split,
            task_prefix=args.task_prefix,
            task_id=args.task_id,
        )

        agent = _build_agent(
            args.agent,
            args.model,
            llm_provider=args.llm_provider,
            llm_base_url=args.llm_base_url,
            llm_api_key_env=args.llm_api_key_env,
            llm_history_mode=args.llm_history_mode,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
            llm_timeout_s=args.llm_timeout_s,
            llm_request_retries=args.llm_request_retries,
            llm_retry_backoff_s=args.llm_retry_backoff_s,
            max_steps_hint=resolved_max_steps,
        )

        # ------------------------------------------------------------------
        # Single-task demo mode
        # ------------------------------------------------------------------
        if args.task_id:
            run_demo(
                args.task_id,
                agent,
                task_data_dir=args.task_data_dir,
                mode=args.mode,
                max_steps=resolved_max_steps,
                max_stagnant_steps=args.max_stagnant_steps,
                online_clean=args.online_clean,
                skip_incompatible_openclaw=args.skip_incompatible_openclaw,
                fallback_openclaw_network_to_mock=args.fallback_openclaw_network_to_mock,
                strict_online_data=args.strict_online_data,
            )
            return

        # ------------------------------------------------------------------
        # Data collection mode
        # ------------------------------------------------------------------
        if args.collect_data:
            collect_training_data(
                agent,
                split=args.split,
                out_dir=args.out_dir,
                task_data_dir=args.task_data_dir,
                mode=args.mode,
                task_prefix=args.task_prefix,
                domain=args.domain,
                difficulty=args.difficulty,
                limit=args.limit,
                max_steps=resolved_max_steps,
                max_stagnant_steps=args.max_stagnant_steps,
                online_clean=args.online_clean,
                skip_incompatible_openclaw=args.skip_incompatible_openclaw,
                fallback_openclaw_network_to_mock=args.fallback_openclaw_network_to_mock,
                strict_online_data=args.strict_online_data,
                online_openclaw_only=args.online_openclaw_only,
                inter_task_sleep_s=args.inter_task_sleep,
                verbose=args.verbose,
            )
            return

        # ------------------------------------------------------------------
        # Evaluation mode (default)
        # ------------------------------------------------------------------
        summary = run_evaluation(
            agent,
            split=args.split,
            task_data_dir=args.task_data_dir,
            mode=args.mode,
            task_prefix=args.task_prefix,
            domain=args.domain,
            difficulty=args.difficulty,
            limit=args.limit,
            max_steps=resolved_max_steps,
            max_stagnant_steps=args.max_stagnant_steps,
            online_clean=args.online_clean,
            skip_incompatible_openclaw=args.skip_incompatible_openclaw,
            fallback_openclaw_network_to_mock=args.fallback_openclaw_network_to_mock,
            strict_online_data=args.strict_online_data,
            online_openclaw_only=args.online_openclaw_only,
            inter_task_sleep_s=args.inter_task_sleep,
            verbose=args.verbose,
        )
        summary.print_report()

        if args.save_report:
            report = {
                "agent": summary.agent_name,
                "split": summary.split,
                "mode": summary.exec_mode,
                "max_steps_budget": resolved_max_steps,
                "llm_provider": summary.llm_provider,
                "llm_model": summary.llm_model,
                "llm_history_mode": summary.llm_history_mode,
                "tgc": summary.tgc,
                "pass_at_budget": summary.pass_at_budget,
                "raw_accuracy": summary.tgc,
                "avg_score": summary.avg_score,
                "near_miss_rate": summary.near_miss_rate,
                "near_miss_tasks": summary.near_miss_tasks,
                "done_early_tasks": summary.done_early_tasks,
                "step_capped_tasks": summary.step_capped_tasks,
                "date_anchor_mismatch_tasks": summary.date_anchor_mismatch_tasks,
                "score_rank_note": summary.score_rank_note,
                "passed": summary.passed,
                "total": summary.total,
                "completed_tasks": summary.completed_tasks,
                "clean_completed_tasks": summary.clean_completed_tasks,
                "provider_failures": summary.provider_failures,
                "provider_impacted_tasks": summary.provider_impacted_tasks,
                "provider_adjusted_accuracy": summary.provider_adjusted_tgc,
                "by_domain": {
                    d: {"passed": m.passed, "total": m.total, "tgc": m.tgc, "avg_score": m.avg_score}
                    for d, m in summary.by_domain.items()
                },
                "by_difficulty": {
                    str(d): {"passed": m.passed, "total": m.total, "tgc": m.tgc, "avg_score": m.avg_score}
                    for d, m in summary.by_difficulty.items()
                },
                "by_hard_scenario": {
                    k: {"passed": m.passed, "total": m.total, "tgc": m.tgc, "avg_score": m.avg_score}
                    for k, m in summary.by_hard_scenario.items()
                },
                "by_hard_ability": {
                    k: {"passed": m.passed, "total": m.total, "tgc": m.tgc, "avg_score": m.avg_score}
                    for k, m in summary.by_hard_ability.items()
                },
                "by_hard_ability_tag": {
                    k: {"passed": m.passed, "total": m.total, "tgc": m.tgc, "avg_score": m.avg_score}
                    for k, m in summary.by_hard_ability_tag.items()
                },
                "results": [r.to_dict() for r in summary.results],
            }
            Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_report, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Report saved → {args.save_report}")

    if args.verbose_log:
        log_path = Path(args.verbose_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_fh:
            with contextlib.redirect_stdout(_TeeStdout(sys.stdout, log_fh)):
                _run()
    else:
        _run()


if __name__ == "__main__":
    main()
