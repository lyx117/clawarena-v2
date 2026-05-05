from __future__ import annotations

from openclaw_env.backend.base import CommandResult
from openclaw_env.skills.base import Skill
from openclaw_env.skills.registry import SkillRegistry
from openclaw_env.skills.runtime import SkillRuntime


class _CountingSkill(Skill):
    def __init__(self, prefixes: tuple[str, ...], name: str) -> None:
        super().__init__(prefixes=prefixes)
        self.name = name
        self.init_calls = 0

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        self.init_calls += 1

    def execute(self, command: str) -> CommandResult:
        return CommandResult(stdout=f"{self.name}:{command}", stderr="", exit_code=0)

    def get_state(self) -> dict[str, object]:
        return {self.name: True}

    def cleanup(self) -> None:
        pass


def test_runtime_dispatches_by_prefix():
    registry = SkillRegistry()
    weather = _CountingSkill(("weather",), "weather")
    registry.register_skill(weather)
    rt = SkillRuntime(registry)
    rt.initialize("/tmp", {})
    result = rt.execute_cli("weather get --location 'NYC'")
    assert result.exit_code == 0
    assert result.stdout.startswith("weather:weather get")


def test_runtime_unknown_prefix_returns_error_with_available_list():
    registry = SkillRegistry()
    registry.register_skill(_CountingSkill(("calendar",), "calendar"))
    rt = SkillRuntime(registry)
    result = rt.execute_cli("unknown do")
    assert result.exit_code == 1
    assert "Unknown command prefix 'unknown'" in result.stderr
    assert "calendar" in result.stderr


def test_runtime_initializes_shared_skill_once_for_multiple_prefixes():
    registry = SkillRegistry()
    cal = _CountingSkill(("calendar", "gcalcli"), "calendar")
    registry.register_skill(cal)
    rt = SkillRuntime(registry)
    rt.initialize("/tmp", {})
    assert cal.init_calls == 1
    state = rt.get_state()
    assert state == {"calendar": True}
