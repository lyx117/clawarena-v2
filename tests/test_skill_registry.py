from __future__ import annotations

import pytest

from openclaw_env.backend.base import CommandResult
from openclaw_env.skills.base import Skill
from openclaw_env.skills.registry import SkillRegistry


class _NoopSkill(Skill):
    def __init__(self, prefixes: tuple[str, ...]) -> None:
        super().__init__(prefixes=prefixes)

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        pass

    def execute(self, command: str) -> CommandResult:
        return CommandResult(stdout=command, stderr="", exit_code=0)

    def cleanup(self) -> None:
        pass


def test_register_and_resolve_prefix():
    registry = SkillRegistry()
    skill = _NoopSkill(("weather",))
    registry.register("weather", skill)
    assert registry.resolve("weather") is skill


def test_register_skill_registers_all_prefixes_once():
    registry = SkillRegistry()
    skill = _NoopSkill(("calendar", "gcalcli"))
    registry.register_skill(skill)
    assert registry.resolve("calendar") is skill
    assert registry.resolve("gcalcli") is skill
    assert len(registry.iter_skills()) == 1


def test_register_duplicate_prefix_raises():
    registry = SkillRegistry()
    registry.register("weather", _NoopSkill(("weather",)))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("weather", _NoopSkill(("weather",)))


def test_list_prefixes_sorted():
    registry = SkillRegistry()
    registry.register("tasks", _NoopSkill(("tasks",)))
    registry.register("calendar", _NoopSkill(("calendar",)))
    assert registry.list_prefixes() == ["calendar", "tasks"]
