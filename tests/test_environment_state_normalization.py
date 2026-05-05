from __future__ import annotations

from openclaw_env.core.environment import _as_named_mapping, _merge_named_entities


def test_as_named_mapping_handles_list_by_name_or_id():
    raw = [
        {"id": "main", "model": "vllm/Qwen"},
        {"name": "researcher", "model": "anthropic/claude-opus-4-6"},
    ]
    mapping = _as_named_mapping(raw)
    assert "main" in mapping
    assert "researcher" in mapping
    assert mapping["researcher"]["model"] == "anthropic/claude-opus-4-6"


def test_merge_named_entities_backfills_state():
    state_agents = {}
    effects = [{"name": "researcher", "model": "anthropic/claude-opus-4-6"}]
    _merge_named_entities(state_agents, effects)
    assert state_agents["researcher"]["model"] == "anthropic/claude-opus-4-6"


def test_as_named_mapping_handles_wrapped_dict_with_list_values():
    raw = {
        "channels": [
            {"id": "telegram", "status": "connected"},
            {"name": "discord", "status": "disconnected"},
        ]
    }
    mapping = _as_named_mapping(raw)
    assert mapping["telegram"]["status"] == "connected"
    assert mapping["discord"]["status"] == "disconnected"
