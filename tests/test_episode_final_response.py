from __future__ import annotations

from examples.train_and_eval import ExpertAgent, run_episode


def test_weather_episode_generates_final_response():
    result = run_episode(
        task_id="weather_get_online_1",
        agent=ExpertAgent(),
        mode="multi",
        max_steps=10,
        record_trajectory=True,
    )
    assert result.success is True
    assert result.final_response.startswith("WEATHER_RESULT:")
    payload = result.to_dict()
    assert "final_response" in payload
    assert payload["final_response"] == result.final_response


def test_calendar_episode_generates_final_response():
    result = run_episode(
        task_id="calendar_today_online_1",
        agent=ExpertAgent(),
        mode="multi",
        max_steps=10,
        record_trajectory=True,
    )
    assert result.steps >= 1
    assert result.final_response.startswith("CALENDAR_RESULT:")
