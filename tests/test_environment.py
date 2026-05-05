"""Integration tests for OpenClawEnv."""

from __future__ import annotations

import pytest

from openclaw_env.core.environment import OpenClawEnv
from openclaw_env.core.task import load_task_ids


# Use a few concrete task IDs that are always generated
_TASK_IDS = [
    "agent_create_1",
    "msg_send_text_1",
    "check_system_status_1",
    "setup_workspace_1",
]


class TestEnvironmentBasic:
    def test_reset_returns_observation(self):
        with OpenClawEnv(task_id="agent_create_1", backend="mock") as env:
            obs = env.reset()
            assert obs.task_instruction != ""
            assert obs.step_number == 0

    def test_step_increments_count(self):
        with OpenClawEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            obs, reward, done, info = env.step("openclaw status")
            assert obs.step_number == 1
            assert info["step"] == 1

    def test_done_after_done_signal(self):
        with OpenClawEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            _, _, done, _ = env.step("done")
            assert done is True

    def test_context_manager(self):
        with OpenClawEnv(task_id="msg_send_text_1", backend="mock") as env:
            obs = env.reset()
            assert env.task is not None

    def test_step_requires_reset(self):
        env = OpenClawEnv(task_id="agent_create_1", backend="mock")
        with pytest.raises(RuntimeError, match="reset"):
            env.step("openclaw status")

    def test_step_after_done_raises(self):
        with OpenClawEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            env.step("done")
            with pytest.raises(RuntimeError, match="done"):
                env.step("openclaw status")


class TestExpertAgent:
    """Test that expert solutions achieve full scores."""

    @pytest.mark.parametrize("task_id", _TASK_IDS)
    def test_expert_solution_passes(self, task_id):
        with OpenClawEnv(task_id=task_id, backend="mock") as env:
            obs = env.reset()
            task = env.task
            assert task is not None
            assert task.ground_truth is not None

            for command in task.ground_truth.solution_commands:
                obs, reward, done, info = env.step(command)
                if done:
                    break

            result = env.evaluate()
            assert result.success is True, (
                f"Expert solution failed for {task_id}: "
                f"{[(d.name, d.message) for d in result.details if not d.passed]}"
            )
            assert result.score == pytest.approx(1.0)

    def test_agent_create_passes(self):
        with OpenClawEnv(task_id="agent_create_1", backend="mock") as env:
            obs = env.reset()
            task = env.task
            for cmd in task.ground_truth.solution_commands:
                env.step(cmd)
            result = env.evaluate()
            assert result.success is True

    def test_msg_send_passes(self):
        with OpenClawEnv(task_id="msg_send_text_1", backend="mock") as env:
            obs = env.reset()
            task = env.task
            for cmd in task.ground_truth.solution_commands:
                env.step(cmd)
            result = env.evaluate()
            assert result.success is True

    def test_wrong_action_fails(self):
        """Wrong action should fail evaluation."""
        with OpenClawEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            # Do something unrelated instead
            env.step("openclaw status")
            result = env.evaluate()
            assert result.success is False


class TestSafetyGuard:
    def test_dangerous_command_blocked(self):
        with OpenClawEnv(
            task_id="agent_create_1",
            backend="mock",
            raise_on_safety_violation=True,
        ) as env:
            env.reset()
            obs, reward, done, info = env.step("rm -rf /")
            assert info.get("safety_violation") is True
            assert done is True

    def test_safe_command_allowed(self):
        with OpenClawEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            obs, reward, done, info = env.step("openclaw status")
            assert info.get("safety_violation") is None


class TestTrajectory:
    def test_trajectory_recorded(self):
        with OpenClawEnv(
            task_id="agent_create_1", backend="mock", record_trajectory=True
        ) as env:
            env.reset()
            env.step("openclaw agents add --name alice --model openai/gpt-4o")
            env.evaluate()

            traj = env.trajectory
            assert traj is not None
            assert len(traj.steps) == 1
            assert traj.steps[0].action == "openclaw agents add --name alice --model openai/gpt-4o"

    def test_trajectory_dict(self):
        with OpenClawEnv(
            task_id="agent_create_1", backend="mock", record_trajectory=True
        ) as env:
            env.reset()
            env.step("openclaw status")

            traj = env.trajectory
            d = traj.to_dict()
            assert "task_id" in d
            assert "trajectory" in d
            assert len(d["trajectory"]) == 1


class TestStateSnapshot:
    def test_save_and_restore(self):
        with OpenClawEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            snapshot_id = env.save_state()
            assert snapshot_id is not None

            env.step("openclaw agents add --name temp --model openai/gpt-4o")
            env.load_state(snapshot_id)
            # After restore, backend is re-initialized from snapshot


class TestMaxSteps:
    def test_truncated_at_max_steps(self):
        with OpenClawEnv(
            task_id="agent_create_1", backend="mock", max_steps=3
        ) as env:
            env.reset()
            for _ in range(3):
                obs, reward, done, info = env.step("openclaw status")
                if done:
                    break
            assert done is True
            assert info.get("truncated") is True
