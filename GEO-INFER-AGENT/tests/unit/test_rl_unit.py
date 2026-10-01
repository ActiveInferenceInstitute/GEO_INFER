#!/usr/bin/env python3

"""
Unit tests for the reinforcement-learning module: Experience, QTable,
ReplayBuffer, RLState, and RLAgent action handlers / persistence.
"""

import asyncio
import json
import os
import tempfile
import unittest

import numpy as np

from geo_infer_agent.models.rl import (
    Experience,
    QTable,
    RLAgent,
    RLState,
    ReplayBuffer,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestExperience(unittest.TestCase):
    """Tests for Experience serialization."""

    def test_scalar_state_roundtrip(self) -> None:
        exp = Experience(state=3, action=1, reward=0.5, next_state=4, done=False)
        d = exp.to_dict()
        restored = Experience.from_dict(d)
        self.assertEqual(restored.state, 3)
        self.assertEqual(restored.action, 1)
        self.assertEqual(restored.reward, 0.5)
        self.assertEqual(restored.next_state, 4)
        self.assertFalse(restored.done)

    def test_array_state_roundtrip(self) -> None:
        exp = Experience(
            state=np.array([1.0, 2.0]),
            action=0,
            reward=-0.1,
            next_state=np.array([2.0, 3.0]),
            done=True,
        )
        d = exp.to_dict()
        self.assertEqual(d["state"], [1.0, 2.0])
        restored = Experience.from_dict(d)
        np.testing.assert_array_equal(restored.state, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(restored.next_state, np.array([2.0, 3.0]))
        self.assertTrue(restored.done)


class TestQTable(unittest.TestCase):
    """Tests for QTable value access, bounds, and serialization."""

    def test_default_and_update_value(self) -> None:
        table = QTable(state_size=3, action_size=2, default_value=0.25)
        self.assertAlmostEqual(table.get_value(0, 0), 0.25)
        table.update_value(1, 1, 0.9)
        self.assertAlmostEqual(table.get_value(1, 1), 0.9)

    def test_out_of_bounds_reads_and_writes(self) -> None:
        table = QTable(state_size=2, action_size=2)
        self.assertEqual(table.get_value(5, 0), 0.0)
        self.assertEqual(table.get_value(0, 5), 0.0)
        # Out-of-bounds write is a silent no-op, not an IndexError.
        table.update_value(5, 0, 1.0)
        self.assertEqual(table.get_value(5, 0), 0.0)

    def test_get_best_action(self) -> None:
        table = QTable(state_size=2, action_size=3)
        table.update_value(0, 2, 1.0)
        self.assertEqual(table.get_best_action(0), 2)
        self.assertEqual(table.get_best_action(99), 0)

    def test_roundtrip(self) -> None:
        table = QTable(state_size=2, action_size=2, default_value=0.5)
        table.update_value(1, 0, 1.5)
        restored = QTable.from_dict(table.to_dict())
        self.assertEqual(restored.state_size, 2)
        self.assertEqual(restored.action_size, 2)
        self.assertAlmostEqual(restored.get_value(1, 0), 1.5)


class TestReplayBuffer(unittest.TestCase):
    """Tests for ReplayBuffer capacity, sampling, and serialization."""

    def _exp(self, reward: float) -> Experience:
        return Experience(state=0, action=0, reward=reward, next_state=1, done=False)

    def test_capacity_eviction(self) -> None:
        buffer = ReplayBuffer(capacity=2)
        for i in range(3):
            buffer.add(self._exp(float(i)))
        self.assertEqual(buffer.size(), 2)
        self.assertEqual(buffer.to_dict()["capacity"], 2)
        # Oldest experience was evicted.
        rewards = [exp.reward for exp in buffer.buffer]
        self.assertNotIn(0.0, rewards)

    def test_sample_respects_size(self) -> None:
        buffer = ReplayBuffer(capacity=10)
        for i in range(3):
            buffer.add(self._exp(float(i)))
        batch = buffer.sample(10)
        self.assertEqual(len(batch), 3)

    def test_roundtrip(self) -> None:
        buffer = ReplayBuffer(capacity=5)
        buffer.add(self._exp(0.7))
        restored = ReplayBuffer.from_dict(buffer.to_dict())
        self.assertEqual(restored.size(), 1)
        self.assertAlmostEqual(restored.buffer[0].reward, 0.7)


class TestRLState(unittest.TestCase):
    """Tests for RLState Q-learning behavior and serialization."""

    def test_update_q_values_undiscounted_terminal(self) -> None:
        state = RLState(state_size=4, action_size=2)
        state.learning_rate = 0.5
        state.discount_factor = 0.9
        state.update_q_values(
            Experience(state=0, action=1, reward=1.0, next_state=1, done=True)
        )
        # Terminal: next_q == 0 → q = 0 + 0.5 * (1 - 0) = 0.5
        self.assertAlmostEqual(state.q_table.get_value(0, 1), 0.5)

    def test_update_q_values_uses_next_state_max(self) -> None:
        state = RLState(state_size=4, action_size=2)
        state.learning_rate = 1.0
        state.discount_factor = 0.5
        state.q_table.update_value(1, 0, 2.0)
        state.epsilon = state.epsilon_min  # avoid epsilon-decay bookkeeping drift
        state.update_q_values(
            Experience(state=0, action=0, reward=0.0, next_state=1, done=False)
        )
        # next_q = 2.0 → q = 0 + 1.0 * (0 + 0.5*2 - 0) = 1.0
        self.assertAlmostEqual(state.q_table.get_value(0, 0), 1.0)

    def test_epsilon_decays_until_min(self) -> None:
        state = RLState(state_size=2, action_size=2)
        state.epsilon_decay = 0.5
        state.epsilon_min = 0.2
        state.epsilon = 0.3
        state.update_q_values(
            Experience(state=0, action=0, reward=0.0, next_state=1, done=True)
        )
        self.assertAlmostEqual(state.epsilon, 0.15)
        state.update_q_values(
            Experience(state=0, action=0, reward=0.0, next_state=1, done=True)
        )
        # Now below epsilon_min: no further decay.
        self.assertAlmostEqual(state.epsilon, 0.15)

    def test_get_state_index_stable(self) -> None:
        state = RLState(state_size=8, action_size=2)
        self.assertEqual(state._get_state_index(3), 3)
        arr = np.array([1.0, 2.0, 3.0])
        idx1 = state._get_state_index(arr)
        idx2 = state._get_state_index(np.array([[1.0, 2.0, 3.0]]).reshape(3))
        self.assertEqual(idx1, idx2)
        self.assertTrue(0 <= state._get_state_index("arbitrary") < 8)

    def test_select_action_epsilon_zero_exploits(self) -> None:
        state = RLState(state_size=4, action_size=3)
        state.epsilon = 0.0
        state.q_table.update_value(state._get_state_index(2), 1, 5.0)
        self.assertEqual(state.select_action(2), 1)

    def test_record_episode_reward(self) -> None:
        state = RLState(state_size=2, action_size=2)
        state.record_episode_reward(1.0, episode_done=False)
        state.record_episode_reward(2.0, episode_done=True)
        self.assertEqual(state.current_episode, 1)
        self.assertEqual(state.episode_rewards, [3.0])
        self.assertAlmostEqual(state.total_reward, 0.0)

    def test_train_from_buffer_requires_full_batch(self) -> None:
        state = RLState(state_size=2, action_size=2)
        state.batch_size = 2
        state.train_from_buffer()  # empty buffer → no-op
        self.assertEqual(state.training_iterations, 0)
        state.update_q_values(
            Experience(state=0, action=0, reward=0.1, next_state=1, done=True)
        )
        state.train_from_buffer(batch_size=5)  # still smaller than batch → no-op
        self.assertEqual(state.training_iterations, 0)
        state.train_from_buffer(batch_size=1)
        self.assertEqual(state.training_iterations, 1)

    def test_roundtrip(self) -> None:
        state = RLState(state_size=3, action_size=2, buffer_capacity=5)
        state.learning_rate = 0.2
        state.update_q_values(
            Experience(state=0, action=1, reward=0.7, next_state=1, done=True)
        )
        restored = RLState.from_dict(state.to_dict())
        self.assertAlmostEqual(restored.learning_rate, 0.2)
        self.assertEqual(restored.q_table.state_size, 3)
        self.assertEqual(restored.replay_buffer.size(), 1)
        self.assertAlmostEqual(
            restored.q_table.get_value(0, 1), state.q_table.get_value(0, 1)
        )


class TestRLAgentConfig(unittest.TestCase):
    """Tests for RLAgent construction and parameter configuration."""

    def test_config_parameters_applied(self) -> None:
        agent = RLAgent(
            agent_id="rl-cfg",
            config={
                "state_size": 12,
                "action_size": 4,
                "buffer_capacity": 7,
                "learning_rate": 0.3,
                "discount_factor": 0.8,
                "epsilon": 0.4,
                "epsilon_decay": 0.9,
                "epsilon_min": 0.05,
                "batch_size": 16,
                "train_frequency": 2,
                "train_batch_size": 8,
            },
        )
        self.assertEqual(agent.state.q_table.state_size, 12)
        self.assertEqual(agent.state.q_table.action_size, 4)
        self.assertEqual(agent.state.replay_buffer.capacity, 7)
        self.assertAlmostEqual(agent.state.learning_rate, 0.3)
        self.assertAlmostEqual(agent.state.discount_factor, 0.8)
        self.assertAlmostEqual(agent.state.epsilon, 0.4)
        self.assertAlmostEqual(agent.state.epsilon_decay, 0.9)
        self.assertAlmostEqual(agent.state.epsilon_min, 0.05)
        self.assertEqual(agent.state.batch_size, 16)
        self.assertEqual(agent.train_frequency, 2)
        self.assertEqual(agent.train_batch_size, 8)


class TestRLAgentBehavior(unittest.TestCase):
    """Tests for RLAgent perception, decision, and action handling."""

    def _make_agent(self, **extra) -> RLAgent:
        config = {"state_size": 6, "action_size": 3, "epsilon": 0.0, **extra}
        return RLAgent(agent_id="rl-behave", config=config)

    def test_perceive_extracts_state_and_vector_state(self) -> None:
        agent = self._make_agent()
        obs = _run(agent.perceive())
        self.assertEqual(obs, {})
        agent.config["sensor_readings"] = {"state": 4}
        _run(agent.perceive())
        self.assertEqual(agent.state.current_state, 4)
        agent.config["sensor_readings"] = {"vector_state": [1.0, 2.0]}
        _run(agent.perceive())
        np.testing.assert_array_equal(agent.state.current_state, np.array([1.0, 2.0]))

    def test_decide_requires_current_state(self) -> None:
        agent = self._make_agent()
        self.assertIsNone(_run(agent.decide()))

    def test_decide_with_action_mapping(self) -> None:
        agent = self._make_agent(
            action_mapping={"2": {"action_type": "move", "action_id": "north"}}
        )
        agent.state.current_state = 1
        agent.state.q_table.update_value(agent.state._get_state_index(1), 2, 1.0)
        action = _run(agent.decide())
        self.assertEqual(action["action_type"], "move")
        self.assertEqual(action["selected_idx"], 2)

    def test_decide_default_action_format(self) -> None:
        agent = self._make_agent()
        agent.state.current_state = 0
        action = _run(agent.decide())
        self.assertEqual(action["action_type"], "execute")
        self.assertIn("selected_idx", action)

    def test_act_records_experience_and_learns(self) -> None:
        agent = self._make_agent(train_frequency=1, train_batch_size=1)
        agent.state.current_state = 0
        action = {
            "action_type": "wait",
            "parameters": {"duration": 0},
            "selected_idx": 1,
        }
        result = _run(agent.act(action))
        self.assertEqual(result["status"], "success")
        self.assertEqual(agent.state.training_iterations, 1)
        # Replay training re-adds the sampled experience on top of the
        # one recorded by act() itself.
        self.assertEqual(agent.state.replay_buffer.size(), 2)

    def test_query_state_performance_and_q_values(self) -> None:
        agent = self._make_agent()
        agent.state.record_episode_reward(2.0, episode_done=True)
        result = _run(
            agent.act(
                {
                    "action_type": "query_state",
                    "parameters": {"query_type": "performance"},
                }
            )
        )
        self.assertEqual(result["total_episodes"], 1)
        self.assertAlmostEqual(result["avg_reward_100"], 2.0)

        agent.state.current_state = 2
        result = _run(
            agent.act(
                {"action_type": "query_state", "parameters": {"query_type": "q_values"}}
            )
        )
        self.assertEqual(result["current_state_idx"], agent.state._get_state_index(2))
        self.assertEqual(len(result["q_values"]), 3)

    def test_query_state_unknown_type(self) -> None:
        agent = self._make_agent()
        result = _run(
            agent.act(
                {"action_type": "query_state", "parameters": {"query_type": "bogus"}}
            )
        )
        self.assertEqual(result["status"], "error")

    def test_set_learning_params(self) -> None:
        agent = self._make_agent()
        result = _run(
            agent.act(
                {
                    "action_type": "set_learning_params",
                    "parameters": {
                        "learning_rate": 0.9,
                        "epsilon": 0.5,
                        "epsilon_decay": 0.8,
                        "discount_factor": 0.7,
                    },
                }
            )
        )
        self.assertEqual(result["status"], "success")
        self.assertAlmostEqual(agent.state.learning_rate, 0.9)
        self.assertAlmostEqual(agent.state.epsilon, 0.5)

    def test_set_learning_params_noop(self) -> None:
        agent = self._make_agent()
        result = _run(
            agent.act({"action_type": "set_learning_params", "parameters": {}})
        )
        self.assertEqual(result["status"], "warning")

    def test_initialize_loads_initial_state_and_model(self) -> None:
        agent = self._make_agent()
        agent.state.update_q_values(
            Experience(state=0, action=0, reward=1.0, next_state=1, done=True)
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.json")
            agent._save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

            fresh = self._make_agent(initial_state=7)
            with open(model_path) as handle:
                saved = json.load(handle)
            fresh.config["model_path"] = model_path
            _run(fresh.initialize())
            self.assertEqual(fresh.state.current_state, 7)
            self.assertAlmostEqual(
                fresh.state.q_table.get_value(0, 0),
                saved["q_table"]["q_table"][0][0],
            )

    def test_load_model_corrupt_file_keeps_state(self) -> None:
        agent = self._make_agent()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write("not json")
            path = handle.name
        try:
            agent._load_model(path)  # logs error, keeps existing state
        finally:
            os.unlink(path)
        self.assertEqual(agent.state.q_table.state_size, 6)

    def test_shutdown_saves_model_when_configured(self) -> None:
        agent = self._make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.json")
            agent.config["model_save_path"] = path
            _run(agent.shutdown())
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
