import json

import pytest

from cardiac_rl.config import load_config
from cardiac_rl.evaluation import aggregate, evaluate
from cardiac_rl.inference import calibrate, write_example
from cardiac_rl.training import train


def test_train_checkpoint_resume_evaluate_and_infer(tmp_path):
    config = load_config("configs/smoke.yaml")
    config["mechanics"]["grid_size"] = 5
    config["data"].update(train_cases=8, validation_cases=3, test_cases=2,
                         noise_levels=[0.], observed_fractions=[1.])
    config["environment"]["horizon"] = 7
    config["training"].update(timesteps=32, n_envs=2, learning_starts=8, batch_size=8,
                             buffer_size=128, eval_frequency=16, eval_cases=2, checkpoint_frequency=16,
                             hidden_sizes=[16, 16], train_freq=1, gradient_steps=1)
    config["evaluation"].update(budgets=[8], direct_train_samples=16, direct_epochs=1)
    data, run = tmp_path / "data", tmp_path / "runs" / "seed_0"
    first = train(config, data, run, seed=0, device="cpu")
    assert first["timesteps"] == 32
    with pytest.raises(ValueError, match="Run already exists"):
        train(config, data, run, seed=0, device="cpu")
    resumed = train(config, data, run, seed=0, device="cpu", steps=48, resume=True)
    assert resumed["timesteps"] == 48 and resumed["resumed"]
    assert resumed["training_forward_calls"] > first["training_forward_calls"]
    results = evaluate(config, data, run, backend="reference", seed=0)
    assert (results / "benchmark.png").exists()
    summary = json.loads((results / "summary.json").read_text())
    assert len(summary) == 6
    assert all(row["mean_calls"] <= 8 for row in summary)
    aggregate(tmp_path / "runs", tmp_path / "aggregate.json")
    observation = tmp_path / "observations.json"
    write_example(config, data, observation)
    result = calibrate(config, data, run, observation, tmp_path / "estimate.json", budget=8)
    assert 1 <= result["a_kpa"] <= 8 and 2 <= result["af_kpa"] <= 24
    assert result["forward_calls"] <= 8


def test_timed_training_preserves_configuration_and_recovers_fallback(tmp_path):
    config = load_config("configs/smoke.yaml")
    config["mechanics"]["grid_size"] = 5
    config["data"].update(train_cases=8, validation_cases=2, test_cases=2)
    config["training"].update(n_envs=1, hidden_sizes=[16, 16], batch_size=8,
                             buffer_size=64, learning_starts=8, eval_cases=1,
                             eval_frequency=8, checkpoint_frequency=8)
    data, run = tmp_path / "data", tmp_path / "run"
    timed = train(config, data, run, seed=0, device="cpu", max_seconds=.000001, threads=2)
    assert timed["status"] == "time_budget_completed"
    assert timed["transition_target"] is None
    assert timed["cpu_threads"] == 2
    assert timed["timesteps"] < config["training"]["timesteps"]
    assert (run / "last.state.json").exists()
    # An intact periodic pair must remain usable after an invalid final model.
    for suffix in (".zip", "_replay.pkl", ".state.json"):
        (run / ("checkpoint"+suffix)).write_bytes((run / ("last"+suffix)).read_bytes())
    (run / "last.zip").write_bytes(b"incomplete scheduler-killed checkpoint")
    resumed = train(config, data, run, seed=0, device="cpu", resume=True, max_seconds=.000001)
    assert resumed["status"] == "time_budget_completed"
    assert resumed["recovered_start_timesteps"] == timed["timesteps"]
