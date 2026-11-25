from dataclasses import replace
import csv
import json

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from cardiac_rl.data import load_cases, make_task
from cardiac_rl.evaluation import aggregate
from cardiac_rl.hybrid.core import CalibrationOracle, LMStepper, converged_lm, converged_lsq, mechanics_inverse
from cardiac_rl.hybrid.environment import HybridEnv
from cardiac_rl.hybrid.evaluation import evaluate_hybrid
from cardiac_rl.hybrid.features import compress_measurements
from cardiac_rl.hybrid.setup import configuration, prepare_hybrid
from cardiac_rl.hybrid.training import train_hybrid


class MidpointInitializer:
    prior_std = np.array([.2, .2])

    def predict(self, features):
        return np.array([.5, .5])


@pytest.fixture(scope="module")
def prepared_hybrid(tmp_path_factory):
    config = configuration("configs/hybrid_smoke.yaml")
    config["mechanics"]["grid_size"] = 5
    config["data"].update(train_cases=8, validation_cases=3, test_cases=3)
    directory = tmp_path_factory.mktemp("hybrid")
    model = prepare_hybrid(config, directory)
    return config, directory, model, load_cases(config, directory, "train")


def test_compact_features_recover_affine_strain_and_ignore_missing_values(prepared_hybrid):
    config, _, model, cases = prepared_hybrid
    clean = model.reference([.3, .7])
    mask = np.ones(model.output_size, dtype=bool)
    result = compress_measurements(model, clean, mask, config["hybrid"]["noise_floor"])
    np.testing.assert_allclose(result.slopes, (model.solve_stretches([.3, .7])-1).ravel(), atol=1e-10)
    assert result.features.shape == (25,)
    task = make_task(cases[0][0], cases[1][0], .01, .25, 981)
    a = compress_measurements(model, task.observed, task.mask, .0001)
    modified = task.observed.copy()
    modified[~task.mask] = np.nan
    b = compress_measurements(model, modified, task.mask, .0001)
    np.testing.assert_array_equal(a.features, b.features)


def test_model_specific_linear_inverse_recovers_clean_parameters(prepared_hybrid):
    config, _, model, _ = prepared_hybrid
    for z in ([.1, .2], [.4, .8], [.9, .1]):
        clean = model.reference(z)
        features = compress_measurements(model, clean, np.ones_like(clean, dtype=bool), config["hybrid"]["noise_floor"])
        estimate = mechanics_inverse(model, features)
        np.testing.assert_allclose(estimate, z, atol=1e-6)


def test_hybrid_gym_contract_and_evaluation_do_not_use_truth(prepared_hybrid):
    config, _, model, cases = prepared_hybrid
    check_env(HybridEnv(config, model, cases, MidpointInitializer()))
    task = make_task(cases[0][0], cases[1][0], .01, .25, 19)
    changed = replace(task, z=np.array([np.nan, np.nan]), clean=np.full_like(task.clean, 9999), noise=999)
    environments = [HybridEnv(config, model, cases, MidpointInitializer(), training=False) for _ in range(2)]
    observations = [environment.reset(options={"task": item, "budget": 8})[0]
                    for environment, item in zip(environments, [task, changed])]
    np.testing.assert_array_equal(*observations)
    for _ in range(3):
        if environments[0].done:
            break
        outcomes = [env.step([0., .5, -1.]) for env in environments]
        np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
        assert outcomes[0][1] == outcomes[1][1] == 0
        assert outcomes[0][4]["forward_calls"] <= 8


def test_lm_probes_are_counted_and_stopping_is_explicit(prepared_hybrid):
    config, _, model, cases = prepared_hybrid
    task = make_task(cases[0][0], cases[1][0], 0., 1., 81)
    oracle = CalibrationOracle(model, task, 4, .0001, np.array([.5, .5]), [.2, .2], backend="reference")
    stepper = LMStepper(oracle, [.5, .5])
    initial_cost = oracle.best_cost
    stepper.step(.01, .5)
    assert 1 <= oracle.calls <= 4
    assert oracle.best_cost <= initial_cost
    assert len(oracle.trace) == oracle.calls
    env = HybridEnv(config, model, cases, MidpointInitializer(), training=False)
    env.reset(options={"task": task, "budget": 8})
    _, _, terminated, truncated, info = env.step([0, 0, 1])
    assert terminated and not truncated and info["forward_calls"] == 1
    assert info["stop_reason"] == "policy_stop"


def test_converged_baselines_do_not_spend_budget_on_restarts(prepared_hybrid):
    config, _, model, _ = prepared_hybrid
    clean = model.reference([.5, .5])
    task = make_task([.5, .5], clean, 0., 1., 7)
    for function in (converged_lm, converged_lsq):
        oracle = CalibrationOracle(model, task, 64, .0001, [.5, .5], [.2, .2], 0., "reference")
        estimate = function(oracle, np.array([.5, .5]), config["hybrid"])
        np.testing.assert_allclose(estimate, [.5, .5], atol=1e-8)
        assert oracle.calls <= 4


def test_final_test_requires_explicit_freeze_acknowledgement(prepared_hybrid, tmp_path):
    config, directory, _, _ = prepared_hybrid
    with pytest.raises(ValueError, match="Final test is sealed"):
        evaluate_hybrid(config, directory, tmp_path, split="test")


def test_legacy_data_directory_is_never_reused(prepared_hybrid):
    config, directory, _, _ = prepared_hybrid
    import copy
    changed = copy.deepcopy(config)
    changed["hybrid"]["exclude_datasets"] = [str(directory)]
    with pytest.raises(ValueError, match="legacy datasets"):
        prepare_hybrid(changed, directory)


def test_hybrid_training_resume_and_validation_pipeline(tmp_path):
    config = configuration("configs/hybrid_smoke.yaml")
    config["mechanics"]["grid_size"] = 5
    config["data"].update(train_cases=8, validation_cases=3, test_cases=3,
                         noise_levels=[0., .01], observed_fractions=[1., .5])
    config["training"].update(timesteps=64, learning_starts=8, batch_size=8, buffer_size=128,
                             n_envs=1, eval_frequency=16, checkpoint_frequency=16,
                             hidden_sizes=[16, 16], train_freq=1, gradient_steps=1)
    config["hybrid"].update(initializer_samples=32, initializer_epochs=2, initializer_hidden_sizes=[16, 16],
                           initializer_patience=2, early_stopping_patience=1, minimum_timesteps=16,
                           minimum_validation_improvement=100., validation_cases_per_condition=1)
    data, run = tmp_path / "data", tmp_path / "seed_0"
    result = train_hybrid(config, data, run, seed=0, device="cpu")
    assert result["status"] == "validation_early_stopping"
    assert result["timesteps"] < 64
    assert result["test_split_used_in_training"] is False
    resumed = train_hybrid(config, data, run, seed=0, device="cpu", resume=True, steps=80)
    assert resumed["timesteps"] > result["timesteps"]
    unchanged = train_hybrid(config, data, run, seed=0, device="cpu", resume=True, steps=resumed["timesteps"])
    assert unchanged["new_timesteps"] == 0 and unchanged["status"] == "completed"
    methods = ["mechanics_direct", "lsq_midpoint", "neural_only", "neural_lsq", "neural_map_lsq", "neural_lm", "hybrid_sac"]
    output = evaluate_hybrid(config, data, run, methods=methods, split="validation", limit=1)
    with (output / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4*2*len(methods)
    assert all(int(row["forward_calls"]) <= int(row["budget"]) for row in rows)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["split"] == "validation" and protocol["known_noise_supplied"] is False
    assert evaluate_hybrid(config, data, run, methods=methods, split="validation", limit=1) == output
    # This is an isolated tiny unit-test fixture, not the configured research test set.
    test_output = evaluate_hybrid(config, data, run, methods=["mechanics_direct", "hybrid_sac"],
                                  split="test", confirm_test=True, limit=1)
    assert json.loads((test_output / "protocol.json").read_text())["split"] == "test"
    (run / "best.zip").write_bytes((run / "best.zip").read_bytes()+b"changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        evaluate_hybrid(config, data, run, split="test", confirm_test=True)


def test_aggregate_never_mixes_validation_and_final_test(tmp_path):
    for seed in range(2):
        for split, count in (("validation", 2), ("test", 3)):
            directory = tmp_path / f"seed_{seed}/evaluation_{split}_reference"
            directory.mkdir(parents=True)
            rows = [{"seed": seed, "case_id": case, "backend": "reference", "method": "hybrid_sac",
                     "budget": 8, "noise": 0., "observed_fraction": 1., "split": split,
                     "mean_relative_parameter_error": .1} for case in range(count)]
            with (directory / "results.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    result = aggregate(tmp_path, tmp_path / "aggregate.json")
    assert {(group["split"], group["parameter_cases"]) for group in result} == {("validation", 2), ("test", 3)}
