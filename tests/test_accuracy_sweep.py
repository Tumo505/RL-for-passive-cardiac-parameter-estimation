import copy
import csv
from dataclasses import replace
import json

import numpy as np
import pytest
import yaml

from cardiac_rl.data import make_task
from cardiac_rl.hybrid.core import CalibrationOracle, converged_lm, converged_lsq, polish_candidate
from cardiac_rl.hybrid.initializer import initializer_fingerprint
from cardiac_rl.hybrid.setup import configuration, prepare_hybrid
from cardiac_rl import sweep


def test_initializer_signature_shares_only_compatible_training():
    config = configuration("configs/hybrid_efficient.yaml")
    changed = copy.deepcopy(config)
    changed["hybrid"]["call_penalty"] = 0.
    changed["hybrid"]["early_stopping_patience"] = 1
    assert initializer_fingerprint(config) == initializer_fingerprint(changed)
    changed["data"]["train_cases"] += 1
    assert initializer_fingerprint(config) != initializer_fingerprint(changed)


def test_polishing_counts_probes_and_uses_estimated_noise_only(tmp_path):
    config = configuration("configs/hybrid_smoke.yaml")
    config["mechanics"]["grid_size"] = 5
    config["data"].update(train_cases=4, validation_cases=2, test_cases=2)
    model = prepare_hybrid(config, tmp_path)
    z = np.array([.2, .8])
    task = make_task(z, model.reference(z), .01, 1., 7)
    changed = replace(task, z=np.full(2, np.nan), clean=np.full_like(task.clean, np.nan), noise=999.)
    outcomes = []
    for item in (task, changed):
        oracle = CalibrationOracle(model, item, 8, .01, [.5, .5], [.2, .2], backend="reference")
        oracle.evaluate([.5, .5])
        cost = oracle.best_cost
        estimate, reason = polish_candidate(oracle, config["hybrid"], noise_limit=.02)
        assert reason == "polished" and 1 < oracle.calls <= 4
        assert oracle.best_cost <= cost
        assert oracle.diagnostics["polishing_forward_calls"] == oracle.calls-1
        outcomes.append(estimate)
    np.testing.assert_array_equal(*outcomes)
    oracle = CalibrationOracle(model, task, 8, .05, [.5, .5], [.2, .2], backend="reference")
    oracle.evaluate([.5, .5])
    assert polish_candidate(oracle, config["hybrid"], .02)[1] == "estimated_noise_above_limit"
    assert oracle.calls == 1
    oracle = CalibrationOracle(model, task, 3, .01, [.5, .5], [.2, .2], backend="reference")
    oracle.evaluate([.5, .5])
    assert polish_candidate(oracle, config["hybrid"])[1] == "insufficient_budget"
    for solver in (converged_lm, converged_lsq):
        # At the exact clean solution the discrepancy rule should stop at one query.
        clean = make_task(z, model.reference(z), 0., 1., 7)
        oracle = CalibrationOracle(model, clean, 32, .0001, z, [.2, .2], backend="reference")
        solver(oracle, z, config["hybrid"], noise_stop=True)
        assert oracle.calls == 1 and oracle.diagnostics["noise_stopped"]


@pytest.fixture(scope="module")
def completed_sweep(tmp_path_factory):
    directory = tmp_path_factory.mktemp("accuracy_sweep")
    base = configuration("configs/hybrid_smoke.yaml")
    base["mechanics"]["grid_size"] = 5
    base["data"].update(train_cases=8, validation_cases=2, test_cases=2,
                        noise_levels=[0., .01], observed_fractions=[1.])
    base["training"].update(timesteps=32, learning_starts=8, batch_size=8, buffer_size=128,
        n_envs=1, eval_frequency=16, checkpoint_frequency=16, hidden_sizes=[16, 16], train_freq=1, gradient_steps=1)
    base["hybrid"].update(initializer_samples=32, initializer_epochs=2, initializer_hidden_sizes=[16, 16],
        initializer_patience=2, early_stopping_patience=2, minimum_timesteps=16,
        validation_cases_per_condition=2, enable_accuracy_controls=True)
    profiles = {}
    for name, penalty in (("efficient", .002), ("balanced", .0002), ("accurate", 0.)):
        config = copy.deepcopy(base)
        config["hybrid"]["call_penalty"] = penalty
        path = directory/f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        profiles[name] = str(path)
    suite_path = directory/"suite.yaml"
    suite_path.write_text(yaml.safe_dump({"profiles": profiles, "seeds": 1, "seed_start": 0, "primary_budget": 8}, sort_keys=False))
    root, data = directory/"runs", directory/"data"
    report = sweep.run(suite_path, root, data, local=True)
    return suite_path, root, data, report


def test_full_three_profile_sweep_selects_without_research_test_leakage(completed_sweep):
    suite, root, data, report = completed_sweep
    assert report["test_used"] is False
    assert len(report["scores"]) == 18  # nine RL candidates, nine fixed baselines
    candidates = [s for s in report["scores"] if s["method"] in sweep.RL_METHODS]
    assert report["selected"]["mean_log_parameter_error"] == min(s["mean_log_parameter_error"] for s in candidates)
    assert not list(root.glob("*/seed_*/evaluation_test_*"))
    assert sweep.select(suite, root) == report
    for profile in ("accurate", "balanced"):
        metadata = json.loads((root/profile/"seed_0/initializer.json").read_text())
        assert metadata["reused"] and metadata["training_seconds"] == 0


def test_frozen_final_test_runs_only_selected_rl_and_all_baselines(completed_sweep, monkeypatch):
    suite, root, data, report = completed_sweep
    commands = []
    monkeypatch.setattr(sweep, "execute", lambda command: commands.append(command))
    # Synthetic two-case unit-test fixture only; no production test is touched.
    sweep.final_test(suite, root, data)
    command = commands[0]
    assert command[command.index("--methods")+1:] == [*sweep.BASELINES, report["selected"]["method"]]
    assert command[command.index("--evaluation-split")+1] == "test"
    assert "--confirm-test" in command and "--evaluate-only" in command
    with pytest.raises(ValueError, match="can no longer change"):
        sweep.select(suite, root)
    with pytest.raises(ValueError, match="Cannot retrain"):
        sweep.run(suite, root, data, local=True)
    selected = root/report["selected"]["profile"]/"seed_0/best.zip"
    original = selected.read_bytes()
    try:
        selected.write_bytes(original+b"mutation")
        with pytest.raises(ValueError, match="modified"):
            sweep.final_test(suite, root, data)
    finally:
        selected.write_bytes(original)


def test_suite_rejects_mismatched_datasets_and_unsafe_names(tmp_path):
    first = configuration("configs/hybrid_efficient.yaml")
    second = copy.deepcopy(first)
    second["seed"] += 1
    paths = [tmp_path/f"{i}.yaml" for i in range(2)]
    for path, config in zip(paths, [first, second]):
        path.write_text(yaml.safe_dump(config))
    suite = tmp_path/"suite.yaml"
    suite.write_text(yaml.safe_dump({"profiles": {"a": str(paths[0]), "b": str(paths[1])}, "seeds": 4, "primary_budget": 8}))
    with pytest.raises(ValueError, match="must share data"):
        sweep.load_suite(suite)
