from dataclasses import replace

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from cardiac_rl.baselines import Objective, optimise
from cardiac_rl.config import load_config
from cardiac_rl.data import load_cases, make_task, prepare
from cardiac_rl.environment import CalibrationEnv


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    config = load_config("configs/smoke.yaml")
    config["mechanics"]["grid_size"] = 5
    config["data"].update(train_cases=8, validation_cases=3, test_cases=3)
    directory = tmp_path_factory.mktemp("cases")
    model = prepare(config, directory)
    cases = load_cases(config, directory, "train")
    return config, directory, model, cases


def test_gym_api_and_seed_reproducibility(prepared):
    config, _, model, cases = prepared
    env = CalibrationEnv(config, model, cases)
    check_env(env)
    first, _ = env.reset(seed=99)
    second, _ = env.reset(seed=99)
    np.testing.assert_array_equal(first, second)


def test_no_truth_or_missing_target_leak(prepared):
    config, _, model, cases = prepared
    task = make_task(cases[0][0], cases[1][0], .01, .25, 17)
    modified = replace(task, z=np.array([.99, .01]), clean=np.full_like(task.clean, 999))
    env = CalibrationEnv(config, model, cases)
    first, _ = env.reset(options={"task": task, "horizon": 5})
    second, _ = env.reset(options={"task": modified, "horizon": 5})
    np.testing.assert_array_equal(first, second)
    assert np.all(first[:model.output_size][~task.mask] == 0)


def test_budget_exhaustion_is_terminal_and_counted(prepared):
    config, _, model, cases = prepared
    task = make_task(cases[0][0], cases[1][0], 0, 1, 1)
    env = CalibrationEnv(config, model, cases)
    env.reset(options={"task": task, "horizon": 3})
    for _ in range(3):
        _, reward, done, truncated, info = env.step([0, 0])
        assert np.isfinite(reward)
    assert done and not truncated
    assert info["forward_calls"] == 4
    with pytest.raises(RuntimeError):
        env.step([0, 0])


@pytest.mark.parametrize("method", ["least_squares", "differential_evolution", "bayesian", "random"])
def test_all_optimisers_respect_hard_budget(prepared, method):
    _, _, model, cases = prepared
    task = make_task(cases[0][0], cases[1][0], 0, 1, 12)
    objective = Objective(model, task, 16)
    z, calls, trace = optimise(method, objective, 4)
    assert 1 <= calls <= 16
    assert np.all((z >= 0) & (z <= 1))
    assert all(a["loss"] >= b["loss"] for a, b in zip(trace, trace[1:]))


def test_parameter_splits_disjoint_and_config_checked(prepared):
    config, directory, _, _ = prepared
    sets = [set(map(tuple, load_cases(config, directory, split)[0])) for split in ("train", "validation", "test")]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    config = {**config, "seed": config["seed"]+1}
    with pytest.raises(ValueError):
        load_cases(config, directory, "train")


def test_reference_inverse_recovers_known_parameters(prepared):
    _, _, model, cases = prepared
    task = make_task(cases[0][0], cases[1][0], 0, 1, 12)
    objective = Objective(model, task, 64, backend="reference")
    z, _, _ = optimise("least_squares", objective, 4)
    np.testing.assert_allclose(z, task.z, atol=1e-4)
