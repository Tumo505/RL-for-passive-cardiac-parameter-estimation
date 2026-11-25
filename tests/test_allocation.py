import sys
from types import SimpleNamespace

import pytest

import cardiac_rl.allocation as allocation_module
from cardiac_rl.allocation import duration_seconds, parse_pbs_status, worker_plan


def test_pbs_walltime_including_elapsed_and_allocated_resources():
    status = parse_pbs_status("""Job Id: 123.pbs
    Resource_List.walltime = 168:00:00
    resources_used.walltime = 01:02:03
    Resource_List.ncpus = 28
    Resource_List.mem = 240gb
""")
    assert status["walltime_seconds"] == 604800
    assert status["used_seconds"] == 3723
    assert status["allocated_cpus"] == 28
    assert status["allocated_memory"] == "240gb"


def test_select_fallback_and_short_allocation_are_not_assumed_unlimited():
    status = parse_pbs_status("""
    Resource_List.walltime = 00:15:00
    Resource_List.select = 1:ncpus=28:mem=240gb:ngpus=4
""")
    assert status["walltime_seconds"] == 900
    assert status["allocated_cpus"] == 28
    with pytest.raises(ValueError, match="walltime"):
        parse_pbs_status("Resource_List.ncpus = 28")


@pytest.mark.parametrize("gpus,devices", [(1, [0, 0, 0, 0]), (2, [0, 1, 0, 1]), (4, [0, 1, 2, 3])])
def test_all_allocated_cpu_threads_and_visible_gpus_are_distributed(gpus, devices):
    plan = worker_plan(28, gpus)
    assert sum(worker["threads"] for worker in plan) == 28
    assert [worker["device"] for worker in plan] == [f"cuda:{device}" for device in devices]


def test_unequal_cpu_partition_does_not_oversubscribe():
    plan = worker_plan(27, 1)
    assert [worker["threads"] for worker in plan] == [7, 7, 7, 6]
    with pytest.raises(ValueError):
        worker_plan(2, 1)


def test_pbs_duration_validates_minutes():
    assert duration_seconds("168:00:00") == 604800
    with pytest.raises(ValueError):
        duration_seconds("01:70:00")


@pytest.mark.parametrize("evaluate_only", [False, True])
@pytest.mark.parametrize("workflow", ["legacy", "hybrid"])
def test_allocation_launches_timed_workers_then_full_evaluation(tmp_path, monkeypatch, evaluate_only, workflow):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "run"
    (root / "seed_0").mkdir(parents=True)
    (root / "seed_0/run.json").write_text("{}")
    monkeypatch.setenv("PBS_JOBID", "123.example")
    monkeypatch.delenv("PBS_NP", raising=False)
    argv = ["allocation", "--output", str(root), "--workflow", workflow]
    if evaluate_only:
        argv.append("--evaluate-only")
        (root / "allocation.json").write_text("original training allocation")
        for seed in range(4):
            (root / f"seed_{seed}").mkdir(exist_ok=True)
            for name in ("run.json", "best.json", "best.zip", "initializer.pt", "frozen.json"):
                (root / f"seed_{seed}" / name).write_text("test artifact")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(allocation_module, "load_config", lambda path: {})
    monkeypatch.setattr(allocation_module, "prepare", lambda *args: None)
    import cardiac_rl.hybrid.setup as hybrid_setup
    monkeypatch.setattr(hybrid_setup, "configuration", lambda path: {"training": {"timesteps": 1000000}})
    monkeypatch.setattr(hybrid_setup, "prepare_hybrid", lambda *args: None)
    monkeypatch.setattr(allocation_module.os, "sched_getaffinity", lambda pid: set(range(28)), raising=False)
    monkeypatch.setattr(allocation_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(allocation_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(allocation_module.torch.cuda, "get_device_name", lambda device: "test GPU")
    monkeypatch.setattr(allocation_module.signal, "signal", lambda *args: None)
    status = "Resource_List.walltime = 168:00:00\nResource_List.ncpus = 28\n"
    monkeypatch.setattr(allocation_module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=status))
    launches, aggregates = [], []

    class FinishedProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            launches.append((command, kwargs["env"]))

        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(allocation_module.subprocess, "Popen", FinishedProcess)
    monkeypatch.setattr(allocation_module, "aggregate", lambda *args: aggregates.append(args))
    allocation_module.main()
    assert len(launches) == (4 if evaluate_only else 8)
    for index, (command, environment) in enumerate([] if evaluate_only else launches[:4]):
        assert command[3] == "train"
        assert command[2] == ("cardiac_rl.hybrid_cli" if workflow == "hybrid" else "cardiac_rl.cli")
        assert "--max-seconds" in command and "--steps" not in command
        assert command[command.index("--device")+1] == "cuda:0"
        assert environment["OMP_NUM_THREADS"] == "7"
        assert ("--resume" in command) == (index == 0)
    for command, _ in (launches if evaluate_only else launches[4:]):
        assert command[3] == "evaluate"
        assert command[command.index("--backend")+1] == "reference"
        if workflow == "hybrid":
            assert command[command.index("--split")+1] == "validation"
            assert "--confirm-test" not in command
    assert len(aggregates) == 1
    if evaluate_only:
        assert (root / "allocation.json").read_text() == "original training allocation"
        assert (root / "evaluation_allocation.json").exists()


def test_short_pbs_allocation_is_rejected_before_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("PBS_JOBID", "123.example")
    monkeypatch.setattr(sys, "argv", ["allocation"])
    monkeypatch.setattr(allocation_module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="Resource_List.walltime = 00:15:00\n"))
    with pytest.raises(SystemExit) as error:
        allocation_module.main()
    assert error.value.code == 2
