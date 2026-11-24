from __future__ import annotations

import argparse
import json

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="SAC inverse passive cardiac mechanics benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default="configs/default.yaml")
        sub.add_argument("--data-dir", default="data/default")
        if name != "prepare":
            sub.add_argument("--output", default="runs/default/seed_0")
            sub.add_argument("--seed", type=int, default=0)
        if name == "train":
            sub.add_argument("--device", default="auto")
            sub.add_argument("--steps", type=int, help="Target TOTAL transitions, including any resumed training")
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--max-seconds", type=float, help="Wall-clock training budget; no default transition cap in this mode")
            sub.add_argument("--threads", type=int, help="Runtime CPU thread override; preserves configuration compatibility")
        if name == "evaluate":
            sub.add_argument("--backend", choices=("grid", "reference"), default="grid")
            sub.add_argument("--methods", nargs="+", default=None)
            sub.add_argument("--limit", type=int, help="Maximum held-out parameter cases, before noise/mask expansion")
            sub.add_argument("--direct-device", default="cpu")
            sub.add_argument("--threads", type=int)
    sub = commands.add_parser("aggregate")
    sub.add_argument("--root", default="runs/default")
    sub.add_argument("--output", default="runs/default/aggregate.json")
    sub = commands.add_parser("example")
    sub.add_argument("--config", default="configs/default.yaml")
    sub.add_argument("--data-dir", default="data/default")
    sub.add_argument("--output", default="observations.example.json")
    sub = commands.add_parser("calibrate")
    sub.add_argument("--config", default="configs/default.yaml")
    sub.add_argument("--data-dir", default="data/default")
    sub.add_argument("--model-dir", required=True)
    sub.add_argument("--observations", required=True)
    sub.add_argument("--output", default="calibration.json")
    sub.add_argument("--budget", type=int)
    sub.add_argument("--backend", choices=("grid", "reference"), default="grid")
    args = parser.parse_args()
    if args.command == "aggregate":
        from .evaluation import aggregate
        aggregate(args.root, args.output)
        print(args.output)
        return
    config = load_config(args.config)
    if args.command == "example":
        from .inference import write_example
        write_example(config, args.data_dir, args.output)
        print(args.output)
    elif args.command == "calibrate":
        from .inference import calibrate
        result = calibrate(config, args.data_dir, args.model_dir, args.observations, args.output, args.budget, args.backend)
        print(json.dumps({key: result[key] for key in ("a_kpa", "af_kpa", "forward_calls")}, indent=2))
    elif args.command == "prepare":
        from .data import prepare
        prepare(config, args.data_dir)
        print(f"Prepared {args.data_dir}")
    elif args.command == "train":
        from .training import train
        result = train(config, args.data_dir, args.output, args.seed, args.device, args.steps, args.resume,
                       args.max_seconds, args.threads)
        print(json.dumps(result, indent=2))
        if result["status"] == "interrupted":
            raise SystemExit(75)
    else:
        from .evaluation import evaluate
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        output = evaluate(config, args.data_dir, args.output, args.methods, args.backend,
                          args.limit, args.seed, args.direct_device, args.threads)
        print(f"Evaluation saved to {output}")


if __name__ == "__main__":
    main()
