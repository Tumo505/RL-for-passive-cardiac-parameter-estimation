from __future__ import annotations

import argparse
import json

from .evaluation import aggregate
from .hybrid.evaluation import METHODS, evaluate_hybrid
from .hybrid.setup import configuration, prepare_hybrid
from .hybrid.training import train_hybrid


def main():
    parser = argparse.ArgumentParser(description="Hybrid v2: fresh data, learned initialization, RL-controlled mechanics refinement")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", default="configs/hybrid.yaml")
        sub.add_argument("--data-dir", default="data/hybrid_v2")
        if command != "prepare":
            sub.add_argument("--output", default="runs/hybrid_v2/seed_0")
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--threads", type=int)
        if command == "train":
            sub.add_argument("--device", default="auto")
            sub.add_argument("--steps", type=int)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--max-seconds", type=float)
        if command == "evaluate":
            sub.add_argument("--split", choices=("validation", "test"), default="validation")
            sub.add_argument("--confirm-test", action="store_true")
            sub.add_argument("--backend", choices=("grid", "reference"), default="reference")
            sub.add_argument("--methods", nargs="+", choices=METHODS)
            sub.add_argument("--limit", type=int)
            sub.add_argument("--overwrite", action="store_true")
            sub.add_argument("--direct-device", default="cpu", help="Compatibility argument; initializer is already trained and online inference stays on CPU")
    sub = commands.add_parser("aggregate")
    sub.add_argument("--root", default="runs/hybrid_v2")
    sub.add_argument("--output", default="runs/hybrid_v2/aggregate.json")
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.root, args.output)
        print(args.output)
        return
    config = configuration(args.config)
    if args.command == "prepare":
        prepare_hybrid(config, args.data_dir)
        print(f"Prepared fresh hybrid data in {args.data_dir}")
    elif args.command == "train":
        result = train_hybrid(config, args.data_dir, args.output, args.seed, args.device,
                              args.steps, args.resume, args.max_seconds, args.threads)
        print(json.dumps(result, indent=2))
        if result["status"] == "interrupted":
            raise SystemExit(75)
    else:
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        destination = evaluate_hybrid(config, args.data_dir, args.output, args.seed, args.split,
            args.confirm_test, args.methods, args.backend, args.limit, args.threads, args.overwrite)
        print(f"Hybrid evaluation saved to {destination}")


if __name__ == "__main__":
    main()
