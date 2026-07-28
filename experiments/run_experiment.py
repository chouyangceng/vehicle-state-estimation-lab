from __future__ import annotations

import argparse

from vehicle_state_estimation.experiments import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic state estimation experiment")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", default="artifacts/state-estimation")
    args = parser.parse_args()
    output = run(seed=args.seed, steps=args.steps, output_dir=args.output)
    print(f"results written to {output}")


if __name__ == "__main__":
    main()
