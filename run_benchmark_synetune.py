# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
from argparse import ArgumentParser

from ax_paper.baselines.synetune import METHODS, run_synetune
from ax_paper.problems.synetune_blackbox import runtime_scalings


def main(
    method_name: str,
    benchmark_name: str,
    dataset_name: str,
    min_seed: int,
    max_seed: int,
    num_workers: int,
    max_num_evaluations: int,
    max_wallclock_time: float,
) -> None:
    for seed in range(min_seed, max_seed):
        run_synetune(
            method_name=method_name,
            benchmark_name=benchmark_name,
            dataset_name=dataset_name,
            seed=seed,
            num_workers=num_workers,
            max_num_evaluations=max_num_evaluations,
            max_wallclock_time=max_wallclock_time,
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("method_name", type=str, choices=METHODS.keys())
    parser.add_argument("benchmark_name", type=str, choices=runtime_scalings.keys())
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--parallelism", type=int, default=4, required=True)
    parser.add_argument(
        "--max_cumulative_steps",
        type=int,
        default=1000,
        required=False,
    )
    parser.add_argument(
        "--timeout_hours",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--min_seed",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--max_seed",
        type=int,
        default=10,
        help="Exclusive: seeds in range(min_seed, max_seed) will be run",
        required=False,
    )

    args = parser.parse_args()

    main(
        method_name=args.method_name,
        benchmark_name=args.benchmark_name,
        dataset_name=args.dataset_name,
        min_seed=args.min_seed,
        max_seed=args.max_seed,
        num_workers=args.parallelism,
        max_wallclock_time=args.timeout_hours,
        max_num_evaluations=args.max_cumulative_steps,
    )
