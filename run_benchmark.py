# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import json
import shutil
from datetime import datetime
from logging import WARNING
from pathlib import Path

from ax.benchmark.benchmark import benchmark_replication
from ax.benchmark.benchmark_result import BenchmarkResult
from ax.benchmark.problems.registry import get_benchmark_problem
from ax.storage.json_store.encoder import object_to_json
from ax.utils.common.random import with_rng_seed
from ax_paper.problems import AX_PAPER_PROBLEM_REGISTRY, AX_PAPER_PROBLEMS
from ax_paper.run_benchmark_helpers import (
    BASELINE_REGISTRY,
    command_line_name_to_method_name,
)
from runtime_data import (
    PAST_AVERAGE_RUNTIME_BY_BENCHMARK,
    PAST_AVERAGE_RUNTIME_BY_METHOD,
)


def dump_benchmark_results_to_json(
    results: list[BenchmarkResult],
    directory: Path,
    method_name: str,
    problem_name: str,
) -> str:
    """
    Encode results as JSON and dump to file.

    Args:
        results: A list of BenchmarkResults.
        method_name: The method (in this case, likely a library).
        problem_name: The name of a `BenchmarkProblem`.

    Returns:
        file name, in format "{method_name}|{problem_name}|{timestamp}"
    """
    timestamp = datetime.now().strftime("%Y-%m-%d+%H:%M:%S%z")
    fname = f"{method_name}|{problem_name}|{timestamp}"
    serialized = object_to_json(obj=results)
    with directory.joinpath(f"{fname}.json").open("w") as f:
        json.dump(serialized, f)
    return fname


def run_benchmark(
    method_name: str,
    problem_name: str,
    min_seed: int,
    max_seed: int,
    num_trials: int,
    max_pending_trials: int,
    timeout_hours: float,
    continue_after_timeout: bool,
) -> None:
    ax_benchmark_dir = Path(__file__).parent.resolve()
    results_dir: Path = ax_benchmark_dir.joinpath("results")
    if not results_dir.exists():
        results_dir.mkdir()

    results = []
    for seed in range(min_seed, max_seed):
        try:
            with with_rng_seed(seed + 12345):
                # Initialize the problem with a fixed seed to ensure any randomness is
                # consistent across replications of different methods.
                problem = get_benchmark_problem(
                    problem_key=problem_name,
                    registry=AX_PAPER_PROBLEM_REGISTRY,
                    num_trials=num_trials,
                )

            # When problems have different runtimes for different trials, trials
            # should be run in parallel; otherwise the variable runtimes would be
            # uninteresting.
            if problem.step_runtime_function is None:
                if max_pending_trials > 1:
                    raise RuntimeError(
                        f"Running trials in parallel ({max_pending_trials=}) "
                        "for problem with constant runtime. This is unexpected.",
                    )
            elif max_pending_trials != 4:
                raise RuntimeError(
                    f"Running trials in sequence ({max_pending_trials=}) "
                    "for problem with variable runtime. This is unexpected.",
                )

            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Running {seed=}, {problem_name=}, {method_name=}, {time=}")
            method = BASELINE_REGISTRY[method_name](
                benchmark_problem=problem,
                seed=seed,
                timeout_hours=timeout_hours,
                max_pending_trials=max_pending_trials,
            )
            result = benchmark_replication(
                problem=problem,
                method=method,
                seed=seed,
                orchestrator_logging_level=WARNING,
            )
        except KeyboardInterrupt:
            if len(results) == 0:
                raise KeyboardInterrupt(
                    "KeyboardInterrupt before any replications completed. Exiting."
                )
            print(
                "Received KeyboardInterrupt. Exiting benchmark loop and writing"
                " partial results to disk."
            )
            break

        results.append(result)
        # If the length of the result is less than the number of trials, then it
        # must have timed out; don't run any more replications.
        n_trials_completed = max(result.experiment.trials.keys()) + 1
        timed_out = n_trials_completed < num_trials
        if (not continue_after_timeout) and timed_out:
            print(
                f"Result for seed {seed} timed out after {len(result.optimization_trace)} "
                "trials. Exiting. Partial results will be written to disk."
            )
            break

    # Cleanup tmp dir, where SMAC dumps its internal logs.
    tmp_dir: Path = ax_benchmark_dir.joinpath("tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # Dump results to file & print file name.
    file_name = dump_benchmark_results_to_json(
        results=results,
        # Run this from the directory you want the file written to
        directory=results_dir,
        method_name=method.name,
        problem_name=problem.name,
    )
    print(file_name)


def predict_runtime(
    method_name: str,
    problem_name: str,
    min_seed: int,
    max_seed: int,
    num_trials: int,
) -> None:
    if num_trials < 50:
        print("Warning: num_trials < 50; predictions are based on running 50 trials.")
    method_name = command_line_name_to_method_name.get(method_name, method_name)
    problem_name = get_benchmark_problem(
        problem_key=problem_name, registry=AX_PAPER_PROBLEM_REGISTRY
    ).name
    if (method_name, problem_name) in PAST_AVERAGE_RUNTIME_BY_BENCHMARK:
        predicted_per_seed_time = PAST_AVERAGE_RUNTIME_BY_BENCHMARK[
            (method_name, problem_name)
        ]
    else:
        print("No past results for this benchmark. Predicting based only on method.")
        predicted_per_seed_time = PAST_AVERAGE_RUNTIME_BY_METHOD[method_name]
    predicted_time_s = (max_seed - min_seed) * predicted_per_seed_time
    predicted_time_h = predicted_time_s / 3600
    print(f"Predicted time: {predicted_time_h:1f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method_name", type=str, choices=BASELINE_REGISTRY.keys())
    parser.add_argument("problem_name", type=str, choices=AX_PAPER_PROBLEMS)
    parser.add_argument("--min_seed", type=int, default=0, required=False)
    parser.add_argument(
        "--max_seed",
        type=int,
        default=10,
        help="Exclusive: seeds in range(min_seed, max_seed) will be run",
        required=False,
    )
    parser.add_argument("--num_trials", type=int, default=50, required=False)
    parser.add_argument("--max_pending_trials", type=int, default=1, required=False)
    parser.add_argument("--predict_runtime", action="store_true")
    parser.add_argument("--timeout_hours", type=float, default=1.0, required=False)
    parser.add_argument(
        "--continue_after_timeout",
        action="store_true",
        help="Whether to attempt subsequent replications after one times out.",
    )
    args = parser.parse_args()
    if args.predict_runtime:
        predict_runtime(
            method_name=args.method_name,
            problem_name=args.problem_name,
            min_seed=args.min_seed,
            max_seed=args.max_seed,
            num_trials=args.num_trials,
        )
    else:
        run_benchmark(
            method_name=args.method_name,
            problem_name=args.problem_name,
            min_seed=args.min_seed,
            max_seed=args.max_seed,
            num_trials=args.num_trials,
            max_pending_trials=args.max_pending_trials,
            timeout_hours=args.timeout_hours,
            continue_after_timeout=args.continue_after_timeout,
        )
