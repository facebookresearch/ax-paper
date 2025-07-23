# Copyright (c) Meta Platforms, Inc. and affiliates.

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.problems.registry import BenchmarkProblemRegistryEntry
from ax.benchmark.problems.synthetic.discretized.mixed_integer import (
    _get_problem_from_common_inputs,
)
from botorch.test_functions.synthetic import AckleyMixed, Labs


def get_labs(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
    dim: int = 50,
) -> BenchmarkProblemRegistryEntry:
    def factory_fn(num_trials: int = num_trials) -> BenchmarkProblem:
        return _get_problem_from_common_inputs(
            bounds=[(0.0, 1.0) for _ in range(dim)],
            dim_int=dim,
            metric_name="Merit Factor",
            lower_is_better=False,  # Maximization problem
            observe_noise_sd=observe_noise_sd,
            test_problem_class=Labs,  # pyre-ignore[6]. Pyre is correct, but not worth fixing
            benchmark_name="Discrete Labs",
            num_trials=num_trials,
            optimal_value=4.0,  # This is more of an educated guess than a known value
            baseline_value=1.4149786013042274,  # From `compute_baseline_value_from_sobol`
        )

    return BenchmarkProblemRegistryEntry(
        factory_fn=factory_fn,
        factory_kwargs={},
    )


def get_ackley_mixed(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
    dim: int = 53,
) -> BenchmarkProblemRegistryEntry:
    def factory_fn(num_trials: int = num_trials) -> BenchmarkProblem:
        return _get_problem_from_common_inputs(
            bounds=[(0.0, 1.0) for _ in range(dim)],
            dim_int=dim - 3,
            metric_name="Ackley",
            lower_is_better=True,  # Minimization problem
            observe_noise_sd=observe_noise_sd,
            test_problem_class=AckleyMixed,  # pyre-ignore[6]
            benchmark_name="Ackley Mixed",
            num_trials=num_trials,
            optimal_value=0.0,  # This is more of an educated guess than a known value
            baseline_value=2.4836502123821997,  # From `compute_baseline_value_from_sobol`
        )

    return BenchmarkProblemRegistryEntry(
        factory_fn=factory_fn,
        factory_kwargs={},
    )


#  ------------ Registry ------------

DISCRETE_MIXED_PROBLEM_REGISTRY: dict[str, BenchmarkProblemRegistryEntry] = {
    "labs": get_labs(),
    "ackley_mixed": get_ackley_mixed(),
}
