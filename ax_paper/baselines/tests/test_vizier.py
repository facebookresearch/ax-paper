# Copyright (c) Meta Platforms, Inc. and affiliates.

from unittest import TestCase

from ax.benchmark.benchmark import benchmark_replication
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.problems.registry import get_benchmark_problem
from ax_paper.run_benchmark_helpers import BASELINE_REGISTRY
from ax_paper.utils import skip_if_import_error


def optimize_with_benchmark_method(benchmark_problem: BenchmarkProblem) -> list[float]:
    method = BASELINE_REGISTRY["Vizier"](
        benchmark_problem=benchmark_problem,
        seed=0,
        timeout_hours=0.1,
        max_pending_trials=1,
    )
    result = benchmark_replication(problem=benchmark_problem, method=method, seed=0)
    return result.optimization_trace.tolist()


class TestVizier(TestCase):
    @skip_if_import_error
    def test_w_branin(self) -> None:
        num_trials = 3
        problem = get_benchmark_problem("branin", num_trials=num_trials)
        benchmark_results = optimize_with_benchmark_method(benchmark_problem=problem)
        self.assertEqual(len(benchmark_results), num_trials)

    @skip_if_import_error
    def test_w_hartmann(self) -> None:
        num_trials = 3
        problem = get_benchmark_problem("hartmann6", num_trials=num_trials)
        benchmark_results = optimize_with_benchmark_method(benchmark_problem=problem)
        self.assertEqual(len(benchmark_results), num_trials)
