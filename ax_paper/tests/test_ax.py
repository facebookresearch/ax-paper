# Copyright (c) Meta Platforms, Inc. and affiliates.

from unittest import TestCase

from ax.benchmark.benchmark import benchmark_replication
from ax.benchmark.problems.registry import get_benchmark_problem
from ax_paper.ax import get_ax_benchmark_method  # pyre-ignore[21]
from pyre_extensions import none_throws


class TestAxBenchmarkMethod(TestCase):
    def test_with_center(self) -> None:
        problem = get_benchmark_problem("branin", num_trials=6)
        method = get_ax_benchmark_method(
            benchmark_problem=problem, sample_center=True, max_pending_trials=1
        )
        self.assertEqual(len(method.generation_strategy._nodes), 3)
        self.assertEqual(
            method.generation_strategy._nodes[0].node_name, "CenterOfSearchSpace"
        )
        result = benchmark_replication(problem=problem, method=method, seed=0)
        summary = none_throws(result.experiment).to_df()
        self.assertEqual(
            summary["generation_node"].tolist(),
            ["CenterOfSearchSpace", "Sobol", "Sobol", "Sobol", "Sobol", "MBM"],
        )
