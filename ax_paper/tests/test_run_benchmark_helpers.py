# Copyright (c) Meta Platforms, Inc. and affiliates.

from unittest import TestCase

from ax.benchmark.problems.registry import get_benchmark_problem
from ax_paper.run_benchmark_helpers import (
    BASELINE_REGISTRY,
    command_line_name_to_method_name,
)
from ax_paper.utils import skip_if_import_error


class TestRunBenchmarkHelpers(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ax_names = [
            "Ax-FAST",
            "Ax-FAST-Center",
            "Ax-FAST-Center-ES",
            "Random",
            "Random+Center",
            "Random+Center+ES",
        ]
        self.hebo_names = ["HEBO", "HEBO+Center"]

    def _test_baseline_naming(self, command_line_method_name: str) -> None:
        problem = get_benchmark_problem("branin")

        method_name = command_line_name_to_method_name.get(
            command_line_method_name, command_line_method_name
        )
        method = BASELINE_REGISTRY[command_line_method_name](
            benchmark_problem=problem, seed=0, timeout_hours=1, max_pending_trials=1
        )
        self.assertEqual(method.max_pending_trials, 1)
        self.assertEqual(method.name, method_name)
        self.assertEqual(method.timeout_hours, 1)

        method = BASELINE_REGISTRY[command_line_method_name](
            benchmark_problem=problem, seed=3, timeout_hours=6, max_pending_trials=4
        )
        self.assertEqual(method.max_pending_trials, 4)
        self.assertEqual(method.name, method_name)
        self.assertEqual(method.timeout_hours, 6)

    @skip_if_import_error
    def test_non_hebo_baselines(self) -> None:
        """Baselines available in the 'benchmark-other' conda environment."""
        other_names = (
            set(BASELINE_REGISTRY.keys()) - set(self.hebo_names) - set(self.ax_names)
        )
        for command_line_method_name in other_names:
            self._test_baseline_naming(
                command_line_method_name=command_line_method_name
            )

    def test_ax_naming(self) -> None:
        """Baselines always available."""
        for name in self.ax_names:
            self._test_baseline_naming(command_line_method_name=name)

    @skip_if_import_error
    def test_hebo(self) -> None:
        """Baselines available in the 'benchmark-hebo' conda environment."""
        for name in self.hebo_names:
            self._test_baseline_naming(command_line_method_name=name)
