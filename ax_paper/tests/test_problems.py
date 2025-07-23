# Copyright (c) Meta Platforms, Inc. and affiliates.

from unittest import TestCase

import torch
from ax.benchmark.benchmark_problem import get_continuous_search_space
from ax.benchmark.benchmark_test_functions.botorch_test import BoTorchTestFunction
from ax.benchmark.problems.registry import get_benchmark_problem
from ax_paper.problems import AX_PAPER_PROBLEM_REGISTRY, AX_PAPER_PROBLEMS
from pyre_extensions import assert_is_instance


class TestProblemRegistry(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Prevent costly surrogate construction and data downloads
        self.problem_keys = [
            elt
            for elt in AX_PAPER_PROBLEMS
            if "LCBench" not in elt and "MNIST" not in elt and "SyneTune" not in elt
        ]

    def test_problem_registry(self) -> None:
        for problem_key in self.problem_keys:
            with self.subTest(problem_key=problem_key):
                problem = get_benchmark_problem(
                    problem_key, registry=AX_PAPER_PROBLEM_REGISTRY
                )
                self.assertIsNotNone(problem.baseline_value)
                self.assertIsInstance(problem.optimal_value, (float, int))

    def test_problem_naming(self) -> None:
        for problem_name in self.problem_keys:
            if problem_name.startswith("DTLZ") or problem_name.startswith("ZDT"):
                problem = get_benchmark_problem(
                    problem_name, registry=AX_PAPER_PROBLEM_REGISTRY
                )

                botorch_problem = assert_is_instance(
                    problem.test_function, BoTorchTestFunction
                ).botorch_problem

                name, obj_str, dim_str = problem_name.split("_")[:3]
                self.assertIn(name[:-1], {"DTLZ", "ZDT"})
                self.assertTrue(name[-1].isnumeric())

                self.assertEqual(obj_str[-1], "m")
                self.assertEqual(dim_str[-1], "d")
                n_obj = int(obj_str[:-1])
                self.assertEqual(botorch_problem.num_objectives, n_obj)
                n_dim = int(dim_str[:-1])
                self.assertEqual(botorch_problem.dim, n_dim)

                # test baseline and optimal values
                self.assertGreater(problem.optimal_value, problem.baseline_value)
                self.assertGreater(problem.optimal_value, 0.0)
                self.assertGreaterEqual(problem.baseline_value, 0.0)

    def test_no_duplicate_problem_names(self) -> None:
        problem_names = [
            get_benchmark_problem(
                problem_key=problem_key, registry=AX_PAPER_PROBLEM_REGISTRY
            ).name
            for problem_key in self.problem_keys
        ]
        self.assertEqual(len(problem_names), len(set(problem_names)))

    def test_shifted_function(self) -> None:
        # BBOB problems are shifted by default.
        bbob = get_benchmark_problem("BBOB01_01_2d", registry=AX_PAPER_PROBLEM_REGISTRY)
        self.assertEqual(
            bbob.search_space, get_continuous_search_space([(-10.0, 10.0)] * 2)
        )
        botorch_problem = assert_is_instance(bbob.test_function, BoTorchTestFunction)
        self.assertTrue(botorch_problem.use_shifted_function)
        self.assertIsInstance(botorch_problem._offset, torch.Tensor)
        self.assertTrue(
            torch.allclose(
                botorch_problem.tensorize_params({"x0": 0, "x1": 0}),
                -botorch_problem._offset,
            )
        )
