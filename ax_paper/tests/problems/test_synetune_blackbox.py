# Copyright (c) Meta Platforms, Inc. and affiliates.

from unittest import TestCase

from ax.benchmark.problems.registry import get_benchmark_problem
from ax_paper.problems.synetune_blackbox import PROBLEM_REGISTRY
from pytest import mark
from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.neighbors._regression import KNeighborsRegressor
from syne_tune.blackbox_repository.blackbox_surrogate import BlackboxSurrogate
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular


class TestSyneTuneBlackbox(TestCase):
    @mark.skip(reason="Slow")
    def test_from_registry(self) -> None:
        expected_blackbox_classes = {
            "SyneTune/fcnet/protein_structure": BlackboxTabular,
            "SyneTune/nasbench201/cifar10": BlackboxTabular,
            "SyneTune/lcbench/KNeighborsRegressor/Fashion-MNIST": BlackboxSurrogate,
            "SyneTune/lcbench/RandomForestRegressor/Fashion-MNIST": BlackboxSurrogate,
        }
        expected_surrogate_classes = {
            "SyneTune/lcbench/RandomForestRegressor/Fashion-MNIST": RandomForestRegressor,
            "SyneTune/lcbench/KNeighborsRegressor/Fashion-MNIST": KNeighborsRegressor,
        }
        for key, expected_blackbox_class in expected_blackbox_classes.items():
            problem = get_benchmark_problem(problem_key=key, registry=PROBLEM_REGISTRY)
            blackbox = problem.test_function.blackbox
            self.assertIsInstance(blackbox, expected_blackbox_class)
            if key in expected_surrogate_classes:
                self.assertIsInstance(
                    blackbox.surrogate, expected_surrogate_classes[key]
                )
