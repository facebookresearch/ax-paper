# Copyright (c) Meta Platforms, Inc. and affiliates.

from unittest import TestCase

import numpy as np

from ax.benchmark.problems.registry import get_benchmark_problem
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.utils import run_trials_with_gs
from ax_paper.run_benchmark_helpers import BASELINE_REGISTRY
from ax_paper.utils import skip_if_import_error


class TestOptuna(TestCase):
    @skip_if_import_error
    def test_correct_data_attached(self, sample_center: bool = False) -> None:
        problem = get_benchmark_problem("branin")
        method = BASELINE_REGISTRY["Optuna"](
            benchmark_problem=problem,
            max_pending_trials=1,
            seed=0,
            timeout_hours=4.0,
            sample_center=sample_center,
        )
        gs = method.generation_strategy
        experiment = get_branin_experiment()
        num_trials = 5
        run_trials_with_gs(experiment=experiment, gs=gs, num_trials=num_trials)
        # Re-fit with latest data.
        gs.current_node._fit(experiment=experiment)
        study = gs.current_node._study
        self.assertEqual(len(study.trials), num_trials)
        optuna_df = study.trials_dataframe()
        ax_df = experiment.to_df()
        # Check that Optuna has the same data as Ax.
        self.assertTrue(
            np.array_equal(
                optuna_df[["value", "params_x1", "params_x2"]].to_numpy(),
                ax_df[["branin", "x1", "x2"]].to_numpy(),
            )
        )
        if sample_center:
            # Make sure center was evaluated.
            self.assertEqual(
                experiment.trials[0].arms[0].parameters, {"x1": 2.5, "x2": 7.5}
            )

    def test_optuna_with_center(self) -> None:
        self.test_correct_data_attached(sample_center=True)
