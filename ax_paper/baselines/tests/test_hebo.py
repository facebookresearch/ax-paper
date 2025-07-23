# Copyright (c) Meta Platforms, Inc. and affiliates.

from unittest import TestCase

import numpy as np
import pandas as pd
from ax.benchmark.benchmark import benchmark_replication
from ax.benchmark.problems.registry import get_benchmark_problem
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.utils import run_trials_with_gs
from ax_paper.run_benchmark_helpers import BASELINE_REGISTRY
from ax_paper.utils import skip_if_import_error
from pytest import mark
from scipy import stats


class TestHebo(TestCase):
    @mark.skip(reason="slow")
    def test_against_native(self, use_gp: bool = True) -> None:
        egn_results = [_optimize_with_egn(seed=i) for i in range(10)]
        native_results = [_optimize_native() for _ in range(10)]

        _t_stat, p_val = stats.ttest_ind(egn_results, native_results)
        self.assertGreater(p_val, 0.05)

    @skip_if_import_error
    def test_correct_data_attached(self, sample_center: bool = False) -> None:
        problem = get_benchmark_problem("branin")
        experiment = get_branin_experiment()
        gs = BASELINE_REGISTRY["HEBO+Center" if sample_center else "HEBO"](
            benchmark_problem=problem,
            max_pending_trials=1,
            seed=0,
            timeout_hours=4.0,
        ).generation_strategy
        num_trials = 5
        run_trials_with_gs(experiment=experiment, gs=gs, num_trials=num_trials)
        # Re-fit HEBO with latest data.
        gs.current_node._fit(experiment=experiment)
        # Compare HEBO data to Ax data.
        hebo_X = gs.current_node._opt.X
        hebo_Y = gs.current_node._opt.y
        exp_df = experiment.to_df()
        # Negating Ys because of minimization.
        self.assertEqual(hebo_Y.flatten().tolist(), (-exp_df["branin"]).to_list())
        self.assertEqual(hebo_X.to_dict(), exp_df[["x1", "x2"]].to_dict())
        self.assertEqual(len(hebo_X), num_trials)
        self.assertEqual(len(hebo_Y), num_trials)
        if sample_center:
            # Verify that center was generated.
            first_param = experiment.trials[0].arms[0].parameters
            self.assertEqual(first_param, {"x1": 2.5, "x2": 7.5})
            # Check that HEBO is aware of the center trial.
            hebo_X = gs.current_node._opt.X
            self.assertEqual(hebo_X.loc[0].to_dict(), {"x1": 2.5, "x2": 7.5})

    def test_with_center(self) -> None:
        self.test_correct_data_attached(sample_center=True)


def _optimize_with_egn(seed: int) -> float:
    problem = get_benchmark_problem("branin", num_trials=30)

    result = benchmark_replication(
        problem=problem,
        method=BASELINE_REGISTRY["HEBO"](
            benchmark_problem=problem, seed=seed, timeout_hours=4.0
        ),
        seed=seed,
    )

    return result.optimization_trace[-1]


def _optimize_native() -> float:
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.general import GeneralBO

    problem = get_benchmark_problem("branin", num_trials=30)

    def obj(params: pd.DataFrame) -> np.ndarray:
        return (
            problem.test_function.evaluate_true(params.to_dict(orient="records").pop())
            .numpy()
            .reshape(-1, 1)
        )

    space = DesignSpace().parse(
        [
            {"name": "x0", "type": "num", "lb": -5.0, "ub": 10.0},
            {"name": "x1", "type": "num", "lb": 0.0, "ub": 15.0},
        ]
    )
    opt = GeneralBO(space)

    for _ in range(problem.num_trials):
        rec = opt.suggest(n_suggestions=1)
        opt.observe(rec, obj(rec))

    return opt.y.min()
