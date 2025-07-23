# Copyright (c) Meta Platforms, Inc. and affiliates.

import random
import shutil
from pathlib import Path
from unittest import TestCase

import numpy as np
from ax.benchmark.benchmark import benchmark_replication
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.problems.registry import get_benchmark_problem
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, ScalarizedObjective
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.utils import run_trials_with_gs
from ax_paper.problems.base import AX_PAPER_PROBLEM_REGISTRY
from ax_paper.run_benchmark_helpers import BASELINE_REGISTRY
from ax_paper.utils import skip_if_import_error
from pytest import mark


def get_smac_native(benchmark_problem: BenchmarkProblem, use_gp: bool):
    from ax_paper.baselines.smac import _search_space_to_config_space
    from smac import (
        BlackBoxFacade as BBFacade,
        HyperparameterOptimizationFacade as HPOFacade,
        Scenario,
    )

    experiment = Experiment(
        search_space=benchmark_problem.search_space,
        optimization_config=benchmark_problem.optimization_config,
    )
    config_space = _search_space_to_config_space(search_space=experiment.search_space)

    opt_config = experiment.optimization_config
    if opt_config.outcome_constraints:
        raise NotImplementedError
    ax_objective = opt_config.objective
    if isinstance(ax_objective, ScalarizedObjective):
        raise NotImplementedError

    smac_objectives = (
        [m.name for m in ax_objective.metrics]
        if isinstance(ax_objective, MultiObjective)
        else ax_objective.metric.name
    )
    scenario = Scenario(
        configspace=config_space,
        objectives=smac_objectives,
        n_trials=benchmark_problem.num_trials,
        deterministic=True,
        output_directory=Path(f"tmp/{random.random()}"),
    )
    if use_gp:
        return BBFacade(scenario=scenario, target_function="None", overwrite=True)
    else:
        return HPOFacade(scenario=scenario, target_function="None", overwrite=True)


def optimize_with_native(
    benchmark_problem: BenchmarkProblem, use_gp: bool
) -> list[float]:
    from smac.runhistory.dataclasses import TrialValue

    negate = not benchmark_problem.optimization_config.objective.minimize
    objectives = []
    smac = get_smac_native(benchmark_problem, use_gp=use_gp)
    for _ in range(benchmark_problem.num_trials):
        info = smac.ask()
        params = dict(info.config)
        cost = benchmark_problem.test_function.evaluate_true(params)
        objectives.append(cost.item())
        if negate:
            cost = -cost
        value = TrialValue(cost=cost)
        smac.tell(info, value)
    return np.minimum.accumulate(objectives).tolist()


def optimize_with_benchmark_method(
    benchmark_problem: BenchmarkProblem, use_gp: bool
) -> list[float]:
    method = BASELINE_REGISTRY["SMAC-BB" if use_gp else "SMAC-HPO"](
        benchmark_problem=benchmark_problem,
        max_pending_trials=1,
        seed=0,
        use_gp=use_gp,
        timeout_hours=4.0,
    )
    result = benchmark_replication(problem=benchmark_problem, method=method, seed=0)
    return result.optimization_trace.tolist()


class TestSmac(TestCase):
    def tearDown(self) -> None:
        # Cleanup tmp dir, where SMAC dumps its internal logs.
        ax_benchmark_dir = Path(__file__).parent.parent.parent.parent.resolve()
        tmp_dir: Path = ax_benchmark_dir.joinpath("tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    @mark.skip(reason="slow")
    @skip_if_import_error
    def test_against_native(self, use_gp: bool = True) -> None:
        problem = get_benchmark_problem("branin", num_trials=10)
        native_results = optimize_with_native(benchmark_problem=problem, use_gp=use_gp)
        benchmark_results = optimize_with_benchmark_method(
            benchmark_problem=problem, use_gp=use_gp
        )
        for n, b in zip(native_results, benchmark_results, strict=True):
            self.assertAlmostEqual(n, b)

    def test_against_native_hpo(self) -> None:
        self.test_against_native(use_gp=False)

    @skip_if_import_error
    def test_smac_data(self, sample_center: bool = False) -> None:
        problem = get_benchmark_problem("branin")
        experiment = get_branin_experiment()
        gs = BASELINE_REGISTRY["SMAC-HPO+Center" if sample_center else "SMAC-HPO"](
            benchmark_problem=problem,
            max_pending_trials=1,
            seed=0,
            use_gp=False,
            timeout_hours=4.0,
        ).generation_strategy
        num_trials = 5
        run_trials_with_gs(experiment=experiment, gs=gs, num_trials=num_trials)
        # Re-fit with latest data.
        gs.current_node.update_generator_state(
            experiment=experiment, data=experiment.lookup_data()
        )
        # Compare SMAC data to Ax data.
        runhistory = gs.current_node.smac.runhistory
        self.assertEqual(runhistory._finished, num_trials)
        for trial_key in runhistory:
            index = trial_key.config_id
            config = runhistory.get_config(index)
            parameters = dict(config)
            if index == 1 and sample_center:  # SMAC index starts from 1.
                # Make sure we sampled the center.
                self.assertEqual(parameters, {"x1": 2.5, "x2": 7.5})
            self.assertEqual(
                parameters, experiment.trials[index - 1].arms[0].parameters
            )
            cost = runhistory._cost_per_config[index]
            ax_data, _ = experiment.lookup_data_for_trial(trial_index=index - 1)
            # Negate for minimization.
            self.assertEqual(ax_data.df["mean"].item(), -cost)

    def test_smac_with_center(self) -> None:
        self.test_smac_data(sample_center=True)

    @skip_if_import_error
    def test_mixed_search_space(self) -> None:
        problem = get_benchmark_problem(
            problem_key="ackley_mixed",
            registry=AX_PAPER_PROBLEM_REGISTRY,
            # We only need one trial as the mixed Ackley problem will error out given
            # how SMAC reorders the parameters.
            num_trials=1,
        )
        method = BASELINE_REGISTRY["SMAC-HPO"](
            benchmark_problem=problem,
            seed=0,
            timeout_hours=4,
            max_pending_trials=1,
        )
        benchmark_replication(problem=problem, method=method, seed=0)
        # Explicitly check parameter ordering
        experiment = Experiment(
            search_space=problem.search_space,
            optimization_config=problem.optimization_config,
        )
        gr = method.generation_strategy.gen_single_trial(experiment=experiment, n=1)
        parameters = gr.arms[0].parameters
        self.assertEqual(
            list(parameters.keys()), list(problem.search_space.parameters.keys())
        )
