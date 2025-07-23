# Copyright (c) Meta Platforms, Inc. and affiliates.

from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.core.search_space import TParameterization
from ax.core.trial import Trial
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax_paper.ax import CenterGenerationNode
from ax_paper.utils import gs_to_benchmark_method
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.abstract_optimizer import AbstractOptimizer
from hebo.optimizers.general import GeneralBO
from hebo.optimizers.hebo import HEBO
from pyre_extensions import assert_is_instance, none_throws


class HEBOGenerationNode(ExternalGenerationNode):
    """
    A GenerationNode that uses HEBO to generate candidates. Should only ever be used
    as a single-node GenerationStrategy.

    Supports the following features:
        - SOO
        - MOO
        - High-dimensional
        - Discrete search spaces
        - Mixed search spaces
    """

    def __init__(self, node_name: str = "HEBOGenerationNode") -> None:
        super().__init__(
            node_name=node_name, should_deduplicate=False, transition_criteria=[]
        )
        self._maybe_opt: AbstractOptimizer | None = None

        self._observed: set[int] = set()

    @property
    def _opt(self) -> AbstractOptimizer:
        return none_throws(self._maybe_opt)

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        # Grab some items off the OptimizationConfig for use configuring the optimizer
        # and transforming the data.
        optimization_config = none_throws(experiment.optimization_config)
        if isinstance(optimization_config.objective, MultiObjective):
            objectives = optimization_config.objective.objectives
        else:
            objectives = [optimization_config.objective]
        outcome_constraints = optimization_config.outcome_constraints

        # NOTE: HEBO explicitly checks against the size of the training data
        # to determine whether to use BO or random search. So, we can safely
        # re-construct the optimizer without worrying about state. See
        # https://github.com/huawei-noah/HEBO/blob/master/HEBO/hebo/optimizers/hebo.py#L122-L124
        space = DesignSpace().parse(
            [
                _parameter_to_design_space_record(parameter=parameter)
                for parameter in experiment.search_space.parameters.values()
            ]
        )

        if len(objectives) + len(outcome_constraints) > 1:
            # Using GeneralBO rather than HEBO because it supports constraints and MOO.
            self._maybe_opt = GeneralBO(
                space=space,
                # HEBO uses these two parameters to determine the shape of the y array
                # it will expect.
                num_obj=len(objectives),
                num_constr=len(outcome_constraints),
            )
        else:
            self._maybe_opt = HEBO(space=space)

        # Transform Data into xs and ys
        # HEBO expects x to be a dataframe with one row per observation and one column
        # per parameter.
        # HEBO expects y to be a 2D array with one row per observation and one column
        # per metric, with objectives on the left and constraints on the right. Metrics
        # must always be reported in the same order, all objectives must be the form of
        # minimize f_n(x), and all constraints must be the form of c_n(x) >= 0.
        x_records = {
            index: none_throws(assert_is_instance(trial, Trial).arm).parameters
            for index, trial in experiment.trials.items()
            if trial.status is TrialStatus.COMPLETED and index not in self._observed
        }
        if len(x_records) == 0:  # If there's nothing to observe exit early
            return

        # Collect metric readings in the order HEBO expects and transform them as
        # necessary.
        data_df = data.df
        obj_ys = []
        for objective in objectives:
            # Create a Nx1 array of objective values, where N is the number of
            # observations.
            raw_ys = np.array(
                [
                    [
                        data_df[
                            (data_df["trial_index"] == index)
                            & (data_df["metric_name"] == objective.metric.name)
                        ]["mean"].item()
                        for index in x_records.keys()
                    ]
                ]
            ).T

            # All objectives must be the form of minimize f_n(x)
            ys = raw_ys if objective.minimize else -raw_ys

            obj_ys.append(ys)

        constr_ys = []
        for constraint in outcome_constraints:
            # Create a Nx1 array of constraint values, where N is the number of
            # observations.
            raw_ys = np.array(
                [
                    [
                        data_df[
                            (data_df["trial_index"] == index)
                            & (data_df["metric_name"] == constraint.metric.name)
                        ]["mean"].item()
                        for index in x_records.keys()
                    ]
                ]
            ).T

            if constraint.op == ">=":
                ys = raw_ys - constraint.bound
            else:
                ys = -raw_ys + constraint.bound

            constr_ys.append(ys)

        # Form the x dataframe and y Nx(M+N) array, where M is the number of objectives
        # and N is the number of constraints.
        x = pd.DataFrame.from_records([*x_records.values()])
        y = np.hstack([*obj_ys, *constr_ys])

        self._opt.observe(x, y)  # Attach the observations to the optimizer.

    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        """
        Args:
            pending_parameters: HEBO has no notion of pending parameters for
                deduplication so we ignore this argument.
        """
        # HEBO prints lots of statements like "jitter = 0.001" and then
        # eventually may print a statement like "jitter is too large, output
        # random predictions." Suppress all except the last.
        f = StringIO()
        with redirect_stdout(f):
            cand = self._opt.suggest().to_dict(orient="records").pop()
        out = f.getvalue()
        for line in out.split("\n"):
            if ("jitter = " not in line) and (len(line) > 0):
                print(line)
        return cand


def get_hebo_benchmark_method(
    *,
    benchmark_problem: BenchmarkProblem,
    max_pending_trials: int,
    seed: int,
    timeout_hours: float,
    sample_center: bool = False,
) -> BenchmarkMethod:
    hebo_node = HEBOGenerationNode()
    if sample_center:
        nodes = [
            CenterGenerationNode(next_node_name=hebo_node.node_name),
            hebo_node,
        ]
    else:
        nodes = [hebo_node]
    gs = GenerationStrategy(
        name="HEBO" + ("+Center" if sample_center else ""), nodes=nodes
    )
    return gs_to_benchmark_method(
        generation_strategy=gs,
        timeout_hours=timeout_hours,
        max_pending_trials=max_pending_trials,
    )


def _parameter_to_design_space_record(parameter: Parameter) -> dict[str, Any]:
    if isinstance(parameter, RangeParameter):
        if parameter.parameter_type == ParameterType.FLOAT:
            if parameter.log_scale:
                return {
                    "name": parameter.name,
                    "type": "pow",
                    "base": 10,
                    "lb": parameter.lower,
                    "ub": parameter.upper,
                }
            return {
                "name": parameter.name,
                "type": "num",
                "lb": parameter.lower,
                "ub": parameter.upper,
            }
        elif parameter.parameter_type == ParameterType.INT:
            # Log scale int parameters are not supported by HEBO
            return {
                "name": parameter.name,
                "type": "int",
                "lb": parameter.lower,
                "ub": parameter.upper,
            }
        else:
            raise NotImplementedError(
                f"HEBO does not support {parameter.parameter_type=}"
            )

    elif isinstance(parameter, ChoiceParameter):
        # Ordered choice parameters are not supported by HEBO
        return {
            "name": parameter.name,
            "type": "cat",
            "categories": parameter.values,
        }
    else:
        raise NotImplementedError(f"HEBO does not support {type(parameter)=}")
