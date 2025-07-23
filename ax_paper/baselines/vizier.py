# Copyright (c) Meta Platforms, Inc. and affiliates.

import random
import time

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, ScalarizedObjective
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace, TParameterization
from ax.core.trial import Trial
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.best_point import _is_row_feasible
from ax_paper.utils import gs_to_benchmark_method
from pyre_extensions import none_throws
from vizier.service import clients, pyvizier as vz


def _search_space_to_vizier_problem(search_space: SearchSpace) -> vz.ProblemStatement:
    if search_space.is_hierarchical:
        raise NotImplementedError
    problem = vz.ProblemStatement()
    for parameter in search_space.parameters.values():
        if isinstance(parameter, RangeParameter):
            scale_type = (
                vz.ScaleType.LOG if parameter.log_scale else vz.ScaleType.LINEAR
            )
            if parameter.parameter_type is ParameterType.FLOAT:
                problem.search_space.root.add_float_param(
                    name=parameter.name,
                    min_value=parameter.lower,
                    max_value=parameter.upper,
                    scale_type=scale_type,
                )
            elif parameter.parameter_type is ParameterType.INT:
                if parameter.lower + 1 == parameter.upper:
                    # Otherwise Vizier suggests the trial over and over again
                    problem.search_space.root.add_categorical_param(
                        name=parameter.name,
                        feasible_values=[str(parameter.lower), str(parameter.upper)],
                    )
                else:
                    problem.search_space.root.add_int_param(
                        name=parameter.name,
                        min_value=parameter.lower,
                        max_value=parameter.upper,
                        scale_type=scale_type,
                    )
            else:
                raise NotImplementedError
        elif isinstance(parameter, ChoiceParameter):
            if parameter.is_ordered:
                problem.search_space.root.add_discrete_param(
                    name=parameter.name,
                    feasible_values=parameter.values,
                )
            else:
                problem.search_space.root.add_categorical_param(
                    name=parameter.name,
                    feasible_values=parameter.values,
                )
        else:
            raise NotImplementedError
    return problem


class VizierGenerationNode(ExternalGenerationNode):
    """This is a generation node that defers to Vizier for candidate generation."""

    def __init__(
        self,
        *,
        name: str = "VizierGenerationNode",
    ) -> None:
        super().__init__(
            node_name=name,
            transition_criteria=None,
            should_deduplicate=False,
        )
        self._client = None

    @property
    def client(self) -> clients.Study:
        return none_throws(self._client, "Client has not been initialized!")

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """Update the searcher state with the trials and data collected so far.
        This method will be called with the up-to-date experiment and data before
        ``get_next_candidate()`` is called to generate the next trial(s). Note
        that ``get_next_candidate()`` may be called multiple times (to generate
        multiple candidates) after a call to  ``update_generator_state()``.

        This will extract all the data from Ax experiment and attach it to the Vizier client.

        Args:
            experiment: The ``Experiment`` object representing the current state of the
                experiment. The key properties includes ``trials``, ``search_space``,
                and ``optimization_config``. The data is provided as a separate arg.
            data: The data / metrics collected on the experiment so far.
        """
        self.binary_parameters = [
            p.name
            for p in experiment.search_space.parameters.values()
            if isinstance(p, RangeParameter)
            and p.parameter_type == ParameterType.INT
            and p.lower + 1 == p.upper
        ]
        # Vizier doesn't respect type int
        self.should_be_rounded = [
            p.name
            for p in experiment.search_space.parameters.values()
            if (isinstance(p, RangeParameter) or isinstance(p, ChoiceParameter))
            and p.parameter_type == ParameterType.INT
            and p.name not in self.binary_parameters
        ]
        problem = _search_space_to_vizier_problem(search_space=experiment.search_space)
        opt_config = experiment.optimization_config
        if isinstance(opt_config.objective, ScalarizedObjective):
            raise NotImplementedError
        objectives = (
            opt_config.objective.objectives
            if isinstance(opt_config.objective, MultiObjective)
            else [opt_config.objective]
        )
        for objective in objectives:
            problem.metric_information.append(
                vz.MetricInformation(
                    name=objective.metric.name,
                    goal=(
                        vz.ObjectiveMetricGoal.MINIMIZE
                        if objective.minimize
                        else vz.ObjectiveMetricGoal.MAXIMIZE
                    ),
                )
            )
        study_config = vz.StudyConfig.from_problem(problem)
        study_config.algorithm = "DEFAULT"
        # NOTE: The study ID has to be unique. Otherwise, it mixes things up under the hood.
        self._client = clients.Study.from_study_config(
            study_config,
            owner="owner",
            study_id=f"dummy_id_{random.random()}_{time.time()}",
        )
        # Attach existing trials to the study.
        data_df = data.df
        for trial_index, ax_trial in experiment.trials.items():
            assert isinstance(ax_trial, Trial)
            if ax_trial.status not in [TrialStatus.RUNNING, TrialStatus.COMPLETED]:
                raise NotImplementedError
            converted_params = {}
            for k, v in ax_trial.arm.parameters.items():
                if k in self.binary_parameters:
                    converted_params[k] = str(v)
                elif k in self.should_be_rounded:
                    converted_params[k] = float(v)
                else:
                    converted_params[k] = v
            if ax_trial.status is TrialStatus.COMPLETED:
                metric_names = list(opt_config.metrics)
                trial_df = data_df[data_df["trial_index"] == trial_index]
                trial_df = trial_df[trial_df["metric_name"].isin(metric_names)]
                metrics = trial_df.set_index("metric_name")["mean"].to_dict()
                is_feasible = _is_row_feasible(
                    df=trial_df, optimization_config=opt_config
                ).all()
                trial = vz.Trial(
                    parameters=converted_params,
                    final_measurement=vz.Measurement(metrics=metrics),
                    # Mark the trial infeasible if it violates outcome constraints
                    infeasibility_reason=(
                        "Violates outcome constraints" if not is_feasible else None
                    ),
                )
            else:
                trial = vz.Trial(
                    parameters=converted_params,
                )
            self._client.add_trial(trial)

    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        """Get the parameters for the next candidate configuration to evaluate.

        Args:
            pending_parameters: A list of parameters of the candidates pending
                evaluation. Ignored here, since the Vizier client is aware
                of all the attached trials.

        Returns:
            A dictionary mapping parameter names to parameter values for the next
            candidate suggested by the method.
        """
        existing_trials = list(self.client.trials())
        count = 1
        while True:
            # If there is a trial that hasn't been run, the client re-suggests it
            # (without re-generating, so this is super cheap).
            # We'll compare the suggestions to the existing trials and request
            # more until we get a fresh suggestion.
            suggestion = self.client.suggest(count=count)[-1]
            if suggestion not in existing_trials:
                new_parameters = {}
                for p in suggestion.parameters.keys():
                    if p in self.binary_parameters or p in self.should_be_rounded:
                        new_parameters[p] = int(suggestion.parameters[p])
                    else:
                        new_parameters[p] = suggestion.parameters[p]
                return new_parameters
            count += 1


def get_vizier_benchmark_method(
    *,
    benchmark_problem: BenchmarkProblem,
    max_pending_trials: int,
    seed: int,
    timeout_hours: float,
) -> BenchmarkMethod:
    gs = GenerationStrategy(name="Vizier", nodes=[VizierGenerationNode()])
    return gs_to_benchmark_method(
        generation_strategy=gs,
        timeout_hours=timeout_hours,
        max_pending_trials=max_pending_trials,
    )
