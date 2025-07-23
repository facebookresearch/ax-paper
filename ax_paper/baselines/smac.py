# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import random
from pathlib import Path

import numpy as np

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace, TParameterization
from ax.core.trial import Trial
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax_paper.ax import CenterGenerationNode
from ax_paper.utils import gs_to_benchmark_method
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from pandas import DataFrame
from pyre_extensions import none_throws
from smac import (
    BlackBoxFacade as BBFacade,
    HyperparameterOptimizationFacade as HPOFacade,
    Scenario,
)
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runhistory.runhistory import logger

# Suppress SMAC logs about not overwriting trials.
logger.setLevel(logging.WARNING)

# This is the `ax_paper` directory where files like `setup.py` live.
root_dir = Path(__file__).parent.parent.parent.resolve()


def _search_space_to_config_space(search_space: SearchSpace) -> ConfigurationSpace:
    if search_space.is_hierarchical:
        raise NotImplementedError

    config_dict = {}
    for parameter in search_space.parameters.values():
        if isinstance(parameter, RangeParameter):
            if parameter.parameter_type is ParameterType.FLOAT:
                config_dict[parameter.name] = Float(
                    name=parameter.name,
                    bounds=(parameter.lower, parameter.upper),
                    log=parameter.log_scale,
                )
            elif parameter.parameter_type is ParameterType.INT:
                config_dict[parameter.name] = Integer(
                    name=parameter.name,
                    bounds=(parameter.lower, parameter.upper),
                    log=parameter.log_scale,
                )
            else:
                raise NotImplementedError
        elif isinstance(parameter, ChoiceParameter):
            config_dict[parameter.name] = Categorical(
                name=parameter.name,
                items=parameter.values,
                ordered=parameter.is_ordered,
            )
        else:
            raise NotImplementedError
    return ConfigurationSpace(config_dict)


def _ax_trial_to_smac_trial_value(
    trial_df: DataFrame, ax_objective: Objective
) -> TrialValue:
    """Converts data from an Ax trial to a SMAC trial value."""
    metrics = trial_df.set_index("metric_name")["mean"].to_dict()
    if isinstance(ax_objective, MultiObjective):
        cost_list = []
        for obj in ax_objective.objectives:
            raw_mean = metrics[obj.metric.name]
            if not obj.minimize:
                raw_mean = -raw_mean
            cost_list.append(raw_mean)
        return TrialValue(cost=cost_list)
    # Single objective case.
    raw_mean = metrics[ax_objective.metric.name]
    if not ax_objective.minimize:
        raw_mean = -raw_mean
    return TrialValue(cost=raw_mean)


class SMACGenerationNode(ExternalGenerationNode):
    """This is a generation node that defers to SMAC3 for candidate generation."""

    def __init__(
        self,
        *,
        num_trials: int,
        seed: int,
        name: str = "SMACGenerationNode",
        use_gp: bool = True,
    ) -> None:
        """Initialize the SMAC generation node.

        Args:
            num_trials: The budget for the experiment. This is used to determine the
                number of initialization trials.
            name: The name of the node.
            use_gp: Whether to use a GP for the surrogate model. If True, BBFacade
                is used, otherwise HPOFacade is used.
        """
        super().__init__(
            node_name=name,
            transition_criteria=None,
            should_deduplicate=False,
        )
        self.num_trials = num_trials
        self.seed = seed
        self._smac = None
        self.use_gp = use_gp

    @property
    def smac(self) -> HPOFacade | BBFacade:
        return none_throws(self._smac, "SMAC has not been initialized!")

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """Update the searcher state with the trials and data collected so far.
        This method will be called with the up-to-date experiment and data before
        ``get_next_candidate()`` is called to generate the next trial(s). Note
        that ``get_next_candidate()`` may be called multiple times (to generate
        multiple candidates) after a call to  ``update_generator_state()``.

        This will extract all the data from Ax experiment and attach it to the SMAC facade.

        Args:
            experiment: The ``Experiment`` object representing the current state of the
                experiment. The key properties includes ``trials``, ``search_space``,
                and ``optimization_config``. The data is provided as a separate arg.
            data: The data / metrics collected on the experiment so far.
        """
        config_space = _search_space_to_config_space(
            search_space=experiment.search_space
        )
        self._ax_parameter_order = list(experiment.search_space.parameters.keys())
        opt_config = experiment.optimization_config
        ax_objective = opt_config.objective
        metric_names = list(opt_config.metrics)
        if self._smac is None:
            if opt_config.outcome_constraints:
                raise NotImplementedError
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
                n_trials=self.num_trials,  # This determines initialization budget (25%)
                # To avoid evaluating the trials again. Setting this since we do not
                # use seeds in function evaluations.
                deterministic=True,
                output_directory=root_dir.joinpath(f"tmp/{random.random()}"),
                seed=self.seed,
            )
            if self.use_gp:
                self._smac = BBFacade(
                    scenario=scenario, target_function="None", overwrite=True
                )
            else:
                self._smac = HPOFacade(
                    scenario=scenario, target_function="None", overwrite=True
                )
        runhistory = self.smac.runhistory

        # Attach existing trials to smac.
        data_df = data.df
        for trial_index, ax_trial in experiment.trials.items():
            assert isinstance(ax_trial, Trial)
            if ax_trial.status not in [TrialStatus.RUNNING, TrialStatus.COMPLETED]:
                raise NotImplementedError
            config = Configuration(
                configuration_space=config_space,
                values=ax_trial.arm.parameters,
            )
            info = TrialInfo(
                config=config,
                seed=0,  # Dummy, errors out otherwise.
            )
            if (
                config_id := runhistory.config_ids.get(config, None)
            ) is not None and config_id in runhistory._cost_per_config:
                # Skip if the trial was already observed.
                continue
            elif ax_trial.status is TrialStatus.COMPLETED:
                # This is a new observation.
                trial_df = data_df[data_df["trial_index"] == trial_index]
                trial_df = trial_df[trial_df["metric_name"].isin(metric_names)]
                value = _ax_trial_to_smac_trial_value(
                    trial_df=trial_df, ax_objective=ax_objective
                )
                runhistory.add(  # add the trial with default status SUCCESS.
                    config=config,
                    cost=value.cost,
                    seed=0,
                    # It fails to register observations without force_update
                    # when we have externally generated trials (e.g., center point).
                    force_update=True,
                )
            elif config_id is None:
                # This is a new running trial that was generated externally.
                # For trials generated by SMAC, config_id will not be None.
                runhistory.add_running_trial(trial=info)

    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        """Get the parameters for the next candidate configuration to evaluate.

        Args:
            pending_parameters: A list of parameters of the candidates pending
                evaluation. Ignored here, since SMAC is aware of all the attached trials.

        Returns:
            A dictionary mapping parameter names to parameter values for the next
            candidate suggested by the method.
        """
        info = self.smac.ask()
        # NOTE: The parameter order may not match the parameter order in the search
        # space as SMAC reorders the parameters by calling sort (e.g., x10 will
        # come before x2). We need to reorder here to make sure we tensorize the
        # parameters correctly before evaluating the underlying objective/constraint.
        info_dict = dict(info.config)
        ordered_info_dict = {k: info_dict[k] for k in self._ax_parameter_order}
        # If any values are np.str_, convert them to str.
        for k, v in ordered_info_dict.items():
            if isinstance(v, np.str_):
                ordered_info_dict[k] = str(v)
        return ordered_info_dict


def get_smac_benchmark_method(
    *,
    benchmark_problem: BenchmarkProblem,
    max_pending_trials: int,
    seed: int,
    use_gp: bool,
    timeout_hours: float,
    sample_center: bool = False,
) -> BenchmarkMethod:
    smac_node = SMACGenerationNode(
        num_trials=benchmark_problem.num_trials, seed=seed, use_gp=use_gp
    )
    if sample_center:
        nodes = [CenterGenerationNode(next_node_name=smac_node.node_name), smac_node]
    else:
        nodes = [smac_node]
    name = "SMAC-BB" if use_gp else "SMAC-HPO"
    if sample_center:
        name = name + "+Center"
    gs = GenerationStrategy(name=name, nodes=nodes)
    return gs_to_benchmark_method(
        generation_strategy=gs,
        timeout_hours=timeout_hours,
        max_pending_trials=max_pending_trials,
    )
