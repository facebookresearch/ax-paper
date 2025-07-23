# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Iterator
from logging import Logger
from typing import Literal

import optuna
import pandas as pd
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, Objective
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.core.trial import Trial
from ax.core.types import TParameterization
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStrategy,
)
from ax.utils.common.logger import get_logger
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)


OPTUNA_SAMPLER_CLASSES: dict[
    str, type[optuna.samplers.TPESampler | optuna.samplers.RandomSampler]
] = {
    "tpe": optuna.samplers.TPESampler,
    "random": optuna.samplers.RandomSampler,
}


class OptunaGenerationNode(ExternalGenerationNode):
    """
    A GenerationNode that uses Optuna to generate candidates. Should only ever be used
    as a single-node GenerationStrategy.

    Supports the following features:
        - SOO
        - MOO
        - High-dimensional
        - Discrete search spaces
        - Mixed search spaces
    """

    def __init__(
        self,
        node_name: str = "OptunaGenerationNode",
        sampler_name: Literal["tpe", "random"] = "tpe",
        pruner: optuna.pruners.BasePruner | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            node_name=node_name,
            should_deduplicate=False,  # NOTE: Setting this to `True` doesn't work.
            transition_criteria=[],
        )

        self._optuna_trials: dict[int, optuna.trial.Trial] = {}

        sampler_cls = OPTUNA_SAMPLER_CLASSES[sampler_name]

        self._sampler = sampler_cls(seed=seed)
        # IMPORTANT: According to docs, if None is specified
        # MedianPruner is used by default
        self._pruner = pruner

        self._maybe_study: optuna.Study | None = None
        self._maybe_distributions: (
            dict[str, optuna.distributions.BaseDistribution] | None
        ) = None
        self._maybe_objectives: list[Objective] | None = None

    @property
    def _study(self) -> optuna.Study:
        return none_throws(self._maybe_study)

    @property
    def _distributions(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return none_throws(self._maybe_distributions)

    @property
    def _objectives(self) -> list[Objective]:
        return none_throws(self._maybe_objectives)

    def _initialize_optuna(self, experiment: Experiment) -> None:
        """Use the Experiment to initialize the Optuna Study and distributions."""
        optimization_config = none_throws(experiment.optimization_config)
        # Raise an exception if there are constraints. Optuna supports parameter
        # constraints but not outcome constraints.
        if len(optimization_config.outcome_constraints) > 0:
            raise ValueError("Optuna does not support outcome constraints.")

        # Extract the objectives from the experiment
        objective = optimization_config.objective
        if isinstance(objective, MultiObjective):
            self._maybe_objectives = objective.objectives
        else:
            self._maybe_objectives = [objective]

        self._maybe_study = optuna.create_study(
            sampler=self._sampler,
            pruner=self._pruner,
            directions=[
                "minimize" if o.minimize else "maximize" for o in self._objectives
            ],
        )

        # Convert the search space to Optuna distributions.
        self._maybe_distributions = {
            name: _parameter_to_distribution(parameter=parameter)
            for name, parameter in experiment.search_space.parameters.items()
        }

    def _attach_existing_ax_trials(self, experiment: Experiment) -> None:
        """Attach pre-existing Ax trials.
        This is done by requesting the trials from Optuna then manally setting
        the trial parameters using the trial's storage mechanism.
        Data for these trials will then be attached in the regular
        `update_generator_state` flow.

        NOTE: We could also use study.add_trial setup but that only supports
        completed trials. We'd need to pick between that and `enqueue_trials`,
        which makes study generate the queued trials next rather than adding them
        as running trials.
        """
        for trial in experiment.trials.values():
            assert isinstance(trial, Trial)
            optuna_trial = self._study.ask()
            for name, value in none_throws(trial.arm).parameters.items():
                distribution = self._distributions[name]
                optuna_trial.storage.set_trial_param(
                    optuna_trial._trial_id, name, value, distribution
                )
                self._optuna_trials[optuna_trial._trial_id] = optuna_trial

    def _optuna_running_trials(self) -> Iterator[int]:
        for frozen_trial in self._study.trials:
            # only FrozenTrials have a state attribute
            if frozen_trial.state != optuna.trial.TrialState.RUNNING:
                continue
            yield frozen_trial.number

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """
        Create a fresh Optuna Study and attach existing data to it. Also convert the
        SearchSpace to Optuna distributions to be used when generating candidates.
        """
        # Set up the study and distributions if they have not been already.
        if self._maybe_study is None:
            self._initialize_optuna(experiment=experiment)
            self._attach_existing_ax_trials(experiment=experiment)

        # Attach data to the study if the trial is expecting it.
        for trial_number in self._optuna_running_trials():
            # trial.report has an effect on only Trials (not FrozenTrials)
            optuna_trial = self._optuna_trials[trial_number]
            ax_trial = assert_is_instance(experiment.trials[trial_number], Trial)
            # If the Ax trial is also still running do nothing.
            if ax_trial.status == TrialStatus.RUNNING:
                if len(self._objectives) > 1:
                    logger.debug(
                        "Optuna Pruners only work with a single objective; "
                        "not attaching intermediate values"
                    )
                    continue
                if data.true_df.empty:
                    logger.info("data.true_df is empty, skipping")
                    continue

                wide_df = data.true_df.pivot(
                    columns=["metric_name"],
                    index=["trial_index", "step"],
                    values="mean",
                )
                (objective,) = self._objectives
                objective_name = objective.metric.name
                # Isolate the single objective
                s: pd.Series = wide_df[objective_name]  # MultiIndex Series
                if ax_trial.index not in s.index.get_level_values(level="trial_index"):
                    logger.debug(f"No data found for Ax trial {ax_trial.index}")
                    continue
                for step, value in s.loc[ax_trial.index].to_dict().items():
                    # IMPORTANT: According to Optuna docs (see link below),
                    # this operation is idempotent for a given trial and step.
                    # Only the value from the first call is stored; subsequent calls are ignored
                    # Ref: optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#report
                    optuna_trial.report(value=value, step=step)

            # If a trial is completed look up its data and attach it to the study.
            elif ax_trial.status == TrialStatus.COMPLETED:
                values_dict = {
                    o["metric_name"]: o["mean"]
                    for o in data.df.loc[
                        data.df["trial_index"] == ax_trial.index
                    ].to_dict(orient="records")
                }
                self._study.tell(
                    trial=trial_number,
                    # Must report values in order of study.directions
                    values=[values_dict[o.metric.name] for o in self._objectives],
                )
            else:
                self._study.tell(
                    trial=trial_number,
                    state=_trial_status_to_trial_state(trial_status=ax_trial.status),
                )

    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        # Intentionally ignoring pending_parameters here since we already attach all
        # trials regardless of the status in `update_generator_state`.
        trial = self._study.ask(self._distributions)
        self._optuna_trials[trial.number] = trial
        return trial.params


class OptunaEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    def should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        optuna_node = assert_is_instance(current_node, OptunaGenerationNode)
        optuna_node.update_generator_state(
            experiment=experiment,
            data=experiment.lookup_data(),
        )
        # NOTE: Ax trial indices and Optuna trial numbers are used interchangeably
        # trial.should_prune() are only behave meaningful for Trials (will always
        # returns False for FrozenTrials)
        return {
            trial_number: "Early-Stopped by Optuna Pruner"
            for trial_number in trial_indices
            if optuna_node._optuna_trials[trial_number].should_prune()
        }


def get_optuna_benchmark_method(
    *,
    benchmark_problem: BenchmarkProblem,
    max_pending_trials: int,
    seed: int | None = None,
    timeout_hours: float,
    sample_center: bool = False,
    sampler_name: Literal["tpe", "random"] = "tpe",
    pruner: optuna.pruners.BasePruner | None = None,
    early_stopping: bool = False,
    distribute_replications: bool = False,
) -> BenchmarkMethod:
    optuna_node = OptunaGenerationNode(
        sampler_name=sampler_name,
        pruner=pruner,
        seed=seed,
    )
    nodes: list[GenerationNode] = []
    if sample_center:
        nodes.append(CenterGenerationNode(next_node_name=optuna_node.node_name))
    nodes.append(optuna_node)
    gs = GenerationStrategy(
        name="Optuna" + ("+Center" if sample_center else ""), nodes=nodes
    )
    early_stopping_strategy = OptunaEarlyStoppingStrategy() if early_stopping else None
    return BenchmarkMethod(
        name=gs.name,
        generation_strategy=gs,
        timeout_hours=timeout_hours,
        distribute_replications=distribute_replications,
        max_pending_trials=max_pending_trials,
        early_stopping_strategy=early_stopping_strategy,
    )


def _parameter_to_distribution(
    parameter: Parameter,
) -> optuna.distributions.BaseDistribution:
    # NOTE: If you add support for new cases here, update
    # _attach_existing_ax_trials as well.
    if isinstance(parameter, RangeParameter):
        if parameter.parameter_type == ParameterType.FLOAT:
            return FloatDistribution(
                low=parameter.lower, high=parameter.upper, log=parameter.log_scale
            )
        elif parameter.parameter_type == ParameterType.INT:
            return IntDistribution(
                low=int(parameter.lower),
                high=int(parameter.upper),
                log=parameter.log_scale,
            )
        else:
            raise NotImplementedError(
                f"{parameter.parameter_type=} is not supported by Optuna."
            )

    elif isinstance(parameter, ChoiceParameter):
        # Note: Optuna does not have a way to express `ChoiceParameter(is_ordered=True)`
        return CategoricalDistribution(parameter.values)
    else:
        raise NotImplementedError(
            f"Parameter type {type(parameter)} is not supported by Optuna."
        )


def _trial_status_to_trial_state(
    trial_status: TrialStatus,
) -> optuna.trial.TrialState:
    if trial_status == TrialStatus.COMPLETED:
        return optuna.trial.TrialState.COMPLETE
    elif trial_status == TrialStatus.ABANDONED:
        return optuna.trial.TrialState.FAIL
    elif trial_status == TrialStatus.FAILED:
        return optuna.trial.TrialState.FAIL
    elif trial_status == TrialStatus.EARLY_STOPPED:
        return optuna.trial.TrialState.PRUNED
    else:
        raise ValueError(f"Cannot convert {trial_status=} to Optuna TrialState.")
