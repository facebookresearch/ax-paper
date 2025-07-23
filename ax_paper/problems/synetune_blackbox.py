# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, InitVar
from typing import Any

import numpy as np
import torch

from ax.benchmark.benchmark_problem import BenchmarkProblem, get_soo_opt_config
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.problems.registry import BenchmarkProblemRegistryEntry
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import _get_parameter_type, ChoiceParameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from syne_tune.blackbox_repository import add_surrogate, load_blackbox
from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.blackbox_repository.simulated_tabular_backend import make_surrogate
from syne_tune.config_space import Categorical, FiniteRange, Float, Integer

# Passed to scikit-learn model initialization
SURROGATE_KWARGS = {
    # Passed to scikit-learn model initialization
    "RandomForestRegressor": {"random_state": 42},
    "KNeighborsRegressor": {"n_neighbors": 1},
}
# Passed to SyneTune's `add_surrogate` method
ADD_SURROGATE_KWARGS = {
    # fit multi-output surrogate to predict learning curve
    "predict_curves": True,
    # consolidate data from all benchmark repetitions to fit
    # one single surrogate
    "separate_seeds": False,
}


DEFAULT_NUM_TRIALS = 50
DEFAULT_ELAPSED_TIME_NAME = "metric_elapsed_time"
REPETITION = 0

runtime_scalings = {
    "lcbench": {
        "Fashion-MNIST": 0.060762135339571664,
        "airlines": 0.06703194110451897,
        "albert": 0.06970224836548143,
        "christine": 0.030640572359308737,
        "covertype": 0.05223329370672019,
    },
    "nasbench201": {
        "cifar10": 0.05246801921474679,
        "cifar100": 0.050610947094724974,
        "ImageNet16-120": 0.018270878984510255,
    },
    "fcnet": {
        "protein_structure": 0.4143615319540671,
        "naval_propulsion": 1.516592503412628,
        "parkinsons_telemonitoring": 3.0975580604553428,
        "slice_localization": 0.2772056003855567,
    },
}


def _to_search_space(config_space: Mapping[str, Any]) -> SearchSpace:
    parameters = []
    for name, domain in config_space.items():
        if isinstance(domain, (Float, Integer)):
            parameter_type = _get_parameter_type(domain.value_type)
            sampler = domain.get_sampler()
            log_scale = isinstance(sampler, (Float._LogUniform, Integer._LogUniform))
            parameter = RangeParameter(
                name=name,
                parameter_type=parameter_type,
                lower=domain.lower,
                upper=domain.upper,
                log_scale=log_scale,
            )
        elif isinstance(domain, FiniteRange):
            parameter_type = _get_parameter_type(domain.value_type)
            if domain.log_scale:
                _values = np.logspace(
                    start=np.log10(domain.lower),
                    stop=np.log10(domain.upper),
                    num=domain.size,
                    endpoint=True,
                )
            else:
                _values = np.linspace(
                    start=domain.lower,
                    stop=domain.upper,
                    num=domain.size,
                    endpoint=True,
                )
            if domain.cast_int:
                _values = np.rint(_values).astype(int)
            values = _values.tolist()
            parameter = ChoiceParameter(
                name=name,
                parameter_type=parameter_type,
                values=values,
                is_ordered=True,
                sort_values=True,
            )
        elif isinstance(domain, Categorical):
            parameter_type = _get_parameter_type(domain.value_type)
            parameter = ChoiceParameter(
                name=name,
                parameter_type=parameter_type,
                values=domain.categories,
                is_ordered=False,
                sort_values=False,
            )
        else:
            # TODO: notably support for Ordinal parameters has not yet
            # been implemented (not difficult)
            raise NotImplementedError(
                f"Parameter `{name}` has unsupported type: {domain}"
            )
        parameters.append(parameter)
    return SearchSpace(parameters=parameters)


@dataclass(kw_only=True)
class SyneTuneBlackboxTestFunction(BenchmarkTestFunction):
    outcome_names: Sequence[str]

    outcome_indices: Sequence[int] = field(init=False)
    n_steps: int = field(init=False)

    blackbox_name: str
    dataset_name: str | None = None
    repetition: int = 0

    # pyre-ignore [16]: Pyre doesn't understand InitVars.
    elapsed_time_name: InitVar[str] = DEFAULT_ELAPSED_TIME_NAME
    elapsed_time_index: int = field(init=False)

    surrogate_name: str | None = None
    blackbox: Blackbox = field(init=False)

    def __post_init__(self, elapsed_time_name) -> None:
        self.runtime_scale_factor = runtime_scalings[self.blackbox_name][
            self.dataset_name
        ]
        _blackbox = load_blackbox(self.blackbox_name)[self.dataset_name]

        if self.surrogate_name is None:
            self.blackbox = _blackbox
        else:
            self.blackbox = add_surrogate(
                _blackbox,
                surrogate=make_surrogate(
                    self.surrogate_name,
                    surrogate_kwargs=SURROGATE_KWARGS[self.surrogate_name],
                ),
                **ADD_SURROGATE_KWARGS,
            )

        _objectives_names = self.blackbox.objectives_names
        if unrecognized := set(self.outcome_names) - set(_objectives_names):
            raise ValueError(
                f"Outcome name(s) not recognized {list(unrecognized)} "
                f"(must be one of {sorted(_objectives_names)})"
            )

        self.elapsed_time_index = _objectives_names.index(elapsed_time_name)
        self.outcome_indices = [
            _objectives_names.index(name) for name in self.outcome_names
        ]
        self.n_steps = max(self.blackbox.fidelity_values).item()

    def get_search_space(self) -> SearchSpace:
        return _to_search_space(self.blackbox.configuration_space)

    def evaluate_true(self, params: Mapping[str, TParamValue]) -> torch.Tensor:
        values = self.blackbox.objective_function(configuration=params, seed=REPETITION)
        Y = values[..., self.outcome_indices].T  # shape: (len(outcome_names), n_steps)
        return torch.from_numpy(Y)

    def step_runtime(self, params: Mapping[str, TParamValue]) -> float:
        values = self.blackbox.objective_function(configuration=params, seed=REPETITION)
        runtimes_per_step = np.diff(values[..., self.elapsed_time_index], prepend=0.0)
        return runtimes_per_step.mean().item() * self.runtime_scale_factor


def get_synetune_blackbox_benchmark_problem(
    metric_name: str,
    blackbox_name: str,
    dataset_name: str | None = None,
    elapsed_time_name: str = DEFAULT_ELAPSED_TIME_NAME,
    surrogate_name: str | None = None,
    lower_is_better: bool = True,
    num_trials: int = DEFAULT_NUM_TRIALS,
    constant_step_runtime: bool = False,
) -> BenchmarkProblem:
    """Create a benchmark problem that encapsulates a SyneTune blackbox problem.

    Args:
        metric_name: Name of the metric to optimize.
        blackbox_name: Name of the blackbox to use.
        dataset_name: Name of the dataset to use within the blackbox.
        elapsed_time_name: Name of the metric that tracks elapsed time.
        surrogate_name: Name of the surrogate model to use. If None, the search space
            is assumed to be evaluated exhaustively, thus no surrogate
            is used and the original blackbox is evaluated directly.
        lower_is_better: Whether lower values of the metric are better.
        num_trials: Number of trials to run.
        constant_step_runtime: Whether to use the actual runtime recorded
            in the benchmark data.

    Returns:
        A wrapped SyneTune blackbox problem.
    """

    long_blackbox_name = (
        blackbox_name
        if surrogate_name is None
        else f"{blackbox_name}::{surrogate_name}"
    )
    name = f"SyneTune-{long_blackbox_name}-{dataset_name}"

    # TODO: populate optimal and baseline values
    optimal_value = float("-inf") if lower_is_better else float("+inf")
    baseline_value = float("+inf") if lower_is_better else float("-inf")

    optimization_config: OptimizationConfig = get_soo_opt_config(
        outcome_names=[metric_name],
        lower_is_better=lower_is_better,
        use_map_metric=True,
    )

    test_function = SyneTuneBlackboxTestFunction(
        blackbox_name=blackbox_name,
        dataset_name=dataset_name,
        outcome_names=[metric_name],
        elapsed_time_name=elapsed_time_name,
        surrogate_name=surrogate_name,
    )
    search_space: SearchSpace = test_function.get_search_space()

    step_runtime_function = (
        None if constant_step_runtime else test_function.step_runtime
    )

    return BenchmarkProblem(
        name=name,
        search_space=search_space,
        optimization_config=optimization_config,
        num_trials=num_trials,
        optimal_value=optimal_value,
        baseline_value=baseline_value,
        test_function=test_function,
        step_runtime_function=step_runtime_function,
    )


PROBLEM_FACTORY_KWARGS = {
    "fcnet": {
        "metric_name": "metric_valid_loss",
        "elapsed_time_name": "metric_elapsed_time",
        "lower_is_better": True,
        # search space is exhaustively evaluated, no surrogate needed
        "surrogate_name": None,
        "blackbox_name": "fcnet",
    },
    "nasbench201": {
        "metric_name": "metric_valid_error",
        "elapsed_time_name": "metric_elapsed_time",
        "lower_is_better": True,
        # search space is exhaustively evaluated, no surrogate needed
        "surrogate_name": None,
        "blackbox_name": "nasbench201",
    },
    "lcbench/RandomForestRegressor": {
        "metric_name": "val_accuracy",
        "elapsed_time_name": "time",
        "lower_is_better": False,
        # finite number of low-discrepancy samples are evaluated,
        # so a surrogate is needed for interpolation
        "surrogate_name": "RandomForestRegressor",
        "blackbox_name": "lcbench",
    },
    "lcbench/KNeighborsRegressor": {
        "metric_name": "val_accuracy",
        "elapsed_time_name": "time",
        "lower_is_better": False,
        "surrogate_name": "KNeighborsRegressor",
        "blackbox_name": "lcbench",
    },
}


# Examples:
# "SyneTune/fcnet/protein_structure"
# "SyneTune/nasbench201/cifar10"
# "SyneTune/lcbench/RandomForestRegressor/Fashion-MNIST"
PROBLEM_REGISTRY = {
    f"SyneTune/{registry_name}/{dataset_name}": BenchmarkProblemRegistryEntry(
        factory_fn=get_synetune_blackbox_benchmark_problem,
        factory_kwargs={
            "dataset_name": dataset_name,
            **kwargs,
        },
    )
    for registry_name, kwargs in PROBLEM_FACTORY_KWARGS.items()
    for dataset_name in runtime_scalings[kwargs["blackbox_name"]].keys()
}
