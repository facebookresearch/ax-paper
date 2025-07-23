# Copyright (c) Meta Platforms, Inc. and affiliates.

import warnings
from functools import partial
from typing import Any

from ax.benchmark.benchmark import compute_baseline_value_from_sobol
from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    BOTORCH_BASELINE_VALUES,
    create_problem_from_botorch,
)
from ax.benchmark.benchmark_step_runtime_function import TBenchmarkStepRuntimeFunction
from ax.benchmark.problems.registry import BenchmarkProblemRegistryEntry
from ax.benchmark.problems.runtime_funcs import int_from_params
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.multi_objective import (
    CarSideImpact,
    DTLZ2,
    Penicillin,
    VehicleSafety,
    ZDT1,
    ZDT2,
    ZDT3,
)
from botorch.test_functions.synthetic import (
    Ackley,
    Branin,
    ConstrainedGramacy,
    Hartmann,
    PressureVessel,
    TensionCompressionString,
    WeldedBeamSO,
)


def get_registry_entry_from_botorch(
    test_problem_class: type[BaseTestProblem],
    baseline_value: float,
    name: str | None = None,
    use_shifted_function: bool = False,
    noise_std: float = 0.0,
    step_runtime_function: TBenchmarkStepRuntimeFunction | None = None,
    observe_noise_sd: bool = False,
    **test_problem_kwargs: Any,
) -> BenchmarkProblemRegistryEntry:
    class_name = test_problem_class.__name__
    if class_name.startswith("DTLZ") or class_name.startswith("ZDT"):
        dim = test_problem_kwargs["dim"]
        num_objectives = test_problem_kwargs.get(
            "num_objectives",
            # pyre-fixme: Unexpected keyword [28]: Unexpected keyword argument
            # `dim` to call `BaseTestProblem.__init__`.
            test_problem_class(dim=dim).num_objectives,
        )
        if name is None:
            name = f"{class_name}_{num_objectives}m_{dim}d"

    def factory_fn(num_trials: int = 50) -> BenchmarkProblem:
        return create_problem_from_botorch(
            test_problem_class=test_problem_class,
            test_problem_kwargs=test_problem_kwargs,
            num_trials=num_trials,
            baseline_value=baseline_value,
            name=name,
            use_shifted_function=use_shifted_function,
            noise_std=noise_std,
            observe_noise_sd=observe_noise_sd,
            step_runtime_function=step_runtime_function,
        )

    return BenchmarkProblemRegistryEntry(
        factory_fn=factory_fn,
        factory_kwargs={},
    )


BOTORCH_PROBLEM_REGISTRY: dict[str, BenchmarkProblemRegistryEntry] = {
    # from log EI paper
    "Ackley_8d_noisy_5pct_observed": get_registry_entry_from_botorch(
        test_problem_class=Ackley,
        name="Ackley_8d_noisy_5pct_observed",
        use_shifted_function=True,
        noise_std=1.0,
        baseline_value=21.27900469429435,
        dim=8,
        observe_noise_sd=True,
    ),
    # from log EI paper
    "Ackley_8d_noisy_5pct": get_registry_entry_from_botorch(
        test_problem_class=Ackley,
        name="Ackley_8d_noisy_5pct",
        use_shifted_function=True,
        noise_std=1.0,
        baseline_value=21.27900469429435,
        dim=8,
    ),
    # from log EI paper
    "Ackley_16d_noisy_2pct": get_registry_entry_from_botorch(
        test_problem_class=Ackley,
        name="Ackley_16d_noisy_2pct",
        use_shifted_function=True,
        noise_std=0.4,
        baseline_value=21.46083170066535,
        dim=16,
    ),
    # from NEHVI paper
    "DTLZ2_2m_6d_noisy_10pct": get_registry_entry_from_botorch(
        test_problem_class=DTLZ2,
        name="DTLZ2_2m_6d_noisy_10pct",
        baseline_value=0.02288348294808305,
        noise_std=0.1,
        dim=6,
    ),
    # from NEI paper
    "Hartmann_6d_noisy": get_registry_entry_from_botorch(
        test_problem_class=Hartmann,
        name="Hartmann_6d_noisy",
        baseline_value=BOTORCH_BASELINE_VALUES[("Hartmann", 6)],
        noise_std=0.2,
        dim=6,
    ),
    "Hartmann_6d_noisy_observed": get_registry_entry_from_botorch(
        test_problem_class=Hartmann,
        name="Hartmann_6d_noisy_observed",
        baseline_value=BOTORCH_BASELINE_VALUES[("Hartmann", 6)],
        noise_std=0.2,
        dim=6,
        observe_noise_sd=True,
    ),
    "DTLZ2_2m_6d": get_registry_entry_from_botorch(
        test_problem_class=DTLZ2,
        baseline_value=0.02288348294808305,
        dim=6,
    ),
    "VehicleSafety": get_registry_entry_from_botorch(
        test_problem_class=VehicleSafety,
        baseline_value=122.87303616691926,
    ),
    "CarSideImpact": get_registry_entry_from_botorch(
        test_problem_class=CarSideImpact,
        baseline_value=161.24278393117623,
    ),
    "Penicillin": get_registry_entry_from_botorch(
        test_problem_class=Penicillin,
        baseline_value=1025462.1581863433,
    ),
    # NOTE: ZDT problems are not shifted, as shifting resulted in NaN objective values.
    "ZDT1_2m_5d": get_registry_entry_from_botorch(
        test_problem_class=ZDT1,
        num_objectives=2,
        dim=5,
        baseline_value=92.88543258394267,
    ),
    "ZDT2_2m_5d": get_registry_entry_from_botorch(
        test_problem_class=ZDT2,
        num_objectives=2,
        dim=5,
        baseline_value=78.11550838220317,
    ),
    "ZDT3_2m_5d": get_registry_entry_from_botorch(
        test_problem_class=ZDT3,
        num_objectives=2,
        dim=5,
        baseline_value=113.87245476788114,
    ),
    "pressure_vessel": get_registry_entry_from_botorch(
        test_problem_class=PressureVessel,
        baseline_value=float("inf"),  # No feasible points found by the heuristic.
    ),
    "tension_compression_string": get_registry_entry_from_botorch(
        test_problem_class=TensionCompressionString,
        baseline_value=float("inf"),  # No feasible points found by the heuristic.
    ),
    "welded_beam_SO": get_registry_entry_from_botorch(
        test_problem_class=WeldedBeamSO,
        baseline_value=float("inf"),  # No feasible points found by the heuristic.
    ),
    # from NEI paper
    "Branin_noise": get_registry_entry_from_botorch(
        name="Branin_noise",
        test_problem_class=Branin,
        baseline_value=BOTORCH_BASELINE_VALUES[("Branin", None)],
        noise_std=5.0,
    ),
    "ZDT1_2m_5d_noise_0_2": get_registry_entry_from_botorch(
        name="ZDT1_2m_5d_noise_0_2",
        test_problem_class=ZDT1,
        num_objectives=2,
        dim=5,
        baseline_value=92.88543258394267,
        # Approximately equal to std of the less variable metric over Sobol
        # trials
        noise_std=0.2,
    ),
    "Branin_variable_runtime": get_registry_entry_from_botorch(
        name="Branin_variable_runtime",
        test_problem_class=Branin,
        baseline_value=BOTORCH_BASELINE_VALUES[("Branin", None)],
        # Each trial takes between 0 and 3 time steps to complete, depending
        # quasi-randomly on the parameterization.
        step_runtime_function=partial(int_from_params, n_possibilities=4),
    ),
    # from NEI paper
    "ConstrainedGramacy_noisy": get_registry_entry_from_botorch(
        name="ConstrainedGramacy_noisy",
        test_problem_class=ConstrainedGramacy,
        baseline_value=float("inf"),  # No feasible points found by the heuristic.
        noise_std=0.2,
    ),
}


def _compute_baseline_values() -> None:
    """This is used to compute the baseline values for the problems in the above
    registry.
    """
    warnings.simplefilter("ignore")  # dummy baseline values may lead to warnings.
    for name, registry_entry in BOTORCH_PROBLEM_REGISTRY.items():
        ax_problem = registry_entry.factory_fn()
        baseline = compute_baseline_value_from_sobol(
            optimization_config=ax_problem.optimization_config,
            search_space=ax_problem.search_space,
            test_function=ax_problem.test_function,
        )
        print(f'"{name}": {baseline}, ')


if __name__ == "__main__":
    _compute_baseline_values()
