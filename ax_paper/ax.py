# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Literal

from ax.adapter.base import DataLoaderConfig
from ax.adapter.registry import Generators, MBM_X_trans, Y_trans

from ax.adapter.transforms.map_key_to_float import MapKeyToFloat
from ax.api.utils.generation_strategy_dispatch import (
    _get_sobol_node,
    choose_generation_strategy,
)
from ax.api.utils.structs import GenerationStrategyDispatchStruct
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.early_stopping.strategies import PercentileEarlyStoppingStrategy
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP


def _construct_generation_strategy(
    generator_spec: GeneratorSpec,
    struct: GenerationStrategyDispatchStruct,
) -> GenerationStrategy:
    """Constructs a Center + Sobol + Modular BoTorch `GenerationStrategy`
    using the provided `generator_spec` for the Modular BoTorch node.
    """
    sobol_node = _get_sobol_node(
        initialization_budget=struct.initialization_budget,
        min_observed_initialization_trials=struct.min_observed_initialization_trials,  # noqa: E501
        initialize_with_center=struct.initialize_with_center,
        use_existing_trials_for_initialization=struct.use_existing_trials_for_initialization,  # noqa: E501
        allow_exceeding_initialization_budget=struct.allow_exceeding_initialization_budget,  # noqa: E501
        initialization_random_seed=struct.initialization_random_seed,
    )
    botorch_node = GenerationNode(
        node_name="MBM",
        generator_specs=[generator_spec],
        should_deduplicate=True,
    )

    name = f"Sobol+{botorch_node.node_name}"

    nodes = []
    if struct.initialize_with_center:
        name = "Center+" + name
        center_node = CenterGenerationNode(next_node_name=sobol_node.node_name)
        nodes.append(center_node)
    nodes.append(sobol_node)
    nodes.append(botorch_node)

    return GenerationStrategy(name=name, nodes=nodes)


def get_ax_benchmark_method(
    benchmark_problem: BenchmarkProblem,
    max_pending_trials: int,
    seed: int | None = None,
    generation_method: Literal["fast", "random_search"] = "fast",
    early_stopping: bool = False,
    sample_center: bool = False,
    distribute_replications: bool = False,
    timeout_hours: float = 1.0,
) -> BenchmarkMethod:
    """
    Define a BenchmarkMethod that uses choose_generation_strategy.

    Args:
        benchmark_problem: Used for providing arguments to
            `choose_generation_strategy`.
        early_stopping: If True, use default PercentileEarlyStoppingStrategy
        sample_center: If True, sample the center of the search space for the
            first point.
    """
    struct = GenerationStrategyDispatchStruct(
        method=generation_method,
        initialize_with_center=sample_center,
        initialization_random_seed=seed,
    )
    if not early_stopping:
        generation_strategy = choose_generation_strategy(struct=struct)
        ess = None
    else:
        surrogate_spec = SurrogateSpec(
            model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)]
        )
        generator_spec = GeneratorSpec(
            generator_enum=Generators.BOTORCH_MODULAR,
            model_kwargs={
                "surrogate_spec": surrogate_spec,
                "botorch_acqf_class": qLogExpectedImprovement,
                "transforms": [MapKeyToFloat] + MBM_X_trans + Y_trans,
                "transform_configs": {
                    "MapKeyToFloat": {"parameters": {"step": {"log_scale": False}}}
                },
                "data_loader_config": DataLoaderConfig(
                    fit_only_completed_map_metrics=False,
                    latest_rows_per_group=1,
                ),
            },
        )
        generation_strategy = _construct_generation_strategy(
            generator_spec=generator_spec, struct=struct
        )
        ess = PercentileEarlyStoppingStrategy(min_progression=5)
    name = f"Ax::{generation_method}" + ("::ES" if early_stopping else "")
    if sample_center:
        name = name + "+Center"

    return BenchmarkMethod(
        name=name,
        generation_strategy=generation_strategy,
        early_stopping_strategy=ess,
        distribute_replications=distribute_replications,
        timeout_hours=timeout_hours,
        max_pending_trials=max_pending_trials,
    )
