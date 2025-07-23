# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Callable
from functools import partial

import optuna

from ax_paper.ax import get_ax_benchmark_method
from ax_paper.utils import _gracefully_import


get_hebo_benchmark_method = _gracefully_import(
    module_name="ax_paper.baselines.hebo",
    function_name="get_hebo_benchmark_method",
    requirement_name="hebo",
)
get_optuna_benchmark_method = _gracefully_import(
    module_name="ax_paper.baselines.optuna",
    function_name="get_optuna_benchmark_method",
    requirement_name="optuna",
)
get_smac_benchmark_method = _gracefully_import(
    module_name="ax_paper.baselines.smac",
    function_name="get_smac_benchmark_method",
    requirement_name="smac",
)
get_vizier_benchmark_method = _gracefully_import(
    module_name="ax_paper.baselines.vizier",
    function_name="get_vizier_benchmark_method",
    requirement_name="vizier",
)

BASELINE_REGISTRY: dict[str, Callable] = {
    "Ax-FAST": partial(get_ax_benchmark_method, generation_method="fast"),
    "Ax-FAST-Center": partial(
        get_ax_benchmark_method,
        generation_method="fast",
        sample_center=True,
    ),
    "Ax-FAST-ES": partial(
        get_ax_benchmark_method,
        generation_method="fast",
        early_stopping=True,
    ),
    "Ax-FAST-Center-ES": partial(
        get_ax_benchmark_method,
        generation_method="fast",
        early_stopping=True,
        sample_center=True,
    ),
    "Random": partial(get_ax_benchmark_method, generation_method="random_search"),
    "Random+Center": partial(
        get_ax_benchmark_method,
        generation_method="random_search",
        sample_center=True,
    ),
    "Random+Center+ES": partial(
        get_ax_benchmark_method,
        generation_method="random_search",
        early_stopping=True,
        sample_center=True,
    ),
    "HEBO": get_hebo_benchmark_method,
    "HEBO+Center": partial(get_hebo_benchmark_method, sample_center=True),
    "Optuna": partial(
        get_optuna_benchmark_method,
        sampler_name="tpe",
        early_stopping=False,
    ),
    "Optuna+Center": partial(
        get_optuna_benchmark_method,
        sampler_name="tpe",
        early_stopping=False,
        sample_center=True,
    ),
    "Optuna-ES(Percentile/Random)": partial(
        get_optuna_benchmark_method,
        sampler_name="random",
        early_stopping=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=5,
            interval_steps=1,
        ),
    ),
    "Optuna-ES(Percentile/TPE)": partial(
        get_optuna_benchmark_method,
        sampler_name="tpe",
        early_stopping=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=5,
            interval_steps=1,
        ),
    ),
    "Optuna-ES(ASHA)": partial(
        get_optuna_benchmark_method,
        sampler_name="random",
        early_stopping=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            reduction_factor=3,
        ),
    ),
    "Optuna-ES(Hyperband)": partial(
        get_optuna_benchmark_method,
        sampler_name="random",
        early_stopping=True,
        pruner=optuna.pruners.HyperbandPruner(
            reduction_factor=3,
        ),
    ),
    "Optuna-ES(BOHB)": partial(
        get_optuna_benchmark_method,
        sampler_name="tpe",
        early_stopping=True,
        pruner=optuna.pruners.HyperbandPruner(
            reduction_factor=3,
        ),
    ),
    "SMAC-BB": partial(get_smac_benchmark_method, use_gp=True),
    "SMAC-BB+Center": partial(
        get_smac_benchmark_method, use_gp=True, sample_center=True
    ),
    "SMAC-HPO": partial(get_smac_benchmark_method, use_gp=False),
    "SMAC-HPO+Center": partial(
        get_smac_benchmark_method, use_gp=False, sample_center=True
    ),
    "Vizier": get_vizier_benchmark_method,
}

command_line_name_to_method_name = {
    "Ax-FAST": "Ax::fast",
    "Ax-FAST-Center": "Ax::fast+Center",
    "Ax-FAST-Center-ES": "Ax::fast::ES+Center",
    "Random": "Ax::random_search",
    "Random+Center": "Ax::random_search+Center",
    "Random+Center+ES": "Ax::random_search::ES+Center",
}
