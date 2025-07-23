# Copyright (c) Meta Platforms, Inc. and affiliates.


import numpy as np

from ax_paper.problems.synetune_blackbox import (
    ADD_SURROGATE_KWARGS,
    PROBLEM_FACTORY_KWARGS,
    SURROGATE_KWARGS,
)

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository import BlackboxRepositoryBackend, load_blackbox
from syne_tune.optimizer.baselines import ASHA, BOHB, DEHB, SyncBOHB, SyncHyperband


# ASHA and BOHB are indispensable baseline methods, Hyperband is
# arguably also important, but it has compatibility issues with
# the LCBench surrogate problem for reasons that are not yet clear.
# The other methods included are beneficial but not critical.
METHODS = {
    "ASHA": ASHA,
    "BOHB": BOHB,
    "DEHB": DEHB,
    "SyncBOHB": SyncBOHB,
    "SyncHyperband": SyncHyperband,
}


def run_synetune(
    method_name: str,
    benchmark_name: str,
    dataset_name: str,
    seed: int,
    num_workers: int,
    max_num_evaluations: int,
    max_wallclock_time: float,
    max_resource_attr: str = "epochs",
) -> None:
    # boilerplate adapted from
    # syne-tune/benchmarking/nursery/benchmark_automl/benchmark_main.py
    # there should be no other source of randomness to control
    # for (e.g. from torch or otherwise)
    np.random.seed(seed)

    config = PROBLEM_FACTORY_KWARGS[benchmark_name]
    blackbox_name, surrogate_name_orig = benchmark_name.split("/")

    metric = config["metric_name"]
    mode = "min" if config["lower_is_better"] else "max"
    elapsed_time_attr = config["elapsed_time_name"]
    surrogate_name = config["surrogate_name"]
    if surrogate_name != surrogate_name_orig:
        raise RuntimeError("Unexpected surrogate name")

    print(
        f"Running '{method_name}' on "
        f"'{benchmark_name}/{dataset_name}' "
        f"({surrogate_name=}, {metric=}, {mode=}, {elapsed_time_attr=})"
    )

    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        dataset=dataset_name,
        elapsed_time_attr=elapsed_time_attr,
        surrogate=surrogate_name,
        # Passed to scikit-learn model initialization
        surrogate_kwargs=SURROGATE_KWARGS[surrogate_name],
        # Passed to SyneTune's `add_surrogate` method
        add_surrogate_kwargs=ADD_SURROGATE_KWARGS,
        seed=0,  # Fix the seed to always use the same repetition from the benchmark data
    )

    blackboxes = load_blackbox(blackbox_name)
    blackbox = blackboxes[dataset_name]

    resource_attr = blackbox.fidelity_name()
    config_space = blackbox.configuration_space_with_max_resource_attr(
        max_resource_attr
    )

    scheduler_cls = METHODS[method_name]
    scheduler = scheduler_cls(
        config_space=config_space,
        metric=metric,
        mode=mode,
        max_resource_attr=max_resource_attr,
        resource_attr=resource_attr,
        random_seed=seed,
    )

    stop_criterion = StoppingCriterion(
        max_wallclock_time=max_wallclock_time,
        max_num_evaluations=max_num_evaluations,
    )

    metadata = {
        "method": method_name,
        "benchmark": benchmark_name,
        "dataset": dataset_name,
        "surrogate": surrogate_name,
        "problem": f"{benchmark_name}/{dataset_name}",
        "seed": seed,
    }

    save_prefix = f"{method_name}|{benchmark_name}-{dataset_name}|{seed:02d}"

    # It is important to set ``sleep_time`` to 0 here (mandatory for simulator backend)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=num_workers,
        sleep_time=0,
        # This callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps
        callbacks=[SimulatorCallback()],
        results_update_interval=600,
        print_update_interval=600,
        tuner_name=save_prefix,
        metadata=metadata,
        save_tuner=False,
    )
    tuner.run()
