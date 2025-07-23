# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Utils can be imported from any other file, so nothing in here should depend on
anything else in ax_paper.
"""

import importlib

from collections.abc import Callable
from warnings import warn

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.generation_strategy.generation_strategy import GenerationStrategy


def gs_to_benchmark_method(
    generation_strategy: GenerationStrategy,
    max_pending_trials: int,
    timeout_hours: float,
    name: str | None = None,
):
    name = generation_strategy.name if name is None else name
    return BenchmarkMethod(
        name=name,
        generation_strategy=generation_strategy,
        max_pending_trials=max_pending_trials,
        timeout_hours=timeout_hours,
    )


# Gracefully handle imports by setting method to empty method if unavailable
def _raise_import_error(library_name: str) -> None:
    raise ImportError(
        f"Failed to import {library_name}. Did you install dependencies from the "
        "correct requirements file?"
    )


def _gracefully_import(
    module_name: str, function_name: str, requirement_name: str
) -> Callable:
    if importlib.util.find_spec(requirement_name):
        return getattr(importlib.import_module(module_name), function_name)
    else:
        return lambda *args, **kwargs: _raise_import_error(requirement_name)


def skip_if_import_error(func: Callable) -> Callable:
    def f(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            warn(
                "Skipping test because module is not installed. Received the "
                f"following error: {e}"
            )

    return f
