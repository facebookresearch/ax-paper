# Copyright (c) Meta Platforms, Inc. and affiliates.

from ax.benchmark.problems.registry import BENCHMARK_PROBLEM_REGISTRY
from ax_paper.problems.bbob import PROBLEM_REGISTRY as BBOB_PROBLEM_REGISTRY
from ax_paper.problems.botorch_problems import BOTORCH_PROBLEM_REGISTRY
from ax_paper.problems.discrete_mixed import DISCRETE_MIXED_PROBLEM_REGISTRY
from ax_paper.problems.synetune_blackbox import (
    PROBLEM_REGISTRY as SYNETUNE_PROBLEM_REGISTRY,
)


AX_PAPER_PROBLEM_REGISTRY = {
    **BENCHMARK_PROBLEM_REGISTRY,  # Keep it at the top, we may overwrite some problems.
    **BOTORCH_PROBLEM_REGISTRY,
    **SYNETUNE_PROBLEM_REGISTRY,
    **BBOB_PROBLEM_REGISTRY,
    **DISCRETE_MIXED_PROBLEM_REGISTRY,
}


AX_PAPER_PROBLEMS = list(AX_PAPER_PROBLEM_REGISTRY.keys())
