# Copyright (c) Meta Platforms, Inc. and affiliates.

from unittest import TestCase

from ax_paper.baselines.synetune import run_synetune


class TestSyneTune(TestCase):
    def test_run(self) -> None:
        run_synetune(
            method_name="ASHA",
            benchmark_name="lcbench/KNeighborsRegressor",
            dataset_name="christine",
            seed=0,
            num_workers=4,
            max_num_evaluations=10,
            max_wallclock_time=5.0,
        )
