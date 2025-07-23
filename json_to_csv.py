# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import os
from pathlib import Path

import pandas as pd
from ax.benchmark.benchmark import get_benchmark_result_with_cumulative_steps
from ax.benchmark.benchmark_result import BenchmarkResult
from ax.core.map_data import MapData
from ax.storage.json_store.decoder import object_from_json

ax_benchmark_dir = Path(__file__).parent.resolve()
directory: Path = ax_benchmark_dir.joinpath("results")


def load_results(fname: str) -> list[BenchmarkResult]:
    serialized = json.loads(open(os.path.join(directory, fname)).read())
    results = object_from_json(serialized)

    # Results logged with timeseries data must be handled separately
    if isinstance(results[0].experiment.lookup_data(), MapData):
        return [
            get_benchmark_result_with_cumulative_steps(
                result=result, optimal_value=float("NaN"), baseline_value=float("NaN")
            )
            for result in results
        ]

    return results


def main() -> None:
    dfs = []
    for fname in os.listdir(directory):
        method_name, problem_name, timestamp = fname.split("|")
        results = load_results(fname=fname)

        records = [
            {
                "ds": timestamp.split("+")[0],
                "ts": timestamp,
                "trial_index": i,
                "method": method_name,
                "problem": problem_name,
                "seed": result.seed,
                "score": result.score_trace[i],
                "optimization_trace": result.optimization_trace[i],
                "oracle_trace": result.oracle_trace[i],
                "inference_trace": result.inference_trace[i],
                "fit_time": result.fit_time,
                "gen_time": result.gen_time,
                "total_time": float("nan"),
                "cost_trace": result.cost_trace[i],
            }
            for result in results
            for i in range(len(result.optimization_trace))
        ]

        df = pd.DataFrame.from_records(records)
        dfs.append(df)

    final = pd.concat(dfs)
    final.to_csv(os.path.join(directory, "benchmark_results.csv"), index=False)


if __name__ == "__main__":
    main()
