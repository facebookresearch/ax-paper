load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

oncall("ae_modopt")

python_library(
    name = "ax",
    srcs = ["ax_paper/ax.py"],
    deps = [
        "//ax/adapter:adapter",
        "//ax/api/utils:generation_strategy_dispatch",
        "//ax/api/utils:structs",
        "//ax/benchmark:benchmark_method",
        "//ax/benchmark:benchmark_problem",
        "//ax/early_stopping:early_stopping",
        "//ax/generation_strategy:center_generation_node",
        "//ax/generation_strategy:generation_strategy",
        "//ax/generation_strategy:generator_spec",
        "//ax/generators:botorch_modular",
        "//pytorch/botorch:botorch",
    ],
)

python_library(
    name = "botorch_problems",
    srcs = ["ax_paper/problems/botorch_problems.py"],
    deps = [
        "//ax/benchmark:benchmark",
        "//ax/benchmark:benchmark_problem",
        "//ax/benchmark:benchmark_problems",
        "//ax/benchmark:benchmark_step_runtime_function",
        "//pytorch/botorch:botorch",
    ],
)

python_library(
    name = "utils",
    srcs = ["ax_paper/utils.py"],
    labels = ["autodeps2_generated"],
    deps = [
        "//ax/benchmark:benchmark_method",
        "//ax/generation_strategy:generation_strategy",
    ],
)

python_library(
    name = "discrete_mixed",
    srcs = ["ax_paper/problems/discrete_mixed.py"],
    labels = ["autodeps2_generated"],
    deps = [
        "//ax/benchmark:benchmark_problem",
        "//ax/benchmark:benchmark_problems",
        "//pytorch/botorch:botorch",
    ],
)

python_library(
    name = "json_to_csv",
    srcs = ["json_to_csv.py"],
    labels = ["autodeps2_generated"],
    deps = [
        "fbsource//third-party/pypi/pandas:pandas",
        "//ax/benchmark:benchmark",
        "//ax/benchmark:benchmark_result",
        "//ax/core:core",
        "//ax/storage/json_store:json_store",
    ],
)
