""":py"""
# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import OrderedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

sns.set(style="whitegrid", font_scale=1.4)

CSV_PATH = "benchmark_raw_results.csv"

""":md
# Set plotting defaults and define problems
"""

""":py"""
problem_categories = {
    "Async": [
        "SyneTune-nasbench201-cifar100",
        "SyneTune-nasbench201-ImageNet16-120",
        "Branin_variable_runtime",
    ],
    "BBOB": [
        "BBOB01_01_20d",
        "BBOB02_01_20d",
        "BBOB03_01_20d",
        "BBOB04_01_20d",
        "BBOB05_01_20d",
        "BBOB06_01_20d",
        "BBOB07_01_20d",
        "BBOB08_01_20d",
        "BBOB09_01_20d",
        "BBOB10_01_20d",
        "BBOB11_01_20d",
        "BBOB12_01_20d",
        "BBOB13_01_20d",
        "BBOB14_01_20d",
        "BBOB15_01_20d",
        "BBOB16_01_20d",
        "BBOB17_01_20d",
        "BBOB18_01_20d",
        "BBOB19_01_20d",
        "BBOB20_01_20d",
        "BBOB21_01_20d",
        "BBOB22_01_20d",
        "BBOB23_01_20d",
        "BBOB24_01_20d",
    ],
    "High Dimensional": [
        "Hartmann_30d",
        "Ackley Mixed",
        "Discrete Labs",
    ],
    "Noisy": [
        "BBOB01_01_20d_noise_1",
        "Branin_noise",
        "Hartmann_6d_noisy",
        "DTLZ2_2m_6d_noisy_10pct",
    ],
    "Constrained": ["TensionCompressionString", "WeldedBeamSO", "PressureVessel"],
    "Other vanilla": ["SixHumpCamel", "Hartmann_6d", "Branin"],
    "Multi-objective": [
        "ZDT1_2m_5d",
        "ZDT2_2m_5d",
        "ZDT3_2m_5d",
        "DTLZ2_2m_6d",
        "VehicleSafety",
        "CarSideImpact",
        "Penicillin",
        "DTLZ2_2m_6d_noisy_10pct",
    ],
    "Mixed/Discrete": [
        "Ackley Mixed",
        "Discrete Labs",
        "SyneTune-nasbench201-cifar100",
        "SyneTune-nasbench201-ImageNet16-120",
    ],
}
problem_to_category = {}
for cat, problems in problem_categories.items():
    for problem in problems:
        if problem in problem_to_category:
            problem_to_category[problem].append(cat)
        else:
            problem_to_category[problem] = [cat]

""":py"""
# Note: Ax::balanced has been removed
METHOD_ORDER = [
    "Ax",
    "Vizier",
    "SMAC-BB",
    "HEBO",
    "Optuna",
    "SMAC-HPO",
    "Random Search",
]
METHOD_ORDER_ES = [
    "Ax",
    "Ax::ES",
    "Random Search::ES",
    "Random Search",
]
combined_order = METHOD_ORDER.copy()

for method in METHOD_ORDER_ES:
    if method not in METHOD_ORDER:
        combined_order.append(method)

method_numbers = dict(zip(combined_order, range(len(combined_order))))


clean_method_names_centered = {
    "Optuna+Center": "Optuna",
    "SMAC-HPO+Center": "SMAC-HPO",
    "Ax::fast+Center": "Ax",
    "Vizier": "Vizier",
    "Ax::random_search+Center": "Random Search",
    "SMAC-BB+Center": "SMAC-BB",
    "HEBO+Center": "HEBO",
    "Ax::fast::ES+Center": "Ax::ES",
    "Ax::random_search::ES+Center": "Random Search::ES",
}
clean_method_names_not_centered = {
    "Optuna": "Optuna",
    "SMAC-HPO": "SMAC-HPO",
    "Ax::fast": "Ax",
    "Vizier": "Vizier",
    "Ax::random_search": "Random Search",
    "SMAC-BB": "SMAC-BB",
    "HEBO": "HEBO",
    "Ax::fast::ES": "Ax::ES",
}

palette = list(sns.color_palette(n_colors=len(combined_order) - 1))
random_search_idx = combined_order.index("Random Search")
palette = palette[:random_search_idx] + [(0.4, 0.4, 0.4)] + palette[random_search_idx:]
colors = OrderedDict(list(zip(combined_order, palette)))
sns.color_palette(palette)

""":md
# Pull data
"""

""":py"""
df = pd.read_csv(CSV_PATH)
df["Method"] = df["Method"].map(
    {**clean_method_names_centered, **clean_method_names_not_centered}
)
df["method_num"] = df["Method"].map(method_numbers)

""":py"""
df[df["trial_index"] == 0][["Fit Time", "Gen Time"]].sum().sum() / 3600

""":md
# Recompute baselines and optimal values
"""

""":py"""


def get_expanded_trace(
    cost_trace: npt.NDArray, optimization_trace: npt.NDArray, max_cost: float
):
    interpolator = interp1d(
        np.concatenate(
            (
                [0],
                cost_trace,
                np.array([cost_trace.max() + 0.01, max_cost]),
            )
        ),
        np.concatenate(
            (
                [-float("inf")],
                optimization_trace,
                np.array([optimization_trace[-1], optimization_trace[-1]]),
            )
        ),
        kind="previous",
    )
    return interpolator(np.arange(max_cost + 1))


""":py"""
best_seen_values = {
    "PressureVessel": 5655.451049556235,
    "TensionCompressionString": 1.375799e-02,
    "Hartmann_6d_noisy": -3.322183962233345,
    "WeldedBeamSO": 1.857019309308374,
    "Branin_variable_runtime": 0.397887,  # True optimal value
    "DTLZ2_2m_6d_noisy_10pct": 0.39296144337238903,
    "SyneTune-nasbench201-ImageNet16-120": 0.5253333449,
    "SyneTune-nasbench201-cifar100": 0.26210001110000003,
}

""":py"""
problem_df = pd.DataFrame.from_records(
    [
        {"Problem": problem, "Category": category}
        for problem, categories in problem_to_category.items()
        for category in categories
    ]
)
constrained_problems = problem_df.loc[
    problem_df["Category"] == "Constrained", "Problem"
].tolist()
nasbench_problems = [
    "SyneTune-nasbench201-cifar100",
    "SyneTune-nasbench201-ImageNet16-120",
]

""":py"""


def get_df_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.merge(problem_df)
    # Expand trace for async problems
    async_idx = df["Problem"] == "Branin_variable_runtime"
    df_async = df[async_idx]

    max_cost = df_async["cost_trace"].max()
    other_cols = [
        elt
        for elt in df_async.columns
        if elt
        not in [
            "trial_index",
            "optimization_trace",
            "inference_trace",
            "score",
            "cost_trace",
        ]
    ]
    new_dfs = []
    for (method, seed), sub_df in df_async.groupby(["Method", "seed"]):
        tmp = sub_df[["cost_trace", "optimization_trace"]].dropna()
        expanded_trace = get_expanded_trace(
            cost_trace=tmp["cost_trace"].to_numpy(),
            optimization_trace=tmp["optimization_trace"].to_numpy(),
            max_cost=max_cost,
        )

        expanded_trace_df = pd.DataFrame(
            {
                "trial_index": np.arange(len(expanded_trace)),
                "optimization_trace": expanded_trace,
                "cost_trace": np.arange(max_cost + 1),
                "score": float("NaN"),
                "inference_trace": float("NaN"),
                **sub_df[other_cols].iloc[0],
            }
        )
        new_dfs.append(expanded_trace_df)

    new_df_async = pd.concat(new_dfs, ignore_index=True)

    df_clean = pd.concat(
        (df[~async_idx], new_df_async.loc[lambda x: x["cost_trace"] > 0]),
        ignore_index=True,
    )

    # Fix up optimization trace
    # Impute values for constrained problems

    constrained = df_clean["Problem"].isin(constrained_problems)
    worst_non_constrained_values = (
        df_clean[
            constrained & np.isfinite(df_clean["optimization_trace"].astype(float))
        ]
        .groupby("Problem")["optimization_trace"]
        .max()
    )

    df_clean["optimization_trace"] = (
        df_clean["optimization_trace"]
        .replace(float("inf"), float("nan"))
        .fillna(df_clean["Problem"].map(worst_non_constrained_values))
    )

    # Recompute score for everyone
    df_clean["max_baseline_trial"] = 4
    # 50 epochs for each of the 5 baseline trials. These are 1-indexed
    df_clean.loc[df_clean["Problem"].isin(nasbench_problems), "max_baseline_trial"] = (
        250
    )
    best_seen = (
        df_clean[
            df_clean["score"] == df_clean.groupby("Problem")["score"].transform("max")
        ][["Problem", "optimization_trace"]]
        .drop_duplicates()
        .set_index("Problem")["optimization_trace"]
        .to_dict()
    )
    best_seen = {**best_seen, **best_seen_values}

    problem_values = (
        df_clean[df_clean["Method"].isin(("Ax", "Random Search"))]
        .loc[
            lambda x: x["trial_index"] == x["max_baseline_trial"],
            ["Method", "Problem", "seed", "optimization_trace"],
        ]
        .groupby("Problem")[["optimization_trace"]]
        .mean()
        .rename(columns={"optimization_trace": "baseline"})
        .reset_index()
    )

    problem_values["best seen"] = problem_values["Problem"].map(best_seen)

    df_clean = df_clean.merge(
        problem_values,
        on="Problem",
        how="outer",
        validate="m:1",
        indicator=True,
    )

    df_clean["New Score"] = (
        100
        * (df_clean["optimization_trace"] - df_clean["baseline"])
        / (df_clean["best seen"] - df_clean["baseline"])
    )
    df_clean["Total Optimization Time"] = df_clean["Fit Time"] + df_clean["Gen Time"]

    return df_clean, problem_values


""":py"""
df_clean, problem_values = get_df_clean(df=df)
assert (df_clean["_merge"] == "both").all()
del df_clean["_merge"]
assert not df_clean["New Score"].isnull().any()

""":md
# Score analysis
"""

""":md
### For shifted problems, restrict to the first 10 seeds for comparability, unless we have at least 100
"""

""":py"""
bbob_problems = sorted([elt for elt in df_clean["Problem"].unique() if "BBOB" in elt])

max_seed_for_shifted_problem_by_benchmark = (
    df_clean[df_clean["Problem"].isin(bbob_problems)]
    .groupby(["Problem", "Method"])[["seed"]]
    .max()
)
assert (
    max_seed_for_shifted_problem_by_benchmark.groupby(level="Problem")["seed"].min()
    == 9
).all()

""":py"""
max_seed = df_clean.groupby(["Problem", "Method"])["seed"].transform("max")
to_drop = (
    (df_clean["Problem"].isin(bbob_problems)) & (max_seed < 99) & (df_clean["seed"] > 9)
)
to_drop.sum()

""":py"""
score_df = (
    df_clean.loc[~to_drop, :]
    .drop(columns=["score"])
    .rename(columns={"New Score": "score"})
    # Note this makes the df longer, as the same problem can appear in multiple categoreies
    .merge(problem_df)
)

""":py"""


def get_rank_df(category: str) -> pd.DataFrame:
    idx = score_df["Category"] == category
    if category in ("Async", "Mixed/Discrete"):
        # We couldn't run SMAC for the NASBench problems, so exclude it for whatever they get
        # analy
        idx &= ~score_df["Method"].isin(("SMAC-BB", "SMAC-HPO"))
    ranks = (
        score_df[idx]
        .set_index(["Problem", "seed", "trial_index", "Method"])[["score"]]
        # make it square -- only compare within same set of problems
        .unstack(level="Method")
        .dropna()
        .stack()
    )
    ranks["rank"] = (
        ranks.astype(float)
        .groupby(level=["Problem", "seed", "trial_index"])["score"]
        .rank(ascending=False)
    )
    ranks.reset_index(inplace=True)
    return ranks.assign(Category=category)


# Ranks need to be considered within a problem category because we will only be
# comparing the methods in that category
# So a benchmark replication can have different rankings depending on which
# category it is being considered in
ranks = pd.concat(
    (get_rank_df(category=category) for category in problem_categories.keys()),
    ignore_index=True,
)
ranks["max_baseline_trial"] = 4
ranks.loc[ranks["Problem"].isin(nasbench_problems), "max_baseline_trial"] = 250

""":py"""
replication_grouped = score_df.sort_values("trial_index").groupby(
    ["Method", "Problem", "Category", "seed"]
)
replication_level_stats = replication_grouped.agg(
    final_score=("score", "last"),
    **{
        col: (col, "first")
        for col in ["Fit Time", "Gen Time", "method_num", "Total Optimization Time"]
    },
)
replication_level_stats["average_score"] = (
    score_df[score_df["trial_index"] > score_df["max_baseline_trial"]]
    .groupby(["Method", "Problem", "Category", "seed"])["score"]
    .mean()
)

rank_grouped = ranks.groupby(["Method", "Problem", "Category", "seed"])
final_rank = rank_grouped["rank"].last()
replication_level_stats["final_rank"] = final_rank
replication_level_stats["average_rank"] = (
    ranks[ranks["trial_index"] > ranks["max_baseline_trial"]]
    .groupby(["Method", "Problem", "Category", "seed"])["rank"]
    .mean()
)

""":py"""
replication_level_long = pd.melt(
    frame=replication_level_stats.reset_index(),
    id_vars=["Method", "Problem", "Category", "seed", "method_num"],
    value_vars=[
        "final_score",
        "Fit Time",
        "Gen Time",
        "Total Optimization Time",
        "average_score",
        "final_rank",
        "average_rank",
    ],
).dropna()

""":md
## Now that we have score_df, let's analyze it
"""

""":py"""
not_es_order = [
    "Ax",
    "SMAC-BB",
    "HEBO",
    "Vizier",
    "SMAC-HPO",
    "Optuna",
    "Random Search",
]
sns.set(style="whitegrid", font_scale=1.4)

col_order = [
    "BBOB",
    "Other vanilla",
    "Mixed/Discrete",
    "Async",
    "Multi-objective",
    "Noisy",
    "Constrained",
    "High Dimensional",
]

""":py"""
sns.set(style="whitegrid", font_scale=1.4)

""":md
# Big plots
"""

""":py"""
g = sns.catplot(
    data=(
        replication_level_stats.reset_index()
        .loc[lambda x: x["Method"] != "Ax::ES"]
        .sort_values("method_num")
    ),
    kind="box",
    orient="h",
    y="Method",
    x="final_score",
    col="Category",
    hue="Method",
    hue_order=not_es_order,
    palette=colors,
    col_wrap=4,
    showfliers=False,
    sharex=True,
    col_order=col_order,
)
for ax in g.axes.flatten():
    ax.set_ylabel(None)
for ax in g.axes.flatten()[4:]:
    ax.set_xlabel("Score at end")
    ax.set_xlim(-25, 101)
g.set_titles(col_template="{col_name}")
plt.tight_layout()
plt.savefig("boxplot_final_value.png")

""":py"""
g = sns.catplot(
    data=(
        replication_level_stats.reset_index()
        .loc[lambda x: x["Method"] != "Ax::ES"]
        .sort_values("method_num")
    ),
    kind="box",
    orient="h",
    y="Method",
    x="Total Optimization Time",
    col="Category",
    hue="Method",
    hue_order=not_es_order,
    palette=colors,
    col_wrap=4,
    showfliers=False,
    sharex=True,
    col_order=col_order,
)
for ax in g.axes.flatten():
    ax.set_ylabel(None)
for ax in g.axes.flatten()[4:]:
    ax.set_xlabel("Time (s)")
g.set_titles(col_template="{col_name}")
plt.tight_layout()
plt.xscale("log")
plt.savefig("boxplot_time.png")

""":md
# Problem-method level (for table)
"""

""":py"""
problem_method_level = (
    replication_level_long.groupby(["Method", "Problem", "variable"])
    .agg(
        method_num=("method_num", "first"),
        mean_value=("value", "mean"),
        n=("value", "count"),
        std=("value", "std"),
    )
    .reset_index()
)
problem_method_level["se"] = problem_method_level["std"] / np.sqrt(
    problem_method_level["n"] - 1
)
problem_method_level["lower"] = (
    problem_method_level["mean_value"] - 2 * problem_method_level["se"]
)
problem_method_level["upper"] = (
    problem_method_level["mean_value"] + 2 * problem_method_level["se"]
)

""":py"""
problem_method_level["Pretty"] = (
    problem_method_level["mean_value"].round().astype(int).astype(str)
    + " ("
    + problem_method_level["se"].round(1).astype(str)
    + ")"
)
score_table = (
    problem_method_level.set_index(["Method", "Problem", "variable"])["Pretty"]
    .unstack(level="variable")
    .drop(columns=["final_rank", "average_rank"])
    .rename(columns={"final_score": "Final Score", "average_score": "Average Score"})
    .reset_index()
    .sort_values(["Problem", "Method"])
    .fillna("")
)
score_table.columns.name = None
score_table

""":py"""
print(score_table.to_latex())

""":md
# Score by trial
"""

""":py"""
in_terms_of_epochs = score_df["Problem"].apply(lambda x: "SyneTune" in x)
syne_tune_scores = (
    score_df[in_terms_of_epochs]
    .loc[lambda x: (x["trial_index"] % 50 == 49)]
    .assign(trial_index=lambda x: x["trial_index"] // 50)
)

df_problem_trial_level = (
    pd.concat((syne_tune_scores, score_df[~in_terms_of_epochs]))
    .loc[lambda x: x["trial_index"] < 50]
    .loc[lambda x: ~((x["Category"] == "Async") & (x["trial_index"] > 17))]
    .groupby(["Problem", "trial_index", "Method", "Category"])
    .agg(
        average_score=("score", "mean"),
        std_score=("score", "std"),
        n=("score", "count"),
    )
)
df_problem_trial_level["se_score"] = df_problem_trial_level["std_score"] / np.sqrt(
    df_problem_trial_level["n"] - 1
)
df_problem_trial_level["lower"] = (
    df_problem_trial_level["average_score"] - 2 * df_problem_trial_level["se_score"]
)
df_problem_trial_level["upper"] = (
    df_problem_trial_level["average_score"] + 2 * df_problem_trial_level["se_score"]
)

""":py"""
in_terms_of_epochs = ranks["Problem"].apply(lambda x: "SyneTune" in x)
syne_tune_ranks = (
    ranks[in_terms_of_epochs]
    .loc[lambda x: (x["trial_index"] % 50 == 49)]
    .assign(trial_index=lambda x: x["trial_index"] // 50)
)

df_problem_trial_level_rank = (
    pd.concat((syne_tune_ranks, ranks[~in_terms_of_epochs]))
    # Make sure everything runs for the same number of trials as everything
    # else in the category, to prevent weird composition effects
    .loc[lambda x: x["trial_index"] < 50]
    .loc[lambda x: ~((x["Category"] == "Async") & (x["trial_index"] > 17))]
    .groupby(["Problem", "trial_index", "Method", "Category"])
    .agg(
        average_rank=("rank", "mean"),
        std_rank=("rank", "std"),
        n=("rank", "count"),
    )
)
df_problem_trial_level_rank["se_rank"] = df_problem_trial_level_rank[
    "std_rank"
] / np.sqrt(df_problem_trial_level_rank["n"] - 1)
df_problem_trial_level_rank["lower"] = (
    df_problem_trial_level_rank["average_rank"]
    - 2 * df_problem_trial_level_rank["se_rank"]
)
df_problem_trial_level_rank["upper"] = (
    df_problem_trial_level_rank["average_rank"]
    + 2 * df_problem_trial_level_rank["se_rank"]
)

""":py"""
g = df_problem_trial_level_rank.groupby(level=["trial_index", "Method", "Category"])
df_cat_trial_level_rank = g[["average_rank"]].mean()
df_cat_trial_level_rank["se_rank"] = np.sqrt(
    g["se_rank"].apply(lambda x: (x**2).sum()) / g["se_rank"].count()
)
df_cat_trial_level_rank["lower"] = (
    df_cat_trial_level_rank["average_rank"] - 2 * df_cat_trial_level_rank["se_rank"]
)
df_cat_trial_level_rank["upper"] = (
    df_cat_trial_level_rank["average_rank"] + 2 * df_cat_trial_level_rank["se_rank"]
)

""":py"""
g = df_problem_trial_level.groupby(level=["trial_index", "Method", "Category"])
df_cat_trial_level = g[["average_score"]].mean()
df_cat_trial_level["se_score"] = np.sqrt(
    g["se_score"].apply(lambda x: (x**2).sum()) / g["se_score"].count()
)
df_cat_trial_level["lower"] = (
    df_cat_trial_level["average_score"] - 2 * df_cat_trial_level["se_score"]
)
df_cat_trial_level["upper"] = (
    df_cat_trial_level["average_score"] + 2 * df_cat_trial_level["se_score"]
)

""":py"""
g = sns.FacetGrid(
    data=df_cat_trial_level.reset_index().rename(
        columns={"average_score": "Score", "trial_index": "Trial Index"}
    ),
    col="Category",
    col_wrap=4,
    col_order=col_order,
    hue="Method",
    hue_order=not_es_order,
    palette=colors,
    sharex=False,
    height=4,
)
g.map(plt.fill_between, "Trial Index", "lower", "upper", alpha=0.2)

g.map(plt.plot, "Trial Index", "Score")
g.set_titles(col_template="{col_name}")
for ax in g.axes:
    ax.set_ylim(-20, 101)
g.add_legend()
plt.savefig("score_by_trial.png")

""":py"""
g = sns.FacetGrid(
    data=df_cat_trial_level_rank.reset_index().rename(
        columns={"average_rank": "Rank", "trial_index": "Trial Index"}
    ),
    col="Category",
    col_wrap=4,
    col_order=col_order,
    hue="Method",
    hue_order=not_es_order,
    palette=colors,
    sharex=False,
    height=4,
)
g.map(plt.fill_between, "Trial Index", "lower", "upper", alpha=0.2)

g.map(plt.plot, "Trial Index", "Rank")
g.set_titles(col_template="{col_name}")
g.add_legend()

""":md
# Score comparisons against Ax
"""

""":py"""
is_shifted = (replication_level_long["Category"] == "BBOB") | (
    replication_level_long["Problem"] == "BBOB01_01_20d_noise_1"
)
shifted_merged = replication_level_long.loc[
    is_shifted
    & (replication_level_long["Method"] != "Ax")
    & (replication_level_long["variable"] == "final_score")
].merge(
    replication_level_long[
        is_shifted
        & (replication_level_long["Method"] == "Ax")
        & (replication_level_long["variable"] == "final_score")
    ]
    .drop(columns=["Method", "method_num"])
    .rename(columns={"value": "Ax_value"})
)
shifted_merged["diff"] = shifted_merged["Ax_value"] - shifted_merged["value"]
shifted_merged_by_method_problem = shifted_merged.groupby(
    ["Method", "Problem", "Category"]
).agg(diff=("diff", "mean"), std=("diff", "std"), count=("diff", "count"))
shifted_merged_by_method_problem["se_diff"] = shifted_merged_by_method_problem[
    "std"
] / np.sqrt(shifted_merged_by_method_problem["count"] - 1)

""":py"""
# Now compare the non-shifted ones, where we don't have to account for the seed
tmp = problem_method_level[
    (~problem_method_level["Problem"].apply(lambda x: "BBOB" in x))
    & (problem_method_level["variable"] == "final_score")
]
problem_method_level_not_shifted = tmp.loc[
    tmp["Method"] != "Ax", ["Problem", "Method", "mean_value", "se"]
].merge(
    tmp.loc[
        tmp["Method"] == "Ax",
        ["Problem", "mean_value", "se"],
    ],
    on=["Problem"],
    suffixes=("", "_Ax"),
)
problem_method_level_not_shifted["diff"] = (
    problem_method_level_not_shifted["mean_value_Ax"]
    - problem_method_level_not_shifted["mean_value"]
)
problem_method_level_not_shifted["se_diff"] = np.sqrt(
    problem_method_level_not_shifted["se"] ** 2
    + problem_method_level_not_shifted["se_Ax"] ** 2
)

""":py"""
problem_method_level_diffs = pd.concat(
    (
        problem_method_level_not_shifted[["Problem", "Method", "diff", "se_diff"]],
        shifted_merged_by_method_problem.reset_index()[
            ["Problem", "Method", "diff", "se_diff"]
        ],
    )
)

""":py"""
grouped = problem_method_level_diffs.merge(problem_df).groupby(["Category", "Method"])
cat_method_level_merged = grouped[["diff"]].mean()
cat_method_level_merged["se_diff"] = np.sqrt(
    grouped["se_diff"].apply(lambda x: (x**2).sum())
) / grouped["se_diff"].apply("count")
cat_method_level_merged["z"] = (
    cat_method_level_merged["diff"] / cat_method_level_merged["se_diff"]
)
cat_method_level_merged["star"] = ""
cat_method_level_merged.loc[np.abs(cat_method_level_merged["z"]) > 1.96, "star"] = "*"
cat_method_level_merged.loc[np.abs(cat_method_level_merged["z"]) > 2.56, "star"] = "**"

""":py"""
table_long = (
    df_cat_trial_level.groupby(level=["Method", "Category"])[["average_score"]]
    .last()
    .reset_index()
).merge(cat_method_level_merged[["star"]].reset_index(), how="outer")
table_long["star"] = table_long["star"].fillna("")

table_long["method_num"] = table_long["Method"].map(method_numbers)
table_long.sort_values("method_num", inplace=True)
table_long["Final Score"] = (
    table_long["average_score"].round(1).astype(str) + table_long["star"]
)

""":py"""
to_print = (
    table_long.set_index(["method_num", "Method", "Category"])["Final Score"]
    .unstack(level="Category")
    .fillna("")
    .sort_index()
    .reset_index(level="method_num", drop=True)[col_order]
    .rename(
        columns={
            "Constrained": "Constr.",
            "High Dimensional": "High-Dim",
            "Multi-objective": "Multi-obj",
        }
    )
)
to_print

""":py"""
print(to_print.to_latex())

""":py"""
