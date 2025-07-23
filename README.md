# About

This repository contains the code for generating the benchmark results in the following paper:

Olson et al. Ax A Platform for Adaptive Experimentation. AutoML Conference, 2025. https://openreview.net/forum?id=U1f6wHtG1g

# Installation

We use Conda environments to manage dependencies. All dependencies are pinned
to a specific version or commit hash to match the versions used while running
the benchmarks. These versions are specified under `requirements/`.
The below instructions are written with the assumption that the commands
are being executed from the root `ax_paper` directory.

HEBO requires Numpy <1.25. This can be incompatible with other baseline
libraries or force downgrades of their dependencies. Hence, there are two
environments you may wish to install.

If you are benchmarking HEBO:

```
conda create --name benchmark-hebo python=3.10 -y && conda activate benchmark-hebo
conda install pytorch==2.5.1 -c conda-forge -y && conda install pip -y
export ALLOW_LATEST_GPYTORCH_LINOP=true && export ALLOW_BOTORCH_LATEST=true
pip install -r requirements/hebo.txt
pip install .
```

Otherwise:

```
conda create --name benchmark-other python=3.10 -y && conda activate benchmark-other
conda install pytorch==2.5.1 -c conda-forge -y && conda install pip -y
export ALLOW_LATEST_GPYTORCH_LINOP=true && export ALLOW_BOTORCH_LATEST=true
pip install -r requirements/optuna_smac_vizier.txt
pip install .
```

Note on Python versions: 3.10 was chosen because it works for all libraries;
HEBO with 3.11 does not seem to work with compiled PyMOO, and the latest version
tested with SMAC's CI is 3.10. Vizier, Ax, and Optuna all seem to work with
3.12.


# Running benchmarks

You can run benchmarks for a specific problem and method combination using the following command.
The `min_seed` and `max_seed` arguments are optional, the `max_seed` is exclusive,
and seeds 0 (`min_seed=0`) to 9 (`max_seed=10`) are used by default.

After running the benchmark, the results will be saved to a JSON file under `ax_paper/results/`.

- `conda activate benchmark-{other|hebo}`
- `python run_benchmark.py ${METHOD_NAME} ${PROBLEM_NAME} --min_seed ${MIN_SEED} --max_seed ${MAX_SEED}`
  (use `-h` for help)
- `conda deactivate`

The following benchmark methods were used in the paper:
- `Random+Center`: Quasi-random Sobol sampling, starting with center point as first trial.
- `Ax-FAST-Center`: Default Ax with center point as the first trial.
- `HEBO+Center`: HEBO with center point as the first trial.
- `Optuna+Center`: Optuna (`TPESampler`) with center point as the first trial.
- `SMAC-BB+Center`: SMAC `BlackBoxFacade` (GP) with center point as the first trial.
- `SMAC-HPO+Center`: SMAC `HyperParameterOptimizationFacade` (random forest) with center point as the first trial.
- `Vizier`: Vizier, which, by default, uses center point as the first trial.

The following problems were included in the paper, with the names listed as expected by the script.
- 20 dimensional BBOB problems: `BBOB{i}_01_20d` for i in 01 .. 24, such as `BBOB01_01_20d`.
- Other vanilla problems: `branin`, `six_hump_camel`, `hartmann6`.
- Constrained problems: `tension_compression_string`, `pressure_vessel`, `welded_beam_SO`.
- Multi-objective: `CarSideImpact`, `ZDT{i}_2m_5d` for i in 1 .. 3, such as `ZDT1_2m_5d`,
  `DTLZ2_2m_6d`, `Penicillin`, `VehicleSafety`.
- Discrete / mixed high-dimensional: `labs`, `ackley_mixed`.
- High-dimensional: `hartmann30`
- Async, Discrete: `SyneTune/nasbench201/cifar100`, `SyneTune/nasbench201/ImageNet16-120`.
- Async: `Branin_variable_runtime`.
- Noisy: `Branin_noise`, `BBOB01_01_20d_noise_1`, `DTLZ2_2m_6d_noisy_10pct`, `Hartmann_6d_noisy`.


## Running multiple benchmarks in a loop

We can utilize bash for loops to queue up multiple benchmark runs to be run back
to back. Suppose we want to run first 10 BBOB problems using SMAC.
`for i in {1..9}; do python run_benchmark.py SMAC-BB BBOB0${i}_01_20d; done`
This is a simple loop that runs through 1 to 9 (inclusive!) and executes the
following command, replacing `${i}` with the value of the variable (and
completes the string with it). Note that the brackets in `${i}` are important.
Without it, we'd try to access a variable `$i_01_20d`, which doesn't exist. If
things don't nicely line up by a simple integer modification, you can also loop
over a space separated list of inputs. Here's the syntax for this:
`for i in branin hartmann6 hartmann3; do python run_benchmark.py SMAC-HPO $i; done`

To detect any typos, you can run the command with the `--predict_runtime` flag
first.


# Reproducing the plots
After running the benchmarks, you can use `python json_to_csv.py` to combine all
outputs into a CSV file that will be used by the plotting notebooks below.

While running the benchmarks, we used a number of machines and utilized a database
to combine the outputs from all runs. With the code submission, we are including
a CSV export of all results that were included in the paper, `benchmark_raw_results.csv`,
in the same format that is produced by the script above. These can be used with the
Python notebook `analyze_results.py` to produce the plots in the paper.

These plots are included:
- Final value plots: `boxplot_final_value.png`
- Performance after a given trial: `score_by_trial.png`
- Performance over runtime: `boxplot_time.png`


# Adding a new baseline

- Update requirements to include any new dependencies.
- Add a file defining the baseline as a benchmark method in `ax_paper/baselines`.
- Register the baseline in `run_benchmark.py`

## License
ax-paper is MIT licensed, as found in the LICENSE file.
