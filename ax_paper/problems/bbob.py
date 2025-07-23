# Copyright (c) Meta Platforms, Inc. and affiliates.

import warnings
from itertools import product

import cocoex
import torch
from ax.benchmark.benchmark import compute_baseline_value_from_sobol
from ax.benchmark.problems.registry import BenchmarkProblemRegistryEntry
from ax_paper.problems.botorch_problems import get_registry_entry_from_botorch
from botorch.test_functions.synthetic import SyntheticTestFunction

# values computed using `_bbob_baseline_value_from_sobol`
BASELINE_VALUES = {
    "BBOB01_01_2d": 97.65045199218892,
    "BBOB01_01_3d": 121.6477691995001,
    "BBOB01_01_5d": 170.1763800930022,
    "BBOB01_01_20d": 799.5481376800786,
    "BBOB02_01_2d": 1793232.1424886202,
    "BBOB02_01_3d": 2122601.761282105,
    "BBOB02_01_5d": 3625499.5587004363,
    "BBOB02_01_20d": 32262893.216258988,
    "BBOB03_01_2d": -369.15365910215013,
    "BBOB03_01_3d": -278.05504405636145,
    "BBOB03_01_5d": 148.68263205351457,
    "BBOB03_01_20d": 7774.022783840776,
    "BBOB04_01_2d": -270.5882755671895,
    "BBOB04_01_3d": 621.4161018237426,
    "BBOB04_01_5d": 1881.6228103197493,
    "BBOB04_01_20d": 22476.915267123313,
    "BBOB05_01_2d": -1.4085848259580045,
    "BBOB05_01_3d": 6.6918063072761536,
    "BBOB05_01_5d": 28.15762012892331,
    "BBOB05_01_20d": 342.3075903743136,
    "BBOB06_01_2d": 154.37165803294945,
    "BBOB06_01_3d": 4069.616829872721,
    "BBOB06_01_5d": 150779.60728290584,
    "BBOB06_01_20d": 2335495.736298011,
    "BBOB07_01_2d": 190.7103053261892,
    "BBOB07_01_3d": 444.45638270338975,
    "BBOB07_01_5d": 670.667791455432,
    "BBOB07_01_20d": 5238.141576306133,
    "BBOB08_01_2d": 2631.820412920882,
    "BBOB08_01_3d": 37552.629576174666,
    "BBOB08_01_5d": 271787.864185937,
    "BBOB08_01_20d": 5364513.84892657,
    "BBOB09_01_2d": 4461.513002276701,
    "BBOB09_01_3d": 41369.354121773664,
    "BBOB09_01_5d": 270819.68079841573,
    "BBOB09_01_20d": 3978921.9226436955,
    "BBOB10_01_2d": 2225914.293601513,
    "BBOB10_01_3d": 2263617.4828678966,
    "BBOB10_01_5d": 4788743.962587911,
    "BBOB10_01_20d": 27539741.613298006,
    "BBOB11_01_2d": 5040753.938219986,
    "BBOB11_01_3d": 3002073.779227793,
    "BBOB11_01_5d": 5270588.288932207,
    "BBOB11_01_20d": 4326043.471825021,
    "BBOB12_01_2d": 3538202.6463643517,
    "BBOB12_01_3d": 44048704.35211659,
    "BBOB12_01_5d": 230962580.78900987,
    "BBOB12_01_20d": 16671068029.504038,
    "BBOB13_01_2d": 488.35416548530355,
    "BBOB13_01_3d": 841.2401738895642,
    "BBOB13_01_5d": 1707.7917733765134,
    "BBOB13_01_20d": 4919.873624640831,
    "BBOB14_01_2d": -38.630834924970806,
    "BBOB14_01_3d": -29.76540413834961,
    "BBOB14_01_5d": 10.87584840567021,
    "BBOB14_01_20d": 357.232976474084,
    "BBOB15_01_2d": 1082.100854347556,
    "BBOB15_01_3d": 1593.4262322428963,
    "BBOB15_01_5d": 1581.459560858877,
    "BBOB15_01_20d": 5539.352118480569,
    "BBOB16_01_2d": 117.90619660918829,
    "BBOB16_01_3d": 140.89532593279685,
    "BBOB16_01_5d": 162.84090141214202,
    "BBOB16_01_20d": 186.64406783643605,
    "BBOB17_01_2d": 20.580240508234752,
    "BBOB17_01_3d": 75.11748587997354,
    "BBOB17_01_5d": 132.55663658447008,
    "BBOB17_01_20d": 906.9848366622093,
    "BBOB18_01_2d": 163.8016991505402,
    "BBOB18_01_3d": 219.99293072433068,
    "BBOB18_01_5d": 316.28996391581023,
    "BBOB18_01_20d": 1777.6933411432185,
    "BBOB19_01_2d": -62.33207556186006,
    "BBOB19_01_3d": -16.896518236747866,
    "BBOB19_01_5d": 73.34257709855933,
    "BBOB19_01_20d": 539.8534054108585,
    "BBOB20_01_2d": 18957.065406492406,
    "BBOB20_01_3d": 27559.891558220632,
    "BBOB20_01_5d": 77788.57581026363,
    "BBOB20_01_20d": 648448.7738915058,
    "BBOB21_01_2d": 55.051289132824685,
    "BBOB21_01_3d": 79.10467517967263,
    "BBOB21_01_5d": 122.11997383956111,
    "BBOB21_01_20d": 199.37106104952892,
    "BBOB22_01_2d": -963.4233380875415,
    "BBOB22_01_3d": -938.4170934245403,
    "BBOB22_01_5d": -912.1805113038274,
    "BBOB22_01_20d": -809.1331525059098,
    "BBOB23_01_2d": 31.823058321323774,
    "BBOB23_01_3d": 31.606113546791466,
    "BBOB23_01_5d": 27.75620927205412,
    "BBOB23_01_20d": 93.09344007283303,
    "BBOB24_01_2d": 852.7666721786961,
    "BBOB24_01_3d": 10371.945913479713,
    "BBOB24_01_5d": 67231.13519137717,
    "BBOB24_01_20d": 749148.0580280392,
}


class BBOB(SyntheticTestFunction):
    # bounds for all dimension and problems are [-5, +5]
    LOWER_BOUND = -5.0
    UPPER_BOUND = +5.0

    def __init__(
        self,
        function_number: int,
        function_instance: int = 1,
        dim: int = 2,
        noise_std: float | None = None,
        negate: bool = False,
    ) -> None:
        """BBOB test functions from the COCO platform.

        There are 24 functions, each with different instances (1-5 and 71-80) and dimensionalities
        2, 3, 5, 10, 20, or 40. See https://coco-platform.org/testsuites/bbob/overview.html
        for more details.

        Args:
            function_number: The number of the function to be used, must be between 1 and 24.
            function_instance: The instance of the function, must be between 1 and 5 or 71 and 80.
            dim: The dimensionality of the function, must be one of 2, 3, 5, 10, 20, or 40.
            noise_std: The standard deviation of the noise to be added, if any.
            negate: Whether to negate the function values.
            dtype: The data type for the tensor operations, default is torch.double.
        """

        if not 0 < function_number <= 24:
            raise ValueError("Function number must be between 1 and 24.")
        if not 0 < function_instance <= 5 and not 70 < function_instance <= 80:
            raise ValueError("Function instance must be between 1 and 5 or 71 and 80.")
        if dim not in {2, 3, 5, 10, 20, 40}:
            raise ValueError("Dimension must be 2, 3, 5, 10, 20, or 40.")
        self._cocoex_problem = cocoex.BareProblem(
            suite_name="bbob",
            function=function_number,
            dimension=dim,
            instance=function_instance,
        )
        self.dim = dim
        self.continuous_inds = list(range(dim))
        bounds = [(BBOB.LOWER_BOUND, BBOB.UPPER_BOUND) for _ in range(dim)]
        self._optimal_value = self._cocoex_problem.best_value()
        self._optimizers = [tuple(self._cocoex_problem.best_parameter())]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self._cocoex_problem(X.tolist()), dtype=self.bounds.dtype)


def get_bbob_name_and_registry_entry(
    function_number: int,
    function_instance: int,
    dim: int,
    noise_std: float = 0.0,
    name: str | None = None,
) -> tuple[str, BenchmarkProblemRegistryEntry]:
    base_name = _bbob_name(
        function_number=function_number,
        function_instance=function_instance,
        dim=dim,
    )
    name = base_name if name is None else name
    try:
        baseline_value = BASELINE_VALUES[base_name]
    except KeyError:
        raise NotImplementedError(
            f"Problem {name} not yet supported (baseline value unknown)"
        )
    entry = get_registry_entry_from_botorch(
        test_problem_class=BBOB,
        baseline_value=baseline_value,
        name=name,
        function_number=function_number,
        function_instance=function_instance,
        dim=dim,
        use_shifted_function=True,
        noise_std=noise_std,
    )
    return name, entry


def _bbob_name(function_number: int, function_instance: int, dim: int) -> str:
    return f"BBOB{function_number:02d}_{function_instance:02d}_{dim:d}d"


PROBLEM_REGISTRY = dict(
    get_bbob_name_and_registry_entry(*args)
    for args in product(
        range(1, 25),
        (1,),
        (2, 3, 5, 20),
    )
)
name, entry = get_bbob_name_and_registry_entry(
    function_number=1,
    function_instance=1,
    dim=20,
    # Approximately equal to the objective std from 5 Sobol trials
    noise_std=1.0,
    name="BBOB01_01_20d_noise_1",
)
PROBLEM_REGISTRY[name] = entry


def _bbob_baseline_value_from_sobol(
    function_number: int,
    function_instance: int,
    dim: int,
    use_shifted_function: bool = True,
):
    """
    This function was used to compute `BASELINE_VALUES`.
    """
    warnings.simplefilter("ignore")  # dummy baseline value will lead to warnings.
    name = _bbob_name(
        function_number=function_number,
        function_instance=function_instance,
        dim=dim,
    )
    registry_entry = get_registry_entry_from_botorch(
        test_problem_class=BBOB,
        baseline_value=float("inf"),  # dummy
        name=name,
        function_number=function_number,
        function_instance=function_instance,
        dim=dim,
        use_shifted_function=use_shifted_function,
    )
    ax_problem = registry_entry.factory_fn()
    return compute_baseline_value_from_sobol(
        optimization_config=ax_problem.optimization_config,
        search_space=ax_problem.search_space,
        test_function=ax_problem.test_function,
    )


def print_all_baselines():
    """Computes all baseline values using `_bbob_baseline_value_from_sobol`
    and prints the values in a way that they can be simply copy-pasted
    into the baseline value dict at the top of this file.
    """
    for function_number, function_instance, dim in product(
        range(1, 25),
        (1,),
        (2, 3, 5, 20),
    ):
        name = _bbob_name(
            function_number=function_number,
            function_instance=function_instance,
            dim=dim,
        )
        baseline = _bbob_baseline_value_from_sobol(
            function_number=function_number,
            function_instance=function_instance,
            dim=dim,
        )
        print(f'"{name}": {baseline}, ')


if __name__ == "__main__":
    print_all_baselines()
