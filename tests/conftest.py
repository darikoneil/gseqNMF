import sys
from os import devnull, environ
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

#: The answer to life, the universe, and everything
MEANING_OF_LIFE: int = 42

#: Disable Numba JIT during tests for consistency and easier debugging
environ["NUMBA_DISABLE_JIT"] = "1"

#: Disable tqdm progress bars during tests
# environ["TQDM_DISABLE"] = "True"


class BlockPrinting:
    """
    Simple context manager that blocks printing
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(devnull, "w")  # noqa: PTH123

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        sys.stdout.close()
        sys.stdout = self._stdout


class Dataset:
    """
    Test dataset class for loading and storing test datasets.
    """

    _test_dataset_directory: Path = Path(__file__).parent.joinpath("datasets")

    def __init__(
        self,
        name: str,
        data: np.ndarray,
        W: np.ndarray,  # noqa: N803
        H: np.ndarray,  # noqa: N803
        cost: np.ndarray,
        loadings: np.ndarray,
        power: np.ndarray,
        **kwargs,
    ):
        if "parameters" in kwargs:
            kwargs = {**kwargs, **kwargs.pop("parameters").item()}
        self.name: str = name
        self.data: np.ndarray = data
        self.W: np.ndarray = W
        self.H: np.ndarray = H
        self.cost: np.ndarray = cost
        self.loadings: np.ndarray = loadings
        self.power: np.ndarray = power
        self.parameters: dict[str, Any] = kwargs
        for param, value in kwargs.items():
            if isinstance(value, np.ndarray) and value.shape == ():
                self.parameters[param] = value.item()
        # HACK: Sklearn's validation checks fall for single element numpy arrays
        #  This is a temporary fix until we can assess whether we need to modify
        #  the parameter constraints or whether it's a bug in the storage of the test
        #  dataset.

    @classmethod
    def load(cls, name: str) -> "Dataset":
        dataset = dict(
            np.load(
                cls._test_dataset_directory.joinpath(f"{name}.npz"), allow_pickle=True
            )
        )
        return cls(name, **dataset)


@pytest.fixture(scope="session")
def example_dataset() -> Dataset:
    """
    Fixture that loads a pre-computed test dataset.

    :return: Dataset object with test data and parameters.
    """
    return Dataset.load("example_dataset")


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("numpy", id="cpu"),
        pytest.param("cupy", id="gpu"),
    ],
)
def xp_imp(request: FixtureRequest) -> ModuleType:
    """
    Fixture that provides the requested array implementation (numpy or cupy).

    :param request: Pytest fixture request object.
    :return: The requested array implementation module.
    """
    return pytest.importorskip(request.param)
