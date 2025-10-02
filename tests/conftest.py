import sys
from os import devnull
from pathlib import Path
from typing import Any

import numpy as np
import pytest

MEANING_OF_LIFE: int = 42  # The answer to life, the universe, and everything


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

    @classmethod
    def load(cls, name: str) -> "Dataset":
        dataset = dict(
            np.load(
                cls._test_dataset_directory.joinpath(f"{name}.npz"), allow_pickle=True
            )
        )
        return cls(name, **dataset)


@pytest.fixture(scope="class")
def test_dataset() -> Dataset:
    return Dataset.load("example_dataset")
