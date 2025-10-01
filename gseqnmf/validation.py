from enum import Enum
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval, StrOptions

try:
    import cupy as cp
except ImportError:
    cp = None


#: Define a custom type alias `NDArrayLike` to represent a flexible array-like type.
#: This type can be:
#: - Any object that satisfies the `ArrayLike` protocol (e.g., lists, tuples, etc.)
#: - A NumPy ndarray (`np.ndarray`)
#: - A CuPy ndarray (`cp.ndarray`) if CuPy is available
type NDArrayLike = ArrayLike | np.ndarray | "cp.ndarray"


class INIT_METHOD(Enum):  # noqa: N801
    """
    Enumeration of initialization methods for the GseqNMF algorithm.
    """

    RANDOM = "random"
    EXACT = "exact"
    NNDSVD = "nndsvd"

    @staticmethod
    def parse(value: str | Enum | None) -> "INIT_METHOD":
        if isinstance(value, INIT_METHOD):
            return value
        if value is None:
            return INIT_METHOD.RANDOM
        try:
            return INIT_METHOD(value.lower())
        except ValueError as exc:
            msg = f"Unknown initialization method: {value}. "
            msg += f"Available methods are: {INIT_METHOD.options()}"
            raise ValueError(msg) from exc

    @staticmethod
    def options() -> list[str]:
        return [m.value for m in INIT_METHOD]


#: Options for initialization methods in the GseqNMF algorithm (Docs/Hints).
INITIALIZATION_METHODS: type[str] = Literal["random", "exact", "nndsvd"]

#: Constraints for parameters
PARAMETER_CONSTRAINTS: dict[str, list] = {
    "n_components": [int, Interval(Real, left=1, right=None, closed="left")],
    "sequence_length": [int, Interval(Real, left=2, right=None, closed="left")],
    "lam": [float, Interval(Real, left=0, right=None, closed="left")],
    "max_iter": [int, Interval(Real, left=1, right=None, closed="left")],
    "tol": [float, Interval(Real, left=0, right=None, closed="left")],
    "alpha_W": [float, Interval(Real, left=0, right=None, closed="left")],
    "lam_W": [float, Interval(Real, left=0, right=None, closed="left")],
    "alpha_H": [float, Interval(Real, left=0, right=None, closed="left")],
    "lam_H": [float, Interval(Real, left=0, right=None, closed="left")],
    "shift": [bool],
    "sort": [bool],
    "update_W": [bool],
    "init": [StrOptions(set(INIT_METHOD.options())), INIT_METHOD, None],
    "random_state": [int, None],
}
