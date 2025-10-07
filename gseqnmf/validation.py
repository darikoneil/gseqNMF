from enum import Enum
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval, StrOptions

from gseqnmf.exceptions import SeqNMFInitializationError

CUPY_INSTALLED: bool = False

try:
    import cupy as cp

    CUPY_INSTALLED: bool = True
except ImportError:  # pragma: no cover
    cp = None  # pragma: no cover


__all__ = [
    "INITIALIZATION_METHODS",
    "INIT_METHOD",
    "PARAMETER_CONSTRAINTS",
    "NDArrayLike",
]


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
        except (ValueError, AttributeError) as exc:
            msg = f"Unknown initialization method: {value}. "
            msg += f"Available methods are: {INIT_METHOD.options()}"
            raise SeqNMFInitializationError(msg) from exc

    @staticmethod
    def options() -> list[str]:
        return [m.value for m in INIT_METHOD]

    # DOC-ME: Write docstring explained each method.


#: Options for initialization methods in the GseqNMF algorithm (Docs/Hints).
INITIALIZATION_METHODS: type[str] = Literal["random", "exact", "nndsvd"]


class RECON_METHOD(Enum):  # noqa: N801
    """
    Enumeration of reconstruction method for the GseqNMF algorithm.
    """

    NORMAL = "normal"
    FAST = "fast"

    @staticmethod
    def parse(value: str | Enum | None) -> "RECON_METHOD":
        if isinstance(value, RECON_METHOD):
            return value
        if value is None:
            return RECON_METHOD.FAST
        try:
            return RECON_METHOD(value.lower())
        except (ValueError, AttributeError) as exc:
            msg = f"Unknown reconstruction solver: {value}. "
            msg += f"Available solvers are: {RECON_METHOD.options()}"
            raise SeqNMFInitializationError(msg) from exc

    @staticmethod
    def options() -> list[str]:
        return [m.value for m in RECON_METHOD]

    # DOC-ME: Write docstring explained each method.


#: Options for recon methods in the GseqNMF algorithm (Docs/Hints).
RECONSTRUCTION_METHODS: type[str] = Literal["normal", "fast"]


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
    "recon": [StrOptions(set(RECON_METHOD.options())), RECON_METHOD, None],
    "random_state": [int, None],
}


def cuda_available()  -> bool:
    """Check if a CUDA-capable GPU is available for CuPy.

    Returns
    -------
    bool
        True if a CUDA-capable GPU is available, False otherwise.
    """
    if not CUPY_INSTALLED:
        return False
    # noinspection PyBroadException
    try:
        return cp.cuda.is_available()
    except Exception:  # pragma: no cover  # noqa: BLE001
        return False


def device_available() -> bool:
    """Check if a GPU device is available for CuPy.

    Returns
    -------
    bool
        True if a GPU device is available, False otherwise.
    """
    if not CUPY_INSTALLED:
        return False
    # noinspection PyBroadException
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False
