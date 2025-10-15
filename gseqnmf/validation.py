from abc import ABCMeta, abstractmethod
from enum import Enum, EnumMeta
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval, StrOptions

from gseqnmf.exceptions import (
    SeqNMFInitializationError,
)

CUPY_INSTALLED: bool = False

try:
    import cupy as cp

    CUPY_INSTALLED: bool = True
except ImportError:  # pragma: no cover
    cp = None  # pragma: no cover
else:
    from cupy.cuda import is_available as cuda_is_available
    from cupy.cuda.runtime import getDeviceCount


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


class _AbstractEnum(ABCMeta, EnumMeta):
    """
    Metaclass for creating abstract enumerations.
    """


# noinspection PyAbstractClass
class _GSEQNMF_METHOD_BASE(Enum, metaclass=_AbstractEnum):  # noqa: N801
    """
    Base class for GseqNMF method enumerations.
    """

    @classmethod
    def options(cls) -> list[str]:
        """
        List all available options in the class

        :return: A list of options as strings.
        """
        return [method.value for method in cls]

    @classmethod
    def parse(cls, value: str | Enum | None) -> "_GSEQNMF_METHOD_BASE":
        """
        Parse the input value and return the corresponding method.

        :param value: The input value to parse.
        :return: The corresponding enum
        :raises SeqNMFInitializationError: If the input value cannot be parsed.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.default()
        try:
            return cls(value.lower())
        except (ValueError, AttributeError) as exc:
            msg = f"Unknown {cls.__name__} option: {value}. "
            msg += f"Available methods are: {cls.options()}"
            raise SeqNMFInitializationError(msg) from exc

    @classmethod
    @abstractmethod
    def default(cls) -> "_GSEQNMF_METHOD_BASE":
        """
        Get the default method for the class.

        :return: The default method as an instance of the class.
        """
        ...


class INIT_METHOD(_GSEQNMF_METHOD_BASE):  # noqa: N801
    """
    Enumeration of initialization methods for the GseqNMF algorithm.
    """

    RANDOM = "random"
    EXACT = "exact"
    NNDSVD = "nndsvd"

    @classmethod
    def default(cls) -> "INIT_METHOD":
        return cls.RANDOM

    # DOC-ME: Write docstring explained each method.


#: Options for initialization methods in the GseqNMF algorithm (Docs/Hints).
INITIALIZATION_METHODS: type[str] = Literal["random", "exact", "nndsvd"]


class RECON_METHOD(_GSEQNMF_METHOD_BASE):  # noqa: N801
    """
    Enumeration of reconstruction method for the GseqNMF algorithm.
    """

    NORMAL = "normal"
    FAST = "fast"

    @classmethod
    def default(cls) -> "RECON_METHOD":
        return cls.FAST

    # DOC-ME: Write docstring explained each method.


#: Options for recon methods in the GseqNMF algorithm (Docs/Hints).
RECONSTRUCTION_METHODS: type[str] = Literal["normal", "fast"]


class W_UPDATE_METHOD(_GSEQNMF_METHOD_BASE):  # noqa: N801
    """
    Enumeration of update methods for the W matrix in the GseqNMF algorithm.
    """

    FIXED = "fixed"
    PARTIAL = "partial"
    FULL = "full"

    @classmethod
    def default(cls) -> "W_UPDATE_METHOD":
        return cls.FULL

    # DOC-ME: Write docstring explained each method.


#: Options for W update methods in the GseqNMF algorithm (Docs/Hints).
W_UPDATE_METHODS: type[str] = Literal["fixed", "partial", "full"]


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
    "W_update": [StrOptions(set(W_UPDATE_METHOD.options())), W_UPDATE_METHOD, None],
    "init": [StrOptions(set(INIT_METHOD.options())), INIT_METHOD, None],
    "recon": [StrOptions(set(RECON_METHOD.options())), RECON_METHOD, None],
    "random_state": [int, None],
}


def cuda_available() -> bool:
    """
    Check if a CUDA-capable GPU is available for CuPy.

    :raises GPUNotSupported:
        If CuPy is not installed.
    :return: True if a CUDA is available, False otherwise.
    """
    if not CUPY_INSTALLED:
        return False
    # noinspection PyBroadException
    try:
        return cuda_is_available()
    except Exception:  # pragma: no cover  # noqa: BLE001
        return False


def device_available() -> bool:
    """
    Check if a GPU device is available for CuPy.

    :raises GPUNotAvailable:
        If CuPy is installed but no GPU device is available.
    :return: True if a GPU device is available, False otherwise.
    """
    if not CUPY_INSTALLED:
        return False
    # noinspection PyBroadException
    try:
        return getDeviceCount() > 0
    except Exception:  # noqa: BLE001 # pragma: no cover
        return False


def is_valid_device(device_id: int = 0) -> bool:
    """
    Check if the specified GPU device ID is valid.

    :param device_id: The ID of the GPU device to check (default is 0).
    :return: True if the device ID is valid, False otherwise.
    """
    if not CUPY_INSTALLED:
        return False
    return 0 <= device_id < getDeviceCount()
