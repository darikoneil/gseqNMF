import warnings
from collections.abc import Callable
from types import ModuleType

import numpy as np
from tqdm import tqdm

from gseqnmf.exceptions import (
    GPUNotAvailableError,
    GPUNotSupportedError,
    InvalidGPUDeviceError,
)
from gseqnmf.validation import (
    CUPY_INSTALLED,
    NDArrayLike,
    device_available,
    is_valid_device,
)

if CUPY_INSTALLED:
    from cupy.cuda import Device
else:
    Device = None  # pragma: no cover

__all__ = [
    "calculate_loading_power",
    "calculate_power",
    "reconstruct",
    "rmse",
]


"""
========================================================================================
User-Exposed Miscellaneous Helpers & Functions
========================================================================================
"""


def calculate_power(
    X: NDArrayLike,  # noqa: N803
    x_hat: NDArrayLike,
    epsilon: float = np.finfo(float).eps,
    padding_index: slice | None = None,
    xp: ModuleType = np,
) -> float:
    """
    Calculate the percent power explained by the reconstruction x_hat of X.

    :param X: Original data matrix.
    :param x_hat: Reconstructed data matrix.
    :param epsilon: Small constant to avoid division by zero.
    :param padding_index: Optional slice to select unpadded region.
    :param xp: Array module (e.g., numpy or cupy) for computation.
    :return: Percent power explained (float).
    """
    epsilon = xp.asarray(epsilon)
    if padding_index is not None:
        X_unpad = X[:, padding_index]  # noqa: N806
        x_hat_unpad = x_hat[:, padding_index]
    else:
        X_unpad = X  # noqa: N806
        x_hat_unpad = x_hat
    denom = xp.sum(X_unpad**2) + epsilon
    return 100 * (xp.sum(X_unpad**2) - xp.sum((X_unpad - x_hat_unpad) ** 2)) / denom


def calculate_loading_power(
    X: NDArrayLike,  # noqa: N803
    W: NDArrayLike,  # noqa: N803
    H: NDArrayLike,  # noqa: N803
    epsilon: float = np.finfo(float).eps,
    padding_index: slice | None = None,
    xp: ModuleType = np,
) -> NDArrayLike:
    """
    Calculate the percent power explained by each component's loading.

    :param X: Original data matrix.
    :param W: Basis tensor (n_features x n_components x sequence_length).
    :param H: Coefficient matrix (n_components x n_samples).
    :param epsilon: Small constant to avoid division by zero.
    :param padding_index: Optional slice to select unpadded region.
    :param xp: Array module (e.g., numpy or cupy) for computation.
    :return: Array of percent power explained per component.
    """
    epsilon = xp.asarray(epsilon)
    if padding_index is not None:
        X_unpad = X[:, padding_index]  # noqa: N806
        H_unpad = H[:, padding_index]  # noqa: N806
    else:
        X_unpad = X  # noqa: N806
        H_unpad = H  # noqa: N806
    denom = xp.sum(X**2) + epsilon
    n_features, n_components, sequence_length = W.shape
    n_components_H, n_samples = H_unpad.shape  # noqa: N806
    assert n_components == n_components_H, "Number of components in W and H must match."
    if sequence_length > n_samples:
        msg = (
            f"sequence_length ({sequence_length}) "
            f"cannot be greater than n_samples ({n_samples})."
        )
        raise ValueError(msg)
    loadings = xp.zeros((n_components,), dtype=W.dtype)
    for k in range(n_components):
        x_hat_k = xp.zeros((n_features, n_samples), dtype=W.dtype)
        for idx in range(sequence_length):
            if idx == 0:
                x_hat_k += W[:, k, idx][:, xp.newaxis] @ H_unpad[k, :][xp.newaxis, :]
            else:
                x_hat_k[:, idx:] += (
                    W[:, k, idx][:, xp.newaxis]
                    @ H_unpad[k, : n_samples - idx][xp.newaxis, :]
                )
        loadings[k] = (xp.sum(X_unpad**2) - xp.sum((X_unpad - x_hat_k) ** 2)) / denom
    return loadings


def calculate_sequenciness() -> None:
    """
    Placeholder for calculating sequenciness metric.

    :return: Not implemented.
    """
    msg = "Sequenciness calculation is not implemented yet."
    raise NotImplementedError(msg)
    # TODO: Implement the sequenciness calculation algorithm.
    # DOC-ME: Add docstring for calculate_sequenciness function
    # TEST: Add tests for calculate_sequenciness function in test_support.py


# noinspection PyUnusedLocal
def reconstruct(
    W: NDArrayLike,  # noqa: N803
    H: NDArrayLike,  # noqa: N803
    h_shifted: NDArrayLike | None = None,  # noqa: ARG001
    xp: ModuleType = np,
) -> NDArrayLike:
    """
    Reconstruct the data matrix from W and H.

    :param W: Basis matrix (n_features x n_components x sequence_length).
    :param H: Coefficient matrix (n_components x n_samples).
    :param h_shifted: Preallocated array for shifted H (not used; API compatibility).
    :param xp: Array module (e.g., numpy or cupy) for computation.
    :return: Reconstructed data matrix (n_features x n_samples).
    """
    n_features, _, sequence_length = W.shape
    _, n_samples = H.shape
    x_hat = xp.zeros((n_features, n_samples), dtype=W.dtype)
    for idx in range(sequence_length):
        x_hat += xp.dot(W[:, :, idx], xp.roll(H, idx - 1, axis=1))
    return x_hat


def reconstruct_fast(
    W: NDArrayLike,  # noqa: N803
    H: NDArrayLike,  # noqa: N803
    h_shifted: NDArrayLike | None = None,
    xp: ModuleType = np,
) -> NDArrayLike:
    """
    Reconstruct the data matrix from W and H. This version is optimized to perform
    the reconstruction via a single operation, increasing performance at the cost of
    higher memory usage.

    :param W: Basis matrix (n_features x n_components x sequence_length).
    :param H: Coefficient matrix (n_components x n_samples).
    :param h_shifted: Preallocated array for shifted H
        (sequence_length x n_components x n_samples).
    :param xp: Array module (e.g., numpy or cupy) for computation.
    :return: Reconstructed data matrix (n_features x n_samples).
    """
    """
    ====================================================================================
    Ideally, this function should be passed a pre-allocated h_shifted array to avoid
    repeated memory allocation (sequence_length x n_components x n_samples). The
    algorithm has a similar time complexity as the standard reconstruction, but more
    optimally leverages optimized BLAS routines and cache locality. Unfortunately, this
    implementation has a higher space complexity at ~O(CLS), where C is the number of
    components, L is the sequence length, and S is the number of samples. This can be
    pretty problematic for large datasets. In those cases, the standard reconstruction
    function should be used instead, which has ~O(FS), where F is the number of
    features.
    ====================================================================================
    """
    _, n_components, sequence_length = W.shape
    _, n_samples = H.shape

    if h_shifted is None:
        h_shifted = xp.zeros((sequence_length, n_components, n_samples), dtype=H.dtype)
    # else:
    #     h_shifted.fill(0)

    sample_index = xp.arange(n_samples)
    sequence_index = xp.arange(sequence_length)[:, None]
    idx = (sample_index[None, :] - (sequence_index - 1)) % n_samples
    h_shifted[:] = xp.swapaxes(H[:, idx], 0, 1)

    return xp.tensordot(W, h_shifted, axes=([1, 2], [1, 0]))


def rmse(
    X: np.ndarray,  # noqa: N803
    x_hat: np.ndarray,
    padding_index: slice | None = None,
    xp: ModuleType = np,
) -> np.ndarray:
    """
    Compute the root mean squared error (RMSE) between X and x_hat.

    :param X: Original data matrix.
    :param x_hat: Reconstructed data matrix.
    :param padding_index: Optional slice to select unpadded region.
    :param xp: Array module (e.g., numpy or cupy) for computation.
    :return: RMSE value (float).
    """
    if padding_index is not None:
        X_unpad = X[:, padding_index]  # noqa: N806
        x_hat_unpad = x_hat[:, padding_index]
    else:
        X_unpad = X  # noqa: N806
        x_hat_unpad = x_hat
    return xp.sqrt(xp.mean((X_unpad - x_hat_unpad) ** 2))


"""
========================================================================================
Internal Utility Helpers & Functions
========================================================================================
"""


#: Labels for hyperparameters using Unicode characters for better readability.
HYPERPARAMETER_LABELS: dict[str, str] = {
    "lam": f"{chr(0x03BB)}",
    "alpha_H": f"{chr(0x0237A)}{chr(0x1D34)}",
    "alpha_W": f"{chr(0x03BB)}{chr(0x1D42)}",
    "lam_H": f"{chr(0x03BB)}{chr(0x1D34)}",
    "lam_W": f"{chr(0x03BB)}{chr(0x1D42)}",
}


#: Number of iterations to consider for smoothing the cost function when checking
#: for convergence.
COST_SMOOTHING_WINDOW: int = 5


def add_x_ortho_h_penalty(
    wt_x: NDArrayLike,
    lam: float,
    penalty: NDArrayLike,
    conv_func: Callable,
    off_diagonal: NDArrayLike,
    xp: ModuleType = np,
) -> None:
    if lam > 0.0:
        penalty[:] = xp.dot(lam * off_diagonal, conv_func(wt_x))
    else:
        penalty.fill(0.0)


def add_events_penalty(
    H: NDArrayLike,  # noqa: N803
    penalty: NDArrayLike,
    lam_H: float,  # noqa: N803
    off_diagonal: NDArrayLike,
    conv_func: Callable,
) -> None:
    if lam_H <= 0.0:
        return
    penalty += np.dot(lam_H * off_diagonal, conv_func(H))


def check_convergence(
    iteration: int,
    max_iter: int,
    tol: float,
    cost_vector: NDArrayLike,
) -> bool:
    """
    Check for convergence based on the change in cost function.

    :param iteration: Current iteration number.
    :param max_iter: Maximum number of iterations.
    :param tol: Tolerance for convergence.
    :param cost_vector: Array of cost values from previous iterations.
    :return: True if converged, False otherwise.
    """
    if iteration == max_iter:
        return True
    return bool(
        iteration > COST_SMOOTHING_WINDOW
        and np.nanmean(cost_vector[iteration - 5 : -1]) - cost_vector[-1] < tol
    )


def create_textbar(
    n_components: int,
    sequence_length: int,
    max_iter: int,
    **hyperparameters: dict[str, float],
) -> str:
    """
    Create a progress bar with a descriptive label for tracking iterations.

    :param n_components: Number of components in the model.
    :param sequence_length: Length of the sequence being processed.
    :param max_iter: Maximum number of iterations for the progress bar.
    :param hyperparameters: Dictionary of hyperparameters with their values.
        - Keys should match the labels in HYPERPARAMETER_LABELS.
        - Values are floats representing the hyperparameter values.
    :return: A tqdm progress bar object with a descriptive label.
    """
    desc = f"n_components = {n_components}, sequence length = {sequence_length}"
    labels = []
    for hyperparameter, value in hyperparameters.items():
        if value == 0:
            continue
        if value <= 0.1:
            labels.append(f", {HYPERPARAMETER_LABELS[hyperparameter]} = {value:.3e}")
        else:
            labels.append(f", {HYPERPARAMETER_LABELS[hyperparameter]} = {value:.3f}")
    if len(labels) > 0:
        labels = "".join(labels) if len(labels) > 1 else labels[0]
        desc += labels
    return tqdm(
        range(1, max_iter + 1),
        total=max_iter,
        bar_format="{desc}",
        desc=desc,
        position=1,
    )


def sort_indices(
    W: NDArrayLike,  # noqa: N803
    H: NDArrayLike,  # noqa: N803
    loadings: NDArrayLike,
) -> tuple[NDArrayLike, NDArrayLike, NDArrayLike]:
    """
    Sort components based on their loadings in descending order.

    :param W: Basis matrix (n_features x n_components x sequence_length).
    :param H: Coefficient matrix (n_components x n_samples).
    :param loadings: Array of loadings for each component.
    :return: Tuple of sorted (W, H, loadings).
    """
    sorting_indices = np.flip(np.argsort(loadings), 0)
    W = W[:, sorting_indices, :]  # noqa: N806
    H = H[sorting_indices, :]  # noqa: N806
    loadings = loadings[sorting_indices]
    return W, H, loadings


def shift_factors(
    W: np.ndarray,  # noqa: N803
    H: np.ndarray,  # noqa: N803
    xp: ModuleType = np,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift factors in W and H to center their mass.

    :param W: Factor tensor (n_features x n_components x sequence_length).
    :param H: Loading matrix (n_components x n_samples).
    :param xp: Array module (e.g., numpy or cupy) for computation.
    :return: Tuple (shifted W, shifted H).
    """
    warnings.simplefilter("ignore")
    n_features, _, sequence_length = W.shape
    n_components, _ = H.shape
    center = int(max(xp.floor(sequence_length / 2), 1))
    w_pad = xp.concatenate(
        (
            xp.zeros([n_features, n_components, sequence_length]),
            W,
            xp.zeros([n_features, n_components, sequence_length]),
        ),
        axis=2,
    )
    for i in range(n_components):
        temp = xp.sum(xp.squeeze(W[:, i, :]), axis=0)
        try:
            cmass = int(
                xp.max(
                    xp.floor(
                        xp.sum(temp * xp.arange(1, sequence_length + 1)) / xp.sum(temp)
                    ),
                    axis=0,
                )
            )
        except ValueError:
            cmass = center
        w_pad[:, i, :] = xp.roll(xp.squeeze(w_pad[:, i, :]), center - cmass, axis=1)
        H[i, :] = xp.roll(H[i, :], cmass - center, axis=0)
    return w_pad[:, :, sequence_length:-sequence_length], H
    # OPTIMIZE: We can make the standard W a view of w_pad to save memory.
    # TEST: Add tests for shift_factors function in test_support.py


def pad_data(
    X: NDArrayLike,  # noqa: N803
    sequence_length: int,
    xp: ModuleType = np,
) -> NDArrayLike:
    """
    Pad the data matrix X with zeros on both sides along the time axis.

    :param X: Input data matrix (n_samples x n_features).
    :param sequence_length: Length of the sequences to pad.
    :param xp: Array module (e.g., numpy or cupy) for computation.
    :return: Padded data matrix (n_samples + 2 * sequence_length x n_features).
    """
    return xp.pad(
        X,
        ((0, 0), (sequence_length, sequence_length)),
        mode="constant",
        constant_values=0,
    )


def random_init_W(  # noqa: N802
    X: np.ndarray,  # noqa: N803
    n_components: int,
    sequence_length: int,
    random_state: int | None = None,
    xp: ModuleType = np,
) -> np.ndarray:
    """
    Random initialization of W.
    """
    rng = xp.random.default_rng(random_state)
    n_features = X.shape[0]
    return X.max() * rng.random((n_features, n_components, sequence_length))


def random_init_H(  # noqa: N802
    X: np.ndarray,  # noqa: N803
    n_components: int,
    random_state: int | None = None,
    xp: ModuleType = np,
) -> np.ndarray:
    """
    Random initialization of H.
    """
    rng = xp.random.default_rng(random_state)
    n_samples = X.shape[1]
    return X.max() * rng.random((n_components, n_samples)) / xp.sqrt(n_samples / 3)


def nndsvd_init(
    X: np.ndarray,  # noqa: N803
    n_components: int,
    sequence_length: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Placeholder for NNDSVD initialization of W.

    :param X: Input data matrix.
    :param n_components: Number of components.
    :param sequence_length: Sequence length.
    :param random_state: Optional random seed.
    :return: Not implemented.
    """
    msg = (
        f"NNDSVD initialization is not implemented yet; {X}, "
        f"{n_components}, {sequence_length}, {random_state=}"
    )
    raise NotImplementedError(msg)
    # DOC-ME: Add docstring for nndsvd_init function
    # TODO: Implement the NNDSVD initialization algorithm.
    # TEST: Add tests for nndsvd_init function in test_support.py


def renormalize(
    W: NDArrayLike,  # noqa: N803
    H: NDArrayLike,  # noqa: N803
    sequence_length: int,
    epsilon: float = np.finfo(float).eps,
) -> None:
    """
    Renormalize W and H so that each component in H has unit norm.

    :param W:
    :param H:
    :param sequence_length:
    :param epsilon:
    :return: None. W and H are modified in place.
    """
    norms = np.sqrt(np.sum(np.power(H, 2), axis=1)).T
    H[:] = np.dot(np.diag(np.divide(1.0, norms + epsilon)), H)
    for shift in range(sequence_length):
        W[:, :, shift] = np.dot(W[:, :, shift], np.diag(norms))
    # NOTE: The norms calculation is cheap and doesn't require pre-allocation


def trans_tensor_convolution(
    X: np.ndarray,  # noqa: N803
    x_hat: np.ndarray,
    W: np.ndarray,  # noqa: N803
    wt_x: np.ndarray,
    wt_x_hat: np.ndarray,
    sequence_length: int,
) -> None:
    """
    Compute W⊤ ⊛ X and W⊤ ⊛ X̂ using tensor convolution.

    :param X: Input data matrix (n_features x n_samples).
    :param x_hat: Reconstructed data matrix (n_features x n_samples).
    :param W: Weight tensor (n_features x n_components x sequence_length).
    :param wt_x: Preallocated output for W⊤ ⊛ X (n_components x n_samples).
    :param wt_x_hat: Preallocated output for W⊤ ⊛ X̂ (n_components x n_samples).
    :param sequence_length: Length of the sequences.

    :returns: None. Outputs are written to pre-allocated wt_x and wt_x_hat.
    """  # noqa: RUF002
    """
    ====================================================================================
    This is an optimized implementation of the tranpose tensor convolution operation
    W⊤ ⊛ X and W⊤ ⊛ X̂ used in the seqNMF algorithm. It avoids rolling the large data
    matrices X and X̂ by instead rolling the smaller intermediate results. W should be
    much smaller than X in 99% of use cases, leading to significant performance
    improvements. This implementation has time complexity of ~O(NKLS), where N is
    the number of features, K is the number of components,L is the sequence length,
    and S is the number of samples. The space complexity reduction is ~N/K, since we
    only need to store the small intermediate results instead of making rolled copies of
    the large data matrices. This implementation is substantially faster than the
    fully vectorized Einstein summation, which has time complexity of ~O(NKLS + KLS^2)
    due to the convolution step, and is also more memory efficient since it avoids
    creating large intermediate tensors.
    ====================================================================================
    """  # noqa: RUF001
    wt_x.fill(0)
    wt_x_hat.fill(0)
    for step in range(sequence_length):
        temp_x = W[:, :, step].T @ X
        temp_x_hat = W[:, :, step].T @ x_hat
        shift = -step + 1
        wt_x += np.roll(temp_x, shift, axis=1)
        wt_x_hat += np.roll(temp_x_hat, shift, axis=1)
    # NOTE: The assignments to temp_x and temp_x_hat are cheap and don't require
    #  pre-allocation


def assess_vram(device_id: int = 0) -> tuple[float, float]:
    """
    Assess the free & total VRAM of the specified GPU device.

    :param device_id: The ID of the GPU device to assess (default is 0).
    :raises GPUNotSupportedError: If CuPy is not installed.
    :raises GPUNotAvailableError: If no GPU device is available.
    :raises InvalidGPUDeviceError: If the specified device ID is invalid.
    :return: A tuple containing the free and total VRAM in gigabytes (GB).
    """
    if not CUPY_INSTALLED:
        raise GPUNotSupportedError
    if not device_available():
        raise GPUNotAvailableError
    if not is_valid_device(device_id):
        raise InvalidGPUDeviceError(device_id)
    mem_info = Device(device_id).mem_info()
    free_mem = mem_info[0] / (1024**3)
    total_mem = mem_info[1] / (1024**3)
    return free_mem, total_mem


def set_device(device_id: int = 0) -> None:
    """
    Set the active GPU device for CuPy operations.

    :param device_id: The ID of the GPU device to set (default is 0).
    :raises GPUNotSupportedError: If CuPy is not installed.
    :raises GPUNotAvailableError: If no GPU device is available.
    :raises InvalidGPUDeviceError: If the specified device ID is invalid.
    """
    if not CUPY_INSTALLED:
        raise GPUNotSupportedError
    if not device_available():
        raise GPUNotAvailableError
    if not is_valid_device(device_id):
        raise InvalidGPUDeviceError(device_id)
    Device(device_id).use()
