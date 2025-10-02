import warnings
from types import ModuleType

import numpy as np
from tqdm import tqdm

from gseqnmf.validation import NDArrayLike

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
    X: np.ndarray,  # noqa: N803
    x_hat: np.ndarray,
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
    if padding_index is not None:
        X_unpad = X[:, padding_index]  # noqa: N806
        x_hat_unpad = x_hat[:, padding_index]
    else:
        X_unpad = X  # noqa: N806
        x_hat_unpad = x_hat
    denom = xp.sum(X_unpad**2) + epsilon
    return 100 * (xp.sum(X_unpad**2) - xp.sum((X_unpad - x_hat_unpad) ** 2)) / denom
    # TEST: Add tests for calculate_power function in test_support.py


def calculate_loading_power(
    X: np.ndarray,  # noqa: N803
    W: np.ndarray,  # noqa: N803
    H: np.ndarray,  # noqa: N803
    epsilon: float = np.finfo(float).eps,
    padding_index: slice | None = None,
    xp: ModuleType = np,
) -> np.ndarray:
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
    # TEST: Add tests for calculate_loading_power function in test_support.py


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


def reconstruct(
    W: NDArrayLike,  # noqa: N803
    H: NDArrayLike,  # noqa: N803
    xp: ModuleType = np,
) -> NDArrayLike:
    """
    Reconstruct the data matrix from W and H.

    :param W: Basis matrix (n_features x n_components x sequence_length).
    :param H: Coefficient matrix (n_components x n_samples).
    :param xp: Array module (e.g., numpy or cupy) for computation.
    :return: Reconstructed data matrix (n_features x n_samples).
    """
    n_features, n_components, sequence_length = W.shape
    n_components_H, n_samples = H.shape  # noqa: N806
    assert n_components == n_components_H, (
        f"Mismatch in components: W has {n_components}, H has {n_components_H}."
    )
    if sequence_length > n_samples:
        msg = f"sequence_length ({sequence_length}) cannot be greater than n_samples ({n_samples})."
        raise ValueError(msg)
    x_hat = xp.zeros((n_features, n_samples), dtype=W.dtype)
    for idx in range(sequence_length):
        x_hat += xp.dot(W[:, :, idx], xp.roll(H, idx - 1, axis=1))
    return x_hat
    # TEST: Add tests for reconstruct function in test_support.py


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
    center = int(xp.max([xp.floor(sequence_length / 2), 1]))
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
    print(
        f"NNDSVD initialization is not implemented yet; "
        f"{X}, "
        f"{n_components}, "
        f"{sequence_length}, "
        f"{random_state=}"
    )
    return
    # DOC-ME: Add docstring for nndsvd_init function
    # TODO: Implement the NNDSVD initialization algorithm.
    # TEST: Add tests for nndsvd_init function in test_support.py
