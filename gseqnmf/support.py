import warnings
from types import ModuleType

import numpy as np
from tqdm import tqdm

from gseqnmf.validation import NDArrayLike

__all__ = [
    "calculate_power",
    "compute_loading_percent_power",
    "nndsvd_init",
    "pad_data",
    "random_init_H",
    "random_init_W",
    "reconstruct",
    "rmse",
    "shift_factors",
]

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
    labels = []
    for hyperparameter, value in hyperparameters.items():
        if value == 0:
            continue
        if value <= 0.1:
            labels.append(f"{HYPERPARAMETER_LABELS[hyperparameter]} = {value:.3e}")
        else:
            labels.append(f"{HYPERPARAMETER_LABELS[hyperparameter]} = {value:.3f}")
    if len(labels) > 0:
        labels = ", ".join(labels) if len(labels) > 1 else labels[0]
    return tqdm(
        range(1, max_iter + 1),
        total=max_iter,
        bar_format="{desc}{postfix}",
        desc=f"n_components = {n_components}, "
        f"sequence length = {sequence_length}, "
        f"{labels}",
        position=0,
    )


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


def shift_factors(W: np.ndarray, H: np.ndarray, xp=np) -> tuple[np.ndarray, np.ndarray]:
    """
    Shifts factors in W and H to center their mass.

    :param W: Factor tensor.
    :param H: Loading matrix.
    :return: Tuple (shifted W, shifted H)
    """
    warnings.simplefilter("ignore")
    N, _, L = W.shape
    K, T = H.shape
    if L > 1:
        center = int(xp.max([xp.floor(L / 2), 1]))
        Wpad = xp.concatenate((xp.zeros([N, K, L]), W, xp.zeros([N, K, L])), axis=2)
        for i in range(K):
            temp = xp.sum(xp.squeeze(W[:, i, :]), axis=0)
            try:
                cmass = int(
                    xp.max(
                        xp.floor(xp.sum(temp * xp.arange(1, L + 1)) / xp.sum(temp)),
                        axis=0,
                    )
                )
            except ValueError:
                cmass = center
            Wpad[:, i, :] = xp.roll(xp.squeeze(Wpad[:, i, :]), center - cmass, axis=1)
            H[i, :] = xp.roll(H[i, :], cmass - center, axis=0)
    return Wpad[:, :, L:-L], H


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


def rmse(
    X: np.ndarray,  # noqa: N803
    x_hat: np.ndarray,
    padding_index: slice | None = None,
    xp: ModuleType = np,
) -> np.ndarray:
    """
    Compute the root mean square error between X and x_hat.
    """
    if padding_index is not None:
        X_unpad = X[:, padding_index]  # noqa: N806
        x_hat_unpad = x_hat[:, padding_index]
    else:
        X_unpad = X  # noqa: N806
        x_hat_unpad = x_hat
    return xp.sqrt(xp.mean((X_unpad - x_hat_unpad) ** 2))


def calculate_power(
    X: np.ndarray,  # noqa: N803
    x_hat: np.ndarray,
    epsilon: float = np.finfo(float).eps,
    padding_index: slice | None = None,
    xp: ModuleType = np,
) -> float:
    if padding_index is not None:
        X_unpad = X[:, padding_index]  # noqa: N806
        x_hat_unpad = x_hat[:, padding_index]
    else:
        X_unpad = X  # noqa: N806
        x_hat_unpad = x_hat
    denom = xp.sum(X_unpad**2) + epsilon
    return (xp.sum(X_unpad**2) - xp.sum((X_unpad - x_hat_unpad) ** 2)) / denom


def compute_loading_percent_power(
    X: np.ndarray,  # noqa: N803
    W: np.ndarray,  # noqa: N803
    H: np.ndarray,  # noqa: N803
    epsilon: float = np.finfo(float).eps,
    padding_index: slice | None = None,
    xp: ModuleType = np,
) -> np.ndarray:
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


def nndsvd_init(
    X: np.ndarray,  # noqa: N803
    n_components: int,
    sequence_length: int,
    random_state: int | None = None,
) -> np.ndarray:
    print(
        f"NNDSVD initialization is not implemented yet; "
        f"{X}, "
        f"{n_components}, "
        f"{sequence_length}, "
        f"{random_state=}"
    )
    return


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
