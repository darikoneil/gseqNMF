from collections.abc import Callable
from functools import partial
from warnings import warn

import numpy as np
from scipy.signal import convolve2d
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from gseqnmf.exceptions import SeqNMFInitializationError
from gseqnmf.support import (
    calculate_loading_power,
    calculate_power,
    create_textbar,
    nndsvd_init,
    pad_data,
    random_init_H,
    random_init_W,
    reconstruct,
    rmse,
    shift_factors,
    trans_tensor_convolution,
)
from gseqnmf.validation import (
    INIT_METHOD,
    INITIALIZATION_METHODS,
    PARAMETER_CONSTRAINTS,
)

__all__ = [
    "GseqNMF",
]


class GseqNMF(TransformerMixin, BaseEstimator):
    """
    Sequential Non-negative Matrix Factorization (seqNMF) model.

    Implements the seqNMF algorithm for extracting sequential patterns from data.

    This implementation is based on:

        Mackevicius, E. L., Bahle, A. H., Williams, A. H., Gu, S., Denisenko, N. I.,
        Goldman, M. S., & Fee, M. S. (2019). *Unsupervised discovery of temporal
        sequences in high-dimensional datasets, with applications to neuroscience.*
        eLife, 8, e38471. https://doi.org/10.7554/eLife.38471

    Original seqNMF code: https://github.com/FeeLab/seqNMF

    :param n_components: Number of components to extract.
    :param sequence_length: Length of the sequential patterns.
    :param lam: Regularization parameter for cross-factor competition.
    :param max_iter: Maximum number of iterations.
    :param tol: Tolerance for convergence.
    :param alpha_W: L1 regularization for W.
    :param lam_W: Cross-factor regularization for W.
    :param alpha_H: L1 regularization for H.
    :param lam_H: Cross-factor regularization for H.
    :param shift: Whether to shift factors during updates.
    :param sort: Whether to sort components by loading after fitting.
    :param update_W: Whether to update W during fitting.
    :param init: Initialization method for W and H.
    :param random_state: Random seed for reproducibility.

    :ivar  n_features_in_: Number of features in the input data.
    :ivar  n_samples_in_: Number of samples in the input data.
    :ivar W_: Fitted W matrix.
    :ivar H_: Fitted H matrix.
    :ivar cost_: Training cost per iteration.
    :ivar loadings_: Component loadings.
    :ivar power_: Component powers.
    """

    #: Sklearn parameter validation constraints
    _parameter_constraints: dict = PARAMETER_CONSTRAINTS

    def __init__(
        self,
        n_components: int,
        sequence_length: int,
        lam: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-4,
        alpha_W: float = 0.0,  # noqa: N803
        lam_W: float = 0.0,  # noqa: N803
        alpha_H: float = 0.0,  # noqa: N803
        lam_H: float = 0.0,  # noqa: N803
        shift: bool = True,
        sort: bool = True,
        update_W: bool = True,  # noqa: N803
        init: INITIALIZATION_METHODS | INIT_METHOD = INIT_METHOD.RANDOM,
        random_state: int | None = None,
    ):
        self.n_components: int = n_components
        self.sequence_length: int = sequence_length
        self.lam: float = lam
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.alpha_W: float = alpha_W
        self.lam_W: float = lam_W
        self.alpha_H: float = alpha_H
        self.lam_H: float = lam_H
        self.shift: bool = shift
        self.sort: bool = sort
        self.update_W: bool = update_W
        self.init = init
        self.random_state = random_state
        ###########################################
        self._is_fitted: bool = False
        # NOTE: This is an  sklearn flag to indicate if the model has been fitted.
        self.n_features_in: int | None = None
        self.n_samples_in: int | None = None
        self.W_ = None
        self.H_ = None
        self.cost_ = None
        self.loadings_ = None
        self.power_ = None
        self._validate_params()
        # NOTE: This is sklearn's internal validation routine that leverages the
        #   class attribute _parameter_constraints attribute defined above.

    @staticmethod
    def _initialize(
        X: np.ndarray,  # noqa: N803
        n_components: int,
        sequence_length: int,
        init: INITIALIZATION_METHODS,
        W_init: np.ndarray | None = None,  # noqa: N803
        H_init: np.ndarray | None = None,  # noqa: N803
        random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, INIT_METHOD]:
        """
        Initialize W and H matrices based on the specified method.

        :param X: Input data matrix (n_samples x n_features).
        :param n_components: Number of components.
        :param sequence_length: Length of the sequences.
        :param init: Initialization method ('random', 'exact', 'nndsvd').
        :param W_init: Initial W matrix for 'exact' initialization.
        :param H_init: Initial H matrix for 'exact' initialization.
        :param random_state: Random seed for reproducibility.
        :returns: Tuple of (padded_X, W_init, H_init, init).
        """

        padded_X = pad_data(X, sequence_length)  # noqa: N806
        init = (
            INIT_METHOD.EXACT
            if np.any([factor is not None for factor in [W_init, H_init]])
            else INIT_METHOD.parse(init)
        )
        match init:
            case INIT_METHOD.RANDOM:
                W_init = random_init_W(  # noqa: N806
                    padded_X, n_components, sequence_length, random_state
                )
                H_init = random_init_H(  # noqa: N806
                    padded_X,
                    n_components,
                    random_state,
                )
            case INIT_METHOD.EXACT:
                if W_init is None or H_init is None:
                    msg = "W and H must be provided for 'exact' initialization."
                    raise SeqNMFInitializationError(msg)
                if (
                    W_init.shape[0] != padded_X.shape[0]
                    or W_init.shape[1] != n_components
                    or W_init.shape[2] != sequence_length
                ):
                    msg = (
                        "W must be a 3D array of shape "
                        "(n_features, n_components, sequence_length)."
                    )
                    raise SeqNMFInitializationError(msg)
                if (
                    H_init.shape[0] != n_components
                    or H_init.shape[1] != padded_X.shape[1]
                ):
                    msg = (
                        "H must be a 2D array of shape "
                        "(n_components, n_samples + 2 * sequence_length)."
                    )
                    raise SeqNMFInitializationError(msg)
                W_init = W_init  # noqa: N806
                H_init = H_init  # noqa: N806
            case INIT_METHOD.NNDSVD:
                (W_init, H_init) = nndsvd_init(  # noqa: N806
                    X,
                    n_components,
                    sequence_length,
                )
            case _:
                # noinspection PyUnreachableCode
                msg = (
                    f"Invalid init method: {init}. Choose from {INIT_METHOD.options()}."
                )
                # noinspection PyUnreachableCode
                raise SeqNMFInitializationError(msg)

        return padded_X, W_init, H_init, init

    @staticmethod
    def _prep_handles(
        padded_X: np.ndarray,  # noqa: N803
        sequence_length: int,
    ) -> dict[str, Callable]:
        """
        Prepare function handles for repeated calculations.

        :param padded_X: Padded input data.
        :param sequence_length: Length of the sequences.

        :returns: Dictionary of function handles for cost, loading, power,
            and convolution.
        """
        padding_index = slice(sequence_length, -sequence_length)
        cost_func = partial(
            rmse,
            X=padded_X,
            padding_index=padding_index,
        )
        loading_func = partial(
            calculate_loading_power, X=padded_X, padding_index=padding_index
        )
        power_func = partial(calculate_power, X=padded_X, padding_index=padding_index)
        kernel = np.ones([1, (2 * sequence_length) - 1])
        conv_func = partial(convolve2d, in2=kernel, mode="same")
        tensor_func = partial(trans_tensor_convolution,
                              X=padded_X,
                              sequence_length=sequence_length)
        return {
            "cost": cost_func,
            "loading": loading_func,
            "power": power_func,
            "conv": conv_func,
            "tensor": tensor_func,
        }

    @staticmethod
    def _preallocate(
        n_features: int,
        n_samples: int,
        n_components: int,
        sequence_length: int,
        max_iter: int,
    ) -> dict[str, np.ndarray]:
        """
        Preallocate arrays for intermediate calculations.

        :param n_features: Number of features.
        :param n_samples: Number of samples.
        :param n_components: Number of components.
        :param sequence_length: Length of the sequences.
        :param max_iter: Maximum number of iterations.

        :returns: Dictionary of preallocated arrays.
        """
        n_samples = n_samples + 2 * sequence_length
        cost = np.ones((max_iter + 1, 1)) * np.nan
        x_hat = np.empty((n_features, n_samples))
        xs = np.empty_like(x_hat)
        w_flat = np.empty((n_features, n_components))
        wt_x = np.empty((n_components, n_samples))
        wt_x_hat = np.empty_like(wt_x)
        return {
            "cost": cost,
            "x_hat": x_hat,
            "wt_x": wt_x,
            "wt_x_hat": wt_x_hat,
            "xs": xs,
            "w_flat": w_flat,
        }

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray | None = None,
        W_init: np.ndarray | None = None,  # noqa: N803
        H_init: np.ndarray | None = None,  # noqa: N803
    ) -> "GseqNMF":
        """
        Fit the seqNMF model to the data ``X``.

        :param X: Input data matrix (n_features x n_samples).
        :param y: Ignored, present for API consistency.
        :param W_init: Initial W matrix.
        :param H_init: Initial H matrix.

        :returns: Fitted seqNMF instance.

        :warns UserWarning: If ``y`` is not None.
        """
        if y is not None:
            msg = (
                "y is not used in gseqNMF and should be set to None. "
                "It is present for API consistency by convention."
            )
            warn(msg, UserWarning, stacklevel=2)
        self.n_samples_in, self.n_features_in = X.shape

        X = np.ascontiguousarray(X.T)  # noqa: N806
        # NOTE: sklearn convention is (n_samples, n_features). We transpose and enforce
        #   contiguous array for performance in the underlying routines.
        # OPTIMIZE: We could add a check or flag here to short circuit this step.
        # OPTIMIZE: We could rework some of the math to avoid O(n) space complexity,
        #  but we'd need to check how that impacts cache locality. If we were offloading
        #  to a GPU, I think it O(1) for RAM since we don't need to actually enforce
        #  CPU contiguity in that case.

        """
        ================================================================================
        1.) Initialize. Create zero-padded data matrix to handle edge effects if a
        sequence extends beyond the data boundaries. Create initial values for W and H
        based on the specified initialization method. Note that the shape of H is
        matched to the padded data matrix, NOT the original data matrix.
        ================================================================================
        2.) Make function handles for functions requiring repeated calls with fixed
        parameters, such as calculating cost & loadings. In the future, we can
        either use these handles and the associated builder to incorporate dynamic
        implementation of cpu & gpu calls. Bound numpy arrays will still be passed
        by reference.
        ================================================================================
        3.) Preallocate arrays to hold intermediate calculations. Most users will be
        memory-bound, so we want to avoid unnecessary allocations. This also helps make
        sure that any out of memory errors are caught as early as possible.
        ================================================================================
        4.) Solve by iteratively calling penalized multiplicative updates
        until convergence or reaching the maximum number of iterations.
        ================================================================================
        5.) After fitting, store the final W and H matrices, as well as the training
        cost, loadings, and power.
        ================================================================================
        """
        X, W, H, self.init = self._initialize(  # noqa: N806
            X=X,
            n_components=self.n_components,
            sequence_length=self.sequence_length,
            init=self.init,
            W_init=W_init,
            H_init=H_init,
        )

        _handles = self._prep_handles(X, self.sequence_length)
        cost_func = _handles["cost"]
        loading_func = _handles["loading"]
        power_func = _handles["power"]
        conv_func = _handles["conv"]
        trans_tensor_conv_func = _handles["tensor"]

        _arrays = self._preallocate(
            self.n_features_in,
            self.n_samples_in,
            self.n_components,
            self.sequence_length,
            self.max_iter,
        )
        cost = _arrays["cost"]
        x_hat = _arrays["x_hat"]
        xs = _arrays["xs"]
        w_flat = _arrays["w_flat"]
        wt_x = _arrays["wt_x"]
        wt_x_hat = _arrays["wt_x_hat"]

        epsilon = np.max(X) * 1e-6
        off_diagonal = 1 - np.eye(self.n_components)
        post_fix = {"cost": "N/A"}

        local_lam = self.lam
        local_tol = self.tol
        local_max_iter = self.max_iter
        IS_FIT = False  # noqa: N806

        textbar = create_textbar(
            self.n_components,
            self.sequence_length,
            self.max_iter,
            lam=self.lam,
            alpha_H=self.alpha_H,
            alpha_W=self.alpha_W,
            lam_H=self.lam_H,
            lam_W=self.lam_W,
        )

        pbar = tqdm(
            range(self.max_iter),
            total=self.max_iter,
            desc="Fitting",
            unit="iter",
            initial=0,
            colour="#6dff9b",
            # mininterval=0.25,
            postfix=post_fix,
            position=1,
            # delay=1e-10,
        )
        # BUG: tqdm bar sometimes rendering text only last
        #  or many reprints of progress  bar
        x_hat.fill(0)
        x_hat += self.inverse_transform(W, H)
        cost[0] = cost_func(x_hat=x_hat)

        for iter_ in range(1, self.max_iter + 1):
            if (
                IS_FIT := (  # noqa: N806
                    (iter_ == local_max_iter)
                    or (
                        (iter_ > 5)
                        and (np.nanmean(cost[iter_ - 5 : -1]) - cost[-1] < local_tol)
                    )
                )
            ):
                local_lam = 0

            trans_tensor_conv_func(W=W, x_hat=x_hat, wt_x=wt_x, wt_x_hat=wt_x_hat)
            #: NOTE: This calculation of W⊤⊛X & W⊤⊛X̂ is a bottleneck.  # noqa: RUF003

            if local_lam > 0:
                subgradient_H = np.dot(  # noqa: N806
                    local_lam * off_diagonal, conv_func(wt_x)
                )
                # NOTE: No need to pre-allocate since subgradients are scalar
            else:
                subgradient_H = 0.0  # noqa: N806

            if self.lam_H > 0:
                dHHdH = np.dot(self.lam_H * off_diagonal, conv_func(H))  # noqa: N806
            else:
                dHHdH = 0.0  # noqa: N806
            subgradient_H += self.alpha_H + dHHdH  # noqa: N806

            H *= np.divide(wt_x, wt_x_hat + subgradient_H + epsilon)  # noqa: N806

            if self.shift:
                W, H = shift_factors(W, H)  # noqa: N806
                W += epsilon  # noqa: N806

            norms = np.sqrt(np.sum(np.power(H, 2), axis=1)).T
            H = np.dot(np.diag(np.divide(1.0, norms + epsilon)), H)  # noqa: N806
            for shift in range(self.sequence_length):
                W[:, :, shift] = np.dot(W[:, :, shift], np.diag(norms))

            x_hat = self.inverse_transform(W, H)

            if self.lam_W > 0:
                w_flat = W.sum(axis=2)
            if (local_lam > 0) and self.update_W:
                xs = conv_func(X)

            for shift in range(self.sequence_length):
                h_shifted = np.roll(H, shift - 1, axis=1)
                x_ht = np.dot(X, h_shifted.T)
                x_hat_ht = np.dot(x_hat, h_shifted.T)

                if (local_lam > 0) and self.update_W:
                    subgradient_W = np.dot(  # noqa: N806
                        local_lam * np.dot(xs, h_shifted.T), off_diagonal
                    )
                else:
                    subgradient_W = 0.0  # noqa: N806

                if self.lam_W > 0:
                    dWWdW = np.dot(self.lam_W * w_flat, off_diagonal)  # noqa: N806
                else:
                    dWWdW = 0.0  # noqa: N806
                subgradient_W += self.alpha_W + dWWdW  # noqa: N806

                W[:, :, shift] *= np.divide(x_ht, x_hat_ht + subgradient_W + epsilon)

            cost[iter_] = cost_func(x_hat=x_hat)
            post_fix["cost"] = f"{cost[iter_].item():.4e}"
            pbar.update()
            pbar.set_postfix(post_fix)

            if IS_FIT:
                break
        pbar.close()
        textbar.close()

        self.W_ = W
        self.H_ = H
        self.cost_ = cost
        self.loadings_ = loading_func(W=W, H=H)
        self.power_ = power_func(x_hat=x_hat)

        if self.sort:
            sorting_indices = np.flip(np.argsort(self.loadings_), 0)
            self.W_ = self.W_[:, sorting_indices, :]
            self.H_ = self.H_[sorting_indices, :]
            self.loadings_ = self.loadings_[sorting_indices]
        self._is_fitted = True
        return self

    def get_params(self, deep: bool = True) -> dict:  # noqa: ARG002
        """
        Get parameters for this estimator.

        :param deep: If True, will return the parameters for this estimator and
            contained subobjects.

        :returns: Parameter names mapped to their values.
        """
        return {key: getattr(self, key) for key in self._parameter_constraints}

    def set_params(self, **params) -> "GseqNMF":
        """
        Set the parameters of this estimator.

        :param params: Estimator parameters.

        :returns: Estimator instance.

        :raises AttributeError: If an invalid parameter is provided.
        """
        for key, value in params.items():
            if key not in self._parameter_constraints:
                msg = (
                    f"Invalid parameter {key} for estimator {self.__class__.__name__}."
                )
                raise AttributeError(msg)
            setattr(self, key, value)
        self._validate_params()
        return self

    def fit_transform(
        self,
        X: np.ndarray,  # noqa: N803
        W: np.ndarray | None = None,  # noqa: N803
        H: np.ndarray | None = None,  # noqa: N803
    ) -> np.ndarray:
        """
        Fit the model to ``X`` and return the transformed data.

        :param X: Input data matrix.
        :param W: Initial W matrix.
        :param H: Initial H matrix.

        :returns: Transformed data (usually H matrix).

        :raises NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError

    # noinspection PyUnusedLocal
    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: ARG002, N803
        """
        Transform the data ``X`` using the fitted model.

        :param X: Input data matrix.

        :returns: Transformed data.

        :raises NotImplementedError: This method is not yet implemented.
        """
        check_is_fitted(self)
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def inverse_transform(
        self,
        W: np.ndarray | None = None,  # noqa: N803
        H: np.ndarray | None = None,  # noqa: N803
    ) -> np.ndarray:
        """
        Reconstruct the data from W and H matrices.

        :param W: W matrix. If None, uses the fitted ``W_``.
        :param H: H matrix. If None, uses the fitted ``H_``.

        :returns: Reconstructed data matrix.
        """
        W = W if W is not None else self.W_  # noqa: N806
        H = H if H is not None else self.H_  # noqa: N806
        return reconstruct(W, H)

    # noinspection PyMethodMayBeStatic
    def _more_tags(self) -> dict[str, bool]:
        """
        Return scikit-learn tags for this estimator.

        :returns: Dictionary of tag names mapped to tag values.
        """
        return {"stateless": False}

    def __sklearn_is_fitted__(self) -> bool:
        """
        Return whether the estimator has been fitted.

        :returns: True if fitted, False otherwise.
        """
        return self._is_fitted
