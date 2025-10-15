from collections.abc import Callable
from functools import partial
from types import ModuleType
from warnings import warn

import numpy as np
from scipy.signal import convolve2d
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from gseqnmf.exceptions import (
    GPUNotAvailableError,
    GPUNotSupportedError,
    SeqNMFInitializationError,
)
from gseqnmf.support import (
    add_events_penalty,
    add_x_ortho_h_penalty,
    calculate_loading_power,
    calculate_power,
    check_convergence,
    create_textbar,
    nndsvd_init,
    pad_data,
    random_init_H,
    random_init_W,
    reconstruct,
    reconstruct_fast,
    renormalize,
    rmse,
    shift_factors,
    sort_indices,
    trans_tensor_convolution,
)
from gseqnmf.validation import (
    CUPY_INSTALLED,
    INIT_METHOD,
    INITIALIZATION_METHODS,
    PARAMETER_CONSTRAINTS,
    RECON_METHOD,
    RECONSTRUCTION_METHODS,
    W_UPDATE_METHOD,
    W_UPDATE_METHODS,
    NDArrayLike,
    cuda_available,
    device_available,
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
    :param recon: Reconstruction method.
    :param use_gpu: Whether to use GPU acceleration.

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
        W_update: W_UPDATE_METHODS | W_UPDATE_METHOD = W_UPDATE_METHOD.FULL,  # noqa: N803
        init: INITIALIZATION_METHODS | INIT_METHOD = INIT_METHOD.RANDOM,
        recon: RECONSTRUCTION_METHODS | RECON_METHOD = RECON_METHOD.FAST,
        random_state: int | None = None,
        use_gpu: bool = False,
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
        self.W_update: W_UPDATE_METHODS | W_UPDATE_METHOD = W_UPDATE_METHOD.parse(
            W_update
        )
        self.init: INITIALIZATION_METHODS | INIT_METHOD = INIT_METHOD.parse(init)
        self.recon: RECONSTRUCTION_METHODS | RECON_METHOD = recon
        self.random_state: int | None = random_state
        self.use_gpu: bool = use_gpu
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
        #   class attribute _parameter_constraints attribute defined above. Note, it
        #   does not validate the use_gpu parameter.
        if use_gpu:
            self._validate_gpu()
            self.xp_ = self._import_cupy()
        else:
            self.xp_ = np
        ############################################
        self._set_recon_method()

    @staticmethod
    def _initialize(
        X: np.ndarray,  # noqa: N803
        n_components: int,
        sequence_length: int,
        init: INIT_METHOD,
        W_update: W_UPDATE_METHOD,  # noqa: N803
        W_init: np.ndarray | None = None,  # noqa: N803
        H_init: np.ndarray | None = None,  # noqa: N803
        random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize W and H matrices based on the specified method.

        :param X: Input data matrix (n_samples x n_features).
        :param n_components: Number of components.
        :param sequence_length: Length of the sequences.
        :param init: Initialization method ('random', 'exact', 'nndsvd').
        :param W_update: Whether W is being updated.
        :param W_init: Initial W matrix for 'exact' initialization.
        :param H_init: Initial H matrix for 'exact' initialization.
        :param random_state: Random seed for reproducibility.
        :returns: Tuple of (padded_X, W_init, H_init, init).
        """

        padded_X = pad_data(X, sequence_length)  # noqa: N806
        match init:
            case INIT_METHOD.RANDOM:
                W_init = (  # noqa: N806
                    random_init_W(padded_X, n_components, sequence_length, random_state)
                    if W_update.value != W_UPDATE_METHOD.FIXED.value
                    else W_init
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
        return padded_X, W_init, H_init

    @staticmethod
    def _prep_handles(
        padded_X: NDArrayLike,  # noqa: N803
        sequence_length: int,
        off_diagonal: NDArrayLike,
        # lam_W: float = 0.0,
        lam_H: float = 0.0,  # noqa: N803
        epsilon: float = np.finfo(float).eps,
    ) -> dict[str, Callable]:
        """
        Prepare function handles for repeated calculations.

        :param padded_X: Padded input data.
        :param sequence_length: Length of the sequences.
        :param recon: Reconstruction method ('normal', 'fast').
        :returns: Dictionary of function handles for various calculations.
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
        tensor_func = partial(
            trans_tensor_convolution, X=padded_X, sequence_length=sequence_length
        )
        renormalize_func = partial(
            renormalize, sequence_length=sequence_length, epsilon=epsilon
        )
        add_x_ortho_h_penalty_func = partial(
            add_x_ortho_h_penalty,
            off_diagonal=off_diagonal,
            conv_func=conv_func,
        )
        add_events_penalty_func = partial(
            add_events_penalty,
            lam_H=lam_H,
            off_diagonal=off_diagonal,
            conv_func=conv_func,
        )
        return {
            "cost": cost_func,
            "loading": loading_func,
            "power": power_func,
            "conv": conv_func,
            "tensor": tensor_func,
            "norm": renormalize_func,
            "x_ortho_h": add_x_ortho_h_penalty_func,
            "events": add_events_penalty_func,
        }

    @staticmethod
    def _preallocate(
        n_features: int,
        n_samples: int,
        n_components: int,
        sequence_length: int,
        max_iter: int,
        lam: float,
        lam_W: float,  # noqa: N803
        update_W: bool,  # noqa: N803
        recon: RECON_METHOD,
    ) -> dict[str, np.ndarray]:
        """
        Preallocate arrays for intermediate calculations.

        :param n_features: Number of features.
        :param n_samples: Number of samples.
        :param n_components: Number of components.
        :param sequence_length: Length of the sequences.
        :param max_iter: Maximum number of iterations.
        :param lam: Regularization parameter for x-ortho.
        :param lam_W: Parts-based regularization parameter for W.
        :param update_W: Whether W is being updated.
        :param recon: Reconstruction method.
        :returns: Dictionary of preallocated arrays.
        """
        n_samples = n_samples + 2 * sequence_length
        return {
            "cost": np.ones((max_iter + 1, 1)) * np.nan,
            "h_shifted": np.empty((n_components, n_samples)),
            "recon_h_shifted": np.empty((sequence_length, n_components, n_samples))
            if recon == RECON_METHOD.FAST
            else None,
            "pd_penalty_h": np.empty((n_components, n_samples)),
            "pd_penalty_w": np.empty((n_features, n_components)),
            "x_hat": np.empty((n_features, n_samples)),
            "wt_x": np.empty((n_components, n_samples)),
            "wt_x_hat": np.empty((n_components, n_samples)),
            "xs": np.empty((n_features, n_samples))
            if (lam > 0.0 and update_W.value == W_UPDATE_METHOD.FULL.value)
            else None,
            "w_flat": np.empty((n_features, n_components)) if lam_W > 0.0 else None,
        }

    # noinspection PyUnusedLocal
    @staticmethod
    def _inverse_transform(
        W: np.ndarray,  # noqa: N803
        H: np.ndarray,  # noqa: N803
        h_shifted: np.ndarray | None,
    ) -> np.ndarray:
        """Placeholder for the inverse transform method set in _set_recon_method."""
        msg = "This method should be set in _set_recon_method."
        raise NotImplementedError(msg)

    @staticmethod
    def _validate_gpu() -> ModuleType:
        """
        Validate that GPU support is available.

        :raises NotImplementedError: If GPU support is not available.
        """
        if not CUPY_INSTALLED:
            raise GPUNotSupportedError
        if not cuda_available() or not device_available():
            raise GPUNotAvailableError

    @staticmethod
    def _import_cupy() -> ModuleType:
        """
        Import the CuPy library.
        """
        import cupy as cp

        return cp

    # HACK: This is hacky, but I want to pass the xp module dynamically

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: NDArrayLike,  # noqa: N803
        y: NDArrayLike | None = None,
        W: NDArrayLike | None = None,  # noqa: N803
        H: NDArrayLike | None = None,  # noqa: N803
    ) -> "GseqNMF":
        """
        Fit the seqNMF model to the data ``X``.

        :param X: Input data matrix (n_features x n_samples).
        :param y: Ignored, present for API consistency.
        :param W: Initial W matrix.
        :param H: Initial H matrix.

        :returns: Fitted seqNMF instance.

        :warns UserWarning: If ``y`` is not None.
        """
        if y is not None:
            msg = (
                "y is not used in gseqNMF and should be set to None. "
                "It is present for API consistency by convention."
            )
            warn(msg, UserWarning, stacklevel=2)

        (self.W_, self.H_, self.cost_, self.loadings_, self.power_) = self._fit(
            X=X, W=W, H=H, W_update=self.W_update
        )

        if self.sort:
            self.W_, self.H_, self.loadings_ = sort_indices(
                self.W_, self.H_, self.loadings_
            )

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
        :returns: Transformed data (H)
        """
        # WARN: If we call fit_transform we DON'T want sklearn to automatically call
        #  fit followed by transform. The result with be more accurate if we call fit
        #  and then return H DIRECTLY rather than calling fit with a fixed W. It also
        #  avoids having to essentially fit twice.
        self.fit(X=X, W=W, H=H)
        return self.H_

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """
        Transform the data ``X`` using the fitted model.

        :param X: Input data matrix.
        :returns: Transformed data (n_components x n_samples).
        """
        check_is_fitted(self)
        _, H, _, _, _ = self._fit(  # noqa: N806
            X=X,
            W=self.W_,
            H=self.H_,
            W_update=W_UPDATE_METHOD.FIXED,
        )
        return H

    # noinspection PyMethodMayBeStatic
    def inverse_transform(
        self,
        W: np.ndarray | None = None,  # noqa: N803
        H: np.ndarray | None = None,  # noqa: N803
        h_shifted: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct the data from W and H matrices.

        :param W: W matrix. If None, uses the fitted ``W_``.
        :param H: H matrix. If None, uses the fitted ``H_``.
        :param h_shifted: Pre-shifted H matrix (only used in certain implementations).
        :returns: Reconstructed data matrix.
        """
        # NOTE: Call the _inverse_transform method during internal calls to avoid
        #  premature checking of fitted status.
        W = W if W is not None else self.W_  # noqa: N806
        H = H if H is not None else self.H_  # noqa: N806
        check_is_fitted(self)
        return self._inverse_transform(W=W, H=H, h_shifted=h_shifted)

    def _fit(
        self,
        X: NDArrayLike,  # noqa: N803
        W_update: W_UPDATE_METHOD,  # noqa: N803
        W: NDArrayLike | None = None,  # noqa: N803
        H: NDArrayLike | None = None,  # noqa: N803
        xp: ModuleType = np,  # TODO: drop-in
    ) -> tuple[NDArrayLike, NDArrayLike, NDArrayLike, NDArrayLike, float]:
        """
        Core fitting routine for seqNMF.
        :param X: Input data matrix.
        :param W: Initial W matrix.
        :param W_update: Whether W is being updated.
        :param H: Initial H matrix.
        :param xp: Numerical computing module (numpy or cupy).
        :return: Tuple of (W, H, cost, loadings, power).
        """
        self.n_samples_in, self.n_features_in = X.shape

        X = np.ascontiguousarray(X.T)  # noqa: N806
        # NOTE: sklearn convention is (n_samples, n_features). We transpose and enforce
        #   contiguous array for performance in the underlying routines.
        # OPTIMIZE: We could add some sort of flag here to short circuit this step?
        # OPTIMIZE: We could rework some of the math to avoid O(n) space complexity,
        #  but we'd need to check how that impacts cache locality. If we were
        #  offloading to a GPU, I think it's O(1) for RAM since we don't need to
        #  actually enforce CPU contiguity in that case (I think)? Maybe there's
        #  some transient memory explosion on the GPU when we do this, but worst case
        #  we could always transfer in chunks. We probably want to do that anyway since
        #  VRAM is more limited.

        """
        ================================================================================
        1.) Initialize. Create zero-padded data matrix to handle edge effects if a
        sequence extends beyond the data boundaries. Create initial values for W and H
        based on the specified initialization method. Note that the shape of H is
        matched to the padded data matrix, NOT the original data matrix.
        ================================================================================
        2.) Make function handles for functions requiring repeated calls with fixed
        parameters, such as calculating cost & loadings. We can
        either use these handles and the associated builder to incorporate dynamic
        implementation of cpu & gpu calls. Bound cupy/numpy arrays will still be passed
        as references in the bytecode, so we don't need to worry about sub-optimal
        spatial complexity. We don't call a routine to bind an implementation for these
        specific to the instance so that the memory is reliably released after fitting.
        Alternatively, we could have bound this with the arguments as weak
        references/proxies, but I think this is better encapsulated this way. We also
        don't bind the handles to the instance (i.e. self._cost_func) because I don't
        want to deal with per-parameter re-binding for every set_params call.
        ================================================================================
        3.) Preallocate arrays to hold intermediate calculations. Most users will be
        memory-bound, so we want to avoid unnecessary allocations. Some of these
        obviously improve performance, but the main goal is to avoid ungraceful
        failure due to memory exhaustion.
        ================================================================================
        4.) Solve by iteratively calling regularized multiplicative updates
        until convergence or the maximum number of iterations.
        ================================================================================
        """
        if W_update.value == W_UPDATE_METHOD.FIXED.value:
            X, _, H = self._initialize(  # noqa: N806
                X=X,
                n_components=self.n_components,
                sequence_length=self.sequence_length,
                init=INIT_METHOD.RANDOM,
                W_update=W_update,
                W_init=W,
                H_init=H,
            )
        else:
            X, W, H = self._initialize(  # noqa: N806
                X=X,
                n_components=self.n_components,
                sequence_length=self.sequence_length,
                init=self.init,
                W_update=W_update,
                W_init=W,
                H_init=H,
            )

        epsilon = np.max(X) * 1e-6
        off_diagonal = 1 - np.eye(self.n_components)

        _handles = self._prep_handles(
            X,
            self.sequence_length,
            off_diagonal=off_diagonal,
            # lam_W=self.lam_W,
            lam_H=self.lam_H,
            epsilon=epsilon,
        )
        cost_func = _handles["cost"]
        loading_func = _handles["loading"]
        power_func = _handles["power"]
        conv_func = _handles["conv"]
        trans_tensor_conv_func = _handles["tensor"]
        renormalize_func = _handles["norm"]
        add_events_penalty_func = _handles["events"]
        add_x_ortho_h_penalty_func = _handles["x_ortho_h"]

        _prealloc = self._preallocate(
            n_features=self.n_features_in,
            n_samples=self.n_samples_in,
            n_components=self.n_components,
            sequence_length=self.sequence_length,
            max_iter=self.max_iter,
            lam=self.lam,
            lam_W=self.lam_W,
            update_W=W_update,
            recon=self.recon,
        )
        cost = _prealloc["cost"]
        h_shifted = _prealloc["h_shifted"]
        recon_h_shifted = _prealloc["recon_h_shifted"]
        pd_penalty_h = _prealloc["pd_penalty_h"]
        pd_penalty_w = _prealloc["pd_penalty_w"]
        x_hat = _prealloc["x_hat"]
        xs = _prealloc["xs"]
        w_flat = _prealloc["w_flat"]
        wt_x = _prealloc["wt_x"]
        wt_x_hat = _prealloc["wt_x_hat"]

        x_hat[:] = self._inverse_transform(W=W, H=H, h_shifted=recon_h_shifted)
        cost[0] = cost_func(x_hat=x_hat)
        # NOTE: Initial cost before any updates

        post_fix = {"cost": "N/A"}
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
            postfix=post_fix,
            position=0,
        )

        local_lam = self.lam
        for iter_ in range(1, self.max_iter + 1):
            if IS_FIT := check_convergence(  # noqa: N806
                iteration=iter_, max_iter=self.max_iter, cost_vector=cost, tol=self.tol
            ):
                local_lam = 0
                # NOTE: We set local_lam to 0 at convergence when the IS_FIT flag is
                #  set. The final iteration will complete the updates with no
                #  cross-factor regularization.

            trans_tensor_conv_func(W=W, x_hat=x_hat, wt_x=wt_x, wt_x_hat=wt_x_hat)
            # NOTE: This calculation of Wt⊛X & Wt⊛X̂ is a bottleneck.

            add_x_ortho_h_penalty_func(
                wt_x=wt_x,
                lam=local_lam,
                penalty=pd_penalty_h,
            )
            add_events_penalty_func(
                H=H,
                penalty=pd_penalty_h,
            )
            pd_penalty_h += self.alpha_H
            # NOTE: calculation of partial derivative of penalty w.r.t H is a secondary
            #  bottleneck, specifically the conv2d operation called in the non-scalar
            #  x-ortho & events penalties.

            H *= np.divide(wt_x, wt_x_hat + pd_penalty_h + epsilon)  # noqa: N806
            if self.shift:
                W, H = shift_factors(W, H)  # noqa: N806
                W += epsilon  # noqa: N806
            renormalize_func(W=W, H=H)
            x_hat[:] = self._inverse_transform(W=W, H=H, h_shifted=recon_h_shifted)

            if W_update.value != W_UPDATE_METHOD.FIXED.value:
                if self.lam_W > 0:
                    w_flat[:] = W.sum(axis=2)

                if (local_lam > 0) and W_update.value == W_UPDATE_METHOD.FULL.value:
                    xs[:] = conv_func(X)

                for shift in range(self.sequence_length):
                    h_shifted[:] = np.roll(H, shift - 1, axis=1)
                    x_ht = np.dot(X, h_shifted.T)
                    x_hat_ht = np.dot(x_hat, h_shifted.T)

                    if (local_lam > 0) and W_update.value == W_UPDATE_METHOD.FULL.value:
                        pd_penalty_w[:] = np.dot(
                            local_lam * np.dot(xs, h_shifted.T), off_diagonal
                        )
                    else:
                        pd_penalty_w.fill(0.0)

                    if self.lam_W > 0:
                        pd_penalty_w += np.dot(self.lam_W * w_flat, off_diagonal)

                    pd_penalty_w += self.alpha_W

                    W[:, :, shift] *= np.divide(x_ht, x_hat_ht + pd_penalty_w + epsilon)

            x_hat[:] = self._inverse_transform(W=W, H=H, h_shifted=recon_h_shifted)
            cost[iter_] = cost_func(x_hat=x_hat)
            post_fix["cost"] = f"{cost[iter_].item():.4e}"
            pbar.update()
            pbar.set_postfix(post_fix)

            if IS_FIT:
                break

        pbar.close()
        textbar.close()
        # NOTE: Make sure to close the progress bars to avoid display issues.

        return (
            W,
            H[:, self.sequence_length : -self.sequence_length],
            cost,
            loading_func(W=W, H=H),
            power_func(x_hat=x_hat),
        )

    def _set_recon_method(self) -> None:
        """
        Set the reconstruction method based on the specified parameter.

        :raises SeqNMFInitializationError: If an invalid reconstruction method is
            provided.
        """
        recon = (
            RECON_METHOD.parse(self.recon)
            if not isinstance(self.recon, RECON_METHOD)
            else self.recon
        )
        match recon:
            case RECON_METHOD.NORMAL:
                self._inverse_transform = reconstruct
            case RECON_METHOD.FAST:
                self._inverse_transform = reconstruct_fast
            case _:
                # noinspection PyUnreachableCode
                msg = (
                    f"Invalid reconstruction method: {recon}. "
                    f"Choose from {RECON_METHOD.options()}."
                )
                # noinspection PyUnreachableCode
                raise SeqNMFInitializationError(msg)

    # noinspection PyMethodMayBeStatic
    def _more_tags(self) -> dict[str, bool]:
        """
        Return scikit-learn tags for this estimator.

        :returns: Dictionary of tag names mapped to tag values.
        """
        return {"stateless": False}  # pragma: no cover

    def __sklearn_is_fitted__(self) -> bool:
        """
        Return whether the estimator has been fitted.

        :returns: True if fitted, False otherwise.
        """
        return self._is_fitted
