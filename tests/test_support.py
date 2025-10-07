from collections.abc import Callable

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np
import pytest

from gseqnmf.support import (
    HYPERPARAMETER_LABELS,
    calculate_loading_power,
    calculate_power,
    create_textbar,
    pad_data,
    random_init_H,
    random_init_W,
    reconstruct,
    reconstruct_fast,
    rmse,
    shift_factors,
    trans_tensor_convolution,
)
from tests.conftest import Dataset


class TestCalculatePower:
    @staticmethod
    @pytest.mark.parametrize(
        ("X", "x_hat", "expected_power"),
        [
            pytest.param(
                np.array([[1, 2], [3, 4]]),
                np.array([[1, 2], [3, 4]]),
                100.0,
                id="perfect reconstruction",
            ),
            pytest.param(
                np.array([[1, 2], [3, 4]]),
                np.array([[0, 0], [0, 0]]),
                0.0,
                id="zero reconstruction",
            ),
            pytest.param(
                np.array([[1, 2], [3, 4]]),
                np.array([[1, 1.5], [2.5, 3.5]]),
                97.5,
                id="partial reconstruction",
            ),
        ],
    )
    def test_calculate_power_returns_correct_percent_power(
        X: np.ndarray,  # noqa: N803
        x_hat: np.ndarray,
        expected_power: np.ndarray,
    ) -> None:
        assert calculate_power(X, x_hat) == pytest.approx(expected_power, abs=1e-2)

    @staticmethod
    @pytest.mark.parametrize(
        ("X", "x_hat", "padding_index", "expected_power"),
        [
            pytest.param(
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[1, 2, 3], [4, 5, 6]]),
                slice(1, 3),
                100.0,
                id="perfect reconstruction with padding",
            ),
            pytest.param(
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[0, 0, 0], [0, 0, 0]]),
                slice(1, 3),
                0.0,
                id="zero reconstruction with padding",
            ),
        ],
    )
    def test_calculate_power_handles_padding_correctly(
        X: np.ndarray,  # noqa: N803
        x_hat: np.ndarray,
        padding_index: slice,
        expected_power: float,
    ) -> None:
        assert calculate_power(X, x_hat, padding_index=padding_index) == pytest.approx(
            expected_power, abs=1e-2
        )


def test_calculate_loading_power(example_dataset: Dataset) -> None:
    expected_loadings = example_dataset.loadings
    loadings = calculate_loading_power(
        example_dataset.data,
        example_dataset.W,
        example_dataset.H,
    )
    assert np.allclose(loadings, expected_loadings)


class TestReconstruct:
    @staticmethod
    @pytest.mark.parametrize(
        "implementation",
        [
            pytest.param(reconstruct, id="reconstruct"),
            pytest.param(reconstruct_fast, id="reconstruct_fast"),
        ],
    )
    def test_reconstruct_computes_correct_reconstruction(
        example_dataset: Dataset, implementation: Callable
    ) -> None:
        reconstruction = implementation(example_dataset.W, example_dataset.H)
        assert np.allclose(reconstruction, example_dataset.parameters["x_hat"])

    @staticmethod
    @pytest.mark.parametrize(
        "implementation",
        [
            pytest.param(reconstruct, id="reconstruct"),
            pytest.param(reconstruct_fast, id="reconstruct_fast"),
        ],
    )
    def test_reconstruct_handles_preallocation(
        example_dataset: Dataset, implementation: Callable
    ) -> None:
        h_shifted = np.zeros(
            (
                example_dataset.parameters["sequence_length"],
                example_dataset.parameters["num_components"],
                example_dataset.H.shape[1],
            )
        )
        _ = implementation(example_dataset.W, example_dataset.H, h_shifted=h_shifted)
        if implementation is reconstruct_fast:
            assert np.sum(h_shifted) != 0, "h_shifted was not modified in-place"


@pytest.mark.parametrize(
    ("comparison", "expected"),
    [
        pytest.param("exact", 0.0, id="exact match"),
        pytest.param("off_by_one", 0.00334, id="off by one"),
        pytest.param("all_off_by_one", 1.0, id="all off by one"),
    ],
)
def test_calculate_rmse(
    example_dataset: Dataset, comparison: str, expected: float
) -> None:
    X = example_dataset.data.copy()  # noqa: N806
    if comparison == "exact":
        x_hat = X.copy()
    elif comparison == "off_by_one":
        x_hat = X.copy()
        x_hat[0, 0] += 1
    elif comparison == "all_off_by_one":
        x_hat = X.copy() + 1
    else:
        msg = "Invalid comparison type"
        raise ValueError(msg)
    assert rmse(X, x_hat) == pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize(
    ("n_components", "sequence_length", "hyperparameters", "expected_desc"),
    [
        pytest.param(
            5,
            10,
            {
                "lam": 0.05,
                "alpha_H": 0.2,
                "alpha_W": 0.001,
                "lam_H": 0.5,
                "lam_W": 0.0001,
            },
            f"n_components = 5, "
            f"sequence length = 10, "
            f"{HYPERPARAMETER_LABELS['lam']} = 5.000e-02, "
            f"{HYPERPARAMETER_LABELS['alpha_H']} = 0.200, "
            f"{HYPERPARAMETER_LABELS['alpha_W']} = 1.000e-03, "
            f"{HYPERPARAMETER_LABELS['lam_H']} = 0.500, "
            f"{HYPERPARAMETER_LABELS['lam_W']} = 1.000e-04",
            id="valid hyperparameters",
        ),
        pytest.param(
            3,
            15,
            {"lam": 0.9, "alpha_W": 0.00},
            f"n_components = 3, sequence length = 15, "
            f"{HYPERPARAMETER_LABELS['lam']} = 0.900",
            id="ignore zero hyperparameter",
        ),
        pytest.param(
            2,
            5,
            {"lam": 0.0},
            "n_components = 2, sequence length = 5",
            id="no hyperparameters",
        ),
    ],
)
def test_create_textbar(
    n_components: int,
    sequence_length: int,
    hyperparameters: dict,
    expected_desc: str,
) -> None:
    progress_bar = create_textbar(n_components, sequence_length, 100, **hyperparameters)
    assert progress_bar.desc == expected_desc
    progress_bar.close()


class TestPadData:
    """
    Test suite for the pad_data function in gseqNMF.support.
    """

    @pytest.mark.parametrize(
        ("X", "sequence_length", "expected_shape"),
        [
            pytest.param(np.zeros((5, 15)), 2, (5, 19), id="typical case"),
            pytest.param(np.ones((15, 5)), 3, (15, 11), id="tall case"),
        ],
    )
    def test_pad_data_returns_correct_shape(
        self,
        X: np.ndarray,  # noqa: N803
        sequence_length: int,
        expected_shape: tuple[int, int],
    ) -> None:
        padded_X = pad_data(X, sequence_length)  # noqa: N806
        assert padded_X.shape == expected_shape

    @pytest.mark.parametrize(
        ("X", "sequence_length", "zero_index"),
        [
            pytest.param(
                np.zeros((5, 15)), 2, [slice(0, 2), slice(17, 19)], id="typical case"
            ),
            pytest.param(
                np.ones((15, 5)), 3, [slice(0, 3), slice(8, 11)], id="tall case"
            ),
        ],
    )
    def test_pad_data_pads_with_zeros_correctly(
        self,
        X: np.ndarray,  # noqa: N803
        sequence_length: int,
        zero_index: list[slice, slice],
    ) -> None:
        padded_X = pad_data(X, sequence_length)  # noqa: N806
        assert np.array_equal(
            padded_X[:, zero_index[0]], np.zeros(padded_X[:, zero_index[0]].shape)
        )
        assert np.array_equal(
            padded_X[:, zero_index[1]], np.zeros(padded_X[:, zero_index[1]].shape)
        )


class TestRandomInitW:
    @staticmethod
    @pytest.mark.parametrize(
        ("X", "n_components", "sequence_length", "random_state", "expected_shape"),
        [
            pytest.param(
                np.zeros((10, 20)),
                5,
                3,
                None,
                (10, 5, 3),
                id="typical case no random state",
            ),
            pytest.param(
                np.ones((15, 25)),
                4,
                2,
                42,
                (15, 4, 2),
                id="typical case with random state",
            ),
            pytest.param(
                np.full((8, 12), 7), 3, 1, None, (8, 3, 1), id="single sequence length"
            ),
        ],
    )
    def test_random_init_W_returns_correct_shape_and_values(  # noqa: N802
        X: np.ndarray,  # noqa: N803
        n_components: int,
        sequence_length: int,
        random_state: int | None,
        expected_shape: tuple[int, int, int],
    ) -> None:
        W = random_init_W(X, n_components, sequence_length, random_state)  # noqa: N806
        assert W.shape == expected_shape
        assert (W >= 0).all()
        assert (X.max() >= W).all()

    @staticmethod
    def test_random_init_W_is_deterministic_with_random_state() -> None:  # noqa: N802
        X = np.ones((10, 20))  # noqa: N806
        W1 = random_init_W(X, 5, 3, random_state=42)  # noqa: N806
        W2 = random_init_W(X, 5, 3, random_state=42)  # noqa: N806
        assert np.array_equal(W1, W2)


class TestRandomInitH:
    @staticmethod
    @pytest.mark.parametrize(
        ("X", "n_components", "random_state", "expected_shape"),
        [
            pytest.param(
                np.zeros((10, 20)), 5, None, (5, 20), id="typical case no random state"
            ),
            pytest.param(
                np.ones((15, 25)), 4, 42, (4, 25), id="typical case with random state"
            ),
            pytest.param(np.full((8, 12), 7), 3, None, (3, 12), id="filled array"),
        ],
    )
    def test_random_init_H_returns_correct_shape_and_values(  # noqa: N802
        X: np.ndarray,  # noqa: N803
        n_components: int,
        random_state: int | None,
        expected_shape: tuple[int, int],
    ) -> None:
        H = random_init_H(X, n_components, random_state)  # noqa: N806
        assert H.shape == expected_shape
        assert (H >= 0).all()
        assert (X.max() >= H).all()

    @staticmethod
    def test_random_init_H_is_deterministic_with_random_state() -> None:  # noqa: N802
        X = np.ones((10, 20))  # noqa: N806
        H1 = random_init_H(X, 5, random_state=42)  # noqa: N806
        H2 = random_init_H(X, 5, random_state=42)  # noqa: N806
        assert np.array_equal(H1, H2)


class TestShiftFactors:
    @staticmethod
    @pytest.mark.parametrize(
        ("W", "H", "expected_W", "expected_H"),
        [
            pytest.param(
                np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),
                np.array([[1, 2], [3, 4]]),
                np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),
                np.array([[1, 2], [3, 4]]),
                id="typical case",
            ),
        ],
    )
    def test_shift_factors_correctly_shifts_and_centers(
        W: np.ndarray,  # noqa: N803
        H: np.ndarray,  # noqa: N803
        expected_W: np.ndarray,  # noqa: N803
        expected_H: np.ndarray,  # noqa: N803
    ) -> None:
        shifted_W, shifted_H = shift_factors(W, H)  # noqa: N806
        assert np.allclose(shifted_W, expected_W)
        assert np.allclose(shifted_H, expected_H)


class TestTransTensorConvolution:
    @staticmethod
    @pytest.mark.parametrize(
        ("X", "x_hat", "W", "sequence_length", "expected_wt_x", "expected_wt_x_hat"),
        [
            pytest.param(
                np.array([[1, 2], [3, 4]]),
                np.array([[0, 1], [1, 0]]),
                np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),
                2,
                np.array([[5, 5], [5, 5]]),
                np.array([[2, 0], [0, 2]]),
                id="typical case",
            )
        ],
    )
    def test_trans_tensor_convolution_computes_correct_outputs(
        X: np.ndarray,  # noqa: N803
        x_hat: np.ndarray,
        W: np.ndarray,  # noqa: N803
        sequence_length: int,
        expected_wt_x: np.ndarray,
        expected_wt_x_hat: np.ndarray,
    ) -> None:
        wt_x = np.zeros_like(expected_wt_x)
        wt_x_hat = np.zeros_like(expected_wt_x_hat)
        trans_tensor_convolution(X, x_hat, W, wt_x, wt_x_hat, sequence_length)
        assert np.allclose(wt_x, expected_wt_x)
        assert np.allclose(wt_x_hat, expected_wt_x_hat)
