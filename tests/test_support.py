import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np
import pytest

from gseqnmf.support import pad_data, rmse
from tests.conftest import Dataset


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


@pytest.mark.parametrize(
    ("comparison", "expected"),
    [
        pytest.param("exact", 0.0, id="exact match"),
        pytest.param("off_by_one", 0.00334, id="off by one"),
        pytest.param("all_off_by_one", 1.0, id="all off by one"),
    ],
)
def test_rmse(test_dataset: Dataset, comparison: str, expected: float) -> None:
    X = test_dataset.data.copy()  # noqa: N806
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
