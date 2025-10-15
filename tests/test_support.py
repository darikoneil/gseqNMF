from collections.abc import Callable

import numpy as np

import gseqnmf.support
from gseqnmf.exceptions import (
    GPUNotAvailableError,
    GPUNotSupportedError,
    InvalidGPUDeviceError,
)

try:
    import cupy as cp
except ImportError:
    cp = np
from types import ModuleType

import pytest

from gseqnmf.support import (
    HYPERPARAMETER_LABELS,
    assess_vram,
    calculate_loading_power,
    calculate_power,
    calculate_sequenciness,
    create_textbar,
    nndsvd_init,
    pad_data,
    random_init_H,
    random_init_W,
    reconstruct,
    reconstruct_fast,
    rmse,
    set_device,
    shift_factors,
    trans_tensor_convolution,
)
from gseqnmf.validation import NDArrayLike
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
        X: NDArrayLike,  # noqa: N803
        x_hat: NDArrayLike,
        expected_power: NDArrayLike,
        xp_imp: ModuleType,
    ) -> None:
        xp_imp.testing.assert_allclose(
            calculate_power(
                xp_imp.asarray(X),
                xp_imp.asarray(x_hat),
                epsilon=xp_imp.finfo(float).eps,
                xp=xp_imp,
            ),
            xp_imp.asarray(expected_power),
        )

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
        xp_imp: ModuleType,
    ) -> None:
        xp_imp.testing.assert_allclose(
            calculate_power(
                xp_imp.asarray(X),
                xp_imp.asarray(x_hat),
                epsilon=xp_imp.finfo(float).eps,
                padding_index=padding_index,
                xp=xp_imp,
            ),
            xp_imp.asarray(expected_power),
        )


def test_calculate_loading_power(example_dataset: Dataset, xp_imp: ModuleType) -> None:
    expected_loadings = xp_imp.asarray(example_dataset.loadings).reshape((-1,))
    loadings = calculate_loading_power(
        xp_imp.asarray(example_dataset.data),
        xp_imp.asarray(example_dataset.W),
        xp_imp.asarray(example_dataset.H),
        epsilon=xp_imp.finfo(float).eps,
        xp=xp_imp,
    )
    xp_imp.testing.assert_allclose(
        loadings,
        expected_loadings,
    )


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
        example_dataset: Dataset,
        implementation: Callable,
        xp_imp: ModuleType,
    ) -> None:
        reconstruction = implementation(
            xp_imp.asarray(example_dataset.W.copy()),
            xp_imp.asarray(example_dataset.H.copy()),
            xp=xp_imp,
        )
        xp_imp.testing.assert_allclose(
            reconstruction, xp_imp.asarray(example_dataset.parameters["x_hat"])
        )

    @staticmethod
    @pytest.mark.parametrize(
        "implementation",
        [
            pytest.param(reconstruct, id="reconstruct"),
            pytest.param(reconstruct_fast, id="reconstruct_fast"),
        ],
    )
    def test_reconstruct_handles_preallocation(
        example_dataset: Dataset,
        implementation: Callable,
        xp_imp: ModuleType,
    ) -> None:
        h_shifted = xp_imp.zeros(
            (
                example_dataset.parameters["sequence_length"],
                example_dataset.parameters["num_components"],
                example_dataset.H.shape[1],
            )
        )
        _ = implementation(
            xp_imp.asarray(example_dataset.W),
            xp_imp.asarray(example_dataset.H),
            h_shifted=h_shifted,
            xp=xp_imp,
        )
        if implementation is reconstruct_fast:
            assert xp_imp.sum(h_shifted) != 0, "h_shifted was not modified in-place"


@pytest.mark.parametrize(
    ("comparison", "expected"),
    [
        pytest.param("exact", 0.0, id="exact match"),
        pytest.param("off_by_one", 0.00334, id="off by one"),
        pytest.param("all_off_by_one", 1.0, id="all off by one"),
    ],
)
def test_calculate_rmse(
    example_dataset: Dataset, comparison: str, expected: float, xp_imp: ModuleType
) -> None:
    X = xp_imp.asarray(example_dataset.data.copy())  # noqa: N806
    if comparison == "exact":
        x_hat = xp_imp.asarray(X.copy())
    elif comparison == "off_by_one":
        x_hat = xp_imp.asarray(X.copy())
        x_hat[0, 0] += 1
    elif comparison == "all_off_by_one":
        x_hat = xp_imp.asarray(X.copy()) + 1
    else:
        msg = "Invalid comparison type"
        raise ValueError(msg)
    xp_imp.testing.assert_allclose(
        rmse(X, x_hat, xp=xp_imp), xp_imp.asarray(expected), atol=5e-3
    )


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
        xp_imp: ModuleType,
    ) -> None:
        padded_X = pad_data(xp_imp.asarray(X), sequence_length, xp=xp_imp)  # noqa: N806
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
        xp_imp: ModuleType,
    ) -> None:
        padded_X = pad_data(xp_imp.asarray(X), sequence_length, xp=xp_imp)  # noqa: N806
        assert xp_imp.array_equal(
            padded_X[:, zero_index[0]], xp_imp.zeros(padded_X[:, zero_index[0]].shape)
        )
        assert xp_imp.array_equal(
            padded_X[:, zero_index[1]], xp_imp.zeros(padded_X[:, zero_index[1]].shape)
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
        xp_imp: ModuleType,
    ) -> None:
        W = random_init_W(  # noqa: N806
            xp_imp.asarray(X), n_components, sequence_length, random_state, xp=xp_imp
        )
        assert W.shape == expected_shape
        assert (W >= 0).all()
        assert (X.max() >= W).all()

    @staticmethod
    def test_random_init_W_is_deterministic_with_random_state(  # noqa: N802
        xp_imp: ModuleType,
    ) -> None:
        X = xp_imp.ones((10, 20))  # noqa: N806
        W1 = random_init_W(  # noqa: N806
            xp_imp.asarray(X), 5, 3, random_state=42, xp=xp_imp
        )
        W2 = random_init_W(  # noqa: N806
            xp_imp.asarray(X), 5, 3, random_state=42, xp=xp_imp
        )
        assert xp_imp.array_equal(W1, W2)


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
        xp_imp: ModuleType,
    ) -> None:
        H = random_init_H(  # noqa: N806
            xp_imp.asarray(X),
            n_components,
            random_state,
            xp=xp_imp,
        )
        assert H.shape == expected_shape
        assert (H >= 0).all()
        assert (X.max() >= H).all()

    @staticmethod
    def test_random_init_H_is_deterministic_with_random_state(  # noqa: N802
        xp_imp: ModuleType,
    ) -> None:
        X = xp_imp.ones((10, 20))  # noqa: N806
        H1 = random_init_H(  # noqa: N806
            xp_imp.asarray(X),
            5,
            random_state=42,
            xp=xp_imp,
        )
        H2 = random_init_H(  # noqa: N806
            xp_imp.asarray(X), 5, random_state=42, xp=xp_imp
        )
        assert xp_imp.array_equal(H1, H2)


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
        xp_imp: ModuleType,
    ) -> None:
        shifted_W, shifted_H = shift_factors(  # noqa: N806
            xp_imp.asarray(W), xp_imp.asarray(H), xp=xp_imp
        )
        xp_imp.testing.assert_allclose(shifted_W, xp_imp.asarray(expected_W))
        xp_imp.testing.assert_allclose(shifted_H, xp_imp.asarray(expected_H))


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
        xp_imp: ModuleType,
    ) -> None:
        wt_x = xp_imp.zeros_like(expected_wt_x)
        wt_x_hat = xp_imp.zeros_like(expected_wt_x_hat)
        trans_tensor_convolution(
            xp_imp.asarray(X),
            xp_imp.asarray(x_hat),
            xp_imp.asarray(W),
            xp_imp.asarray(wt_x),
            xp_imp.asarray(wt_x_hat),
            sequence_length,
        )
        xp_imp.testing.assert_allclose(wt_x, xp_imp.asarray(expected_wt_x))
        xp_imp.testing.assert_allclose(wt_x_hat, xp_imp.asarray(expected_wt_x_hat))


class TestNNSVDInit:
    @staticmethod
    def test_not_implemented() -> None:
        with pytest.raises(NotImplementedError):
            _ = nndsvd_init(np.zeros((10, 20)), 5, 100)


class TestGPUUtilities:
    @staticmethod
    def test_set_device(mocker: object) -> None:
        class MockDevice:
            def __init__(self, device_id: int) -> None:
                self.device_id = device_id

            def use(self, device_id: int) -> None:
                self.device_id = device_id

            def __enter__(self) -> int:
                return self.device_id

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
                pass

        mock_set_device = mocker.patch("gseqnmf.support.Device")
        mocker.patch.object(MockDevice, "use")
        mock_set_device.return_value = MockDevice(0)
        set_device(0)
        mock_set_device.assert_called_once_with(0)

    @staticmethod
    def test_set_device_invalid(mocker: object) -> None:
        class MockDevice:
            def __init__(self, device_id: int) -> None:
                self.device_id = device_id

            def use(self, device_id: int) -> None:
                self.device_id = device_id

            def __enter__(self) -> int:
                return self.device_id

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
                pass

        mock_set_device = mocker.patch("gseqnmf.support.Device")
        mocker.patch.object(MockDevice, "use")
        mock_set_device.return_value = MockDevice(0)
        with pytest.raises(InvalidGPUDeviceError):
            set_device(-1)

    @staticmethod
    def test_set_device_no_cupy() -> None:
        gseqnmf.support.CUPY_INSTALLED = False
        with pytest.raises(GPUNotSupportedError):
            set_device(0)
        gseqnmf.support.CUPY_INSTALLED = True

    @staticmethod
    def test_no_device_available(mocker: object) -> None:
        mock_set_device = mocker.patch("gseqnmf.support.device_available")
        mock_set_device.return_value = False
        gseqnmf.support.CUPY_INSTALLED = True
        with pytest.raises(GPUNotAvailableError):
            set_device(0)
        gseqnmf.support.CUPY_INSTALLED = True

    @staticmethod
    def test_assess_vram(mocker: object) -> None:
        class MockDevice:
            def __init__(self, device_id: int) -> None:
                self.device_id = device_id

            # noinspection PyMethodMayBeStatic
            def mem_info(self) -> tuple[float, float]:
                return 4 * 1024**3, 8 * 1024**3

            def __enter__(self) -> int:
                return self.device_id

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
                pass

        mock_set_device = mocker.patch("gseqnmf.support.Device")
        mock_set_device.return_value = MockDevice(0)
        vram = assess_vram(0)
        np.testing.assert_allclose(np.asarray(vram), np.array([4.0, 8.0]))

    @staticmethod
    def test_assess_vram_no_cupy() -> None:
        gseqnmf.support.CUPY_INSTALLED = False
        with pytest.raises(GPUNotSupportedError):
            _ = assess_vram(0)
        gseqnmf.support.CUPY_INSTALLED = True

    @staticmethod
    def test_assess_vram_no_device(mocker: object) -> None:
        gseqnmf.support.CUPY_INSTALLED = True
        mocker_patch = mocker.patch("gseqnmf.support.device_available")
        mocker_patch.return_value = False
        with pytest.raises(GPUNotAvailableError):
            _ = assess_vram(0)
        gseqnmf.support.CUPY_INSTALLED = True

    @staticmethod
    def test_assess_vram_invalid_device(mocker: object) -> None:
        gseqnmf.support.CUPY_INSTALLED = True
        mocker_patch = mocker.patch("gseqnmf.support.is_valid_device")
        mocker_patch.return_value = False
        with pytest.raises(InvalidGPUDeviceError):
            _ = assess_vram(0)
        gseqnmf.support.CUPY_INSTALLED = True


class TestSequenciness:
    @staticmethod
    def test_not_implemented_sequenciness() -> None:
        with pytest.raises(NotImplementedError):
            calculate_sequenciness()
