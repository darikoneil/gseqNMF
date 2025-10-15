import pytest

import gseqnmf.validation  # NOTE: to reset CUPY_INSTALLED
from gseqnmf.exceptions import (
    GPUNotAvailableError,
    GPUNotSupportedError,
    SeqNMFInitializationError,
)
from gseqnmf.gseqnmf import GseqNMF
from gseqnmf.validation import (
    INIT_METHOD,
    RECON_METHOD,
    W_UPDATE_METHOD,
    cuda_available,
    device_available,
    is_valid_device,
)


class TestInitMethod:
    @staticmethod
    @pytest.mark.parametrize(
        ("input_value", "expected_method"),
        [
            pytest.param("random", INIT_METHOD.RANDOM, id="valid_random"),
            pytest.param("exact", INIT_METHOD.EXACT, id="valid_exact"),
            pytest.param("nndsvd", INIT_METHOD.NNDSVD, id="valid_nndsvd"),
            pytest.param(None, INIT_METHOD.RANDOM, id="default_random"),
        ],
    )
    def test_init_method_parse_returns_correct_method(
        input_value: str | None, expected_method: INIT_METHOD
    ) -> None:
        assert INIT_METHOD.parse(input_value) == expected_method

    @staticmethod
    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param("invalid", id="invalid_string"),
            pytest.param(123, id="invalid_type"),
        ],
    )
    def test_init_method_parse_raises_error_for_invalid_method(
        invalid_value: str | int,
    ) -> None:
        with pytest.raises(SeqNMFInitializationError):
            INIT_METHOD.parse(invalid_value)

    @staticmethod
    def test_init_method_options_returns_all_methods() -> None:
        assert set(INIT_METHOD.options()) == {"random", "exact", "nndsvd"}

    @staticmethod
    def test_init_method_parse_short_circuit() -> None:
        method = INIT_METHOD.RANDOM
        assert INIT_METHOD.parse(method) is method


class TestReconMethod:
    @staticmethod
    @pytest.mark.parametrize(
        ("input_value", "expected_method"),
        [
            pytest.param("normal", RECON_METHOD.NORMAL, id="valid_normal"),
            pytest.param("fast", RECON_METHOD.FAST, id="valid_fast"),
            pytest.param(None, RECON_METHOD.FAST, id="default_fast"),
        ],
    )
    def test_recon_method_parse_returns_correct_method(
        input_value: str | None, expected_method: RECON_METHOD
    ) -> None:
        assert RECON_METHOD.parse(input_value) == expected_method

    @staticmethod
    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param("invalid", id="invalid_string"),
            pytest.param(123, id="invalid_type"),
        ],
    )
    def test_recon_method_parse_raises_error_for_invalid_method(
        invalid_value: str | int,
    ) -> None:
        with pytest.raises(SeqNMFInitializationError):
            RECON_METHOD.parse(invalid_value)

    @staticmethod
    def test_recon_method_options_returns_all_methods() -> None:
        assert set(RECON_METHOD.options()) == {"normal", "fast"}

    @staticmethod
    def test_recon_method_parse_short_circuit() -> None:
        method = RECON_METHOD.NORMAL
        assert RECON_METHOD.parse(method) is method


class TestWUpdateMethod:
    @staticmethod
    @pytest.mark.parametrize(
        ("input_value", "expected_method"),
        [
            pytest.param("fixed", W_UPDATE_METHOD.FIXED, id="valid_fixed"),
            pytest.param("partial", W_UPDATE_METHOD.PARTIAL, id="valid_partial"),
            pytest.param("full", W_UPDATE_METHOD.FULL, id="valid_full"),
            pytest.param(None, W_UPDATE_METHOD.FULL, id="default_full"),
        ],
    )
    def test_w_update_method_parse_returns_correct_method(
        input_value: str | None, expected_method: W_UPDATE_METHOD
    ) -> None:
        assert W_UPDATE_METHOD.parse(input_value) == expected_method

    @staticmethod
    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param("invalid", id="invalid_string"),
            pytest.param(123, id="invalid_type"),
        ],
    )
    def test_w_update_method_parse_raises_error_for_invalid_method(
        invalid_value: str | int,
    ) -> None:
        with pytest.raises(SeqNMFInitializationError):
            W_UPDATE_METHOD.parse(invalid_value)

    @staticmethod
    def test_w_update_method_options_returns_all_methods() -> None:
        assert set(W_UPDATE_METHOD.options()) == {"fixed", "partial", "full"}

    @staticmethod
    def test_w_update_method_parse_short_circuit() -> None:
        method = W_UPDATE_METHOD.FIXED
        assert W_UPDATE_METHOD.parse(method) is method


class TestGPUValidation:
    @staticmethod
    def test_gpu_no_device(mocker: object) -> None:
        gseqnmf.validation.CUPY_INSTALLED = True
        mock_cuda = mocker.patch("gseqnmf.validation.cuda_is_available")
        mock_cuda.return_value = True
        mock_device_count = mocker.patch("gseqnmf.validation.getDeviceCount")
        mock_device_count.return_value = False
        assert cuda_available() is True
        assert device_available() is False
        gseqnmf.validation.CUPY_INSTALLED = True

    @staticmethod
    def test_gpu_with_device(mocker: object) -> None:
        gseqnmf.validation.CUPY_INSTALLED = True
        mock_cuda = mocker.patch("gseqnmf.validation.cuda_is_available")
        mock_cuda.return_value = True
        mock_device_count = mocker.patch("gseqnmf.validation.getDeviceCount")
        mock_device_count.return_value = True
        assert cuda_available() is True
        assert device_available() is True
        gseqnmf.validation.CUPY_INSTALLED = True

    @staticmethod
    def test_gpu_no_cupy(mocker: object) -> None:
        gseqnmf.validation.CUPY_INSTALLED = False
        mock_cuda = mocker.patch("gseqnmf.validation.cuda_is_available")
        mock_cuda.return_value = False
        assert cuda_available() is False
        assert device_available() is False
        gseqnmf.validation.CUPY_INSTALLED = True

    @staticmethod
    def test_gpu_with_cupy(mocker: object) -> None:
        gseqnmf.validation.CUPY_INSTALLED = True
        mock_cuda = mocker.patch("gseqnmf.validation.cuda_is_available")
        mock_cuda.return_value = True
        mock_device_count = mocker.patch("gseqnmf.validation.getDeviceCount")
        mock_device_count.return_value = True
        assert cuda_available() is True
        assert device_available() is True
        gseqnmf.validation.CUPY_INSTALLED = True

    @staticmethod
    def test_gpu_not_supported_error() -> None:
        gseqnmf.gseqnmf.CUPY_INSTALLED = False
        with pytest.raises(GPUNotSupportedError):
            GseqNMF(n_components=10, sequence_length=100, use_gpu=True)
        gseqnmf.gseqnmf.CUPY_INSTALLED = True

    @staticmethod
    def test_gpu_not_available_error(mocker: object) -> None:
        gseqnmf.gseqnmf.CUPY_INSTALLED = True
        mock_available = mocker.patch("gseqnmf.gseqnmf.device_available")
        mock_available.return_value = False
        with pytest.raises(GPUNotAvailableError):
            GseqNMF(n_components=10, sequence_length=100, use_gpu=True)
        gseqnmf.gseqnmf.CUPY_INSTALLED = True

    @staticmethod
    def test_gpu_valid_device(mocker: object) -> None:
        gseqnmf.validation.CUPY_INSTALLED = True
        mock_device_count = mocker.patch("gseqnmf.validation.getDeviceCount")
        mock_device_count.return_value = 2
        assert gseqnmf.validation.is_valid_device(0) is True
        assert gseqnmf.validation.is_valid_device(1) is True
        gseqnmf.validation.CUPY_INSTALLED = True

    @staticmethod
    def test_gpu_invalid_device(mocker: object) -> None:
        gseqnmf.validation.CUPY_INSTALLED = True
        mock_device_count = mocker.patch("gseqnmf.validation.getDeviceCount")
        mock_device_count.return_value = 2
        assert gseqnmf.validation.is_valid_device(2) is False
        assert gseqnmf.validation.is_valid_device(-1) is False
        gseqnmf.validation.CUPY_INSTALLED = True

    @staticmethod
    def test_gpu_valid_device_no_cupy() -> None:
        gseqnmf.validation.CUPY_INSTALLED = False
        assert is_valid_device(0) is False
        gseqnmf.validation.CUPY_INSTALLED = True
