import pytest

from gseqnmf.exceptions import SeqNMFInitializationError
from gseqnmf.validation import INIT_METHOD, RECON_METHOD


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
