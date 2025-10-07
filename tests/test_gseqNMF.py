# noqa: N999
import numpy as np
import pytest

from gseqnmf.exceptions import SeqNMFInitializationError
from gseqnmf.gseqnmf import GseqNMF
from gseqnmf.support import pad_data
from tests.conftest import MEANING_OF_LIFE, Dataset


class TestGseqNMF:
    @pytest.fixture(autouse=True)
    def _setup(self, example_dataset: Dataset) -> None:
        self.test_dataset = example_dataset
        self.model = GseqNMF(
            n_components=self.test_dataset.parameters["num_components"],
            sequence_length=self.test_dataset.parameters["sequence_length"],
            max_iter=self.test_dataset.parameters["max_iter"],
            init="random",
            random_state=MEANING_OF_LIFE,
        )

    def _reset_model(self) -> None:
        self.model = GseqNMF(
            n_components=self.test_dataset.parameters["num_sequences"],
            sequence_length=self.test_dataset.parameters["sequence_length"],
            max_iter=self.test_dataset.parameters["max_iter"],
            init="random",
            random_state=MEANING_OF_LIFE,
        )

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            pytest.param(
                ["random"],
                ("MOCK_PADDED_X", "MOCKED_W_RANDOM", "MOCKED_H_RANDOM"),
                id="random",
            ),
            pytest.param(
                ["nndsvd"],
                ("MOCK_PADDED_X", "MOCKED_W_NNDSVD", "MOCKED_H_NNDSVD"),
                id="nndsvd",
            ),
        ],
    )
    def test_initialization_methods(
        self, mocker: object, method: str, expected: tuple[str, str, str]
    ) -> None:
        # MOCKING
        initialize_mock = mocker.patch("gseqnmf.gseqnmf.pad_data")
        initialize_mock.return_value = expected[0]
        random_init_W_mock = mocker.patch("gseqnmf.gseqnmf.random_init_W")  # noqa: N806
        random_init_W_mock.return_value = expected[1]
        random_init_H_mock = mocker.patch("gseqnmf.gseqnmf.random_init_H")  # noqa: N806
        random_init_H_mock.return_value = expected[2]
        nndsvd_nonneg_mock = mocker.patch("gseqnmf.gseqnmf.nndsvd_init")
        nndsvd_nonneg_mock.return_value = expected[1:]

        # noinspection PyTupleAssignmentBalance
        padded_X, W_init, H_init, _ = GseqNMF._initialize(  # noqa: N806, SLF001
            X=self.test_dataset.data.copy(),
            n_components=self.test_dataset.parameters["num_components"],
            sequence_length=self.test_dataset.parameters["sequence_length"],
            init=method[0],
            W_init="MOCKED_W_EXACT" if method[0] == "exact" else None,
            H_init="MOCKED_H_EXACT" if method[0] == "exact" else None,
            random_state=MEANING_OF_LIFE,
        )
        assert (padded_X, W_init, H_init) == expected

    def test_initialize_exact(self) -> None:
        # noinspection PyTupleAssignmentBalance
        _, W_init, H_init, _ = GseqNMF._initialize(  # noqa: N806, SLF001
            X=self.test_dataset.data.copy(),
            n_components=self.test_dataset.parameters["num_components"],
            sequence_length=self.test_dataset.parameters["sequence_length"],
            init="exact",
            W_init=self.test_dataset.W.copy(),
            H_init=pad_data(
                self.test_dataset.H.copy(),
                self.test_dataset.parameters["sequence_length"],
            ),
            random_state=MEANING_OF_LIFE,
        )
        np.testing.assert_equal(W_init, self.test_dataset.W)
        np.testing.assert_equal(
            H_init,
            pad_data(
                self.test_dataset.H.copy(),
                self.test_dataset.parameters["sequence_length"],
            ),
        )

    def test_initialize_exact_failures(self) -> None:
        with pytest.raises(SeqNMFInitializationError):
            # noinspection PyTupleAssignmentBalance
            _ = GseqNMF._initialize(  # noqa: SLF001
                X=self.test_dataset.data.copy(),
                n_components=self.test_dataset.parameters["num_components"],
                sequence_length=self.test_dataset.parameters["sequence_length"],
                init="exact",
                W_init=np.zeros((5, 5, 5)),
                H_init=self.test_dataset.H.copy(),
                random_state=MEANING_OF_LIFE,
            )
        with pytest.raises(SeqNMFInitializationError):
            # noinspection PyTupleAssignmentBalance
            _ = GseqNMF._initialize(  # noqa: SLF001
                X=self.test_dataset.data.copy(),
                n_components=self.test_dataset.parameters["num_components"],
                sequence_length=self.test_dataset.parameters["sequence_length"],
                init="exact",
                W_init=self.test_dataset.W.copy(),
                H_init=np.zeros((5, 5)),
                random_state=MEANING_OF_LIFE,
            )
        with pytest.raises(SeqNMFInitializationError):
            # noinspection PyTupleAssignmentBalance
            _ = GseqNMF._initialize(  # noqa: SLF001
                X=self.test_dataset.data.copy(),
                n_components=self.test_dataset.parameters["num_components"],
                sequence_length=self.test_dataset.parameters["sequence_length"],
                init="exact",
                W_init=None,
                H_init=None,
                random_state=MEANING_OF_LIFE,
            )

    def test_initialize_invalid_method(self) -> None:
        with pytest.raises(
            SeqNMFInitializationError,
        ):
            GseqNMF._initialize(  # noqa: SLF001
                X=self.test_dataset.data.copy(),
                n_components=self.test_dataset.parameters["num_components"],
                sequence_length=self.test_dataset.parameters["sequence_length"],
                init="invalid",
                random_state=MEANING_OF_LIFE,
            )

    def test_get_params(self) -> None:
        params = self.model.get_params()
        expected_params = {
            "n_components": self.test_dataset.parameters["num_components"],
            "sequence_length": self.test_dataset.parameters["sequence_length"],
            "max_iter": self.test_dataset.parameters["max_iter"],
        }
        for param in expected_params:
            assert params[param] == expected_params[param]

    @pytest.mark.parametrize(
        ("params", "expected_values"),
        [
            pytest.param(
                {"n_components": 5, "sequence_length": 100},
                {"n_components": 5, "sequence_length": 100},
                id="n_components and sequence_length",
            ),
            pytest.param(
                {"tol": 1e-5, "max_iter": 200},
                {"tol": 1e-5, "max_iter": 200},
                id="tol and max_iter",
            ),
            pytest.param(
                {"alpha_W": 0.1, "alpha_H": 0.2},
                {"alpha_W": 0.1, "alpha_H": 0.2},
                id="alpha_W and alpha_H",
            ),
        ],
    )
    def test_set_params(self, params: dict, expected_values: dict) -> None:
        """
        ////////////////////////////////////////////////////////////////////////////////
        // The try-except block ensures the model is reset after shenanigans.
        ////////////////////////////////////////////////////////////////////////////////
        """
        self.model.set_params(**params)
        for key, value in expected_values.items():
            try:
                assert getattr(self.model, key) == value
            except Exception as exc:
                self._reset_model()
                if isinstance(exc, AssertionError):
                    msg = f"Failed to set params: {key}: {value}"
                    raise AssertionError(msg) from exc  # noqa: TRY004
                raise

    def test_params_ignores_invalid_parameters(self) -> None:
        with pytest.raises(AttributeError):
            self.model.set_params(invalid_param="Don't Panic!")

    def test_fitting(self) -> None:
        self.model.fit(self.test_dataset.data.T.copy())
        assert self.model.power_ >= self.test_dataset.power
        assert self.model.cost_[-1] <= self.test_dataset.cost[-1]
