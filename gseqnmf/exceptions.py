__all__ = [
    "SeqNMFInitializationError",
]


class SeqNMFInitializationError(Exception):
    """Exception raised for errors in the initialization of the NMF model."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
