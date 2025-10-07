__all__ = [
    "GPUNotAvailableError",
    "GPUNotSupportedError",
    "SeqNMFInitializationError",
]


class SeqNMFInitializationError(Exception):
    """Exception raised for errors in the initialization of the NMF model."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class GPUNotAvailableError(Exception):
    """Exception raised when a GPU is requested but not available."""

    def __init__(self) -> None:
        self.message = (
            "GPU-acceleration requested but unable to identify suitable device."
        )
        super().__init__(self.message)


class GPUNotSupportedError(ImportError):
    """Exception raised when a GPU is requested but not supported."""

    def __init__(self) -> None:
        self.message = (
            "GPU-acceleration requested but not  supported by the "
            "current configuration. Please ensure that CuPY is"
            "installed"
        )
        super().__init__(self.message)
