class FranforecastError(Exception):
    """Base class for errors surfaced to the user."""


class DataError(FranforecastError):
    """Input data cannot be turned into a usable panel."""


class ModelError(FranforecastError):
    """A forecaster could not be fit or could not produce a valid forecast."""
