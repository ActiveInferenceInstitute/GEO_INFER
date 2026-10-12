"""
Base model class for agricultural analysis and prediction.
"""

import abc
import pickle
from pathlib import Path
from typing import Any

from geo_infer_ag.models.secure_serialization import (
    CONTEXT_MODEL_SAVE,
    sign_payload,
    verify_payload,
)


def write_signed_payload(path: str | Path, payload: bytes) -> None:
    """Write serialized model bytes as an authenticated GISP1 envelope.

    Shared save-side helper for ``AgricultureModel`` and the joblib-backed
    subclasses; envelopes are signed under the ``ag.model.save`` context
    (see :mod:`geo_infer_ag.models.secure_serialization`).
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        file_obj.write(sign_payload(payload, context=CONTEXT_MODEL_SAVE))


def read_verified_payload(path: str | Path) -> bytes:
    """Read a model file, verifying its envelope before deserialization.

    Trust boundary: unsigned, truncated, cross-context, or tampered files
    raise before the bytes reach ``pickle.loads`` or ``joblib.load``.
    """
    envelope = Path(path).read_bytes()
    return verify_payload(envelope, context=CONTEXT_MODEL_SAVE)


class AgricultureModel(abc.ABC):
    """
    Abstract base class for agricultural models.

    This class defines the interface for all agricultural models in the GEO-INFER-AG
    module. Subclasses should implement the core predictive functionality.

    Attributes:
        name: Name of the model
        version: Version string
        metadata: Dictionary of model metadata
        required_inputs: List of required input data sources
    """

    def __init__(
        self, name: str, version: str = "0.1.0", config: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize the agricultural model.

        Args:
            name: Unique name for the model
            version: Version string
            config: Optional configuration parameters
        """
        self.name = name
        self.version = version
        self.config = config or {}
        self.metadata: dict[str, Any] = {
            "name": name,
            "version": version,
            "type": self.__class__.__name__,
        }
        self.required_inputs: list[str] = []

    @abc.abstractmethod
    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate predictions using the model.

        Args:
            data: Dictionary of input data

        Returns:
            Dictionary containing prediction results

        Raises:
            RuntimeError: Raised if a subclass does not implement predict.
        """
        raise RuntimeError("Subclasses must implement predict()")

    def validate_inputs(self, data: dict[str, Any]) -> bool:
        """
        Validate that all required inputs are present.

        Args:
            data: Dictionary of input data

        Returns:
            True if all required inputs are present

        Raises:
            ValueError: If required inputs are missing
        """
        for input_name in self.required_inputs:
            if input_name not in data:
                raise ValueError(f"Required input '{input_name}' is missing")
        return True

    @property
    def info(self) -> dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary of model information
        """
        return {
            "name": self.name,
            "version": self.version,
            "type": self.__class__.__name__,
            "required_inputs": self.required_inputs,
            "config": self.config,
        }

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model

        Notes:
            The default implementation serializes the model instance with
            pickle and wraps it in an authenticated GISP1 envelope, which
            ``load`` verifies before deserializing. Subclasses may override
            this for framework-specific formats.
        """
        write_signed_payload(path, pickle.dumps(self))

    @classmethod
    def load(cls, path: str) -> "AgricultureModel":
        """
        Load a model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded model instance

        Raises:
            geo_infer_ag.models.secure_serialization.PayloadSecurityError:
                If the file is not a valid GISP1 envelope (unsigned,
                tampered, or signed under a different key/context).

        Notes:
            The default implementation verifies the GISP1 envelope created
            by ``save`` before unpickling.
        """
        model = pickle.loads(read_verified_payload(path))
        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, expected {cls.__name__}"
            )
        return model
