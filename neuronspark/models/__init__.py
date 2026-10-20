"""Model definitions: :class:`NeuronSparkConfig` (``config_neuronspark``) and the backbone / LM-head models
(``neuronspark_lm``, imported on first access)."""
from __future__ import annotations

from neuronspark.models.config_neuronspark import (
    ARCHITECTURES,
    MODEL_TYPE,
    SPIKING_STATE_DTYPES,
    NeuronSparkConfig,
)

__all__ = [
    "NeuronSparkConfig",
    "NeuronSparkModel",
    "NeuronSparkLMHeadModel",
    "MODEL_TYPE",
    "ARCHITECTURES",
    "SPIKING_STATE_DTYPES",
]

_LAZY = {
    "NeuronSparkModel": "neuronspark.models.neuronspark_lm",
    "NeuronSparkLMHeadModel": "neuronspark.models.neuronspark_lm",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'neuronspark.models' has no attribute {name!r}")
