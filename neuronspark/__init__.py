"""NeuronSpark: a spiking language model (1.27 B parameters) and its operators.

24 layers in a 3:1 ratio: 18 spiking layers (a membrane-potential selective state space with a bipolar
soma read-out) and 6 memory layers (pool attention with a lexical read-out, a spiking FFN of PLIF neurons and a
product-key memory), plus model-level lexical n-gram memory and induction match matrices.

Importing the package has no side effects: it sets no process-wide state, reads no environment variable and registers
nothing with other libraries.  The model classes are imported on first access.
"""
from __future__ import annotations

from neuronspark.models.config_neuronspark import NeuronSparkConfig

__version__ = "1.0.0"

__all__ = ["NeuronSparkConfig", "NeuronSparkModel", "NeuronSparkLMHeadModel", "__version__"]

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
    raise AttributeError(f"module 'neuronspark' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
