"""NeuronSpark: a spiking language model (1.27 B parameters) and its operators.

Dendritic neuron layers and hippocampal layers in a fixed 3:1 pattern.  A dendritic neuron layer is a population of
somata with dendritic compartments (membrane potentials with an input-dependent leak, released through a bipolar soft
threshold); a hippocampal layer adds an episodic slot memory written by PLIF write-gate neurons, a copy pathway, a
spiking FFN of PLIF neurons and a product-key associative memory.  The model level holds a lexical n-gram memory and
induction match matrices.

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
