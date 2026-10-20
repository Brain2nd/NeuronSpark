"""Layer modules of the NeuronSpark model.

Building blocks (this package): ``Linear`` / ``ShortConvWeight`` (``linear``), ``RMSNorm`` / ``frozen_rms_norm``
(``norms``), the RoPE tables and reference (``rope``).  Layers (imported on first access): ``DendriticNeuronLayer`` /
``DendriticNeurons`` (``dendritic_neuron_layer``), ``HippocampalLayer`` (``hippocampal_layer``), ``PLIFNeuron``
(``plif``), ``SpikingFFN`` (``spiking_ffn``), ``LexicalMemory`` (``lexical_memory``), ``ReadoutBalance``
(``readout_balance``), the induction match matrices (``induction``) and the copy-detector statistics
(``copy_detector``).

``mechanism_init_seed(base, layer_idx)`` is the seed of the forked RNG stream from which a mechanism draws its launch
values (the run seed is ``torch.initial_seed()``, read without consuming random numbers).
"""
from __future__ import annotations

import torch

from neuronspark.modules.linear import Linear, ShortConvWeight
from neuronspark.modules.norms import RMSNorm, frozen_rms_norm, rms_norm_ref
from neuronspark.modules.rope import apply_rope_interleaved_ref, precompute_rope_freqs, rope_head_band_tables

__all__ = [
    "Linear", "ShortConvWeight", "RMSNorm", "frozen_rms_norm", "rms_norm_ref", "precompute_rope_freqs",
    "rope_head_band_tables", "apply_rope_interleaved_ref", "mechanism_init_seed",
    "DendriticNeuronLayer", "DendriticNeurons", "HippocampalLayer", "PLIFNeuron", "SpikingFFN", "LexicalMemory",
    "ReadoutBalance",
    "induction_match_matrices", "induction_session_match", "copy_stats_whole_row", "copy_stats_session", "copy_pi",
]


def mechanism_init_seed(base: int, layer_idx: int) -> int:
    """``(torch.initial_seed() * 1_000_003 + base + layer_idx) mod (2^63 - 1)``: different run seeds give different
    launch values, two builds under the same seed give the same values, and the global stream is untouched because
    the draw happens on a forked stream."""
    return (int(torch.initial_seed()) * 1_000_003 + int(base) + int(layer_idx)) % (2 ** 63 - 1)


_LAZY = {
    "DendriticNeuronLayer": "neuronspark.modules.dendritic_neuron_layer",
    "DendriticNeurons": "neuronspark.modules.dendritic_neuron_layer",
    "HippocampalLayer": "neuronspark.modules.hippocampal_layer",
    "PLIFNeuron": "neuronspark.modules.plif",
    "SpikingFFN": "neuronspark.modules.spiking_ffn",
    "LexicalMemory": "neuronspark.modules.lexical_memory",
    "ReadoutBalance": "neuronspark.modules.readout_balance",
    "induction_match_matrices": "neuronspark.modules.induction",
    "induction_session_match": "neuronspark.modules.induction",
    "copy_stats_whole_row": "neuronspark.modules.copy_detector",
    "copy_stats_session": "neuronspark.modules.copy_detector",
    "copy_pi": "neuronspark.modules.copy_detector",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'neuronspark.modules' has no attribute {name!r}")
