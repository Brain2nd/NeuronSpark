"""Utilities: inference cache (``cache``), numerics regimes (``precision``), the per-operator match rule and test
helpers (``testing``), checkpoint loading (``hf``; ``hub``: where a checkpoint's files are, on the Hugging Face Hub or
locally, and the export of a checkpoint directory), launch calibrations (``calibration``) and generation with CUDA-graph
decode (``generation``).  Submodules are imported explicitly, e.g. ``from neuronspark.utils.cache import
InferenceParams``; importing this package imports none of them."""
