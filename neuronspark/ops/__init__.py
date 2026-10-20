"""Operators of the NeuronSpark model.

Layout (as in mamba_ssm): ``ops/linear.py`` (fp32-weight-gradient linear), ``ops/spiking_combined.py`` (training
spiking-layer chunk scan + soma Function), ``ops/spiking_prefill.py`` (prefill spiking-layer scan), ``ops/triton/``
(Triton kernels, the vendored mamba SSD kernels under ``ops/triton/mamba/``) and ``ops/gluon/`` (Gluon kernels for
Blackwell). Each family exposes private launchers, an autograd Function where it has a backward, a plain public function
and a pure-torch ``*_ref``.

Modules call operators through a namespace chosen once::

    from neuronspark import ops
    o = ops.namespace("kernel")        # or "ref": the pure-torch references, same signatures
    hn = o.spiking_norm(h, gain, 1e-5)

``KERNELS`` / ``REFS`` resolve an op name to its callable through :mod:`neuronspark.ops.registry` on first use (the
resolved callables are cached in the namespace: a documented first-call cache).  ``REFS.<name>`` and
``REFS.<name>_ref`` are the same reference function.  Importing this package imports no kernel module.
"""
from __future__ import annotations

from typing import Callable, Dict

from neuronspark.ops.registry import OPS, OpSpec, op_names, resolve

__all__ = ["KERNELS", "REFS", "namespace", "OPS", "OpSpec", "op_names", "resolve"]


class _OpNamespace:
    """Attribute access ``ns.<op>`` -> the op's kernel (``impl="kernel"``) or reference (``impl="ref"``)."""

    def __init__(self, impl: str) -> None:
        if impl not in ("kernel", "ref"):
            raise ValueError(f"impl must be 'kernel' or 'ref', got {impl!r}")
        object.__setattr__(self, "impl", impl)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str) -> Callable:
        if name.startswith("__"):
            raise AttributeError(name)
        cache: Dict[str, Callable] = object.__getattribute__(self, "_cache")
        fn = cache.get(name)
        if fn is None:
            key = name
            if self.impl == "ref" and name.endswith("_ref") and name[:-4] in OPS:
                key = name[:-4]
            if key not in OPS:
                raise AttributeError(f"neuronspark.ops: unknown op {name!r}")
            fn = resolve(key, self.impl)
            cache[name] = fn
        return fn

    def __setattr__(self, name, value) -> None:
        raise AttributeError("operator namespaces are read-only")

    def __dir__(self):
        return sorted(op_names())

    def __repr__(self) -> str:
        return f"<neuronspark.ops namespace impl={self.impl!r} ({len(OPS)} ops)>"


KERNELS = _OpNamespace("kernel")
REFS = _OpNamespace("ref")


def namespace(impl: str = "kernel") -> _OpNamespace:
    """``KERNELS`` for ``impl="kernel"``, ``REFS`` for ``impl="ref"``."""
    if impl == "kernel":
        return KERNELS
    if impl == "ref":
        return REFS
    raise ValueError(f"impl must be 'kernel' or 'ref', got {impl!r}")
