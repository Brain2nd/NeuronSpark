"""Operators of the NeuronSpark model.

This copy of ``neuronspark/ops/__init__.py`` holds the definitions the exported operator uses,
and nothing else of the package file.
"""

from __future__ import annotations

from typing import Callable, Dict
from ...neuronspark.ops.registry import OPS, op_names, resolve

__all__ = ["namespace"]


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
