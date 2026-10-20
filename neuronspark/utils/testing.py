"""The per-operator match rule and test-input helpers.

Match rule (the default correctness rule of flashinfer-bench, ``bench/utils.py compute_error_stats`` with the
``evaluators/default.py check_correctness`` order), applied to every output and every gradient of an operator against
the reference implementation of the same regime:

* both tensors are cast to fp32; ``abs_err = |x - y|`` and ``rel_err = abs_err / (|y| + 1e-8)`` with ``y`` the reference;
* an element fails only if ``abs_err > 1e-2`` **and** ``rel_err > 1e-2``;
* ``matched_ratio = 1 - fails / numel`` (an empty tensor matches with ratio 1.0) and must be **1.0**;
* checked first, in order: shape mismatch (``INCORRECT_SHAPE``), dtype mismatch (``INCORRECT_DTYPE``, unless the
  comparison is declared cross-dtype), an inf or a nan in the tested output (``INCORRECT_NUMERICAL``).

Reported per output: ``matched_ratio``, ``fails``, ``max_abs``, ``max_rel`` (plus the status and the element count).
The rule is the whole correctness standard; nothing else about the values is required.

Large tensors are compared in aligned row blocks (the same per-element arithmetic, bounded memory).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "ATOL",
    "RTOL",
    "EPS",
    "REQUIRED_MATCHED_RATIO",
    "MatchResult",
    "match_stats",
    "assert_match",
    "compare_outputs",
    "assert_all_match",
    "summarize",
    "rand_tensor",
]

ATOL = 1e-2
RTOL = 1e-2
EPS = 1e-8
REQUIRED_MATCHED_RATIO = 1.0
CHUNK = 1 << 24


@dataclass
class MatchResult:
    """Result of the rule for one output."""

    name: str
    status: str                       # "PASSED" | "INCORRECT_SHAPE" | "INCORRECT_DTYPE" | "INCORRECT_NUMERICAL"
    matched_ratio: float = 0.0
    fails: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    numel: int = 0
    shape: Tuple[int, ...] = ()
    dtype: str = ""
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "PASSED"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    def __str__(self) -> str:
        s = (f"{self.name}: {self.status} matched_ratio={self.matched_ratio:.9f} fails={self.fails}/{self.numel} "
             f"max_abs={self.max_abs:.3e} max_rel={self.max_rel:.3e} shape={list(self.shape)} dtype={self.dtype}")
        return s + (f" ({self.detail})" if self.detail else "")


def rand_tensor(shape, dtype: torch.dtype, device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Random operator input (flashinfer-bench ``gen_inputs`` random branch): float dtypes ``torch.randn``; bool
    ``randint(0, 2)``; int8 in [-128, 128); other integer dtypes in [-1024, 1024).  Inputs with a domain (index
    tensors, non-negative quantities) are drawn inside the domain by the caller."""
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device, generator=generator)
    if dtype == torch.int8:
        return torch.randint(-128, 128, shape, dtype=dtype, device=device, generator=generator)
    if dtype in (torch.int16, torch.int32, torch.int64):
        return torch.randint(-1024, 1024, shape, dtype=dtype, device=device, generator=generator)
    raise ValueError(f"rand_tensor: unsupported dtype {dtype}")


def _pieces(t: torch.Tensor, rows: int) -> Iterator[torch.Tensor]:
    """Flat pieces of ``t`` in blocks of ``rows`` leading-dimension rows (a 0-d tensor is one piece)."""
    if t.dim() == 0:
        yield t.reshape(-1)
        return
    for r0 in range(0, int(t.shape[0]), rows):
        yield t[r0:r0 + rows].reshape(-1)


@torch.no_grad()
def _error_stats(sol: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float, float, int, int]:
    """(max_abs, max_rel, matched_ratio, fails, numel) of the rule, chunked; sol and ref have the same shape."""
    n = sol.numel()
    if n == 0:
        return 0.0, 0.0, 1.0, 0, 0
    per_row = max(1, n // max(1, int(sol.shape[0]))) if sol.dim() else 1
    rows = max(1, CHUNK // per_row)
    max_abs = 0.0
    max_rel = 0.0
    fails = 0
    for xa, yb in zip(_pieces(sol, rows), _pieces(ref, rows)):
        a = xa.to(torch.float32)
        b = yb.to(device=a.device, dtype=torch.float32)
        ae = (a - b).abs()
        re = ae / (b.abs() + EPS)
        fails += int(((ae > ATOL) & (re > RTOL)).sum().item())
        max_abs = max(max_abs, float(ae.max().item()))
        max_rel = max(max_rel, float(re.max().item()))
        del a, b, ae, re
    matched = min(max(1.0 - fails / n, 0.0), 1.0)
    return max_abs, max_rel, matched, fails, n


@torch.no_grad()
def match_stats(new: torch.Tensor, ref: torch.Tensor, name: str = "output", *,
                check_dtype: bool = True) -> MatchResult:
    """The rule for one output pair, in the evaluator's order.  ``check_dtype=False`` declares a cross-dtype comparison
    (e.g. against a float64 reference): both sides are still cast to fp32."""
    res = MatchResult(name=name, status="PASSED", shape=tuple(ref.shape), dtype=str(ref.dtype))
    if tuple(new.shape) != tuple(ref.shape):
        res.status, res.detail = "INCORRECT_SHAPE", f"tested shape {list(new.shape)}"
        res.max_abs = res.max_rel = float("inf")
        return res
    if check_dtype and new.dtype != ref.dtype:
        res.status, res.detail = "INCORRECT_DTYPE", f"tested dtype {new.dtype}"
        res.max_abs = res.max_rel = float("inf")
        return res
    new = new.detach()
    ref = ref.detach()
    if new.is_floating_point():
        per_row = max(1, new.numel() // max(1, int(new.shape[0]))) if new.dim() else 1
        rows = max(1, CHUNK // per_row)
        has_inf = has_nan = False
        for c in _pieces(new, rows):
            has_inf = has_inf or bool(torch.isinf(c).any().item())
            has_nan = has_nan or bool(torch.isnan(c).any().item())
        if has_inf or has_nan:
            res.status, res.detail = "INCORRECT_NUMERICAL", "tested output has " + ("inf" if has_inf else "nan")
            res.numel = new.numel()
            res.max_abs = res.max_rel = float("inf")
            return res
    ma, mr, matched, fails, n = _error_stats(new, ref)
    res.max_abs, res.max_rel, res.matched_ratio, res.fails, res.numel = ma, mr, matched, fails, n
    if fails > 0 or matched < REQUIRED_MATCHED_RATIO:
        res.status = "INCORRECT_NUMERICAL"
    return res


def assert_match(new: torch.Tensor, ref: torch.Tensor, name: str = "output", *,
                 check_dtype: bool = True) -> MatchResult:
    """:func:`match_stats` that raises ``AssertionError`` (with the statistics) unless the output passes."""
    res = match_stats(new, ref, name, check_dtype=check_dtype)
    if not res.passed:
        raise AssertionError(str(res))
    return res


_Outputs = Union[Mapping[str, Optional[torch.Tensor]], Sequence[Optional[torch.Tensor]], torch.Tensor]


def _named(outputs: _Outputs, prefix: str) -> List[Tuple[str, Optional[torch.Tensor]]]:
    if isinstance(outputs, torch.Tensor):
        return [(prefix, outputs)]
    if isinstance(outputs, Mapping):
        return [(f"{prefix}.{k}" if prefix else str(k), v) for k, v in outputs.items()]
    return [(f"{prefix}[{i}]", v) for i, v in enumerate(outputs)]


def compare_outputs(new: _Outputs, ref: _Outputs, prefix: str = "", *, check_dtype: bool = True) -> List[MatchResult]:
    """The rule for every output of an operator (tensors, sequences or dicts with the same keys).  A ``None`` on both
    sides (an absent gradient) is skipped; ``None`` on one side only fails the output."""
    a, b = _named(new, prefix or "output"), _named(ref, prefix or "output")
    if [k for k, _ in a] != [k for k, _ in b]:
        raise ValueError(f"compare_outputs: different outputs {[k for k, _ in a]} vs {[k for k, _ in b]}")
    out: List[MatchResult] = []
    for (k, x), (_, y) in zip(a, b):
        if x is None and y is None:
            continue
        if x is None or y is None:
            side = "tested" if x is None else "reference"
            out.append(MatchResult(name=k, status="INCORRECT_SHAPE", max_abs=float("inf"), max_rel=float("inf"),
                                   detail=f"output missing on the {side} side"))
            continue
        out.append(match_stats(x, y, k, check_dtype=check_dtype))
    return out


def summarize(results: Sequence[MatchResult]) -> dict:
    """Overall status (any failing output fails) and the extremes over the outputs."""
    status = "PASSED" if results and all(r.passed for r in results) else "FAILED"
    return {
        "status": status,
        "outputs": len(results),
        "min_matched_ratio": min((r.matched_ratio for r in results), default=1.0),
        "fails": sum(r.fails for r in results),
        "max_abs": max((r.max_abs for r in results), default=0.0),
        "max_rel": max((r.max_rel for r in results), default=0.0),
        "failed_outputs": [r.name for r in results if not r.passed],
    }


def assert_all_match(new: _Outputs, ref: _Outputs, prefix: str = "", *, check_dtype: bool = True) -> List[MatchResult]:
    """:func:`compare_outputs` that raises ``AssertionError`` listing every failing output."""
    results = compare_outputs(new, ref, prefix, check_dtype=check_dtype)
    bad = [str(r) for r in results if not r.passed]
    if bad:
        raise AssertionError("per-operator rule failed:\n  " + "\n  ".join(bad))
    return results
