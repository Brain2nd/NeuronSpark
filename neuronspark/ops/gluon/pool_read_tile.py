"""Tile helpers shared by the Gluon kernels of the fused pool read (:mod:`neuronspark.ops.gluon.pool_read_fwd`,
:mod:`neuronspark.ops.gluon.pool_read_bwd`).

``_window_term``: the induction term of a (rows, columns) tile from the id windows of its rows and of its columns,
``X = ((m_2 beta_2 + m_3 beta_3) + m_4 beta_4)`` in fp32 from +0 (the orders in ORDMASK), with the order masks.  The
order-n match compares the first n entries of the query window (x_t, x_{t-1}, ..) with the key window
(x_{j-1}, x_{j-2}, ..); the comparison is symmetric, so the rows may hold either side.  Below the diagonal (j <= t - 1)
the pads of the windows never match (a key window reaches before the row start only where the query window does not),
so the definition's range conditions reduce to j <= t - 1, which the callers apply on the diagonal tiles.

``_row``: one linear shared-memory row (a TMA box of one row) read as a vector broadcast to every thread.
"""
from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

__all__ = ["LOG2E", "_window_term", "_row"]

LOG2E = gl.constexpr(1.4426950408889634)


@gluon.jit
def _row(buf, N: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr):
    """(N,) vector in ``layout`` from a (1, N) linear shared-memory row."""
    return buf._reinterpret(dtype, [N], gl.SwizzledSharedLayout(1, 1, 1, [0])).load(layout)


@gluon.jit
def _window_term(r0, r1, r2, r3, c0, c1, c2, c3, beta1, beta2, beta3, beta4, BR: gl.constexpr, BC: gl.constexpr,
                 L: gl.constexpr, ORDMASK: gl.constexpr, NMAX: gl.constexpr):
    """(x, m1, m2, m3, m4): the (BR, BC) fp32 induction term and the order masks (false for an order not in
    ORDMASK) from the row windows r0..r3 (SliceLayout(1, L)) and the column windows c0..c3 (SliceLayout(0, L))."""
    run = r0[:, None] == c0[None, :]
    none = run & (r0[:, None] < -2)
    m1 = none
    m2 = none
    m3 = none
    m4 = none
    x = gl.zeros([BR, BC], gl.float32, layout=L)
    if (ORDMASK >> 1) & 1:
        x = x + gl.where(run, beta1, 0.0)
        m1 = run
    if NMAX > 1:
        run = run & (r1[:, None] == c1[None, :])
        if (ORDMASK >> 2) & 1:
            x = x + gl.where(run, beta2, 0.0)
            m2 = run
    if NMAX > 2:
        run = run & (r2[:, None] == c2[None, :])
        if (ORDMASK >> 3) & 1:
            x = x + gl.where(run, beta3, 0.0)
            m3 = run
    if NMAX > 3:
        run = run & (r3[:, None] == c3[None, :])
        if (ORDMASK >> 4) & 1:
            x = x + gl.where(run, beta4, 0.0)
            m4 = run
    return x, m1, m2, m3, m4
