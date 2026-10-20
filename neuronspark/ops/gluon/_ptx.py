"""Round-to-nearest and PTX helpers for Gluon kernels (``triton.experimental.gluon``, Triton 3.5.1).

The same PTX instructions as :mod:`neuronspark.ops.triton._ptx` (``_rn_mul_f32`` / ``_rn_add_f32`` / ``_rn_sub_f32`` /
``_rn_div_f32``); ``_g_sigmoid_rn`` is ``_sigmoid_rn_f32`` with GUARD = 0 (libdevice expf of the negated argument,
add.rn, div.rn), so ``_release_gate_rn_f32`` at EXACT = 0 is ``_g_sigmoid_rn(_g_rn_mul(margin, alpha))``.
``_ex2ftz`` is ``ex2.approx.ftz.f32`` (the SSD decay); ``_tc_fence`` the tcgen05 thread-sync fences.  Kernels import the
helpers they call into their own module.
"""
from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice as _libdevice

__all__ = ["_g_rn_mul", "_g_rn_add", "_g_rn_sub", "_g_rn_div", "_g_chain_combine", "_g_pick_last", "_g_sigmoid_rn",
           "_ex2ftz", "_tc_fence"]


@gluon.jit
def _g_rn_mul(left, right):
    return gl.inline_asm_elementwise(asm="mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[left, right],
                                     dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _g_rn_add(left, right):
    return gl.inline_asm_elementwise(asm="add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[left, right],
                                     dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _g_rn_sub(left, right):
    return gl.inline_asm_elementwise(asm="sub.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[left, right],
                                     dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _g_rn_div(left, right):
    return gl.inline_asm_elementwise(asm="div.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[left, right],
                                     dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _g_chain_combine(la, lx, lp, lv, ra, rx, rp, rv):
    """The reverse-walk adjoint step as a scan combine: L = the state after the previous row (its x), R = this row's
    (a~, c~, -, valid); returns (a~, x after the row, x before the row, valid), i.e.
    x_cur = RN(RN(a~ x_next) + c~); x_next = where(valid, x_cur, x_next)."""
    nx = gl.where(rv != 0.0, _g_rn_add(_g_rn_mul(ra, lx), rx), lx)
    return ra, nx, lx, rv


@gluon.jit
def _g_pick_last(v1, i1, v2, i2):
    """(value, index) of the larger index: commutative and associative, so any reduction order selects the value of
    the last row exactly."""
    return gl.where(i2 > i1, v2, v1), gl.maximum(i1, i2)


@gluon.jit
def _g_sigmoid_rn(x):
    e = _libdevice.exp(-x)
    den = _g_rn_add(1.0, e)
    return _g_rn_div(1.0, den)


@gluon.jit
def _ex2ftz(t):
    return gl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [t], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _tc_fence(AFTER: gl.constexpr, NWF: gl.constexpr):
    """tcgen05.fence::before_thread_sync (AFTER 0) / ::after_thread_sync (AFTER 1), once per thread of the calling partition: the
    PTX ordering of tcgen05 operations across a thread synchronisation (a TMEM store by the compute warps, then the MMA warp's
    tcgen05.mma reading that accumulator; an MMA's commit, then the compute warps' TMEM loads).  Gluon 3.5.1 emits neither."""
    LF: gl.constexpr = gl.BlockedLayout([1], [32], [NWF], [0])
    z = gl.full([32 * NWF], 0, gl.int32, layout=LF)
    if AFTER:
        gl.inline_asm_elementwise("tcgen05.fence::after_thread_sync; mov.b32 $0, $1;", "=r,r", [z], dtype=gl.int32,
                                  is_pure=False, pack=1)
    else:
        gl.inline_asm_elementwise("tcgen05.fence::before_thread_sync; mov.b32 $0, $1;", "=r,r", [z], dtype=gl.int32,
                                  is_pure=False, pack=1)
