"""Tensor-parallel inference on two GPUs: both GPUs work on the same requests, each holding half of every dendritic
neuron layer and half of the readout's vocabulary.

    import torch.distributed as dist
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    model = NeuronSparkLMHeadModel.from_pretrained(path, device=f"cuda:{local_rank}")   # every rank loads the model
    tp = tensor_parallel(model)                    # world size 1: the model itself
    out = tp.generate(prompts, max_length)         # captured prefill and decode steps, as for one GPU

The split (rank r of 2):

* dendritic neuron layer (``mixer_a``): heads [r H/2, (r + 1) H/2) with their channels -- the in-projection's z, x and
  dt rows, the conv taps and bias of the x channels, A, D, dt bias, theta, gamma, the out-projection's columns -- and
  their recurrent state (conv tail (B, d_inner / 2 + 2 N, 3), SSM state (B, H / 2, P, N)); the B and C rows (shared by
  every head) and their conv channels are held and computed by both GPUs.  The gated RMSNorm's per-row sum of squares
  is exchanged inside the norm kernel (:mod:`neuronspark.ops.triton.tp_dendritic`); the out-projection's partial
  product is taken in fp32 and summed across the GPUs, rounded to bf16 once and added to the fp32 residual stream (the
  decode step: one kernel with the next dendritic neuron layer's pre-norm, :func:`neuronspark.ops.triton.tp_exchange
  .tp_join`; the prefill: the partial written into symmetric memory and read by the other GPU once released,
  :func:`neuronspark.ops.triton.tp_bulk.tp_prefill_join`);
* the hippocampal layers, for a batch of B >= 2 rows: rows [0, ceil(B / 2)) on rank 0 and the rest on rank 1, with
  their states (rings, clocks, conv history, neuron potentials); each GPU runs the one-GPU layer on its rows (its rows
  of the position counter, the shared ring pointer) and the rows are joined on both GPUs after the layer (the decode
  step: the layer's last addition, the exchange and the next dendritic neuron layer's pre-norm in one kernel,
  :func:`neuronspark.ops.triton.tp_exchange.tp_tail_gather`, through the layer's ``tail``; the prefill: the rows
  written into symmetric memory and read by the other GPU once released, :func:`neuronspark.ops.triton.tp_bulk
  .tp_prefill_gather`), after the last layer only the positions the readout takes.  A batch of one row runs them whole
  on both GPUs;
* the readout: vocabulary rows [0, S) on rank 0 and [S, V) on rank 1, S = the largest multiple of every readout
  balance width that is at most V / 2 (the balance adds ``c[:, v % w]``: a shard starting at a multiple of w is the
  same function); the logits are gathered (NCCL) into the full (B, R, V) rows on both GPUs, or, for greedy decoding
  (:class:`GreedyTokens`), each GPU's per-row maximum and its index are exchanged and the token picked
  (:func:`neuronspark.ops.triton.tp_exchange.tp_greedy_pick`): the full logits are never formed;
* everything else -- embedding, lexical memory, induction match, copy statistics, the final norm -- runs whole on
  both GPUs on the same values.

Exchanges inside a captured step go through a mailbox in symmetric memory (:class:`TensorParallelComm`): one region
per call site of the step (decode sites and prefill sites in separate ranges), the site counter reset at the start of
every forward.  The prefill's out-projection partials and the split hippocampal layers' rows go through two symmetric
buffers (by site parity) with a flag word per site, and every prefill call advances an epoch counter and starts with a
fence on a call word (:mod:`neuronspark.ops.triton.tp_bulk`); the buffers grow in an eager call, at least doubling, and
are kept until :meth:`TensorParallelComm.close` (a captured call addresses them).  Both GPUs must run the same sequence
of forwards
(the same calls in the same order), and each GPU its forwards one after another (on one stream, or on streams ordered
by events: eager calls and graph replays share the mailbox, the epoch and the buffers); the generation path does, as
both ranks call it with the same arguments, and a sampled token is rank 0's on both GPUs
(:meth:`TensorParallelLM.agree_tokens`).  The decay rates and soma thresholds of a GPU's heads are weight constants
(:class:`neuronspark.utils.cache.WeightConstants`), formed on first use and held by every captured step that reads
them.

The model's mathematics is the one-GPU path's; the sums run in another order (two partial dot products and two
partial sums of squares added in fp32), and a split hippocampal layer runs the one-GPU kernels on half the batch (a
kernel whose choices depend on the batch, e.g. a product's algorithm, can round a row otherwise).  The cache's
contract is kept: the dendritic state dtype of the cache; :meth:`TensorParallelLM.gather_cache` joins the split states
into the one-GPU cache's names and shapes.  ``DecodeGraph`` / ``PrefillGraph`` of :mod:`neuronspark.utils.generation`
capture the tensor-parallel model unchanged.
"""
from __future__ import annotations

import gc
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from neuronspark import ops as _ops
from neuronspark.models.neuronspark_lm import CausalLMOutput, NeuronSparkLMHeadModel
from neuronspark.modules.copy_detector import allocate_copy_state, copy_stats_session
from neuronspark.modules.dendritic_neuron_layer import DendriticNeuronLayer, DendriticNeurons
from neuronspark.modules.hippocampal_layer import HippocampalLayer
from neuronspark.modules.induction import allocate_induction_state, induction_session_match
from neuronspark.ops.linear import _cached_bf16, cache_bf16_, drop_bf16_
from neuronspark.ops.triton.dendritic_step import MAX_DECODE_BATCH
from neuronspark.ops.triton.tp_bulk import tp_fence, tp_prefill_gather, tp_prefill_join, tp_release
from neuronspark.ops.triton.tp_dendritic import k2_words, tp_gated_norm_soma, tp_step_core
from neuronspark.ops.triton.tp_exchange import tp_greedy_pick, tp_join, tp_tail_gather
from neuronspark.utils.cache import (DendriticState, HippocampalState, InferenceParams, RING_MULTIPLE,
                                     WeightConstants, drop_weight_constants, resolve_dendritic_state_dtype)
from neuronspark.utils.generation import GenerationMixin
from neuronspark.utils.precision import regime

__all__ = ["TensorParallelComm", "TensorParallelLM", "GreedyTokens", "tensor_parallel", "vocab_split", "mailbox_sites",
           "DECODE_SITE_WORDS", "DECODE_SITES", "PREFILL_SITE_WORDS", "PREFILL_SITES"]

#: mailbox words of one decode call site (the largest exchange: MAX_DECODE_BATCH rows of D <= 2,048) and the default
#: number of decode sites (:func:`mailbox_sites` gives a model's)
DECODE_SITE_WORDS = MAX_DECODE_BATCH * 2048
DECODE_SITES = 64
#: mailbox words of one prefill call site (one per row of B x T) and the default number of prefill sites
PREFILL_SITE_WORDS = 65536
PREFILL_SITES = 64


def mailbox_sites(model: NeuronSparkLMHeadModel) -> Tuple[int, int]:
    """(decode sites, prefill sites) of the model's calls: a decode step takes 2 per dendritic neuron layer, 1 per
    hippocampal layer (a split batch) and 1 for the greedy pick (which a greedy prefill takes after its own mailbox
    sites); a prefill call takes 1 mailbox site per dendritic neuron layer and 1 buffer site (with its flag word) per
    layer."""
    n_hip = sum(isinstance(layer, HippocampalLayer) for layer in model.backbone.layers)
    n_den = len(model.backbone.layers) - n_hip
    return 2 * n_den + n_hip + 1, n_den + n_hip


def vocab_split(vocab_size: int, widths) -> int:
    """The first vocabulary row of rank 1: the largest multiple of every balance width at most V / 2."""
    step = 1
    for w in widths:
        step = step * int(w) // math.gcd(step, int(w))
    s = (int(vocab_size) // 2) // step * step
    if s <= 0:
        raise ValueError(f"vocab_split: no split of {vocab_size} rows at a multiple of {step}")
    return s


# ================================================================================================ communicator
class TensorParallelComm:
    """The process group of the two GPUs and the mailbox of the in-graph exchanges (module docstring), with
    ``decode_sites`` regions of a decode step and ``prefill_sites`` regions and buffer sites of a prefill call
    (:func:`mailbox_sites` of the model; both GPUs pass the same)."""

    def __init__(self, group=None, device: Optional[torch.device] = None, decode_sites: int = DECODE_SITES,
                 prefill_sites: int = PREFILL_SITES) -> None:
        import torch.distributed._symmetric_memory as symm_mem
        self.group = group if group is not None else dist.group.WORLD
        self.world = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        if self.world != 2:
            raise ValueError(f"TensorParallelComm: two GPUs, got a group of {self.world}")
        if int(decode_sites) < 1 or int(prefill_sites) < 1:
            raise ValueError(f"TensorParallelComm: at least one site of each kind, got {decode_sites} decode and "
                             f"{prefill_sites} prefill sites")
        self.peer = 1 - self.rank
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        self.decode_sites, self.prefill_sites = int(decode_sites), int(prefill_sites)
        name = self.group.group_name
        self._decode_base = 0
        self._prefill_base = self.decode_sites * DECODE_SITE_WORDS
        self._flag_base = self._prefill_base + self.prefill_sites * PREFILL_SITE_WORDS
        self._call_word = self._flag_base + self.prefill_sites
        total = self._call_word + 1
        self.local = symm_mem.empty(total, dtype=torch.int64, device=self.device)
        self.local.zero_()
        self.handle = symm_mem.rendezvous(self.local, name)
        self.remote = self.handle.get_buffer(self.peer, (total,), torch.int64)
        self.epoch = torch.zeros(1, dtype=torch.int64, device=self.device)   # prefill calls so far (on the device)
        self._bulk = None                                # (local, handle, remote, capacity) of the partial buffers
        self._bulk_kept: List[tuple] = []                # smaller buffers a captured call may still address
        torch.cuda.synchronize(self.device)
        dist.barrier(self.group)
        self._prefill = False
        self._site = 0
        self._bulk_site = 0

    def begin(self, prefill: bool) -> None:
        """Start a forward: the next region is the first of the call's range; a prefill call advances the epoch and
        passes the fence on the call word (both GPUs have finished their previous calls)."""
        self._prefill = bool(prefill)
        self._site = 0
        self._bulk_site = 0
        if self._prefill:
            self.epoch.add_(1)
            c = self._call_word
            tp_fence(self.remote[c:c + 1], self.local[c:c + 1], self.epoch)

    def bulk(self, rows: int, cols: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(this GPU's partial buffer (rows, cols) fp32, the peer's, this GPU's flag word, the peer's) of the next
        prefill join site (:mod:`neuronspark.ops.triton.tp_bulk`).  The buffers grow in an eager call (both GPUs make
        the same calls, so both grow together); a captured call needs an eager call of its size first."""
        n = int(rows) * int(cols)
        site = self._bulk_site
        if site >= self.prefill_sites:
            raise RuntimeError(f"TensorParallelComm: a prefill call has {self.prefill_sites} join sites")
        if self._bulk is None or n > self._bulk[3]:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("TensorParallelComm: a captured prefill needs an eager call of its size first (its "
                                   "partial buffers are allocated there)")
            self._grow(n, int(cols))
        self._bulk_site += 1
        local, _, remote, cap = self._bulk
        lo = (site % 2) * cap
        f = self._flag_base + site
        return (local[lo:lo + n].view(int(rows), int(cols)), remote[lo:lo + n].view(int(rows), int(cols)),
                self.local[f:f + 1], self.remote[f:f + 1])

    def _grow(self, n: int, cols: int) -> None:
        """New partial buffers of at least ``n`` values per parity: at least twice the old capacity, up to the largest
        call's (PREFILL_SITE_WORDS rows of ``cols``), so that the kept buffers total less than three times the
        largest."""
        import torch.distributed._symmetric_memory as symm_mem
        old = 0 if self._bulk is None else self._bulk[3]
        cap = max(int(n), min(2 * old, PREFILL_SITE_WORDS * int(cols)))
        cap = -(-cap // 1024) * 1024                     # the second parity's view starts 4 KiB aligned
        local = symm_mem.empty(2 * cap, dtype=torch.float32, device=self.device)
        handle = symm_mem.rendezvous(local, self.group.group_name)
        remote = handle.get_buffer(self.peer, (2 * cap,), torch.float32)
        if self._bulk is not None:
            self._bulk_kept.append(self._bulk)
        self._bulk = (local, handle, remote, cap)
        torch.cuda.synchronize(self.device)
        dist.barrier(self.group)

    def region(self, words: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(the peer's words, this GPU's words) of the next call site, ``words`` long."""
        base, size, sites = ((self._prefill_base, PREFILL_SITE_WORDS, self.prefill_sites) if self._prefill
                             else (self._decode_base, DECODE_SITE_WORDS, self.decode_sites))
        if int(words) > size or self._site >= sites:
            raise RuntimeError(f"TensorParallelComm: call site {self._site} needs {words} words; a "
                               f"{'prefill' if self._prefill else 'decode'} call has {sites} sites of {size}")
        off = base + self._site * size
        self._site += 1
        return self.remote[off:off + int(words)], self.local[off:off + int(words)]

    def close(self) -> None:
        """Free the mailbox on both GPUs together (every rank calls it; before the process group is destroyed): no
        exchange may be pending, and no captured graph that addresses the mailbox may be replayed afterwards."""
        if self.local is None:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(self.group)
        self.remote = self.handle = self.local = None
        self._bulk, self._bulk_kept = None, []
        gc.collect()
        torch.cuda.synchronize(self.device)
        dist.barrier(self.group)


# ================================================================================================ dendritic shard
class _DendriticShard(nn.Module):
    """This GPU's half of one dendritic neuron population (module docstring)."""

    def __init__(self, m: DendriticNeurons, rank: int) -> None:
        super().__init__()
        H, P, N, di = m.nheads, m.head_dim, m.state_size, m.d_inner
        if H % 2:
            raise ValueError(f"tensor_parallel: a dendritic neuron layer of {H} heads does not split in two")
        self.layer_idx = m.layer_idx
        self.H, self.P, self.N, self.di_all = H // 2, P, N, di
        self.di = di // 2
        self.conv_dim = self.di + 2 * N
        self.chunk_size = m.chunk_size
        self.norm_eps = m.norm_eps
        self.theta_floor = m.theta_floor
        h0, c0 = rank * self.H, rank * self.di
        h1, c1 = h0 + self.H, c0 + self.di
        f = lambda t: nn.Parameter(t.detach().clone().contiguous(), requires_grad=False)   # noqa: E731
        w = m.in_proj.weight
        self.in_proj = f(torch.cat([w[c0:c1], w[di + c0:di + c1], w[2 * di:2 * di + 2 * N],
                                    w[2 * di + 2 * N + h0:2 * di + 2 * N + h1]]))
        self.conv_weight = f(torch.cat([m.conv_weight[c0:c1], m.conv_weight[di:di + 2 * N]]))
        self.conv_bias = f(torch.cat([m.conv_bias[c0:c1], m.conv_bias[di:di + 2 * N]]))
        self.dt_bias = f(m.dt_bias[h0:h1])
        self.A_log = f(m.A_log[h0:h1])
        self.D = f(m.D[h0:h1])
        self.gamma = f(m.gamma[c0:c1])
        self.rho = f(m.rho[c0:c1])
        self.W_out = f(m.W_out[:, c0:c1])
        self.pre_norm_gain = m.pre_norm_gain                     # the layer's own (whole on both GPUs)
        self._constants = WeightConstants()                      # (A, theta) per weight version, as the layer's

    def constants(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(A = -exp(A_log) fp32, theta = theta_floor + softplus(rho) fp32) of this GPU's heads and channels."""
        return self._constants.get((self.A_log, self.rho), lambda: (
            -torch.exp(self.A_log.float()), self.theta_floor + torch.nn.functional.softplus(self.rho)))

    def prepare(self) -> None:
        cache_bf16_(self.in_proj)
        cache_bf16_(self.W_out)
        self.constants()

    def release(self) -> None:
        drop_bf16_(self.in_proj)
        drop_bf16_(self.W_out)
        self._constants.drop()

    def allocate(self, batch_size: int, dtype: torch.dtype, device) -> DendriticState:
        return DendriticState.allocate(int(batch_size), self.conv_dim, 4, self.H, self.P, self.N, dtype, device)

    def _w_out(self) -> torch.Tensor:
        w = _cached_bf16(self.W_out)
        return w if w is not None else self.W_out.to(torch.bfloat16)

    def step(self, o, comm: TensorParallelComm, h: torch.Tensor, st: DendriticState, hn: Optional[torch.Tensor],
             next_gain: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hc = h[0] if h[0].is_contiguous() else h[0].contiguous()
        B, Dm = (int(s) for s in hc.shape)
        if hn is None:
            hn = o.dendritic_prenorm_rows(hc, self.pre_norm_gain, self.norm_eps)
        proj = o.linear(hn, self.in_proj)                                            # (B, 2 di + 2 N + H) bf16
        A, theta = self.constants()
        send, recv = comm.region(k2_words(B, self.di))
        xs = tp_step_core(proj, st.conv, st.ssm, self.conv_weight, self.conv_bias, A, self.dt_bias, self.D,
                          theta, self.gamma, self.norm_eps, self.di_all, send, recv)
        part = torch.mm(xs, self._w_out().t(), out_dtype=torch.float32)             # (B, D) fp32 partial
        send, recv = comm.region(B * Dm)
        h_new, hn_next = tp_join(part, hc, send, recv, next_gain, self.norm_eps)
        return h_new.unsqueeze(0), hn_next

    def prefill(self, o, comm: TensorParallelComm, h: torch.Tensor, st: DendriticState) -> torch.Tensor:
        T, B, Dm = (int(s) for s in h.shape)
        di, cd, H, P, N = self.di, self.conv_dim, self.H, self.P, self.N
        hn = o.dendritic_prenorm_rows(h.reshape(T * B, Dm), self.pre_norm_gain, self.norm_eps).view(T, B, Dm)
        proj = o.linear(hn.transpose(0, 1), self.in_proj)                           # (B, T, 2 di + 2 N + H) bf16
        z = proj[..., :di]
        xBC = proj[..., di:di + cd]
        dt = proj[..., di + cd:]
        xbc, tail = o.dendritic_conv_silu_prefill(xBC, self.conv_weight, self.conv_bias, st.conv)
        st.conv.copy_(tail)
        x = xbc[..., :di].view(B, T, H, P)
        Bm = xbc[..., di:di + N].view(B, T, 1, N)
        Cm = xbc[..., di + N:].view(B, T, 1, N)
        A, theta = self.constants()
        y, final_state = o.dendritic_chunk_scan_prefill(x, dt, A, Bm, Cm, self.D, self.dt_bias, st.ssm,
                                                        chunk_size=self.chunk_size)
        st.ssm.copy_(final_state)
        send, recv = comm.region(B * T)
        xo = tp_gated_norm_soma(y.reshape(B * T, di), z.reshape(B * T, di), theta, self.gamma, self.norm_eps,
                                self.di_all, send, recv)
        part, part_peer, flag, flag_peer = comm.bulk(B * T, Dm)
        torch.mm(xo, self._w_out().t(), out_dtype=torch.float32, out=part)          # (B * T, D) fp32 partial
        tp_release(flag_peer, comm.epoch)
        return tp_prefill_join(part, part_peer, h, flag, comm.epoch)


# ================================================================================================ the model
class TensorParallelLM(nn.Module, GenerationMixin):
    """The language model split across two GPUs for inference (module docstring).  ``forward(input_ids,
    inference_params, num_last_tokens)`` -> ``CausalLMOutput(logits (B, R, V) bf16)`` on both GPUs (inference only:
    an inference cache is required)."""

    def __init__(self, model: NeuronSparkLMHeadModel, comm: TensorParallelComm) -> None:
        super().__init__()
        if model.impl != "kernel":
            raise ValueError("tensor_parallel: the kernel implementation is required")
        d, p = mailbox_sites(model)
        if comm.decode_sites < d or comm.prefill_sites < p:
            raise ValueError(f"tensor_parallel: the model's calls take {d} decode and {p} prefill sites, the mailbox "
                             f"has {comm.decode_sites} and {comm.prefill_sites}")
        self.model = model
        self.comm = comm
        self.config = model.config
        self.impl = model.impl
        cfg = model.config
        bb = model.backbone
        self.rank = comm.rank
        self.shards = nn.ModuleDict({str(i): _DendriticShard(layer.mixer_a, comm.rank)
                                     for i, layer in enumerate(bb.layers) if isinstance(layer, DendriticNeuronLayer)})
        if not self.shards:
            raise ValueError("tensor_parallel: the model has no dendritic neuron layer to split")
        V = int(cfg.vocab_size)
        s = vocab_split(V, bb.balance_stack.widths)
        self.vocab_lo, self.vocab_hi = (0, s) if comm.rank == 0 else (s, V)
        self.vocab_widths = (s, V - s)
        self.readout_weight = nn.Parameter(bb.embed_tokens.weight[self.vocab_lo:self.vocab_hi].detach().clone(),
                                           requires_grad=False)
        self._decode_graph = None
        self._prefill_graph = None

    # ------------------------------------------------------------------------------------------------ weights
    def _whole_weights(self) -> List[nn.Parameter]:
        """The projection weights run whole on both GPUs (every inference projection but the split ones)."""
        bb = self.model.backbone
        split = {f"layers.{i}.mixer_a." for i in self.shards}
        return [p for n, p in bb.inference_linear_weights()
                if n != "embed_tokens.weight" and not any(n.startswith(s) for s in split)]

    @torch.no_grad()
    def prepare_inference(self) -> "TensorParallelLM":
        for p in self._whole_weights():
            cache_bf16_(p)
        for sh in self.shards.values():
            sh.prepare()
        cache_bf16_(self.readout_weight)
        return self

    def release_inference(self) -> None:
        self._decode_graph = None
        self._prefill_graph = None
        drop_weight_constants(self)
        for p in self._whole_weights():
            drop_bf16_(p)
        for sh in self.shards.values():
            sh.release()
        drop_bf16_(self.readout_weight)

    def close(self) -> None:
        """Release the inference constants and free the mailbox (every rank calls it, before the process group is
        destroyed); the model is not used again."""
        self.release_inference()
        self.comm.close()

    # ------------------------------------------------------------------------------------------------ cache
    def _rows(self, batch_size: int) -> Optional[Tuple[int, int]]:
        """This GPU's rows [lo, hi) of the hippocampal layers' batch split (rank 0 the first ceil(B / 2), rank 1 the
        rest); None for a batch of one row (both GPUs run it whole)."""
        B = int(batch_size)
        if B < 2:
            return None
        n0 = (B + 1) // 2
        return (0, n0) if self.rank == 0 else (n0, B)

    @staticmethod
    def _half(ip: InferenceParams, lo: int, hi: int) -> InferenceParams:
        """The cache as this GPU's hippocampal layers see it: rows [lo, hi) of the position counter, the ring pointer,
        the layers' states (the hippocampal ones hold hi - lo rows)."""
        return InferenceParams(max_seqlen=ip.max_seqlen, max_batch_size=hi - lo, pos=ip.pos[lo:hi],
                               ring_ptr=ip.ring_ptr, dendritic_state_dtype=ip.dendritic_state_dtype, layers=ip.layers,
                               lexical=ip.lexical, induction=ip.induction, copy=ip.copy,
                               rope_max_seq_len=ip.rope_max_seq_len, announce=False)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int,
                                 dendritic_state_dtype: Optional[Union[str, torch.dtype]] = None,
                                 device=None) -> InferenceParams:
        """The cache of this GPU: the dendritic states of its heads, the hippocampal states of its rows (every row for
        a batch of one), every other state whole."""
        cfg = self.config
        bb = self.model.backbone
        B, W = int(batch_size), int(max_seqlen)
        if not 1 <= B <= MAX_DECODE_BATCH:
            raise ValueError(f"allocate_inference_cache: batch_size must lie in [1, {MAX_DECODE_BATCH}], got {B}")
        if W <= 0 or W % RING_MULTIPLE != 0 or W > int(cfg.rope_max_seq_len):
            raise ValueError(f"allocate_inference_cache: max_seqlen must be a positive multiple of {RING_MULTIPLE} "
                             f"and at most {cfg.rope_max_seq_len}, got {W}")
        dt = resolve_dendritic_state_dtype(cfg.inference_dendritic_state_dtype if dendritic_state_dtype is None
                                           else dendritic_state_dtype)
        dev = bb.embed_tokens.weight.device if device is None else torch.device(device)
        rows = self._rows(B)
        n_hippo = B if rows is None else rows[1] - rows[0]
        layers = {}
        for i, layer in enumerate(bb.layers):
            if isinstance(layer, HippocampalLayer):
                layers[i] = layer.allocate_inference_cache(n_hippo, W, device=dev)
            else:
                layers[i] = self.shards[str(i)].allocate(B, dt, dev)
        s0, s1 = (int(v) for v in cfg.lexical_order1_slice)
        pos, ring_ptr = InferenceParams.allocate_counters(B, dev)
        return InferenceParams(
            max_seqlen=W, max_batch_size=B, pos=pos, ring_ptr=ring_ptr, dendritic_state_dtype=dt, layers=layers,
            lexical=bb.lexical_memory.allocate_inference_cache(B, device=dev),
            induction=allocate_induction_state(B, W, cfg.hippocampal_induction_bias_orders, pad=int(cfg.vocab_size),
                                               device=dev),
            copy=allocate_copy_state(B, W, dim=s1 - s0, device=dev),
            rope_max_seq_len=int(cfg.rope_max_seq_len), announce=False)

    @torch.no_grad()
    def gather_cache(self, ip: InferenceParams) -> Dict[str, torch.Tensor]:
        """Copies of every state tensor of the cache, the dendritic states of both GPUs joined along the heads and the
        hippocampal states of a split batch along the rows (the one-GPU cache's names and shapes), on every rank."""
        out = {}
        B = int(ip.max_batch_size)
        n0 = (B + 1) // 2
        split = self._rows(B) is not None
        for name, t in ip.tensors():
            parts = name.split(".")
            if parts[0] == "layers" and parts[1] in self.shards and parts[2] in ("conv", "ssm"):
                both = [torch.empty_like(t) for _ in range(2)]
                dist.all_gather(both, t.contiguous(), group=self.comm.group)
                if parts[2] == "ssm":
                    t = torch.cat(both, dim=1)
                else:
                    di = self.shards[parts[1]].di
                    t = torch.cat([both[0][:, :di], both[1][:, :di], both[0][:, di:]], dim=1)
            elif parts[0] == "layers" and split and isinstance(ip.layers[int(parts[1])], HippocampalState):
                rows = torch.zeros((n0,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)   # rank 1 padded
                rows[:int(t.shape[0])].copy_(t)
                both = [torch.empty_like(rows) for _ in range(2)]
                dist.all_gather(both, rows, group=self.comm.group)
                t = torch.cat([both[0], both[1][:B - n0]], dim=0)
            else:
                t = t.clone()
            out[name] = t
        return out

    # ------------------------------------------------------------------------------------------------ forward
    def _backbone(self, ids: torch.Tensor, ip: InferenceParams, num_last_tokens: int = 0) -> torch.Tensor:
        """ids (B, T) -> the final norm (B, R, D) fp32 of the last R = ``num_last_tokens`` positions (all for 0)."""
        cfg = self.config
        bb = self.model.backbone
        if torch.is_grad_enabled():
            raise RuntimeError("TensorParallelLM: inference calls run under torch.no_grad()")
        B, T = (int(s) for s in ids.shape)
        if B != int(ip.max_batch_size):
            raise RuntimeError(f"TensorParallelLM: {B} rows for a cache of {ip.max_batch_size} rows")
        ip.check_room(T)
        W = int(ip.max_seqlen)
        D = int(cfg.D)
        R = T if int(num_last_tokens) <= 0 else min(int(num_last_tokens), T)
        o = _ops.namespace(self.impl)
        comm = self.comm
        comm.begin(prefill=T > 1)
        x = bb.embed_tokens(ids).transpose(0, 1)
        if T > 1:
            x = x.contiguous()
        lex = bb.lexical_memory.session_vector(ids, ip.lexical, ops=o)
        ind = induction_session_match(ids, ip.induction, ip.ring_ptr, W, cfg.hippocampal_induction_bias_orders, ops=o)
        copy = copy_stats_session(lex, ip.copy, ip.ring_ptr, ip.pos, W, tuple(cfg.lexical_order1_slice), ops=o)
        lex = lex.to(torch.bfloat16)
        rows = self._rows(B)
        hip = ip
        if rows is not None:                         # the hippocampal layers' inputs: this GPU's rows
            lo, hi = rows
            hip = self._half(ip, lo, hi)
            lex = lex[:, lo:hi].contiguous()
            ind = tuple(m[lo:hi] for m in ind)
            copy = tuple(c[:, lo:hi].contiguous() for c in copy)
        carry: Optional[torch.Tensor] = None
        hn: Optional[torch.Tensor] = None
        n_layers = len(bb.layers)
        for i, layer in enumerate(bb.layers):
            nxt = bb.layers[i + 1] if i + 1 < n_layers else None
            if isinstance(layer, HippocampalLayer):
                hn = None
                if rows is None:
                    x, carry = layer(x, lex=lex, ind=ind, copy=copy, carry=carry, inference_params=hip)
                elif T == 1:
                    (x, hn), carry = layer(x[:, lo:hi], lex=lex, ind=ind, copy=copy, carry=carry,
                                           inference_params=hip, tail=self._tail_join(lo, B, nxt))
                else:
                    x, carry = layer(x[:, lo:hi].contiguous(), lex=lex, ind=ind, copy=copy, carry=carry,
                                     inference_params=hip)
                    x = self._join_rows(x, lo, B, R if nxt is None else T)
                continue
            shard = self.shards[str(i)]
            if T == 1:
                gain = nxt.mixer_a.pre_norm_gain if isinstance(nxt, DendriticNeuronLayer) else None
                x, hn = shard.step(o, comm, x, ip.layers[i], hn, gain)
            else:
                x = shard.prefill(o, comm, x, ip.layers[i])
        if int(x.shape[0]) != R:
            x = x[int(x.shape[0]) - R:]
        x = o.rmsnorm_rows(x.reshape(R * B, D), float(bb.norm.eps), weight=bb.norm.weight).view(R, B, D)
        ip.pos.add_(T)
        ip.ring_ptr.add_(T)
        ip.seqlen_offset += T
        return x.transpose(0, 1)

    def _tail_join(self, lo: int, B: int, nxt: Optional[nn.Module]):
        """The ``tail`` of a split hippocampal layer's decode step: a function of the layer's (h', ffn, y, rho) on this
        GPU's rows [lo, lo + n) -> (every row of the layer's output (1, B, D) fp32, the next dendritic neuron layer's
        pre-norm (B, D) bf16 if the next layer is one, else None), one kernel."""
        gain, eps = ((nxt.mixer_a.pre_norm_gain, float(nxt.mixer_a.norm_eps))
                     if isinstance(nxt, DendriticNeuronLayer) else (None, 1e-5))
        send, recv = self.comm.region(B * int(self.config.D))

        def tail(h, ffn, y, rho):
            ho, hn = tp_tail_gather(h, ffn, y, rho, lo, B, send, recv, gain, eps)
            return ho.unsqueeze(0), hn
        return tail

    def _join_rows(self, y: torch.Tensor, lo: int, B: int, last: int) -> torch.Tensor:
        """Every row of the stream from this GPU's rows ``y`` (T, n, D) fp32 = rows [lo, lo + n) of a split
        hippocampal layer's prefill output: the last ``last`` positions of every row (last, B, D) fp32."""
        comm = self.comm
        T, n, D = (int(s) for s in y.shape)
        if last < T:
            y = y[T - last:]
        part, part_peer, flag, flag_peer = comm.bulk(last * ((B + 1) // 2), D)   # the same size on both GPUs
        part[:last * n].copy_(y.reshape(last * n, D))
        tp_release(flag_peer, comm.epoch)
        return tp_prefill_gather(y, part_peer, flag, comm.epoch, lo, B)

    def _readout(self, hidden: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        """hidden (B, R, D) fp32 -> the balanced logits (B, R, V) bf16, gathered from both GPUs; ``greedy``: the
        greedy tokens (B, R, 1) int64 instead."""
        B, R, D = (int(s) for s in hidden.shape)
        rows = hidden.reshape(B * R, D)
        o = _ops.namespace(self.impl)
        local = self.model.backbone.balance_stack.balance(o.linear(rows, self.readout_weight), rows)
        if greedy:
            value, index = local.max(dim=-1)
            send, recv = self.comm.region(2 * B * R)
            return tp_greedy_pick(value, index, self.vocab_lo, send, recv).view(B, R, 1)
        v0, v1 = self.vocab_widths
        width = max(v0, v1)
        send = torch.zeros((B * R, width), dtype=torch.bfloat16, device=rows.device)
        send[:, :int(local.shape[1])].copy_(local)
        both = torch.empty((2 * B * R, width), dtype=torch.bfloat16, device=rows.device)
        dist.all_gather_into_tensor(both, send, group=self.comm.group)
        logits = torch.cat([both[:B * R, :v0], both[B * R:, :v1]], dim=1)
        return logits.view(B, R, v0 + v1)

    def agree_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Rank 0's sampled tokens on both GPUs (``decode`` calls it after sampling: the two processes' random streams
        differ, the logits do not)."""
        t = tokens.contiguous()
        dist.broadcast(t, src=dist.get_global_rank(self.comm.group, 0), group=self.comm.group)
        return t

    def forward(self, input_ids: torch.Tensor, inference_params: Optional[InferenceParams] = None,
                num_last_tokens: int = 0, greedy: bool = False) -> CausalLMOutput:
        """``greedy``: the output's ``logits`` holds the greedy next tokens (B, R, 1) int64 (see
        :class:`GreedyTokens`)."""
        if inference_params is None:
            raise ValueError("TensorParallelLM: inference only (pass an inference cache)")
        n = int(num_last_tokens)
        if n < 0:
            raise ValueError(f"num_last_tokens must be >= 0, got {num_last_tokens}")
        if greedy:                                           # checked before the cache advances
            B, T = (int(s) for s in input_ids.shape)
            R = T if n == 0 else min(n, T)
            if 2 * B * R > DECODE_SITE_WORDS:
                raise ValueError(f"TensorParallelLM: greedy selection takes at most {DECODE_SITE_WORDS // 2} rows "
                                 f"(batch x positions read out), got {B} x {R}")
        with regime("infer"):
            hidden = self._backbone(input_ids, inference_params, n)
            if greedy:
                self.comm._prefill = False                    # the selection's words: a decode-range site
            logits = self._readout(hidden, greedy)
        return CausalLMOutput(logits=logits)


class GreedyTokens(nn.Module):
    """A :class:`TensorParallelLM` whose forward returns the greedy next tokens in place of the logits (the output's
    ``logits`` (B, R, 1) int64): captured by ``DecodeGraph`` (``prepare_decode_graph(GreedyTokens(tp), ...)``), its
    ``step(tokens)`` returns the next tokens (B, 1).  Own cache and graphs; the weights are the model's."""

    def __init__(self, tp: TensorParallelLM) -> None:
        super().__init__()
        self.tp = tp
        self.config = tp.config
        self._decode_graph = None
        self._prefill_graph = None

    def prepare_inference(self) -> "GreedyTokens":
        self.tp.prepare_inference()
        return self

    def allocate_inference_cache(self, *args, **kwargs) -> InferenceParams:
        return self.tp.allocate_inference_cache(*args, **kwargs)

    def forward(self, input_ids: torch.Tensor, inference_params: Optional[InferenceParams] = None,
                num_last_tokens: int = 0) -> CausalLMOutput:
        return self.tp(input_ids, inference_params, num_last_tokens, greedy=True)


def tensor_parallel(model: NeuronSparkLMHeadModel, group=None) -> Union[NeuronSparkLMHeadModel, TensorParallelLM]:
    """The model split across the two GPUs of ``group`` (default: the world); the model itself when the group has one
    process (or none is initialised)."""
    if not dist.is_available() or not dist.is_initialized():
        return model
    g = group if group is not None else dist.group.WORLD
    if dist.get_world_size(g) == 1:
        return model
    d, p = mailbox_sites(model)
    return TensorParallelLM(model, TensorParallelComm(g, device=model.lm_head_weight.device, decode_sites=d,
                                                      prefill_sites=p))
