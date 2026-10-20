"""Launch-time calibrations of a freshly launched 1MIX model.

A model built with ``init="launch"`` is calibrated once, before training, on the first 64 training rows (e.g. 8
batches of (8, 1024) token ids), in the whole-row regime under ``torch.no_grad()`` and ``regime("train")``
(bf16 autocast, TF32 on).  A loaded checkpoint carries the calibrated values (``pkm_calibrated = 1``); every function
here refuses such a model.  The calibrations write only ``input_neuron2.v_th``, ``pkm_subkeys``, ``pkm_o.weight`` and
the PKM buffers (``pkm_copy_tau``, ``pkm_calibrated``, ``pkm_launch_rms_res`` / ``_rms_y`` / ``_entropy``).

:func:`calibrate_fresh_model` runs, in order:

1. :func:`calibrate_plif_threshold_match`: for every memory layer in depth order (the shallower layers already
   matched), the input x of ``input_neuron2`` is recorded over the calibration rows, the gate operands are recomputed
   in fp32 outside autocast (``z = w_eta x + b_eta + eta_U (eta_V xc)`` with xc the strictly-past-centred input,
   ``eta = sigmoid(z)``, ``a = 1 - eta``, ``drive = eta x``), and each channel's threshold ``v_th`` is bisected
   (24 halvings on ``[1e-4, 2 max|x| + 1]``) on the single-step recurrence ``u = a V + drive``, bipolar release
   ``S = max(u - v_th, 0) - max(-u - v_th, 0)``, soft reset ``V = u - S`` until its release fraction over the tokens
   beyond the first 64 equals the target: ``plif_threshold_match_p`` when positive, otherwise the mean release of the
   population's fastest quartile of channels (by launch time constant ``1 / (1 - beta)``).  Channels that cannot reach
   the target keep their launch threshold.
2. :func:`calibrate_pkm`: pass A records the unit half-queries of every product-key memory and the copy detector's
   per-position maxima; the sub-keys become the unit centroids of a spherical k-means of the half-queries (fp64 on the
   CPU, k-means++ seeding from a generator seeded ``mechanism_init_seed(996_500, layer_idx)``,
   ``memory_pkm_kmeans_iters`` Lloyd iterations, empty clusters re-seeded); ``pkm_copy_tau`` becomes the 0.9 quantile
   of the maxima.  Pass B (with the calibrated sub-keys) measures rms(y - mean_c y) of the read and rms of the
   attention residual; ``pkm_o.weight *= memory_pkm_launch_ratio * rms(res) / rms(y)``; the buffers record the launch
   values and ``pkm_calibrated = 1``.

During every calibration forward a product-key memory that is not calibrated adds nothing to its layer and the layer's
read gain is not applied (the base function), so each layer is calibrated on the operands of the calibrated shallower
layers and of the base function.

Calibration capture object (:class:`LaunchCalibration`): the ``calib`` argument of the whole-row forward, see the
protocol in :mod:`neuronspark.modules.pkm`.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from neuronspark import ops as _ops
from neuronspark.modules import mechanism_init_seed
from neuronspark.modules.pkm import PKM_CALIB_MODES
from neuronspark.utils.precision import regime

__all__ = [
    "LaunchCalibration",
    "calibrate_fresh_model",
    "calibrate_plif_threshold_match",
    "calibrate_pkm",
    "spherical_kmeans",
    "pkm_new_stats",
    "pkm_accumulate_stats",
    "pkm_stats_summary",
    "CALIBRATION_ROWS",
    "THRESHOLD_SKIP",
    "THRESHOLD_ITERATIONS",
    "PKM_KMEANS_SEED_BASE",
    "COPY_TAU_QUANTILE",
]

# rows of the launch calibration (the first 64 training rows)
CALIBRATION_ROWS = 64
# threshold matching: tokens of a row before the release fraction is counted; bisection halvings; the lower end of the
# threshold interval
THRESHOLD_SKIP = 64
THRESHOLD_ITERATIONS = 24
THRESHOLD_LO = 1e-4
# fastest-quartile target of the threshold matching
FAST_QUANTILE = 0.25
# k-means++ seeding stream of the sub-keys: mechanism_init_seed(996_500, layer_idx)
PKM_KMEANS_SEED_BASE = 996_500
# quantile of the copy detector's launch maxima that becomes pkm_copy_tau
COPY_TAU_QUANTILE = 0.90


# ================================================================================================ capture object
class LaunchCalibration:
    """``calib`` of the whole-row forward: ``mode`` None (calibration active, nothing captured: every uncalibrated
    product-key memory adds nothing), ``"queries"`` (pass A) or ``"rms"`` (pass B) for the ``layers`` given.

    ``pkm_mode(layer)`` returns the mode for a captured layer, None otherwise; ``record(layer, name, value)`` stores
    ``value`` under ``name`` (``"pkm_usage"`` is accumulated into the layer's usage statistics instead).
    """

    def __init__(self, mode: Optional[str] = None, layers: Sequence[Any] = ()) -> None:
        if mode not in PKM_CALIB_MODES:
            raise ValueError(f"LaunchCalibration: mode must be one of {PKM_CALIB_MODES}, got {mode!r}")
        if mode is None and layers:
            raise ValueError("LaunchCalibration: mode None captures no layer")
        self.mode = mode
        self._layers = {id(l): l for l in layers}
        self.records: Dict[int, Dict[str, List[Any]]] = {id(l): {} for l in layers}
        self.usage: Dict[int, dict] = {id(l): pkm_new_stats(l) for l in layers} if mode == "rms" else {}

    def pkm_mode(self, layer) -> Optional[str]:
        if self.mode is None or id(layer) not in self._layers:
            return None
        return self.mode

    def record(self, layer, name: str, value) -> None:
        key = id(layer)
        if key not in self._layers:
            raise RuntimeError("LaunchCalibration: a layer outside the calibration recorded a value")
        if name == "pkm_usage":
            if self.mode != "rms":
                raise RuntimeError("LaunchCalibration: usage statistics are recorded in pass 'rms' only")
            u, w, ii, jj = value
            pkm_accumulate_stats(self.usage[key], u, w, ii, jj, int(layer.pkm_n))
            return
        self.records[key].setdefault(name, []).append(value)

    def take(self, layer, name: str) -> List[Any]:
        """The values recorded for ``layer`` under ``name`` (removed from the object)."""
        return self.records[id(layer)].pop(name, [])


# ================================================================================================ PKM usage statistics
def pkm_new_stats(layer) -> dict:
    """Usage accumulator of one product-key memory: per-slot selection counts and softmax weight mass, per (head,
    half, sub-key) selection counts, the summed entropy of the per-(token, head) top-k softmax."""
    dev = layer.pkm_values.device
    H, n = int(layer.pkm_heads), int(layer.pkm_n)
    return {"slot_count": torch.zeros(n * n, dtype=torch.float64, device=dev),
            "slot_mass": torch.zeros(n * n, dtype=torch.float64, device=dev),
            "subkey_selected": torch.zeros(H * 2 * n, dtype=torch.float64, device=dev),
            "entropy_sum": torch.zeros((), dtype=torch.float64, device=dev),
            "token_heads": torch.zeros((), dtype=torch.float64, device=dev),
            "H": H, "n": n}


def pkm_accumulate_stats(st: dict, u: torch.Tensor, w: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor,
                         n: int) -> None:
    """Add one call's selections ``u``, weights ``w`` and sub-key indices ``ii`` / ``jj`` (T, B, H, k) to ``st``."""
    H = int(u.shape[-2])
    u, w = u.reshape(-1, H, u.shape[-1]), w.reshape(-1, H, w.shape[-1]).double()
    ii, jj = ii.reshape(-1, H, ii.shape[-1]), jj.reshape(-1, H, jj.shape[-1])
    N = st["slot_count"].numel()
    st["slot_count"] += torch.bincount(u.reshape(-1), minlength=N).double()
    st["slot_mass"] += torch.bincount(u.reshape(-1), weights=w.reshape(-1), minlength=N)
    hoff = (torch.arange(H, device=u.device) * 2 * n).view(1, H, 1)
    sel = torch.cat([(hoff + ii).reshape(-1), (hoff + n + jj).reshape(-1)])
    st["subkey_selected"] += torch.bincount(sel, minlength=H * 2 * n).double()
    st["entropy_sum"] += -(w * torch.log(w.clamp_min(1e-300))).sum()
    st["token_heads"] += float(w.shape[0] * H)


def pkm_stats_summary(st: dict) -> dict:
    """Usage measures of an accumulator: slot coverage, normalised usage entropy of the weight mass and of the counts,
    the weight share of the top 1% of slots, sub-key coverage (overall and the minimum over head-halves), the mean
    entropy of the top-k softmax per (token, head)."""
    N = st["slot_count"].numel()
    cnt, mass = st["slot_count"].double().cpu(), st["slot_mass"].double().cpu()
    p = mass / mass.sum().clamp_min(1e-300)
    ent = float(-(p[p > 0] * torch.log(p[p > 0])).sum() / math.log(N))
    pc = cnt / cnt.sum().clamp_min(1e-300)
    ent_c = float(-(pc[pc > 0] * torch.log(pc[pc > 0])).sum() / math.log(N))
    top = max(1, int(math.ceil(0.01 * N)))
    sk = st["subkey_selected"].double().cpu().view(st["H"], 2, st["n"]) > 0
    return {"tokens_heads": float(st["token_heads"]), "slot_coverage": float((cnt > 0).double().mean()),
            "usage_entropy_mass": ent, "usage_entropy_count": ent_c,
            "top1pct_mass_share": float(torch.sort(p, descending=True).values[:top].sum()),
            "subkey_coverage": float(sk.double().mean()),
            "subkey_coverage_min_head_half": float(sk.double().mean(dim=-1).min()),
            "softmax_entropy_mean": float(st["entropy_sum"]) / max(float(st["token_heads"]), 1.0)}


# ================================================================================================ helpers
def _backbone(model):
    """The :class:`NeuronSparkModel` of a model (an LM head model or the backbone itself)."""
    return model.backbone if hasattr(model, "backbone") else model


def _refuse_calibrated(backbone) -> None:
    done = [l.layer_idx for l in backbone.memory_layers if float(l.pkm_calibrated) > 0.5 or l.pkm_ready]
    if done:
        raise RuntimeError(f"launch calibration: memory layers {done} are calibrated already (a loaded checkpoint "
                           "carries its calibrated values); the calibration runs on a fresh launch only")


def _check_batches(backbone, batches: Sequence[torch.Tensor], rows: Optional[int]) -> List[torch.Tensor]:
    if not isinstance(batches, (list, tuple)) or not batches:
        raise ValueError("launch calibration: batches must be a non-empty list of (B, T) token-id tensors")
    dev = backbone.embed_tokens.weight.device
    chunk = int(backbone.config.sd_chunk_size)
    out = []
    for b in batches:
        if not isinstance(b, torch.Tensor) or b.dim() != 2 or b.dtype not in (torch.int64, torch.int32):
            raise ValueError("launch calibration: every batch must be a (B, T) int64 token-id tensor")
        if b.device != dev:
            raise ValueError(f"launch calibration: a batch on {b.device}, the model on {dev}")
        if backbone.impl == "kernel" and int(b.shape[1]) % chunk != 0:
            raise ValueError(f"launch calibration: T = {int(b.shape[1])} is not a multiple of sd_chunk_size {chunk}")
        out.append(b.to(torch.int64))
    total = sum(int(b.shape[0]) for b in out)
    if rows is not None and total != int(rows):
        raise ValueError(f"launch calibration: the batches hold {total} rows, the calibration uses {int(rows)}")
    return out


@torch.no_grad()
def _run_forward(backbone, batches: Sequence[torch.Tensor], calib: LaunchCalibration) -> None:
    """One whole-row forward of the backbone per batch (each row from its zero state) with ``calib``."""
    for ids in batches:
        backbone(ids, calib=calib)


def _strict_past_centred(o, x: torch.Tensor) -> torch.Tensor:
    """(T, B, C) fp32 ``x_t - mean_{j<t} x_j`` (zero at t = 0): the inclusive cumulative sum shifted by one position
    divided by max(t, 1)."""
    T = int(x.shape[0])
    xf = x.float()
    xc = o.cumsum0(xf)
    xc = torch.cat([torch.zeros_like(xc[:1]), xc[:-1]], dim=0)
    xn = torch.arange(T, device=x.device, dtype=torch.float32).clamp_min(1.0).view(T, 1, 1)
    has_past = (torch.arange(T, device=x.device) > 0).view(T, 1, 1).to(torch.float32)
    return (xf - xc / xn) * has_past


def spherical_kmeans(X: torch.Tensor, k: int, iters: int, gen: torch.Generator) -> Tuple[torch.Tensor, dict]:
    """Spherical k-means of the unit rows ``X`` (N, d) float64 CPU: k-means++ seeding on the cosine distance
    ``1 - cos`` (generator ``gen``), then ``iters`` Lloyd iterations (assign by the largest cosine, centroid = the
    normalised mean).  An empty cluster is re-seeded at the points of lowest best cosine (stable order); after the last
    iteration a centroid that is nobody's nearest is moved onto such a point (at most k repair rounds).  Returns
    ``(C (k, d) unit float64, info)``."""
    N = int(X.shape[0])
    if N < k:
        raise RuntimeError(f"PKM k-means: {N} calibration queries for {k} sub-keys")
    first = int(torch.randint(N, (1,), generator=gen))
    idx = [first]
    best = X @ X[first]
    for _ in range(1, k):
        d2 = (1.0 - best).clamp_min(0.0).pow(2)
        if not float(d2.sum()) > 0.0:
            raise RuntimeError("PKM k-means: fewer distinct calibration queries than sub-keys")
        i = int(torch.multinomial(d2, 1, generator=gen))
        idx.append(i)
        best = torch.maximum(best, X @ X[i])
    C = X[idx].clone()
    reseeded = 0
    for _ in range(int(iters)):
        sim = X @ C.t()
        bval, a = sim.max(dim=1)
        S = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=k)
        empty = torch.nonzero(cnt == 0).reshape(-1)
        if empty.numel():
            far = torch.sort(bval, stable=True).indices[:empty.numel()]
            S[empty] = X[far]
            reseeded += int(empty.numel())
        C = F.normalize(S, dim=1)
    repaired = 0
    for _ in range(k):
        sim = X @ C.t()
        bval, a = sim.max(dim=1)
        cnt = torch.bincount(a, minlength=k)
        empty = torch.nonzero(cnt == 0).reshape(-1)
        if not empty.numel():
            break
        far = torch.sort(bval, stable=True).indices[:empty.numel()]
        C[empty] = X[far]
        repaired += int(empty.numel())
    sim = X @ C.t()
    bval, a = sim.max(dim=1)
    cnt = torch.bincount(a, minlength=k)
    return C, {"points": N, "reseeded": reseeded, "repaired": repaired, "nonempty": int((cnt > 0).sum()),
               "mean_best_cos": float(bval.mean()), "max_cluster_frac": float(cnt.max()) / N}


# ================================================================================================ threshold matching
def _capture_neuron_input(backbone, neuron, batches: Sequence[torch.Tensor]) -> torch.Tensor:
    """(T, rows, C) fp32: the input of ``neuron`` over the calibration rows (one calibration forward)."""
    store: List[torch.Tensor] = []
    hook = neuron.register_forward_pre_hook(lambda mod, args: store.append(args[0].detach().float()))
    try:
        _run_forward(backbone, batches, LaunchCalibration(None))
    finally:
        hook.remove()
    if not store:
        raise RuntimeError("threshold matching: the population saw no calibration rows")
    x = torch.cat(store, 1)
    if not bool(torch.isfinite(x).all()):
        raise RuntimeError("threshold matching: non-finite calibration operands")
    return x


def _plif_gate_operands(o, neuron, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(a, drive) fp32 of the single-step recurrence from the recorded input x (autocast disabled by the caller)."""
    z = neuron.w_eta.float() * x + neuron.b_eta.float()
    xc = _strict_past_centred(o, x).to(x.dtype) if neuron.centred else x
    z = z + F.linear(F.linear(xc, neuron.eta_V), neuron.eta_U).to(torch.float32)
    eta = torch.sigmoid(z)
    a = 1.0 - eta
    drive = eta * x
    return a, drive


def _simulate_release(a: torch.Tensor, drive: torch.Tensor, theta: torch.Tensor, skip: int) -> Tuple[torch.Tensor, float]:
    """Per-channel release fraction over the tokens t >= ``skip`` and the spike energy of the bipolar soft-reset
    recurrence with thresholds ``theta`` (C,)."""
    th = theta.view(1, 1, -1)
    T = int(drive.shape[0])
    V = torch.zeros_like(drive[0])
    released = torch.zeros(drive.shape[-1], device=drive.device)
    energy = torch.zeros((), device=drive.device, dtype=torch.float64)
    thr = th[0]
    for t in range(T):
        u = a[t] * V + drive[t]
        S = torch.clamp(u - thr, min=0.0)
        S = S - torch.clamp(-u - thr, min=0.0)
        V = u - S
        if t >= skip:
            released += (S != 0).float().mean(0)
            energy += S.double().pow(2).sum()
    return released / max(T - skip, 1), float(energy)


@torch.no_grad()
def calibrate_plif_threshold_match(model, batches: Sequence[torch.Tensor], skip: int = THRESHOLD_SKIP,
                                   iterations: int = THRESHOLD_ITERATIONS) -> Dict[str, dict]:
    """Threshold matching of every memory layer's ``input_neuron2`` (step 1 of the module docstring).  ``batches``:
    the calibration rows as (B, T) token-id tensors on the model's device.  Returns the per-population report."""
    backbone = _backbone(model)
    _refuse_calibrated(backbone)
    batches = _check_batches(backbone, batches, None)
    o = _ops.namespace(backbone.impl)
    p_fixed = float(backbone.config.plif_threshold_match_p)
    names = {id(m): n for n, m in backbone.named_modules()}
    report: Dict[str, dict] = {}
    for layer in backbone.memory_layers:
        neuron = layer.input_neuron2
        name = f"{names[id(layer)]}.input_neuron2"
        x = _capture_neuron_input(backbone, neuron, batches)
        with regime("train"), torch.autocast(x.device.type, enabled=False):
            a, drive = _plif_gate_operands(o, neuron, x)
            theta0 = neuron.v_th.detach().float().clone()
            rate0, energy0 = _simulate_release(a, drive, theta0, skip)
            if not energy0 > 0.0:
                raise RuntimeError(f"{name}: the launch thresholds never release")
            tau = 1.0 / (1.0 - neuron.beta.detach().float()).clamp_min(1e-6)        # launch time constant per channel
            fast = tau <= torch.quantile(tau, FAST_QUANTILE)
            fast_rate = float(rate0[fast].mean())
            target = p_fixed if p_fixed > 0.0 else fast_rate
            lo = torch.full_like(theta0, THRESHOLD_LO)
            hi = x.abs().amax(dim=(0, 1)).clamp_min(1e-3) * 2.0 + 1.0
            rate_max, _ = _simulate_release(a, drive, lo, skip)
            matchable = rate_max >= target
            if not bool(matchable.any()):
                raise RuntimeError(f"{name}: no channel can reach the target release rate {target}")
            for _ in range(int(iterations)):                                         # release falls as theta rises
                mid = (lo + hi) / 2
                rate, _ = _simulate_release(a, drive, mid, skip)
                too_many = rate > target
                lo = torch.where(too_many, mid, lo)
                hi = torch.where(too_many, hi, mid)
            theta = torch.where(matchable, (lo + hi) / 2, theta0)
            rate, energy = _simulate_release(a, drive, theta, skip)
            if not energy > 0.0:
                raise RuntimeError(f"{name}: degenerate spike energy after matching")
        neuron.v_th.copy_(theta.to(neuron.v_th.dtype))
        q = torch.tensor([.05, .5, .95], device=theta.device)
        report[name] = {"rows": int(x.shape[1]), "target": float(target), "fast_quartile_release": fast_rate,
                        "release_launch": float(rate0.mean()), "release_matched": float(rate.mean()),
                        "release_matched_matchable": float(rate[matchable].mean()),
                        "matchable_frac": float(matchable.float().mean()),
                        "release_matched_min": float(rate.min()), "release_matched_max": float(rate.max()),
                        "theta_launch_q05_50_95": [float(v) for v in torch.quantile(theta0, q)],
                        "theta_matched_q05_50_95": [float(v) for v in torch.quantile(theta, q)],
                        "tau_q05_50_95": [float(v) for v in torch.quantile(tau, q)],
                        "spike_energy_ratio": float(energy / energy0), "rms_x": float(x.double().pow(2).mean().sqrt())}
        del x, a, drive
    return report


# ================================================================================================ PKM calibration
@torch.no_grad()
def calibrate_pkm(model, batches: Sequence[torch.Tensor]) -> Dict[str, dict]:
    """Product-key memory calibration of every memory layer (step 2 of the module docstring).  ``batches``: the
    calibration rows as (B, T) token-id tensors on the model's device.  Returns the per-layer report."""
    backbone = _backbone(model)
    _refuse_calibrated(backbone)
    batches = _check_batches(backbone, batches, None)
    layers = backbone.memory_layers
    if not layers:
        raise ValueError("calibrate_pkm: the model has no memory layer")
    names = {id(m): n for n, m in backbone.named_modules()}
    report: Dict[str, dict] = {names[id(l)]: {} for l in layers}
    # ---- pass A: launch half-queries -> spherical k-means sub-keys; copy maxima -> pkm_copy_tau
    calib = LaunchCalibration("queries", layers)
    _run_forward(backbone, batches, calib)
    for l in layers:
        n = names[id(l)]
        qs = calib.take(l, "pkm_queries")
        if not qs:
            raise RuntimeError(f"{n}: no calibration tokens reached the product-key memory")
        Q = torch.cat(qs, 0)                                                   # (tokens, H, 2, dh) fp32 CPU
        if not bool(torch.isfinite(Q).all()):
            raise RuntimeError(f"{n}: non-finite PKM query during calibration")
        maxima = torch.cat(calib.take(l, "copy_max"))
        maxima = maxima[torch.isfinite(maxima)]
        if not maxima.numel():
            raise RuntimeError(f"{n}: no copy source positions in the calibration rows")
        tau = torch.quantile(maxima.float(), COPY_TAU_QUANTILE)
        l.pkm_copy_tau.copy_(tau.to(l.pkm_copy_tau.device))
        report[n]["copy_gate_tau"] = float(tau)
        gen = torch.Generator().manual_seed(mechanism_init_seed(PKM_KMEANS_SEED_BASE, l.layer_idx))
        keys = torch.empty((l.pkm_heads, 2, l.pkm_n, l.pkm_half), dtype=torch.float64)
        infos = []
        for h in range(l.pkm_heads):
            for a in range(2):
                C, info = spherical_kmeans(Q[:, h, a].double(), l.pkm_n, l.pkm_kmeans_iters, gen)
                keys[h, a] = C
                infos.append(info)
        l.pkm_subkeys.copy_(keys.to(device=l.pkm_subkeys.device, dtype=l.pkm_subkeys.dtype))
        report[n].update({"tokens": int(Q.shape[0]),
                          "kmeans_nonempty_min": min(i["nonempty"] for i in infos),
                          "kmeans_reseeded": sum(i["reseeded"] for i in infos),
                          "kmeans_repaired": sum(i["repaired"] for i in infos),
                          "kmeans_mean_best_cos": sum(i["mean_best_cos"] for i in infos) / len(infos),
                          "kmeans_max_cluster_frac": max(i["max_cluster_frac"] for i in infos)})
        del Q, qs
    del calib
    # ---- pass B: rms of the read and of the attention residual -> pkm_o scale; usage with the calibrated sub-keys
    calib = LaunchCalibration("rms", layers)
    _run_forward(backbone, batches, calib)
    for l in layers:
        n = names[id(l)]
        cy, cr = calib.take(l, "ms_y"), calib.take(l, "ms_res")
        if not cy or not cr:
            raise RuntimeError(f"{n}: no calibration tokens reached the product-key memory (pass B)")
        ms_y, ms_r = torch.cat(cy), torch.cat(cr)
        if not (bool(torch.isfinite(ms_y).all()) and bool(torch.isfinite(ms_r).all())):
            raise RuntimeError(f"{n}: non-finite PKM read or attention residual during calibration")
        rms_y, rms_r = float(ms_y.mean().sqrt()), float(ms_r.mean().sqrt())
        if not (rms_y > 0.0 and rms_r > 0.0):
            raise RuntimeError(f"{n}: zero read on the calibration rows (rms y {rms_y}, rms residual {rms_r})")
        s = l.pkm_launch_ratio * rms_r / rms_y
        l.pkm_o.weight.mul_(s)
        usage = pkm_stats_summary(calib.usage[id(l)])
        l.pkm_launch_rms_res.copy_(torch.tensor(rms_r, dtype=torch.float32))
        l.pkm_launch_rms_y.copy_(torch.tensor(rms_y * s, dtype=torch.float32))
        l.pkm_launch_entropy.copy_(torch.tensor(usage["softmax_entropy_mean"], dtype=torch.float32))
        l.pkm_calibrated.copy_(torch.ones((), dtype=torch.float32))
        l.mark_pkm_ready()
        report[n].update({"rms_res_attn": rms_r, "rms_y_draw": rms_y, "scale": s, "rms_y_launch": rms_y * s,
                          "ratio_launch": rms_y * s / rms_r, **usage})
    return report


# ================================================================================================ the launch sequence
def _check_launch_snapshot(backbone) -> None:
    versions = getattr(backbone, "_launch_versions", None)
    if versions is None:
        raise RuntimeError("calibrate_fresh_model: the model was not built with init='launch' (a loaded checkpoint "
                           "carries its calibrated values)")
    now = backbone.parameter_versions()
    changed = sorted(n for n, v in now.items() if versions.get(n) != v)
    if changed or set(now) != set(versions):
        raise RuntimeError(f"calibrate_fresh_model: {len(changed)} parameters were modified since the launch "
                           f"(e.g. {changed[:5]}); the calibration runs on the launch values")


def calibrate_fresh_model(model, batches: Sequence[torch.Tensor], rows: Optional[int] = CALIBRATION_ROWS
                          ) -> Dict[str, Dict[str, dict]]:
    """The launch calibration of a fresh model: :func:`calibrate_plif_threshold_match`, then :func:`calibrate_pkm`,
    on ``batches`` (the first training rows as (B, T) token-id tensors on the model's device; together exactly
    ``rows`` rows unless ``rows`` is None).  Refuses a model whose parameters changed since its construction.  Returns
    ``{"plif_threshold_match": ..., "pkm": ...}``."""
    backbone = _backbone(model)
    _check_launch_snapshot(backbone)
    _refuse_calibrated(backbone)
    batches = _check_batches(backbone, batches, rows)
    threshold = calibrate_plif_threshold_match(model, batches)
    pkm = calibrate_pkm(model, batches)
    return {"plif_threshold_match": threshold, "pkm": pkm}
