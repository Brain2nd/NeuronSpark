"""Why phase A acquires nothing: the write at a fact's later mention and the read at its probe, per hippocampal layer.

  step    = sum_m w_m^2 / (lam + sum_j w_j^2 + usage[u_m])   over the written slots: the fraction of the gap one
            token's RLS write closes
  moved   = <read(V+dV) - read(V), val* - read(V)> / |val* - read(V)|^2   at the write token after the whole document
  overlap = sum_{s addressed at both} min(w_write[s], w_probe[s])         the slot overlap write-time vs probe-time
  dread   = |read_probe(V+dV) - read_probe(V)| / |read_probe(V)|          what the probe's read sees of the writes
and an oracle: dV := the exact one-token least-squares write (no lam, no usage, all 64 slots) at the fact tokens only,
then the probes again -- whether the PKM path can carry the answer at all when the write is complete.
One short session (4 entities x 4 attributes, m mentions, no distractor text) on an arm's final checkpoint."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

B_ = "/workspace/neuronspark-gb300"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--lam", type=float, default=10.0)
ap.add_argument("--write-topm", type=int, default=2)
ap.add_argument("--prior-windows", type=int, default=32)
ap.add_argument("--prior-scale", type=float, default=1.0)
ap.add_argument("--entities", type=int, default=4)
ap.add_argument("--mentions", type=int, default=2)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--session", type=int, default=0)
ap.add_argument("--cap-gib", type=float, default=12.0)
ap.add_argument("--out", default=None)
ap.add_argument("--neuronspark", default=f"{B_}/tmp/stage/NeuronSpark-exp")
ap.add_argument("--data", default=f"{B_}/data_work/babylm/tok")
ap.add_argument("--tokenizer", default=f"{B_}/tokenizer/NeuronSpark-V4-1.16B-Pretrain-Qwen3VL/tokenizer.json")
a = ap.parse_args()
sys.path.insert(0, a.neuronspark); sys.path.insert(0, f"{a.neuronspark}/exp/consolidation")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.2/bin/ptxas")
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from neuronspark import NeuronSparkConfig
from neuronspark.models.neuronspark_lm import NeuronSparkLMHeadModel
from neuronspark.modules.consolidation import Consolidation
from neuronspark.utils.precision import regime
import facts as FX

torch.cuda.set_per_process_memory_fraction(a.cap_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
dev = torch.device("cuda")
torch.manual_seed(a.seed)
path = Path(a.ckpt)
cfg = NeuronSparkConfig.from_dict(json.load(open(path / "config.json")))
model = NeuronSparkLMHeadModel(cfg, device=dev, init="empty")
model.load_state_dict({k: v.to(dev) for k, v in load_file(str(path / "model.safetensors")).items()}, strict=True)
model.eval(); torch.cuda.empty_cache()
tok = Tokenizer.from_file(a.tokenizer)
dstream = np.load(f"{a.data}/dev.u32.npy", mmap_mode="r")
CH = int(cfg.dendritic_chunk_size)
hippo = [m for m in model.modules() if m.__class__.__name__ == "HippocampalLayer"]

# ---- recording wrappers (class level, before the instances exist)
REC = {"on": False, "doc": False, "reads": [], "writes": [], "pool": [], "gates": []}            # reads: (layer_idx, w (T,B,Hk), u); writes: per chunk dicts
_orig_rw, _orig_wc = Consolidation.read_write, Consolidation.write_chunk
def _rw(self, ops, w, u, glog, x_hat, r_pool, gates):
    if REC["on"]:
        REC["reads"].append((self._idx, w.detach().float().cpu(), u.detach().cpu()))
        if gates is not None:
            REC["gates"].append((self._idx, gates[0].detach().float().cpu(), gates[1].detach().float().cpu()))
        if gates is not None:
            REC["pool"].append((self._idx, r_pool[:, 0].detach().float().cpu(), gates[0][:, 0].detach().float().cpu()))
    return _orig_rw(self, ops, w, u, glog, x_hat, r_pool, gates)
def _wc(self, u, w, val, val_star, captured=None, val0=None):
    if REC["on"] and self.write and not self.collect:
        wm = w
        if self.write_topm is not None:
            H = self.layer.pkm_heads
            wh = w.reshape(w.shape[0], H, -1)
            keep = torch.zeros_like(wh, dtype=torch.bool)
            keep.scatter_(-1, wh.topk(min(self.write_topm, wh.shape[-1]), dim=-1).indices, True)
            wm = torch.where(keep.reshape(w.shape), w, torch.zeros_like(w))
        denom = self.lam + wm.pow(2).sum(dim=-1, keepdim=True) + self.usage[u]
        step = (wm.pow(2) / denom).sum(dim=-1)                              # (N,)
        REC["writes"].append({"layer": self._idx, "u": u.cpu(), "w": w.float().cpu(), "wm": wm.float().cpu(),
                              "val": val.float().cpu(), "val_star": val_star.float().cpu(),
                              "val0": (val0 if val0 is not None else val).float().cpu(), "step": step.float().cpu()})
    return _orig_wc(self, u, w, val, val_star, captured, val0)
Consolidation.read_write, Consolidation.write_chunk = _rw, _wc

cons = []
for i, layer in enumerate(hippo):
    c = Consolidation(layer, lam=a.lam, chunk=CH, write=True, write_topm=a.write_topm)
    c._idx = i
    layer.consolidation = c
    cons.append(c)
from neuronspark.ops.triton.pkm_read import pkm_value_read

def pad(ids):
    L = len(ids); width = -(-L // CH) * CH + 1
    x = torch.full((1, width), FX.EOT, dtype=torch.int64); x[0, :L] = torch.tensor(ids, dtype=torch.int64)
    return x[:, :-1].to(dev), L - 1

# ---- usage prior exactly as the harness
if a.prior_windows:
    idx = np.sort(np.random.default_rng(a.seed + 3).permutation(len(dstream) // (a.seq + 1))[:a.prior_windows])
    for c in cons: c.collect = True
    with torch.no_grad():
        for i in range(0, len(idx), 4):
            w_ = torch.from_numpy(np.stack([np.asarray(dstream[int(k) * (a.seq + 1):(int(k) + 1) * (a.seq + 1)], dtype=np.int64)
                                            for k in idx[i:i + 4]])).to(dev)
            with regime("train"):
                model.per_token_ce(w_[:, :-1].contiguous(), w_[:, 1:].contiguous())
    for c in cons:
        c.collect = False; c.prime(a.prior_scale)

values = FX.single_token_values(tok)
sess = FX.build_session(a.session, tok, dstream, n_entities=a.entities, mentions=a.mentions, distractor_tokens=0, seed=a.seed,
                        values=values)
doc = [int(t) for t in sess.doc_ids]
print(f"session {a.session}: {len(sess.facts)} facts, doc {len(doc)} tokens, lam {a.lam}, topm {a.write_topm}, prior {a.prior_windows}", flush=True)

# ---- 1. read the document with the writes on, recording
REC["on"] = True; REC["doc"] = True; REC["reads"].clear(); REC["writes"].clear(); REC["pool"].clear(); REC["gates"].clear()
x, _ = pad(doc)
with torch.no_grad(), regime("train"):
    model.backbone(x, aux_out=[])
REC["on"] = False; REC["doc"] = False
POOL = {l: (torch.cat([r_ for ll, r_, _ in REC["pool"] if ll == l]), torch.cat([f_ for ll, _, f_ in REC["pool"] if ll == l]))
        for l in range(len(hippo))}
DOCG = {l: (torch.cat([f_[:, 0] for ll, f_, _ in REC["gates"] if ll == l]), torch.cat([r_[:, 0] for ll, _, r_ in REC["gates"] if ll == l]))
        for l in range(len(hippo))}                                              # per layer: f (T,), rho (T,) over the document
def pi_of(rho): return -torch.logit((rho / 2).clamp(1e-6, 1 - 1e-6)) / 4.0
L = len(hippo)
wr = {l: [d for d in REC["writes"] if d["layer"] == l] for l in range(L)}
cat = {l: {k: torch.cat([d[k] for d in wr[l]]) for k in ("u", "w", "wm", "val", "val_star", "val0", "step")} for l in range(L)}
usage_after = [c.usage.clone() for c in cons]
dV_doc = [c.dV.clone() for c in cons]

LASTPOOL = []                                            # per probe read: {layer: (r at the last position, f)}
def probe_reads(prompts):
    """writes off: per prompt the top-1 and per layer (w, u) at the last position."""
    outs = []
    for c in cons: c.write = False
    for ids in prompts:
        REC["on"] = True; REC["reads"].clear(); REC["gates"].clear(); REC["pool"].clear()
        x, last = pad(ids)
        with torch.no_grad(), regime("train"):
            hidden = model.backbone(x, aux_out=[])
            top = int(model._readout(hidden[:, last:last + 1])[0, 0].argmax())
        REC["on"] = False
        per_layer = {l: (w_[last, 0], u_[last, 0]) for l, w_, u_ in REC["reads"]}
        outs.append((top, per_layer, {l: (f_[last, 0], r_[last, 0]) for l, f_, r_ in REC["gates"]}))
        LASTPOOL.append({l: (r_[last], f_[last]) for l, r_, f_ in REC["pool"]})
        REC["gates"].clear(); REC["pool"].clear()
    for c in cons: c.write = True
    return outs

def rel_read(w_, u_, l, dv):
    V = hippo[l].pkm_values.detach()
    base = pkm_value_read(w_.view(1, 1, -1).to(dev), u_.view(1, 1, -1).to(dev), V)[0, 0]
    after = pkm_value_read(w_.view(1, 1, -1).to(dev), u_.view(1, 1, -1).to(dev), V + dv)[0, 0]
    return base, after

same = [p for p in sess.probes if p["kind"] == "same"]
probes_doc = probe_reads([p["prompt_ids"] for p in same])
LASTPOOL.clear()
ic_outs = probe_reads([doc + p["prompt_ids"][1:] for p in same])
IC_POOL = list(LASTPOOL)
ic_ok = [top == p["fact"].answer_id for p, (top, _, _) in zip(same, ic_outs)]

rows = []
for p, (top, pl, pg) in zip(same, probes_doc):
    f = p["fact"]
    pos = [i for i, t in enumerate(doc) if t == f.answer_id]                       # the value's positions (mentions)
    rec = {"entity": f.entity, "attr": f.attribute, "value": f.value, "mentions_at": pos, "write_correct": top == f.answer_id,
           "per_mention": []}
    for mi, pv in enumerate(pos):
        t = pv - 1                                                                  # the token that predicts the value
        per = []
        for l in range(L):
            C = cat[l]
            u_w, w_w, wm = C["u"][t], C["w"][t], C["wm"][t]
            val0, vs = C["val0"][t], C["val_star"][t]
            base, after = rel_read(w_w, u_w, l, dV_doc[l])
            gap = vs.to(dev) - val0.to(dev)
            moved = float(torch.dot(after - base, gap) / gap.pow(2).sum().clamp_min(1e-12))
            w_p, u_p = pl[l]
            both = {}
            for s, ww in zip(u_w.tolist(), w_w.tolist()): both[s] = ww
            overlap = sum(min(both[s], wp) for s, wp in zip(u_p.tolist(), w_p.tolist()) if s in both)
            pb, pa = rel_read(w_p, u_p, l, dV_doc[l])
            per.append({"step": float(C["step"][t]), "moved": moved, "overlap": overlap,
                        "dread": float((pa - pb).norm() / pb.norm().clamp_min(1e-12)),
                        "gap_over_val0": float(gap.norm() / val0.to(dev).norm().clamp_min(1e-12))})
        rec["per_mention"].append({"mention": mi, "t": t, "layers": per})
    rows.append(rec)

def mean_over(rows, key, mention):
    vals = [lay[key] for r in rows for pm in r["per_mention"] if pm["mention"] == mention for lay in pm["layers"]]
    return float(np.mean(vals)) if vals else float("nan")

acc_write = float(np.mean([r["write_correct"] for r in rows]))
print(f"write arm (lam {a.lam}): same-template accuracy {acc_write:.3f} over {len(rows)} facts", flush=True)
for mi in range(a.mentions):
    print(f"  mention {mi}: step {mean_over(rows, 'step', mi):.4f}  moved {mean_over(rows, 'moved', mi):.4f}  "
          f"overlap {mean_over(rows, 'overlap', mi):.3f}  dread {mean_over(rows, 'dread', mi):.4f}  "
          f"gap/|val0| {mean_over(rows, 'gap_over_val0', mi):.3f}", flush=True)

# copy confidence pi (rho = 2 sigmoid(-4 pi)) at the fact tokens of the document vs all tokens vs the probes
for l in range(L):
    f_doc, rho_doc = DOCG[l]; pi_doc = pi_of(rho_doc)
    t2 = [r["per_mention"][a.mentions - 1]["t"] for r in rows]; t1 = [r["per_mention"][0]["t"] for r in rows]
    pi_probe = torch.stack([pg[l][1] for _, _, pg in probes_doc]); pi_probe = pi_of(pi_probe)
    pi_ic = pi_of(torch.stack([pg[l][1] for _, _, pg in ic_outs]))
    print(f"  layer {l}: pi at mention-1 value tokens {pi_doc[t1].mean():.3f}, at last-mention value tokens {pi_doc[t2].mean():.3f}, "
          f"all doc tokens {pi_doc.mean():.3f}; f at last mention {f_doc[t2].mean():.3f} (doc {f_doc.mean():.3f}); "
          f"pi at the probes {pi_probe.mean():.3f}, in-context probes {pi_ic.mean():.3f}", flush=True)

# ---- 2. oracles: joint minimum-norm least-squares writes (A X = E exactly, 16 facts at once, no lam, no usage)
def joint_write(l, entries):
    """entries: [(u (Hk,), w (Hk,), e (dv,))] -> dV (slots, dv) with sum_m w_m dV[u_m] = e for every entry."""
    V = hippo[l].pkm_values.detach()
    slots = sorted(set(s for u_, _, _ in entries for s in u_.tolist()))
    col = {s: i for i, s in enumerate(slots)}
    A = torch.zeros(len(entries), len(slots), device=dev)
    for r_, (u_, w_, _) in enumerate(entries):
        for s, ww in zip(u_.tolist(), w_.tolist()):
            A[r_, col[s]] += ww
    E = torch.stack([e for *_, e in entries]).to(dev)
    G = A @ A.t() + 1e-6 * torch.eye(len(entries), device=dev)
    X = A.t() @ torch.linalg.solve(G, E)
    dV = torch.zeros_like(V); dV[torch.tensor(slots, device=dev)] = X
    return dV, len(slots)

for variant in ("A: write-time addressing", "B: probe-time addressing"):
    for c in cons: c.dV.zero_()
    nslots = []
    for l in range(L):
        C = cat[l]
        entries = []
        for r, (top, pl, _) in zip(rows, probes_doc):
            pm = r["per_mention"][a.mentions - 1]; t = pm["t"]
            e = C["val_star"][t] - C["val0"][t]
            if variant.startswith("A"):
                entries.append((C["u"][t], C["w"][t], e))
            else:
                w_p, u_p = pl[l]
                entries.append((u_p, w_p, e))
        dV, n = joint_write(l, entries); cons[l].dV.copy_(dV); nslots.append(n)
    outs = probe_reads([p["prompt_ids"] for p in same])
    ok = [top == p["fact"].answer_id for p, (top, _, _) in zip(same, outs)]
    fid, dreads = [], []
    for r, (top, pl, _) in zip(rows, outs):
        pm = r["per_mention"][a.mentions - 1]; t = pm["t"]
        for l in range(L):
            C = cat[l]
            base, after = rel_read(C["w"][t], C["u"][t], l, cons[l].dV)
            gap = (C["val_star"][t] - C["val0"][t]).to(dev)
            fid.append(float(torch.dot(after - base, gap) / gap.pow(2).sum().clamp_min(1e-12)))
            w_p, u_p = pl[l]
            pb, pa = rel_read(w_p, u_p, l, cons[l].dV)
            dreads.append(float((pa - pb).norm() / pb.norm().clamp_min(1e-12)))
    print(f"oracle {variant}: same-template accuracy {np.mean(ok):.3f} (in-context-correct facts answered: "
          f"{sum(o and i for o, i in zip(ok, ic_ok))}/{sum(ic_ok)}); moved at the write tokens {np.mean(fid):.3f}; "
          f"dread at the probes {np.mean(dreads):.3f}; slots written per layer {nslots}", flush=True)

# ---- 2b. the pool's own contribution at the mention token, transplanted into h' of every hippocampal layer at the
# probe token (an in-place add before the layer's ffn norm; the pool residual r and gain f recorded during the read)
class _Proxy:
    """The layer's operator namespace with row_rmsnorm wrapped (the namespace itself is read-only)."""
    def __init__(self, ns, l): self._ns, self._l = ns, l
    def __getattr__(self, k): return getattr(self._ns, k)
    def row_rmsnorm(self, h, weight, eps):
        if weight is not None and self._l in ADD:
            last, vec = ADD[self._l]
            h[last, 0].add_(vec.to(h.dtype))
        return self._ns.row_rmsnorm(h, weight, eps)
    def read_gain_residual(self, h, r, f):
        if NOPOOL.get("on"):
            return self._ns.read_gain_residual(h, torch.zeros_like(r), f)   # the pool contributes nothing (r = 0)
        return self._ns.read_gain_residual(h, r, f)
ADD = {}; NOPOOL = {}
_orig_ns = {l: hippo[l]._ops for l in range(L)}
for l in range(L): hippo[l]._ops = _Proxy(_orig_ns[l], l)
for c in cons: c.dV.zero_(); c.write = False
for label, gain in (("f r (the design's target quantity)", lambda f: f), ("(1 + f) r (the pool's whole contribution)", lambda f: 1.0 + f)):
    ok = []
    for r, p in zip(rows, same):
        t = r["per_mention"][a.mentions - 1]["t"]
        x, last = pad(p["prompt_ids"])
        ADD.clear()
        for l in range(L):
            rl, fl = POOL[l]
            ADD[l] = (last, (rl[t] * gain(float(fl[t]))).to(dev))
        with torch.no_grad(), regime("train"):
            hidden = model.backbone(x, aux_out=[])
            top = int(model._readout(hidden[:, last:last + 1])[0, 0].argmax())
        ok.append(top == p["fact"].answer_id)
    ADD.clear()
    print(f"transplant {label}: same-template accuracy {np.mean(ok):.3f} (in-context-correct facts answered "
          f"{sum(o_ and i for o_, i in zip(ok, ic_ok))}/{sum(ic_ok)})", flush=True)
# the pool residual the in-context probe itself produces at its last token, transplanted into the closed-book probe
ok = []
for i, p in enumerate(same):
    x, last = pad(p["prompt_ids"])
    ADD.clear()
    for l in range(L):
        r_ic, f_ic = IC_POOL[i][l]
        ADD[l] = (last, (r_ic * (1.0 + float(f_ic))).to(dev))
    with torch.no_grad(), regime("train"):
        hidden = model.backbone(x, aux_out=[])
        top = int(model._readout(hidden[:, last:last + 1])[0, 0].argmax())
    ok.append(top == p["fact"].answer_id)
ADD.clear()
print(f"transplant (1 + f) r of the in-context probe itself (the pool's output for this query with the document in "
      f"front): closed-book accuracy {np.mean(ok):.3f} (in-context-correct facts answered {sum(o_ and i for o_, i in zip(ok, ic_ok))}/{sum(ic_ok)})", flush=True)

# ---- 2c. where the in-context answer lives: the in-context probes with the pool residual removed (r = 0 in all six
# hippocampal layers; the dendritic compartments, the lexical memory and the induction windows keep the context)
NOPOOL["on"] = True
outs = probe_reads([doc + p["prompt_ids"][1:] for p in same])
ok = [top == p["fact"].answer_id for p, (top, _, _) in zip(same, outs)]
NOPOOL["on"] = False
print(f"in-context with the pool residual removed (r = 0, all layers): same-template accuracy {np.mean(ok):.3f} "
      f"(of the in-context-correct facts {sum(o_ and i for o_, i in zip(ok, ic_ok))}/{sum(ic_ok)} still answered)", flush=True)
NOPOOL["on"] = True
outs = probe_reads([p["prompt_ids"] for p in same])
ok0 = [top == p["fact"].answer_id for p, (top, _, _) in zip(same, outs)]
NOPOOL["on"] = False
print(f"closed book with the pool residual removed: same-template accuracy {np.mean(ok0):.3f}", flush=True)
for l in range(L): hippo[l]._ops = _orig_ns[l]
for c in cons: c.write = True

# ---- 3. the in-context reference for the same probes (the document in front)
for c in cons: c.dV.zero_()
acc_ic = float(np.mean(ic_ok))
print(f"in-context (document in front, no writes): same-template accuracy {acc_ic:.3f}", flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows, "acc_write": acc_write, "acc_incontext": acc_ic}, indent=1) + "\n")
