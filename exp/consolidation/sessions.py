"""The level-2 session harness (``docs/experiments/consolidation_level2.md``): K sessions of new facts read in the
whole-row regime, then closed-book cloze probes of every session so far, the invariant-fact set, the dev cross entropy.

Arms: ``write`` (online consolidation on), ``nowrite`` (the consolidation attached with writes off: must equal the base
checkpoint), ``incontext`` (the probes asked with the session document in front: the acquisition ceiling), ``finetune``
(AdamW on the session document, one pass).  One JSON per run.

usage: python sessions.py --ckpt runs/babylm_dp/hebbian/final --arm write --sessions 8 --out runs/consolidation/write.json
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

B_ = "/workspace/neuronspark-gb300"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="directory with config.json and model.safetensors")
ap.add_argument("--arm", required=True, choices=("write", "nowrite", "incontext", "finetune"))
ap.add_argument("--out", required=True)
ap.add_argument("--sessions", type=int, default=8)
ap.add_argument("--entities", type=int, default=16, help="entities per session (facts = 4 x entities)")
ap.add_argument("--mentions", type=int, default=2)
ap.add_argument("--distractor", type=int, default=32)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--write-topm", type=int, default=None, help="write only the m strongest slots per head (read keeps k)")
ap.add_argument("--prior-windows", type=int, default=0, help="dev windows whose slot usage primes the RLS precision")
ap.add_argument("--prior-scale", type=float, default=1.0)
ap.add_argument("--invariant", type=int, default=2048, help="dev positions tested for the invariant set")
ap.add_argument("--dev-windows", type=int, default=128)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--finetune-lr", type=float, default=3e-5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--neuronspark", default=f"{B_}/tmp/stage/NeuronSpark-exp")
ap.add_argument("--data", default=f"{B_}/data_work/babylm/tok")
ap.add_argument("--tokenizer", default=f"{B_}/tokenizer/NeuronSpark-V4-1.16B-Pretrain-Qwen3VL/tokenizer.json")
ap.add_argument("--cap-gib", type=float, default=14.0)
ap.add_argument("--blimp", default=None, help="BLiMP jsonl directory: scored before the first and after the last session")
a = ap.parse_args()
sys.path.insert(0, a.neuronspark); sys.path.insert(0, str(Path(__file__).resolve().parent))
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
model.eval()
torch.cuda.empty_cache()                                                      # the loaded copies' fragments
tok = Tokenizer.from_file(a.tokenizer)
dstream = np.load(f"{a.data}/dev.u32.npy", mmap_mode="r")
CH = int(cfg.dendritic_chunk_size)
hippo = [m for m in model.modules() if m.__class__.__name__ == "HippocampalLayer"]
cons = []
if a.arm in ("write", "nowrite"):
    for layer in hippo:
        c = Consolidation(layer, lam=a.lam, chunk=a.chunk, write=(a.arm == "write"), write_topm=a.write_topm)
        layer.consolidation = c
        cons.append(c)
opt = None
if a.arm == "finetune":
    sys.path.insert(0, f"{B_}/trainer")
    from nstrain.model import param_groups
    groups = param_groups(model, 0.1, 8.0)
    for g in groups:
        g["lr"] = a.finetune_lr * g["lr_mult"]
    opt = torch.optim.AdamW(groups, lr=a.finetune_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)


class writes_off:
    """Evaluation forwards never write: only the session document is read with the writes on."""
    def __enter__(self):
        self.saved = [c.write for c in cons]
        for c in cons:
            c.write = False
    def __exit__(self, *exc):
        for c, w in zip(cons, self.saved):
            c.write = w


def pad_rows(rows, fill=FX.EOT):
    """(B, L) int64 with L a multiple of the chunk size + 1; returns ids and the index of each row's last real token."""
    L = max(len(r) for r in rows)
    width = -(-L // CH) * CH + 1
    ids = torch.full((len(rows), width), fill, dtype=torch.int64)
    last = []
    for j, r in enumerate(rows):
        ids[j, :len(r)] = torch.tensor(r, dtype=torch.int64)
        last.append(len(r) - 1)
    return ids, last


def ce_rows(rows, labels, step=2048):
    """Per-token CE (n,) fp32 of hidden rows (n, D) fp32 against labels (n,), as the LM head's ``per_token_ce`` forms it
    but in row chunks of ``step`` (its own chunk is 14k rows: the (rows, V) logits of a long document in one piece)."""
    E = model.backbone.embed_tokens.weight
    parts = []
    for i in range(0, int(rows.shape[0]), step):
        lg = model._o.linear(rows[i:i + step], E)                                   # (step, V) bf16
        parts.append(model.backbone.balance_stack.ce_per_token(lg, rows[i:i + step], labels[i:i + step]))
    return parts[0] if len(parts) == 1 else torch.cat(parts)


@torch.no_grad()
def predict_next(prompts, tokens_per_call=32 * 256):
    """Top-1 next token after each prompt (list of id lists); prompts are batched by their padded width.  Only the last
    real position of every row goes through the readout (the logits of all positions were the one big tensor)."""
    tops = []
    i = 0
    with writes_off():
      while i < len(prompts):
        width = -(-max(len(r) for r in prompts[i:i + 32]) // CH) * CH
        micro = max(1, tokens_per_call // width)
        chunk = prompts[i:i + micro]
        i += micro
        ids, last = pad_rows(chunk)
        x = ids[:, :-1].to(dev)
        with regime("train"):
            hidden = model.backbone(x, aux_out=[])                              # (b, T, D) fp32
            rows = hidden[torch.arange(len(last), device=dev), torch.tensor(last, device=dev)]
            logits = model._readout(rows.unsqueeze(1))                          # (b, 1, V) bf16
        tops += logits[:, 0].argmax(-1).tolist()
    return tops


@torch.no_grad()
def read_document(doc_ids):
    """The session document through the whole-row forward (writes happen inside when the arm writes)."""
    if len(doc_ids) > int(cfg.rope_max_seq_len):
        raise RuntimeError(f"session document of {len(doc_ids)} tokens exceeds rope_max_seq_len {cfg.rope_max_seq_len}")
    ids, _ = pad_rows([list(doc_ids)])
    x, y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
    n = len(doc_ids) - 1
    with regime("train"):
        hidden = model.backbone(x, aux_out=[])                                  # the writes happen in here
        per = ce_rows(hidden[0, :n], y[0, :n])
    return float(per.mean())


def finetune_document(doc_ids):
    """One AdamW step on the document, as consecutive rows of ``a.seq`` tokens (the arm's training length; every target
    predicted once, the gradient accumulated over the rows, the loss the token mean over the document)."""
    model.train()
    ids = list(doc_ids)
    n_tok = len(ids) - 1
    tot = 0.0
    for i in range(0, n_tok, a.seq):
        r = ids[i:i + a.seq + 1]
        padded, _ = pad_rows([r])
        x, y = padded[:, :-1].to(dev), padded[:, 1:].to(dev)
        y = y.clone(); y[0, len(r) - 1:] = -100
        with regime("train"):
            lo = model.loss(x, y)
            (lo.loss * ((len(r) - 1) / n_tok)).backward()
        tot += float(lo.ce) * (len(r) - 1)
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    model.eval()
    return tot / n_tok


@torch.no_grad()
def dev_ce():
    idx = np.sort(np.random.default_rng(a.seed + 1).permutation(len(dstream) // (a.seq + 1))[:a.dev_windows])
    tot, n = 0.0, 0
    with writes_off():
      for i in range(0, len(idx), 4):
        w = torch.from_numpy(np.stack([np.asarray(dstream[int(k) * (a.seq + 1):(int(k) + 1) * (a.seq + 1)], dtype=np.int64)
                                       for k in idx[i:i + 4]])).to(dev)
        with regime("train"):
            per = model.per_token_ce(w[:, :-1].contiguous(), w[:, 1:].contiguous())
        tot += float(per.sum()); n += per.numel()
    return tot / n


@torch.no_grad()
def blimp_eval():
    """BLiMP accuracy (overall and macro over phenomena): the good sentence's summed log-prob beats the bad one's."""
    if not a.blimp:
        return None
    files = sorted(Path(a.blimp).glob("*.jsonl"))
    tot_ok, tot_n, per = 0, 0, {}
    with writes_off():
        for fpath in files:
            rows = [json.loads(l) for l in fpath.read_text().splitlines() if l.strip()]
            scores = []
            for key in ("sentence_good", "sentence_bad"):
                enc = [[FX.EOT] + e.ids for e in tok.encode_batch([r[key] for r in rows], add_special_tokens=False)]
                out = []
                for i in range(0, len(enc), 64):
                    ids, last = pad_rows(enc[i:i + 64])
                    tg = torch.full_like(ids[:, 1:], -100)
                    for j, e in enumerate(enc[i:i + 64]):
                        tg[j, :len(e) - 1] = ids[j, 1:len(e)]
                    with regime("train"):
                        pt = model.per_token_ce(ids[:, :-1].to(dev), tg.to(dev))
                    out += (-pt.sum(1)).tolist()
                scores.append(out)
            ok = sum(1 for g, b in zip(*scores) if g > b)
            per[fpath.stem] = ok / len(rows); tot_ok += ok; tot_n += len(rows)
    return {"overall": tot_ok / tot_n, "macro": float(np.mean(list(per.values()))), "pairs": tot_n, "phenomena": per}


def invariant_set():
    items = FX.invariant_items(dstream, a.invariant, 255, a.seed + 2)         # 256-token windows: prompt 255 + answer
    prompts = [[int(x) for x in dstream[s:s + L]] for s, L in items]
    answers = [int(dstream[s + L]) for s, L in items]
    tops = predict_next(prompts)
    keep = [(p, ans) for p, ans, t in zip(prompts, answers, tops) if t == ans]
    return keep


def eval_probes(sessions, with_doc=None):
    """Accuracy per kind over the given sessions' probes; with_doc: the document ids put in front of every prompt."""
    res = {}
    for s in sessions:
        prompts = []
        for p in s.probes:
            pr = p["prompt_ids"]
            if with_doc is not None:
                room = int(cfg.rope_max_seq_len) - len(pr)
                pr = list(with_doc[-room:]) + pr[1:]                        # the document, then the prompt (no second EOT)
            prompts.append(pr)
        tops = predict_next(prompts)
        for p, t in zip(s.probes, tops):
            k = (s.index, p["kind"])
            r = res.setdefault(k, [0, 0])
            r[0] += int(t == p["answer_id"]); r[1] += 1
    return {f"s{si}/{kind}": {"acc": v[0] / v[1], "n": v[1]} for (si, kind), v in sorted(res.items())}


t0 = time.time()
if cons and a.prior_windows:
    idx = np.sort(np.random.default_rng(a.seed + 3).permutation(len(dstream) // (a.seq + 1))[:a.prior_windows])
    for c in cons:
        c.collect = True
    with torch.no_grad():
        for i in range(0, len(idx), 4):
            w = torch.from_numpy(np.stack([np.asarray(dstream[int(k) * (a.seq + 1):(int(k) + 1) * (a.seq + 1)], dtype=np.int64)
                                           for k in idx[i:i + 4]])).to(dev)
            with regime("train"):
                model.per_token_ce(w[:, :-1].contiguous(), w[:, 1:].contiguous())
    for c in cons:
        c.collect = False
        c.prime(a.prior_scale)
    print(f"usage prior from {len(idx)} dev windows x {a.seq} tokens, scale {a.prior_scale}: "
          f"slots touched {[int((c.usage > 0).sum()) for c in cons]}, usage median {[round(float(c.usage[c.usage > 0].median()), 3) for c in cons]}", flush=True)
values = FX.single_token_values(tok)
sessions = [FX.build_session(i, tok, dstream, n_entities=a.entities, mentions=a.mentions,
                             distractor_tokens=a.distractor, seed=a.seed, values=values) for i in range(a.sessions)]
inv = invariant_set()
base_ce = dev_ce()
blimp0 = blimp_eval()
out = {"ckpt": str(path), "rule": cfg.dendritic_plasticity, "arm": a.arm, "args": vars(a), "n_invariant": len(inv),
       "doc_tokens": [int(len(s.doc_ids)) for s in sessions], "base": {"dev_ce": base_ce, "blimp": blimp0},
       "before": eval_probes(sessions), "rounds": []}
print(f"{a.arm}: {len(inv)} invariant items of {a.invariant}, base dev ce {base_ce:.4f}, docs {out['doc_tokens'][:3]}... tokens",
      flush=True)
acquired_prev = 0
inv_prompts = [p for p, _ in inv]; inv_ans = [ans for _, ans in inv]
for k, s in enumerate(sessions):
    if a.arm == "finetune":
        doc_ce = finetune_document(s.doc_ids)
    else:
        doc_ce = read_document(s.doc_ids)                                   # writes inside for the write arm
    seen = sessions[:k + 1]
    probes = eval_probes(seen, with_doc=(s.doc_ids if a.arm == "incontext" else None))
    inv_ok = sum(int(t == ans) for t, ans in zip(predict_next(inv_prompts), inv_ans))
    ce = dev_ce()
    acquired = sum(v["acc"] * v["n"] for key, v in probes.items() if key.endswith("/same"))
    forgotten = len(inv) - inv_ok
    fuar = forgotten / acquired if acquired > 0 else float("inf")
    rnd = {"session": k, "doc_ce": doc_ce, "probes": probes, "invariant_kept": inv_ok / max(len(inv), 1), "dev_ce": ce,
           "fuar": fuar, "writes": [c.writes for c in cons],
           "captured": [float(np.mean([e["captured"] for e in c.log[-64:]])) if c.log else None for c in cons],
           "f_mean": [float(np.mean(c.log_f[-16:])) if c.log_f else None for c in cons],
           "dV_rms": [float(c.dV.pow(2).mean().sqrt()) for c in cons], "elapsed_s": time.time() - t0}
    out["rounds"].append(rnd)
    cur = probes[f"s{k}/same"]["acc"]; rep = probes[f"s{k}/rephrase"]["acc"]; hop = probes[f"s{k}/twohop"]["acc"]
    first = probes["s0/same"]["acc"]
    print(f"session {k}: doc ce {doc_ce:.3f} | this session same {cur:.3f} rephrase {rep:.3f} twohop {hop:.3f} | "
          f"s0 same {first:.3f} | invariant kept {rnd['invariant_kept']:.4f} dev ce {ce:.4f} (base {base_ce:.4f}) | "
          f"FUAR {fuar:.3f}" + (f" | captured {rnd['captured']}" if cons else ""), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
out["after"] = {"blimp": blimp_eval()}
if blimp0 is not None:
    print(f"BLiMP overall {blimp0['overall']:.4f} -> {out['after']['blimp']['overall']:.4f}", flush=True)
Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
print(f"DONE {a.arm} in {time.time() - t0:.0f} s -> {a.out}", flush=True)
