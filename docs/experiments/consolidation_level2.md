# Level 2 experiment: online consolidation of in-context knowledge into the product-key memory

Branch `exp/consolidation` (experimental; nothing here is on `main`).

## Goal

The pretrained model keeps improving from what it reads at inference, without gradient steps: knowledge corrected
inside a context is **retained after the context is reset**, accumulates over contexts, and the pretrained knowledge is
not lost.  Explicitly not the goal: extrapolating the context length, or never resetting and living on an unbounded
state.

Two lifetimes only.  The context state (dendritic compartments, hippocampal event pool, induction and copy rings) is
cleared at a reset.  The persistent store (the product-key memory's values, later the weights) is written online during
the forward and is never cleared.

## Mechanism under test

The hippocampal layer already contains the two stores and the signal between them (`hippocampal_layer.py`, whole-row
forward):

```
r      = pool attention residual                         what the context recalls for this token
h'     = h + f r,   f = 1 + 4 pi c                        the pool's contribution to the stream
x_hat  = ffn_norm(h')
val    = sum_m w_m V[u_m]                                 PKM read: top-k slots u, softmax weights w, values V
r_pkm  = val rsqrt(mean val^2 + eps) (1 + g_norm) sigmoid(glog)
y      = r_pkm W_o^T
h_out  = h' + ffn + (y rho - mean(y rho)),  rho = 2 sigmoid(-4 pi)
```

Online consolidation (new): at every token, in every hippocampal layer, the PKM is taught to reproduce the pool's
contribution for the current query, so that the same query later -- in a new context, with the pool empty -- gets the
same contribution from the PKM:

```
target   y* = y + f r / rho                               PKM alone must supply what pool + PKM supplied together
in value space:
         r* = lstsq(W_o, y*)                              the output projection's pseudo-inverse (fixed per checkpoint)
         val* = rms(val) unit(r* / ((1 + g_norm) sigmoid(glog)))
write    minimise ||sum_m w_m (V[u_m] + dV[u_m]) - val*||^2 + lambda ||dV||^2 over the addressed slots only
         -> recursive least squares on the k addressed slots (Sherman-Morrison, per-slot precision), closed form
```

The teacher is local (pool read against PKM read in the same layer); there is no global error and no gradient.  The
writes go to a delta table `dV` (slots x dv per layer, fp32) added to the values at read time: `V + dV`.

Lifetimes: at inference `dV` persists across resets (this is the point).  In training the rows of a batch are unrelated,
so `dV` is cleared per row; the model still learns to rely on within-row consolidation (a fact stated early in a row is
read back from the PKM later in the row), and the rule, addressing and read-out are the same code in both regimes.

Granularity: the whole-row regime computes a row in parallel, so the write is applied per chunk (the chunk size of the
layer's kernels): reads inside a chunk see `dV` from the previous chunks.  The decode regime writes per token.

Consolidation needs a second exposure inside the context: the first mention writes the pool (novelty); a later mention
makes the pool recall and gives the PKM its target.  The fact corpus below therefore controls the number of mentions.

## What is measured (from the literature, mapped to this scale)

| question | metric | source |
|---|---|---|
| after reading and a reset, is the new knowledge there | closed-book accuracy on the read facts; floor = no write, ceiling = facts in context | SEAL (SQuAD without context), EntiGraph (QuALITY closed book), CKL NewLAMA |
| knowledge or string | same-phrasing vs rephrased cloze; two-hop composition | knowledge editing Generalization / Portability, WikiBigEdit Rephrase / Multi-hop |
| is the pretrained model intact | facts the base model knew (InvariantLAMA analogue); held-out perplexity; a downstream suite; unrelated prompts unchanged (Locality) | CKL InvariantLAMA, knowledge-editing Locality, Gupta et al. 2024 (downstream after sequential edits), sparse memory finetuning (NQ retention) |
| does it accumulate without overwriting | retention of session k's facts after K sessions; FUAR = forgotten / (updated + acquired) per session; capacity curve | GRACE / WISE sequential editing, CKL FUAR, MemoryLLM retention over updates, StreamingQA (recent and past) |

At the 650M BabyLM tier facts are probed by **cloze exact match** of the final token (LAMA style), not free-form QA.

### Fact corpus (synthetic, BabyLM-register English)

* Entities: invented names (never in the training data); attributes from closed sets (capital, colour, profession,
  pet, number); each fact is one (entity, attribute, value) triple with a unique value per attribute within a session.
* Session document: `F` facts, each mentioned `m` times (factor: m in {1, 2, 4}) in varied sentence templates, interleaved
  with distractor BabyLM dev text so that the document length is realistic (2-4K tokens).
* Probes per fact: same template as a mention (memorisation), an unseen template (rephrase), a two-hop template
  ("the pet of the person who lives in <capital>", composition of two facts).  The answer is the final token; exact
  match against the value's first token.
* Sessions: K = 8 per run, disjoint entities; after every session all previous sessions' probes are re-run.

### Retention probes

* Invariant facts: 512 dev-set sentences whose final token the base checkpoint predicts correctly (top-1); re-tested
  after every session (share still correct).
* Held-out perplexity: the arm's dev cross entropy (256 fixed windows) after every session.
* Downstream: BLiMP (full filtered set) before and after the K sessions.

### Arms and baselines

* `write`: online consolidation on (the mechanism).
* `no-write`: the same sessions with the write disabled (floor; must equal the base checkpoint on every probe).
* `in-context`: the probes asked with the session document in the context (ceiling for acquisition).
* `finetune`: AdamW on the session document (1 epoch, the training LR); the classical comparison for acquisition and,
  above all, for forgetting (Gupta et al.).

### Pre-registered acceptance

1. Acquisition: `write` same-phrasing cloze accuracy exceeds `no-write` by at least 20 points at m = 2 (absolute, over
   K sessions); rephrase accuracy exceeds `no-write`; two-hop reported, no threshold.
2. Retention: invariant-fact accuracy stays within 2 points of the base over K sessions; dev cross entropy within
   0.01 nat; BLiMP within 1 point.  `finetune` is expected to break at least one of these -- that contrast is the
   result.
3. Accumulation: session-1 facts after session 8 keep at least half of their session-1 accuracy; FUAR < 1 at every
   session.
4. The two-exposure prediction: m = 1 gives no acquisition, m >= 2 does.  (If m = 1 works, the mechanism is not what
   this document says it is.)

## Phases

* **A. Post hoc on the BabyLM arms' checkpoints** (`runs/babylm_dp/{hebbian,predictive}/final`): the write rule and the
  harness, with the model as trained (no consolidation in its training).  Answers whether the write rule and the
  addressing carry knowledge across a reset at all, and calibrates lambda and the write rate.
* **B. Trained with the rule live**: a BabyLM arm trained with per-row consolidation in the forward, against the arm
  without it (same seed, data order and protocol as the plasticity arms); then the harness.  This is the level-2
  claim proper: the pretraining forward and the inference forward run the same writes.

## Implementation plan (this branch)

1. `neuronspark/modules/consolidation.py`: the delta table and the recursive least-squares write (per layer: `dV`
   (slots, dv) fp32, per-slot precision; `write(u, w, val, val_target, lambda)`), the output projection's pseudo-inverse
   cached per checkpoint, the target computation from `(r, f, rho, y)`.
2. Hippocampal layer: a `consolidation` attribute (None by default: the layer is unchanged); when set, the PKM read adds
   `dV[u]`, and the whole-row forward writes per chunk, the decode step per token.
3. The harness `exp/consolidation/`: fact corpus generator, cloze probes, retention probes, session loop, the four arms,
   FUAR and the curves; one JSON per run, one table per experiment.
4. Tests (verify repository): writes off equals the unchanged layer under the per-operator rule; a write moves the read
   of the same query toward its target and touches only the addressed slots; a second pass writes less; reset clears.
5. Phase B needs a per-row delta table (B, slots, dv) and a batched read of `V + dV[b]` (the rows of a training batch
   are unrelated and must not see each other's writes); phase A runs one document at a time and uses one table.

## Calibration on the development checkpoint (Hebbian arm, update 4000 of 18.4k; 3 sessions x 32 facts, m = 2)

Acquisition cannot be judged there (the model is far from trained; every arm reads 0-3%), the forgetting side can:

| setting | dev CE drift after 3 sessions | invariant facts kept |
|---|---|---|
| teach `r (1 + f)`, lam 1, all 16 slots | +0.17 nat | 0.86 |
| teach `r f`, lam 1, all 16 | +0.17 | 0.86 |
| teach `r f`, lam 10, all 16 | +0.077 | 0.95 |
| teach `r f`, lam 10, 2 slots per head | +0.024 | 0.96 |
| teach `r f`, lam 10, 2 slots, usage prior (32 dev windows) | +0.008 | 0.99 |
| teach `r f`, lam 100, 2 slots, usage prior | +0.002 | 1.00 |

Decisions: the target is the gated recall `r f`; writes go to the 2 strongest slots per head; the RLS precision is primed
from 32 dev windows; lam in {10, 100} is carried into the phase-A matrix (acquisition decides between them).  The
scale anchor of the target is the read of `V` alone (anchoring on `V + dV` fed back and diverged).  `captured` (the
share of the target the PKM's output subspace can express) is 0.96-0.97 with the gated target.
