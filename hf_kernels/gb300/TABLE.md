# The GB300 kernel packages: how each was checked

51 operators of tag `gb300-operators-20261116` (5c2720b), each exported with the exporter of Kernel Design
Agents (NVlabs/kda, `scripts/export_huggingface.py`) and kernel-builder 0.18.3, loaded with
`kernels.get_local_kernel(<package>, trust_remote_code=True)` in a process without a GPU and without this repository,
and then run on an NVIDIA B200 (torch 2.9.1+cu130, triton 3.5.1) against the operator of that commit.

* files: the repository files in the package (the operator's module and its static import closure).
* validated: the operator's recorded calls in the 1.3B model (one training step, one prefill of 8 x 1,024 tokens, one
  decode step; an input file for an operator the model does not call there): the outputs, the gradients and the
  arguments written in place against the repository's operator and against its pure-PyTorch reference, under the
  rule of `neuronspark/utils/testing.py`.
* kernel time: CUPTI device time per call, repository / package in ms (difference), where it was read.

| operator | files | exported | loaded | validated | kernel time: tree / exported, ms |
|---|---|---|---|---|---|
| `linear` | 5 | yes (`neuronspark-linear v1`) | yes | PASSED: 86 calls (train 29, prefill 29, decode 28) | not asked |
| `dendritic_norm` | 5 | yes (`neuronspark-dendritic-norm v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `dendritic_resid_norm` | 5 | yes (`neuronspark-dendritic-resid-norm v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `dendritic_chunk_scan_soma` | 18 | yes (`neuronspark-dendritic-chunk-scan-soma v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `dendritic_prenorm_rows` | 8 | yes (`neuronspark-dendritic-prenorm-rows v1`) | yes | PASSED: 2 calls (prefill 1, decode 1) | not asked |
| `dendritic_conv_silu_prefill` | 4 | yes (`neuronspark-dendritic-conv-silu-prefill v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `dendritic_chunk_scan_prefill` | 11 | yes (`neuronspark-dendritic-chunk-scan-prefill v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `dendritic_gated_norm_soma` | 6 | yes (`neuronspark-dendritic-gated-norm-soma v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `dendritic_step_core` | 8 | yes (`neuronspark-dendritic-step-core v1`) | yes | PASSED: 1 call (decode 1) | not asked |
| `dendritic_res_add` | 4 | yes (`neuronspark-dendritic-res-add v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `row_rmsnorm` | 5 | yes (`neuronspark-row-rmsnorm v1`) | yes | PASSED: 2 calls (train 2) | not asked |
| `rmsnorm_rows` | 5 | yes (`neuronspark-rmsnorm-rows v1`) | yes | PASSED: 11 calls (prefill 6, decode 5) | not asked |
| `rmsnorm_gate_rows` | 5 | yes (`neuronspark-rmsnorm-gate-rows v1`) | yes | PASSED: 2 calls (prefill 1, decode 1) | not asked |
| `out_norm_gate` | 5 | yes (`neuronspark-out-norm-gate v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `nov_norm` | 5 | yes (`neuronspark-nov-norm v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `pool_heads` | 6 | yes (`neuronspark-pool-heads v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `value_split` | 6 | yes (`neuronspark-value-split v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `silu_rope` | 8 | yes (`neuronspark-silu-rope v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `rope_paired` | 8 | yes (`neuronspark-rope-paired v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `rope_interleaved` | 8 | yes (`neuronspark-rope-interleaved v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `value_residual` | 6 | yes (`neuronspark-value-residual v1`) | yes | PASSED: 2 calls (train 2) | not asked |
| `zc_center` | 6 | yes (`neuronspark-zc-center v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `read_gain_residual` | 6 | yes (`neuronspark-read-gain-residual v1`) | yes | PASSED: 3 calls (train 1, prefill 1, decode 1) | not asked |
| `pkm_tail` | 6 | yes (`neuronspark-pkm-tail v1`) | yes | PASSED: 3 calls (train 1, prefill 1, decode 1) | not asked |
| `cumsum0` | 5 | yes (`neuronspark-cumsum0 v1`) | yes | PASSED: 2 calls (train 2) | not asked |
| `exact_cumsum0` | 5 | yes (`neuronspark-exact-cumsum0 v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `centred_running` | 5 | yes (`neuronspark-centred-running v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `short_conv_train` | 5 | yes (`neuronspark-short-conv-train v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `short_conv_silu_prefill` | 5 | yes (`neuronspark-short-conv-silu-prefill v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `short_conv_step` | 5 | yes (`neuronspark-short-conv-step v1`) | yes | PASSED: 1 call (decode 1) | not asked |
| `pool_read` | 13 | yes (`neuronspark-pool-read v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `ring_write_prefill` | 5 | yes (`neuronspark-ring-write-prefill v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `ring_recall_prefill` | 5 | yes (`neuronspark-ring-recall-prefill v1`) | yes | PASSED: 2 calls (prefill 2) | not asked |
| `ring_read_t1` | 5 | yes (`neuronspark-ring-read-t1 v1`) | yes | PASSED: 1 call (decode 1) | not asked |
| `xq_form_t1` | 5 | yes (`neuronspark-xq-form-t1 v1`) | yes | PASSED: 1 call (case 1) | not asked |
| `plif_centred_input` | 9 | yes (`neuronspark-plif-centred-input v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `plif_scan_train` | 9 | yes (`neuronspark-plif-scan-train v1`) | yes | PASSED: 2 calls (train 2) | not asked |
| `plif_gate_prefill` | 9 | yes (`neuronspark-plif-gate-prefill v1`) | yes | PASSED: 2 calls (prefill 2) | not asked |
| `plif_scan_prefill` | 9 | yes (`neuronspark-plif-scan-prefill v1`) | yes | PASSED: 3 calls (prefill 3) | not asked |
| `pkm_topk_train` | 4 | yes (`neuronspark-pkm-topk-train v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `pkm_value_read` | 5 | yes (`neuronspark-pkm-value-read v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `pkm_topk_prefill` | 4 | yes (`neuronspark-pkm-topk-prefill v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `pkm_weighted_sum` | 5 | yes (`neuronspark-pkm-weighted-sum v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `pkm_topk_softmax_t1` | 4 | yes (`neuronspark-pkm-topk-softmax-t1 v1`) | yes | PASSED: 1 call (decode 1) | not asked |
| `copy_switch_aux` | 4 | yes (`neuronspark-copy-switch-aux v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `balance_ce_per_token` | 6 | yes (`neuronspark-balance-ce-per-token v1`) | yes | PASSED: 1 call (train 1) | not asked |
| `balance_add` | 4 | yes (`neuronspark-balance-add v1`) | yes | PASSED: 2 calls (prefill 1, decode 1) | not asked |
| `copy_stats_prefill` | 4 | yes (`neuronspark-copy-stats-prefill v1`) | yes | PASSED: 1 call (prefill 1); the -inf entries of out.0, out.1 coincide, the rule on the other entries | not asked |
| `xq_queries_prefill` | 8 | yes (`neuronspark-xq-queries-prefill v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `lex_read_prefill` | 5 | yes (`neuronspark-lex-read-prefill v1`) | yes | PASSED: 1 call (prefill 1) | not asked |
| `vr_mix_prefill` | 4 | yes (`neuronspark-vr-mix-prefill v1`) | yes | PASSED: 2 calls (prefill 2) | not asked |
