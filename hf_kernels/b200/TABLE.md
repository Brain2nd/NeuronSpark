# The B200 kernel packages: how each was checked

81 operators of branch `b200`, each exported with the exporter of Kernel Design Agents (NVlabs/kda,
`scripts/export_huggingface.py`) and kernel-builder 0.18.3, loaded with `kernels.get_local_kernel(<package>,
trust_remote_code=True)` in a process without a GPU and without this repository, and then run on an NVIDIA B200
(torch 2.9.1+cu130, triton 3.5.1) against the operator of the commit it was exported from.

74 packages are of commit b955637; the seven that contain `neuronspark/ops/triton/pool_ring.py` (`pool_attend`,
`pool_read`, `ring_read_t1`, `ring_recall_prefill`, `ring_write_prefill`, `xq_form_t1`, `xq_queries_prefill`) are of
93d2f38, the only file that differs between the two commits being that module.

* files: the repository files in the package (the operator's module and its static import closure).
* validated: the operator's recorded calls in the 1.3B model (one training step, one prefill of 8 x 1,024 tokens, one
  decode step; an input file for an operator the model does not call there): the outputs, the gradients and the
  arguments written in place against the repository's operator and against its pure-PyTorch reference, under the
  rule of `neuronspark/utils/testing.py`.
* kernel time: CUPTI device time per call, repository / package in ms (difference), forward and backward, of the first
  recorded call of each regime; DIFFER marks a difference outside the resolution of the comparison (the larger of 1 %
  and the spread of the trial medians).

One comparison is marked DIFFER: `ring_recall_prefill`, the extra queries' call (the package 1.5 % faster in each of
three runs).  Its values equal the repository's bit for bit on both calls, and its kernels are the repository's
compiled kernels (the same Triton cache keys); two loads of that kernel timed against each other in a process of
their own agree within 0.1 % (263.25 / 263.06 us and 183.98 / 183.99 us), at the package's time.

| operator | files | exported | loaded | validated | kernel time: tree / exported, ms |
|---|---|---|---|---|---|
| `linear` | 7 | yes (`neuronspark-linear v1`) | yes | PASSED: 73 calls (train 29, prefill 27, decode 17) | train w00: fwd 0.2926 / 0.2963 (+1.24 %), bwd 0.6085 / 0.6095 (+0.17 %); prefill w00: fwd 0.2805 / 0.2822 (+0.61 %); decode w00: fwd 0.0155 / 0.0157 (+1.12 %) |
| `linear_add` | 7 | yes (`neuronspark-linear-add v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0102 / 0.0104 (+1.80 %) |
| `dendritic_norm` | 5 | yes (`neuronspark-dendritic-norm v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0194 / 0.0194 (-0.06 %), bwd 0.0489 / 0.0488 (-0.19 %) |
| `dendritic_resid_norm` | 5 | yes (`neuronspark-dendritic-resid-norm v1`) | yes | PASSED: 2 calls (train 1, decode 1) | train w00: fwd 0.0278 / 0.0275 (-1.09 %), bwd 0.0615 / 0.0616 (+0.11 %); decode w00: fwd 0.0028 / 0.0028 (+0.39 %) |
| `dendritic_chunk_scan_soma` | 19 | yes (`neuronspark-dendritic-chunk-scan-soma v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.4994 / 0.4988 (-0.12 %), bwd 1.1914 / 1.1931 (+0.14 %) |
| `dendritic_prenorm_rows` | 8 | yes (`neuronspark-dendritic-prenorm-rows v1`) | yes | PASSED: 2 calls (prefill 1, decode 1) | prefill w00: fwd 0.0214 / 0.0213 (-0.44 %); decode w00: fwd 0.0026 / 0.0026 (+0.00 %) |
| `dendritic_conv_silu_prefill` | 4 | yes (`neuronspark-dendritic-conv-silu-prefill v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0820 / 0.0820 (+0.01 %) |
| `dendritic_chunk_scan_prefill` | 11 | yes (`neuronspark-dendritic-chunk-scan-prefill v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.3456 / 0.3452 (-0.12 %) |
| `dendritic_gated_norm_soma` | 6 | yes (`neuronspark-dendritic-gated-norm-soma v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.1065 / 0.1065 (+0.04 %) |
| `dendritic_step_core` | 8 | yes (`neuronspark-dendritic-step-core v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0352 / 0.0353 (+0.19 %) |
| `dendritic_res_add` | 4 | yes (`neuronspark-dendritic-res-add v1`) | yes | PASSED: 2 calls (prefill 1, decode 1) | prefill w00: fwd 0.0223 / 0.0223 (+0.11 %); decode w00: fwd 0.0023 / 0.0023 (+1.05 %) |
| `row_rmsnorm` | 5 | yes (`neuronspark-row-rmsnorm v1`) | yes | PASSED: 1 call (case 1) | case row_rmsnorm: call 0.0804 / 0.0802 (-0.31 %) |
| `row_rmsnorm_tee` | 5 | yes (`neuronspark-row-rmsnorm-tee v1`) | yes | PASSED: 3 calls (train 3) | train w00: fwd 0.0144 / 0.0147 (+2.25 %), bwd 0.0399 / 0.0399 (-0.15 %) |
| `rmsnorm_rows` | 5 | yes (`neuronspark-rmsnorm-rows v1`) | yes | PASSED: 7 calls (prefill 5, decode 2) | prefill w00: fwd 0.0196 / 0.0191 (-2.23 %); decode w00: fwd 0.0041 / 0.0041 (+1.38 %) |
| `rmsnorm_gate_rows` | 5 | yes (`neuronspark-rmsnorm-gate-rows v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0323 / 0.0325 (+0.45 %) |
| `out_norm_gate` | 5 | yes (`neuronspark-out-norm-gate v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0396 / 0.0396 (+0.03 %), bwd 0.0889 / 0.0888 (-0.08 %) |
| `silu_rope` | 8 | yes (`neuronspark-silu-rope v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0235 / 0.0235 (-0.06 %), bwd 0.0302 / 0.0302 (+0.19 %) |
| `rope_interleaved` | 8 | yes (`neuronspark-rope-interleaved v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0176 / 0.0176 (+0.35 %) |
| `value_residual` | 6 | yes (`neuronspark-value-residual v1`) | yes | PASSED: 2 calls (train 2) | train w00: fwd 0.0234 / 0.0234 (-0.15 %), bwd 0.0719 / 0.0720 (+0.14 %) |
| `centre_rows` | 6 | yes (`neuronspark-centre-rows v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0112 / 0.0111 (-0.14 %), bwd 0.0112 / 0.0114 (+2.27 %) |
| `read_gain_residual` | 6 | yes (`neuronspark-read-gain-residual v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0679 / 0.0679 (-0.04 %) |
| `pkm_tail` | 6 | yes (`neuronspark-pkm-tail v1`) | yes | PASSED: 2 calls (train 1, prefill 1) | train w00: fwd 0.0321 / 0.0321 (-0.24 %), bwd 0.0224 / 0.0225 (+0.36 %); prefill w00: fwd 0.0321 / 0.0322 (+0.36 %) |
| `novelty_read_gain` | 4 | yes (`neuronspark-novelty-read-gain v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0358 / 0.0357 (-0.20 %), bwd 0.0950 / 0.0946 (-0.32 %) |
| `lex_read_train` | 4 | yes (`neuronspark-lex-read-train v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0253 / 0.0254 (+0.38 %), bwd 0.0454 / 0.0459 (+1.01 %) |
| `lexical_train` | 8 | yes (`neuronspark-lexical-train v1`) | yes | PASSED: 1 call (case 1) | case lexical_train: call 0.2127 / 0.2141 (+0.66 %) |
| `cumsum0` | 5 | yes (`neuronspark-cumsum0 v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0135 / 0.0132 (-2.10 %), bwd 0.0132 / 0.0131 (-0.42 %) |
| `centred_past` | 5 | yes (`neuronspark-centred-past v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0333 / 0.0335 (+0.47 %), bwd 0.0684 / 0.0682 (-0.31 %) |
| `centred_running` | 5 | yes (`neuronspark-centred-running v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0429 / 0.0430 (+0.27 %) |
| `short_conv_train` | 5 | yes (`neuronspark-short-conv-train v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0421 / 0.0421 (+0.02 %), bwd 0.1162 / 0.1163 (+0.09 %) |
| `short_conv_silu_prefill` | 5 | yes (`neuronspark-short-conv-silu-prefill v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0852 / 0.0852 (+0.01 %) |
| `pool_attend` | 80 | yes (`neuronspark-pool-attend v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.3070 / 0.3072 (+0.06 %), bwd 0.9195 / 0.9195 (-0.00 %) |
| `shared_operand` | 5 | yes (`neuronspark-shared-operand v1`) | yes | PASSED: 2 calls (train 2) | train w00: fwd 0.0144 / 0.0143 (-1.00 %), bwd 0.0338 / 0.0334 (-1.15 %) |
| `clock_increments` | 5 | yes (`neuronspark-clock-increments v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0080 / 0.0080 (+0.27 %), bwd 0.0447 / 0.0448 (+0.26 %) |
| `pool_read` | 80 | yes (`neuronspark-pool-read v1`) | yes | PASSED: 1 call (case 1) | case pool_read.py: call 1.5293 / 1.5318 (+0.16 %) |
| `ring_write_prefill` | 6 | yes (`neuronspark-ring-write-prefill v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0420 / 0.0420 (-0.07 %) |
| `ring_recall_prefill` | 6 | yes (`neuronspark-ring-recall-prefill v1`) | yes | PASSED: 2 calls (prefill 2) | prefill w00: fwd 0.2635 / 0.2648 (+0.50 %); prefill w01: fwd 0.1865 / 0.1837 (-1.47 %, DIFFER) |
| `ring_read_t1` | 6 | yes (`neuronspark-ring-read-t1 v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0669 / 0.0671 (+0.25 %) |
| `xq_form_t1` | 6 | yes (`neuronspark-xq-form-t1 v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0043 / 0.0043 (+1.00 %) |
| `plif_centred_input` | 9 | yes (`neuronspark-plif-centred-input v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0515 / 0.0517 (+0.40 %), bwd 0.0741 / 0.0742 (+0.10 %) |
| `plif_scan_train` | 9 | yes (`neuronspark-plif-scan-train v1`) | yes | PASSED: 2 calls (train 2) | train w00: fwd 0.0871 / 0.0874 (+0.41 %), bwd 0.1009 / 0.1007 (-0.28 %) |
| `plif_gate_prefill` | 9 | yes (`neuronspark-plif-gate-prefill v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0312 / 0.0317 (+1.64 %) |
| `plif_scan_prefill` | 9 | yes (`neuronspark-plif-scan-prefill v1`) | yes | PASSED: 2 calls (prefill 2) | prefill w00: fwd 0.0556 / 0.0553 (-0.51 %) |
| `plif_prefill_pair` | 9 | yes (`neuronspark-plif-prefill-pair v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0819 / 0.0820 (+0.12 %) |
| `pkm_unit_queries` | 4 | yes (`neuronspark-pkm-unit-queries v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0144 / 0.0143 (-0.69 %), bwd 0.0117 / 0.0118 (+0.52 %) |
| `pkm_gained_keys` | 4 | yes (`neuronspark-pkm-gained-keys v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.0038 / 0.0039 (+2.65 %), bwd 0.0106 / 0.0107 (+1.67 %) |
| `pkm_topk_train` | 4 | yes (`neuronspark-pkm-topk-train v1`) | yes | PASSED: 2 calls (train 1, prefill 1) | train w00: fwd 0.1394 / 0.1394 (-0.02 %), bwd 0.0227 / 0.0227 (+0.09 %); prefill w00: fwd 0.1400 / 0.1390 (-0.66 %) |
| `pkm_value_read` | 4 | yes (`neuronspark-pkm-value-read v1`) | yes | PASSED: 2 calls (train 1, prefill 1) | train w00: fwd 0.0898 / 0.0889 (-1.02 %), bwd 0.2399 / 0.2402 (+0.08 %); prefill w00: fwd 0.0882 / 0.0880 (-0.22 %) |
| `pkm_norm_gate` | 4 | yes (`neuronspark-pkm-norm-gate v1`) | yes | PASSED: 2 calls (train 1, prefill 1) | train w00: fwd 0.0143 / 0.0143 (+0.49 %), bwd 0.0557 / 0.0559 (+0.35 %); prefill w00: fwd 0.0142 / 0.0142 (+0.15 %) |
| `pkm_topk_softmax_t1` | 4 | yes (`neuronspark-pkm-topk-softmax-t1 v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0053 / 0.0052 (-0.86 %) |
| `copy_switch_aux` | 4 | yes (`neuronspark-copy-switch-aux v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.1887 / 0.1885 (-0.09 %), bwd 0.3840 / 0.3842 (+0.06 %) |
| `balance_ce_per_token` | 7 | yes (`neuronspark-balance-ce-per-token v1`) | yes | PASSED: 1 call (train 1) | train w00: fwd 0.4760 / 0.4761 (+0.02 %), bwd 0.9068 / 0.9069 (+0.01 %) |
| `balance_add` | 4 | yes (`neuronspark-balance-add v1`) | yes | PASSED: 2 calls (prefill 1, decode 1) | prefill w00: fwd 0.0036 / 0.0036 (-1.91 %); decode w00: fwd 0.0079 / 0.0079 (-0.13 %) |
| `copy_stats_prefill` | 4 | yes (`neuronspark-copy-stats-prefill v1`) | yes | PASSED: 1 call (prefill 1); the -inf entries of out.0, out.1 coincide, the rule on the other entries | prefill w00: fwd 0.0768 / 0.0766 (-0.26 %) |
| `xq_queries_prefill` | 9 | yes (`neuronspark-xq-queries-prefill v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0312 / 0.0312 (+0.21 %) |
| `lex_read_prefill` | 5 | yes (`neuronspark-lex-read-prefill v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0264 / 0.0262 (-0.89 %) |
| `vr_mix_prefill` | 4 | yes (`neuronspark-vr-mix-prefill v1`) | yes | PASSED: 2 calls (prefill 2) | prefill w00: fwd 0.0312 / 0.0312 (-0.13 %) |
| `copy_gate_prefill` | 4 | yes (`neuronspark-copy-gate-prefill v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0041 / 0.0037 (-8.82 %) |
| `pool_clock_prefill` | 4 | yes (`neuronspark-pool-clock-prefill v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0321 / 0.0324 (+0.68 %) |
| `plif_step` | 9 | yes (`neuronspark-plif-step v1`) | yes | PASSED: 3 calls (decode 3) | decode w00: fwd 0.0056 / 0.0056 (+0.10 %) |
| `centred_step` | 9 | yes (`neuronspark-centred-step v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0026 / 0.0026 (+0.63 %) |
| `ring_write_step` | 12 | yes (`neuronspark-ring-write-step v1`) | yes | PASSED: 2 calls (decode 2) | decode w00: fwd 0.0112 / 0.0112 (-0.36 %) |
| `read_gain_step` | 5 | yes (`neuronspark-read-gain-step v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0055 / 0.0054 (-1.81 %) |
| `pkm_tail_step` | 5 | yes (`neuronspark-pkm-tail-step v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0029 / 0.0029 (-0.55 %) |
| `induction_step` | 4 | yes (`neuronspark-induction-step v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0116 / 0.0116 (-0.43 %) |
| `induction_match_prefill` | 4 | yes (`neuronspark-induction-match-prefill v1`) | yes | PASSED: 1 call (prefill 1) | prefill w00: fwd 0.0811 / 0.0810 (-0.10 %) |
| `lexical_step` | 4 | yes (`neuronspark-lexical-step v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0131 / 0.0129 (-1.44 %) |
| `pkm_scores_step` | 4 | yes (`neuronspark-pkm-scores-step v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0032 / 0.0031 (-0.33 %) |
| `pkm_value_step` | 4 | yes (`neuronspark-pkm-value-step v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0085 / 0.0085 (-0.06 %) |
| `copy_stats_step` | 5 | yes (`neuronspark-copy-stats-step v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0204 / 0.0204 (-0.22 %) |
| `attn_out_step` | 5 | yes (`neuronspark-attn-out-step v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0032 / 0.0032 (+1.76 %) |
| `lex_read_step` | 5 | yes (`neuronspark-lex-read-step v1`) | yes | PASSED: 1 call (decode 1) | decode w00: fwd 0.0040 / 0.0039 (-1.22 %) |
| `tp_step_core` | 12 | yes (`neuronspark-tp-step-core v1`) | yes | PASSED: 1 call (case 1) | case tp_step_core: call 0.0112 / 0.0113 (+0.79 %) |
| `tp_join` | 6 | yes (`neuronspark-tp-join v1`) | yes | PASSED: 1 call (case 1) | case tp_join: call 0.0049 / 0.0050 (+3.58 %) |
| `tp_tail_gather` | 6 | yes (`neuronspark-tp-tail-gather v1`) | yes | PASSED: 1 call (case 1) | case tp_tail_gather: call 0.0037 / 0.0037 (-0.29 %) |
| `tp_greedy_pick` | 6 | yes (`neuronspark-tp-greedy-pick v1`) | yes | PASSED: 1 call (case 1) | case tp_greedy_pick: call 0.0030 / 0.0028 (-6.46 %) |
| `tp_gated_norm_soma` | 12 | yes (`neuronspark-tp-gated-norm-soma v1`) | yes | PASSED: 1 call (case 1) | case tp_gated_norm_soma: call 0.0743 / 0.0744 (+0.07 %) |
| `tp_fence` | 7 | yes (`neuronspark-tp-fence v1`) | yes | PASSED: 1 call (case 1) | case tp_fence: call 0.0060 / 0.0061 (+1.81 %) |
| `tp_release` | 7 | yes (`neuronspark-tp-release v1`) | yes | PASSED: 1 call (case 1) | case tp_release: call 0.0054 / 0.0054 (+1.13 %) |
| `tp_prefill_join` | 7 | yes (`neuronspark-tp-prefill-join v1`) | yes | PASSED: 1 call (case 1) | case tp_prefill_join: call 0.0378 / 0.0378 (+0.10 %) |
| `tp_prefill_gather` | 7 | yes (`neuronspark-tp-prefill-gather v1`) | yes | PASSED: 1 call (case 1) | case tp_prefill_gather: call 0.0194 / 0.0193 (-0.51 %) |
| `tp_state_pull` | 7 | yes (`neuronspark-tp-state-pull v1`) | yes | PASSED: 1 call (case 1) | case tp_state_pull: call 0.0295 / 0.0296 (+0.29 %) |
