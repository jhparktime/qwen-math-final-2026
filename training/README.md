# Training reproduction order

These artifacts document the exact staged route to the submitted adapter. They
must never be run on leaderboard or final-test questions.

1. `EXP-0007_Upstage_TrainOnly_CoT.py`
   - Cell-formatted Colab script for Solar Pro 4 training-only teacher traces.
   - Reads only `clean_train.csv`; the API key comes from a Colab secret or an
     environment variable and is never stored in the repository.
2. `RFT-0002_R1_native_k4_parallel_3session_a100.ipynb`
   - Four native Qwen rollouts per organizer training question.
   - Strict terminal-boxed label verification and at most two retained traces.
3. `RFT-0002_R1_AB_BF16_LoRA_A100.ipynb`
   - Fresh R1 LoRA A/B training. Lane B checkpoint-178 is the R2 parent.
4. `RFT-0003A_Pro4_Hint_Qwen_Generation_A100.ipynb`
   - Selects low-success organizer-train questions, supplies an existing verified
     Pro4 solution as a private hint, and stores only Qwen-rewritten responses
     that pass strict label verification.
5. `RFT-0004B_R2_Pro4_Hint_LowDrift_LoRA_A100.ipynb`
   - Continues Lane B on the frozen 65% rewritten-hard / 35% R1-anchor corpus.
   - The resulting `adapter_final` is the frozen R2 parent of the submitted R3 solver.
6. `RFT-0008B_R3_Pro4Hint_Rollout_A100.ipynb`
   - Freezes the R2 Pro4-hint solver and generates four Qwen-only rewrites for
     hard organizer-training questions, using the same training-only hint protocol.
   - Strictly filters terminal boxed answers and excludes all fixed holdout IDs
     and templates before materializing the R3 corpus.
7. `RFT-0008D_R3Mix_R2Continuation_r16_A100.ipynb`
   - Continues the R2 rank-16 LoRA on the audited R1/R2/R3 Qwen-response mixture.
   - Writes checkpoints, `adapter_final`, corpus/report hashes, TensorBoard logs,
     and W&B offline logs. This `adapter_final` is the submitted R3 solver.

The public copies of the R3 notebooks are source-only: cell outputs were
cleared before publication. They contain neither organizer evaluation records,
test/leaderboard predictions, teacher API credentials, nor raw training traces.

## Post-submission candidate (not the frozen submitted adapter)

`RFT-0015_R2Pro4_PublicMath_Continuation_A100.py` and its companion notebook
test the previously unrun ablation: continuing the frozen R2 Pro4-hint adapter
on the strict verified DATA-0004 public-math corpus only. It records public
source provenance and W&B-offline logs, and must be promoted only after frozen
tune/dev Exact Match evaluation. It never reads organizer holdouts, leaderboard,
or private-test questions.

All Qwen model loads in these archived notebooks are pinned to commit
`aa8e72537993ba99e69dfaafa59ed015b17504d1` in this repository copy. Each
notebook records source hashes, filters, holdout guards, checkpoints, and local
TensorBoard/W&B-offline paths.

The organizer datasets and generated trace files are deliberately not committed.
Their required filenames and hashes are asserted inside the notebooks.
