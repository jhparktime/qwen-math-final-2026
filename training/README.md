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
   - The resulting `adapter_final` is the submitted solver.

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
