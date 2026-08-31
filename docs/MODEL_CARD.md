# Model card — frozen R3 deployment candidate

## Solver

- Base: `Qwen/Qwen2.5-3B-Instruct`
- Pinned base revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Adapter run: `RFT-0008D-r3mix-r2continue-r16-a100`
- Adapter SHA-256: `3b13039776a5e77567d8a0e3b8425b762bae747d5d195cd82966a3a87597633f`
- Adapter type: rank-16 LoRA continuation from the R2 Pro4-hint solver

## Training lineage

1. R1 retained native Qwen rollouts on organizer training questions only when
   the terminal boxed integer matched the organizer label.
2. R2 used training-only Solar Pro 4 derivations as private hints on difficult
   training questions. Qwen rewrote the reasoning; teacher text was never an
   assistant target, and only strict Qwen rewrites entered SFT.
3. R3 froze the R2 solver, generated additional Qwen rewrites on hard
   organizer-training questions with the same training-only hint protocol, and
   continued LoRA training on the audited R1/R2/R3 mixture.

The organizer evaluation and leaderboard questions were excluded from training.
Commercial API use was limited to training-data construction; no evaluation or
final-test question is transmitted to an API.

## Frozen inference

- SC16: vLLM engine seed 0; per-request base seed 3; temperature 1.0,
  top-p 0.95, 2,048 output tokens
- Vote: normalized integer plurality; earliest sampled answer breaks a tie
- Adaptive length: only capped sample indices are regenerated at 4,096 and
  then 8,192 tokens; the frozen stronger-consensus router requires top-count
  gain >=2, margin gain >=1, and extended top count >=4
- PAL4 fallback: per-request base seed 3; only original SC16 margin <= 1;
  accept an executable PAL answer only when at least 3 of 4 executions agree;
  a valid PAL replacement has precedence over adaptive length
- Output: one exact signed integer string per input ID

Public leaderboard reference: `0.79783` for this frozen R3 SC16 + adaptive
length + PAL3 configuration. It is descriptive only and is not used during
final inference.

## Limitations

The solver can produce convincing but incorrect reasoning. Longer output is
not automatically better, so adaptive replacement is consensus-gated and does
not override a valid PAL result. PAL code is restricted to a small Python
standard-library allowlist with CPU, wall-time, memory, file-size, and
file-descriptor limits.
