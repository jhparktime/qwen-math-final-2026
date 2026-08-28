# Model card

## Final solver

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Base revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Adapter run: `RFT-0004B-r2-pro4-hint-lowdrift-lora`
- Adapter weight SHA-256: `e4a22286b3b6a3108c0f2a374012601309abee6511b96b2a108749d432909f11`
- Adapter type: LoRA continuation from the R1 Lane-B checkpoint

The adapter was trained only from organizer-provided training questions and
derived training-only reasoning traces. Evaluation questions were excluded.

## Training summary

1. R1 sampled four native Qwen solutions per organizer training question.
2. Only Qwen solutions whose terminal boxed integer matched the organizer label
   passed the strict filter; at most two traces per question were retained.
3. For the R2 hard subset, pre-existing Solar Pro 4 training-only solutions were
   used as private hints. Qwen rewrote each derivation in its own style.
4. Only rewritten Qwen responses that passed the same terminal boxed-answer
   filter entered SFT. Teacher text was not used as the assistant target.
5. The R2 continuation used 65% hard rewritten traces and 35% untouched R1
   anchors, assistant-only loss, BF16, learning rate `1e-5`, effective batch 48,
   and 0.5 epoch.

Commercial API generation was restricted to training data construction. No
leaderboard or final-test question is sent to an API.

## Frozen inference

- SC16: temperature 1.0, top-p 0.95, 2,048 output tokens
- Vote: normalized integer plurality
- Tie break: earliest sampled answer among tied top counts
- Length recovery: only capped sample indices are reconsidered at 4,096 and,
  when still capped, 8,192 tokens
- PAL fallback: original SC16 margin at most 1 and executable PAL agreement at
  least 3 of 4
- Final answer format: one exact signed integer string

Public leaderboard reference: R2+PAL `0.80144`. This value is descriptive and
is not used by the inference code.

## Limitations

The solver can produce persuasive but incorrect arithmetic. Longer generations
are therefore routed conservatively and never override a successful PAL
decision. PAL execution is restricted to a small Python standard-library
allowlist with time, memory, file-size, and file-descriptor limits.
