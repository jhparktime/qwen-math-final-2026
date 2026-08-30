# Data provenance and compliance

## Organizer data

The model uses the organizer-provided `deep_chal_math_train.csv` for training.
Organizer leaderboard and test files are not redistributed in this repository.
They are read only by the local inference command.

Some archived training notebooks refer to `test_v1.csv`. This is a fixed
training-derived internal holdout created before model training, not the
organizer's final private test file.

## Generated training traces

| Source | Scope | Use |
|---|---|---|
| Qwen/Qwen2.5-3B-Instruct | Organizer train only | Native R1 rollout and Qwen-form rewritten R2 traces |
| Solar Pro 4 through Upstage API | Organizer train only | Private reasoning hint during R2 data construction |

Solar Pro 4 was never called on leaderboard or test questions. Its response was
not the SFT assistant target: the permitted Qwen base generated the final
training response, which was retained only when its terminal boxed integer
matched the organizer training label.

## Explicit exclusions

- No leaderboard or final-test answer lookup
- No web search, retrieval, or external API during inference
- No non-Qwen base model or external-model ensemble during inference
- No manual editing of validation, leaderboard, or final-test predictions
- No organizer dataset files committed to Git

The raw generated training corpus is not required for final inference. Training
reproduction requires access to the organizer training file and the audited
training-only trace artifacts described above. The public source-code sequence
is documented in [`training/README.md`](../training/README.md), including the
R3 rollout and continuation notebooks that produced the submitted adapter.
