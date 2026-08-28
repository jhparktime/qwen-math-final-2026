# Input placement

Do not commit organizer datasets. Place the final question-only CSV at:

```text
inputs/deep_chal_math_test.csv
```

The file must contain unique `id` values and a non-empty `question` column. A
non-empty `answer` column is rejected by the inference script.
