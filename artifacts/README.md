# Frozen R3 adapter

Download the frozen R3 adapter before entering offline inference:

```bash
python scripts/fetch_frozen_adapter.py
```

The command downloads both release assets, validates each SHA-256, and writes:

```text
artifacts/r3_r2continue_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

| Asset | SHA-256 |
|---|---|
| `adapter_config.json` | `a292598b9f11c1395270c8eb3013852d949a2a5b1315b8691cf08f8ddfa269df` |
| `adapter_model.safetensors` | `3b13039776a5e77567d8a0e3b8425b762bae747d5d195cd82966a3a87597633f` |

The 119.8 MB weight is stored as a [GitHub Release asset](https://github.com/jhparktime/qwen-math-final-2026/releases/tag/r3-r2continue-v1), not ordinary Git. Do not substitute a different checkpoint.
