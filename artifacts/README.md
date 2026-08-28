# Frozen adapter

Place the frozen R2 Pro4-hint LoRA directory here before inference:

```text
artifacts/r2_pro4_hint_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

Required weight SHA-256:

```text
e4a22286b3b6a3108c0f2a374012601309abee6511b96b2a108749d432909f11
```

The inference script rejects a missing or mismatched adapter. The adapter must
be distributed through Git LFS or a public release asset before final handoff;
do not silently substitute another checkpoint.
