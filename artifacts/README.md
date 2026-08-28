# Frozen adapter

`adapter_config.json` is versioned with this repository. Download the frozen
weight once, before entering the offline inference stage:

```bash
python scripts/fetch_frozen_adapter.py
```

The command retrieves the pinned GitHub Release asset and validates its SHA-256.
The resulting directory is:

```text
artifacts/r2_pro4_hint_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

Required weight SHA-256:

```text
e4a22286b3b6a3108c0f2a374012601309abee6511b96b2a108749d432909f11
```

The inference script rejects a missing or mismatched adapter. The 119.8 MB
weight is stored as a GitHub Release asset rather than ordinary Git because it
exceeds GitHub's 100 MB file limit; do not silently substitute another
checkpoint.

To create that Release from the original mounted Drive artifact, use
`notebooks/UPLOAD_FROZEN_ADAPTER_TO_GITHUB.ipynb`. The one-time GitHub device
login remains in the Colab runtime and no credential is committed.
