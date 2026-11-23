# Reproducibility checklist

- Record commit SHA, Python/package versions, config snapshot, and random seeds.
- Record exact input file names and SHA-256 hashes.
- Persist train/validation/test row counts and class distributions.
- Fit all preprocessing on train only.
- Save per-agent feature lists and fitted model artifacts.
- Save predictions and metrics per run ID.
- Never overwrite prior experiment outputs.
- Do not report results until the complete pipeline has actually run.
