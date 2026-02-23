# Experiments Layer

This package adds a consistent API for simulator training/evaluation runs with reproducibility and architecture switching.

## What is included

- **Config schema** (`config.py`) for:
  - algorithm
  - network family (MLP/CNN/terrain-attention)
  - attention toggles
  - seed
  - terrain curriculum stages
- **Architecture families** (`templates/*.prototxt`):
  - `mlp_baseline`
  - `cnn_baseline`
  - `terrain_attention` (attention-gated terrain features)
- **Metrics capture** (`metrics.py`):
  - reward curves from training logs
  - reward variance/stability
  - success rates by terrain (from evaluation summaries)
  - wall-clock runtime
- **Reproducibility controls**:
  - fixed `rand_seed` forwarding
  - deterministic checkpoint naming convention
  - config snapshot (`config_snapshot.json`) per run
- **C++ runtime playback/eval handoff**:
  - generated `cpp_runtime_eval_args.txt` points the runtime at trained checkpoints.

## Usage

```bash
python3 experiments/run_experiment.py --config experiments/configs/dog_attention_experiment.json
```

Outputs are grouped under `experiments/results/<name>_<algo>_seed<seed>_<timestamp>/`.
