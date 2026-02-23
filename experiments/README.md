# Experiments Layer

This package adds a consistent API for simulator training/evaluation runs with reproducibility, architecture switching, and backend transition milestones.

## What is included

- **Config schema** (`config.py`) for:
  - algorithm
  - network family (MLP/CNN/terrain-attention)
  - backend selection (`legacy_caffe` or `modern_backend`)
  - transition milestone settings (A/B/C/D)
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
  - optional loss/gradient extraction for training sanity checks
- **Reproducibility controls**:
  - fixed `rand_seed` forwarding
  - deterministic checkpoint naming convention
  - config snapshot (`config_snapshot.json`) per run
- **C++ runtime playback/eval handoff**:
  - generated `cpp_runtime_eval_args.txt` points the runtime at trained checkpoints and selected backend.

## Transition milestones

`ExperimentRunner` writes a `milestones.json` report with the following checks:

1. **Milestone A** inference parity (`legacy_caffe` vs `modern_backend`) using same checkpoint and eval terrain list.
2. **Milestone B** single-step training sanity from training logs (loss decrease and non-zero gradients when present in logs).
3. **Milestone C** short-horizon training smoke run (no crashes, produces reward points).
4. **Milestone D** benchmark matrix readiness marker (`experiments/benchmark_matrix.py` for terrain × architecture sweeps).
5. **Legacy removal gate** is marked true only when all milestone checks pass; docs update is explicitly required.

## Usage

```bash
python3 experiments/run_experiment.py --config experiments/configs/dog_attention_experiment.json
```

Outputs are grouped under `experiments/results/<name>_<algo>_seed<seed>_<timestamp>/`.

## Full benchmark matrix

```bash
python3 experiments/benchmark_matrix.py --config experiments/configs/dog_attention_experiment.json
```

This executes a terrain × architecture matrix and writes a manifest JSON under `experiments/results/`.
