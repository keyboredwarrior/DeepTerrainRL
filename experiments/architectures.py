from __future__ import annotations

from pathlib import Path

from .config import ArchitectureConfig


def resolve_architecture_template(config: ArchitectureConfig) -> str:
    if config.family == "mlp_baseline":
        return "experiments/templates/policy_mlp_baseline.prototxt"
    if config.family == "cnn_baseline":
        return "experiments/templates/policy_cnn_baseline.prototxt"
    if config.family == "terrain_attention":
        return "experiments/templates/policy_terrain_attention.prototxt"
    raise ValueError(f"Unsupported architecture family: {config.family}")


def ensure_templates_exist() -> None:
    required = [
        "experiments/templates/policy_mlp_baseline.prototxt",
        "experiments/templates/policy_cnn_baseline.prototxt",
        "experiments/templates/policy_terrain_attention.prototxt",
    ]
    for rel in required:
        path = Path(rel)
        if not path.exists():
            raise FileNotFoundError(f"Missing architecture template: {path}")
