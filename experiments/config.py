from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal
import json

ArchitectureFamily = Literal["mlp_baseline", "cnn_baseline", "terrain_attention"]
PolicyBackend = Literal["legacy_caffe", "modern_backend"]


@dataclass
class TerrainCurriculumConfig:
    terrain_types: List[str]
    stage_iters: int = 0
    total_iters: int = 0


@dataclass
class ArchitectureConfig:
    family: ArchitectureFamily
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])
    conv_channels: List[int] = field(default_factory=lambda: [32, 32])
    use_attention: bool = False
    attention_heads: int = 4
    attention_dim: int = 128


@dataclass
class TransitionMilestoneConfig:
    enable_parity_check: bool = True
    parity_tolerance: float = 0.05
    enable_single_step_sanity: bool = True
    min_nonzero_gradients: int = 1
    enable_short_horizon_sanity: bool = True
    short_horizon_iters: int = 100
    enable_full_benchmark_matrix: bool = False


@dataclass
class ExperimentConfig:
    name: str
    algorithm: str
    seed: int
    binary_path: str
    character: str
    controller: str
    state_file: str
    character_file: str
    architecture: ArchitectureConfig
    curriculum: TerrainCurriculumConfig
    policy_checkpoint_solver: str
    policy_backend: PolicyBackend = "legacy_caffe"
    legacy_mode_enabled: bool = True
    max_iters: int = 2000
    output_root: str = "experiments/results"
    terrain_file: str = "data/terrain/mixed.txt"
    milestones: TransitionMilestoneConfig = field(default_factory=TransitionMilestoneConfig)

    @staticmethod
    def from_file(path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        payload["architecture"] = ArchitectureConfig(**payload["architecture"])
        payload["curriculum"] = TerrainCurriculumConfig(**payload["curriculum"])
        payload["milestones"] = TransitionMilestoneConfig(**payload.get("milestones", {}))
        return ExperimentConfig(**payload)

    def snapshot(self, dst: str | Path) -> None:
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
