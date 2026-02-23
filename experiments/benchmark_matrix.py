#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import List

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner


DEFAULT_ARCHITECTURES: List[str] = ["mlp_baseline", "cnn_baseline", "terrain_attention"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmark matrix across terrains and architecture variants")
    parser.add_argument("--config", required=True, help="Base experiment config JSON")
    parser.add_argument("--architectures", nargs="*", default=DEFAULT_ARCHITECTURES)
    args = parser.parse_args()

    base_config = ExperimentConfig.from_file(args.config)
    matrix_results = []

    for arch in args.architectures:
        for terrain in base_config.curriculum.terrain_types:
            cfg = copy.deepcopy(base_config)
            cfg.architecture.family = arch
            cfg.terrain_file = terrain
            cfg.name = f"{base_config.name}_{Path(terrain).stem}_{arch}"
            cfg.milestones.enable_full_benchmark_matrix = True
            artifacts = ExperimentRunner(cfg).run()
            matrix_results.append(
                {
                    "architecture": arch,
                    "terrain": terrain,
                    "run_dir": artifacts.run_dir,
                    "comparison": artifacts.comparison_file,
                    "milestones": artifacts.milestone_report_file,
                }
            )

    out_file = Path(base_config.output_root) / f"{base_config.name}_full_benchmark_matrix.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(matrix_results, indent=2, sort_keys=True), encoding="utf-8")
    print(out_file.as_posix())


if __name__ == "__main__":
    main()
