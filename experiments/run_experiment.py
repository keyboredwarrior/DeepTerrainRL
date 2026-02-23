#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DeepTerrainRL experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    args = parser.parse_args()

    config = ExperimentConfig.from_file(args.config)
    runner = ExperimentRunner(config)
    artifacts = runner.run()
    print(json.dumps(artifacts.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
