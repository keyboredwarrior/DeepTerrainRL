from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import subprocess
import time
from typing import Dict, List

from .architectures import ensure_templates_exist, resolve_architecture_template
from .config import ExperimentConfig
from .metrics import (
    build_comparison_report,
    parse_training_log,
    read_eval_summary,
    write_comparison_report,
    write_reward_curve,
)


@dataclass
class RunArtifacts:
    run_dir: str
    checkpoint_path: str
    cpp_eval_args_file: str
    reward_curve_file: str
    comparison_file: str


class ExperimentRunner:
    """Consistent API that wraps simulator train/eval runs with reproducibility defaults."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self) -> RunArtifacts:
        ensure_templates_exist()
        run_dir = self._build_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)

        config_snapshot = run_dir / "config_snapshot.json"
        self.config.snapshot(config_snapshot)

        train_args_path, checkpoint_path = self._write_train_args(run_dir)
        train_log = run_dir / "train.log"

        start = time.time()
        self._execute([self.config.binary_path, "-arg_file=", train_args_path.as_posix()], train_log)
        wall_clock_s = time.time() - start

        points = parse_training_log(train_log)
        reward_curve = run_dir / "reward_curve.csv"
        write_reward_curve(points, reward_curve)

        eval_summaries = self._run_eval_sweep(run_dir, checkpoint_path)
        comparison = build_comparison_report(points, eval_summaries, wall_clock_s)
        comparison_path = run_dir / "comparison.json"
        write_comparison_report(comparison, comparison_path)

        cpp_args = self._write_cpp_eval_args(run_dir, checkpoint_path)
        return RunArtifacts(
            run_dir=str(run_dir),
            checkpoint_path=str(checkpoint_path),
            cpp_eval_args_file=str(cpp_args),
            reward_curve_file=str(reward_curve),
            comparison_file=str(comparison_path),
        )

    def _build_run_dir(self) -> Path:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        name = f"{self.config.name}_{self.config.algorithm}_seed{self.config.seed}_{stamp}"
        return Path(self.config.output_root) / name

    def _write_train_args(self, run_dir: Path) -> tuple[Path, Path]:
        arch_path = resolve_architecture_template(self.config.architecture)
        checkpoint_name = f"{self.config.name}_{self.config.algorithm}_{self.config.architecture.family}_seed{self.config.seed}.h5"
        checkpoint_path = run_dir / checkpoint_name
        args_path = run_dir / "train_args.txt"

        args = {
            "scenario": f"train_{self.config.algorithm}",
            "output_path": checkpoint_path.as_posix(),
            "character_file": self.config.character_file,
            "state_file": self.config.state_file,
            "char_type": self.config.character,
            "char_ctrl": self.config.controller,
            "terrain_file": self.config.terrain_file,
            "policy_arch_config": arch_path,
            "policy_checkpoint": self.config.policy_checkpoint_solver,
            "trainer_max_iter": str(self.config.max_iters),
            "trainer_curriculum_iters": str(self.config.curriculum.total_iters),
            "trainer_curriculum_stage_iters": str(self.config.curriculum.stage_iters),
            "rand_seed": str(self.config.seed),
        }
        self._write_args_file(args_path, args)
        return args_path, checkpoint_path

    def _run_eval_sweep(self, run_dir: Path, checkpoint_path: Path):
        summaries = []
        for terrain in self.config.curriculum.terrain_types:
            terrain_key = Path(terrain).stem
            eval_summary_path = run_dir / f"eval_{terrain_key}.json"
            eval_args_path = run_dir / f"eval_{terrain_key}_args.txt"
            eval_log = run_dir / f"eval_{terrain_key}.log"

            args = {
                "scenario": "poli_eval",
                "character_file": self.config.character_file,
                "state_file": self.config.state_file,
                "char_type": self.config.character,
                "char_ctrl": self.config.controller,
                "terrain_file": terrain,
                "policy_arch_config": resolve_architecture_template(self.config.architecture),
                "policy_model": checkpoint_path.as_posix(),
                "poli_eval_rand_seed": str(self.config.seed),
                "poli_eval_max_episodes": "20",
                "poli_eval_success_dist": "1.0",
                "poli_eval_output": eval_summary_path.as_posix(),
            }
            self._write_args_file(eval_args_path, args)
            self._execute([self.config.binary_path, "-arg_file=", eval_args_path.as_posix()], eval_log)
            summaries.append(read_eval_summary(eval_summary_path))
        return summaries

    def _write_cpp_eval_args(self, run_dir: Path, checkpoint_path: Path) -> Path:
        args_path = run_dir / "cpp_runtime_eval_args.txt"
        args = {
            "scenario": "poli_eval",
            "character_file": self.config.character_file,
            "state_file": self.config.state_file,
            "char_type": self.config.character,
            "char_ctrl": self.config.controller,
            "terrain_file": self.config.terrain_file,
            "policy_arch_config": resolve_architecture_template(self.config.architecture),
            "policy_model": checkpoint_path.as_posix(),
        }
        self._write_args_file(args_path, args)
        return args_path

    def _write_args_file(self, path: Path, args: Dict[str, str]) -> None:
        lines = [f"-{k}= {v}" for k, v in args.items()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _execute(self, cmd: List[str], log_path: Path) -> None:
        with open(log_path, "w", encoding="utf-8") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
