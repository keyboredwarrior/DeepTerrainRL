from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import subprocess
import time
from typing import Dict, List, Any

from .architectures import ensure_templates_exist, resolve_architecture_template
from .config import ExperimentConfig
from .metrics import (
    build_comparison_report,
    parse_gradient_norm_series,
    parse_loss_series,
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
    milestone_report_file: str


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

        train_args_path, checkpoint_path = self._write_train_args(run_dir, max_iters=self.config.max_iters)
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

        milestone_report = self._run_transition_milestones(run_dir, train_log, checkpoint_path, eval_summaries)
        milestone_path = run_dir / "milestones.json"
        milestone_path.write_text(json.dumps(milestone_report, indent=2, sort_keys=True), encoding="utf-8")

        cpp_args = self._write_cpp_eval_args(run_dir, checkpoint_path)
        return RunArtifacts(
            run_dir=str(run_dir),
            checkpoint_path=str(checkpoint_path),
            cpp_eval_args_file=str(cpp_args),
            reward_curve_file=str(reward_curve),
            comparison_file=str(comparison_path),
            milestone_report_file=str(milestone_path),
        )

    def _build_run_dir(self) -> Path:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        name = f"{self.config.name}_{self.config.algorithm}_seed{self.config.seed}_{stamp}"
        return Path(self.config.output_root) / name

    def _write_train_args(self, run_dir: Path, max_iters: int) -> tuple[Path, Path]:
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
            "policy_backend": self.config.policy_backend,
            "legacy_mode": "true" if self.config.legacy_mode_enabled else "false",
            "trainer_max_iter": str(max_iters),
            "trainer_curriculum_iters": str(self.config.curriculum.total_iters),
            "trainer_curriculum_stage_iters": str(self.config.curriculum.stage_iters),
            "rand_seed": str(self.config.seed),
        }
        self._write_args_file(args_path, args)
        return args_path, checkpoint_path

    def _run_eval_sweep(self, run_dir: Path, checkpoint_path: Path, backend: str | None = None):
        summaries = []
        selected_backend = backend or self.config.policy_backend
        for terrain in self.config.curriculum.terrain_types:
            terrain_key = Path(terrain).stem
            eval_summary_path = run_dir / f"eval_{selected_backend}_{terrain_key}.json"
            eval_args_path = run_dir / f"eval_{selected_backend}_{terrain_key}_args.txt"
            eval_log = run_dir / f"eval_{selected_backend}_{terrain_key}.log"

            args = {
                "scenario": "poli_eval",
                "character_file": self.config.character_file,
                "state_file": self.config.state_file,
                "char_type": self.config.character,
                "char_ctrl": self.config.controller,
                "terrain_file": terrain,
                "policy_arch_config": resolve_architecture_template(self.config.architecture),
                "policy_model": checkpoint_path.as_posix(),
                "policy_backend": selected_backend,
                "legacy_mode": "true" if selected_backend == "legacy_pytorch" else "false",
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
            "policy_backend": self.config.policy_backend,
            "legacy_mode": "true" if self.config.legacy_mode_enabled else "false",
        }
        self._write_args_file(args_path, args)
        return args_path

    def _run_transition_milestones(self, run_dir: Path, train_log: Path, checkpoint_path: Path, eval_summaries: List[Any]) -> Dict[str, Any]:
        report: Dict[str, Any] = {}

        if self.config.milestones.enable_parity_check:
            report["milestone_a_inference_parity"] = self._check_inference_parity(run_dir, checkpoint_path)

        if self.config.milestones.enable_single_step_sanity:
            report["milestone_b_single_step_training"] = self._check_single_step_training(train_log)

        if self.config.milestones.enable_short_horizon_sanity:
            report["milestone_c_short_horizon_training"] = self._check_short_horizon(run_dir)

        report["milestone_d_full_benchmark_matrix"] = {
            "passed": bool(self.config.milestones.enable_full_benchmark_matrix and len(eval_summaries) > 0),
            "enabled": self.config.milestones.enable_full_benchmark_matrix,
            "note": "Use experiments/benchmark_matrix.py for full terrain x architecture sweeps.",
        }

        report["legacy_removal_gate"] = {
            "can_remove_legacy_mode": all(v.get("passed", False) for k, v in report.items() if k.startswith("milestone_")),
            "docs_update_required": True,
        }
        return report

    def _check_inference_parity(self, run_dir: Path, checkpoint_path: Path) -> Dict[str, Any]:
        legacy = self._run_eval_sweep(run_dir, checkpoint_path, backend="legacy_pytorch")
        modern = self._run_eval_sweep(run_dir, checkpoint_path, backend="modern_backend")

        terrain_deltas: Dict[str, Dict[str, float]] = {}
        within_tolerance = True
        tol = self.config.milestones.parity_tolerance
        for lhs, rhs in zip(legacy, modern):
            success_delta = abs(lhs.success_rate - rhs.success_rate)
            dist_delta = abs(lhs.avg_dist - rhs.avg_dist)
            terrain_deltas[lhs.terrain] = {
                "success_rate_delta": success_delta,
                "avg_dist_delta": dist_delta,
            }
            within_tolerance = within_tolerance and success_delta <= tol and dist_delta <= tol

        return {
            "passed": within_tolerance,
            "tolerance": tol,
            "terrain_deltas": terrain_deltas,
        }

    def _check_single_step_training(self, train_log: Path) -> Dict[str, Any]:
        losses = parse_loss_series(train_log)
        grads = parse_gradient_norm_series(train_log)
        has_loss_drop = len(losses) > 1 and losses[-1] < losses[0]
        nonzero_grad_count = sum(1 for g in grads if g != 0.0)
        min_grads = self.config.milestones.min_nonzero_gradients
        return {
            "passed": has_loss_drop and nonzero_grad_count >= min_grads,
            "loss_decreased": has_loss_drop,
            "num_loss_samples": len(losses),
            "nonzero_gradients": nonzero_grad_count,
            "min_nonzero_gradients": min_grads,
        }

    def _check_short_horizon(self, run_dir: Path) -> Dict[str, Any]:
        short_dir = run_dir / "short_horizon"
        short_dir.mkdir(parents=True, exist_ok=True)
        train_args_path, _ = self._write_train_args(short_dir, max_iters=self.config.milestones.short_horizon_iters)
        short_log = short_dir / "train_short.log"

        crashed = False
        try:
            self._execute([self.config.binary_path, "-arg_file=", train_args_path.as_posix()], short_log)
        except subprocess.CalledProcessError:
            crashed = True

        short_points = parse_training_log(short_log) if short_log.exists() else []
        return {
            "passed": (not crashed) and len(short_points) > 0,
            "crashed": crashed,
            "reward_points": len(short_points),
            "short_horizon_iters": self.config.milestones.short_horizon_iters,
        }

    def _write_args_file(self, path: Path, args: Dict[str, str]) -> None:
        lines = [f"-{k}= {v}" for k, v in args.items()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _execute(self, cmd: List[str], log_path: Path) -> None:
        with open(log_path, "w", encoding="utf-8") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
