from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import re
from statistics import mean, pstdev
from typing import Dict, Iterable, List

ITER_RE = re.compile(r"^Iter\s+(?P<iter>\d+)\s*$")
REWARD_RE = re.compile(r"^Avg\s+Tuple\s+Reward:\s+(?P<reward>-?\d+(?:\.\d+)?)")
LOSS_RE = re.compile(r"loss[:=]\s*(?P<loss>-?\d+(?:\.\d+)?)", re.IGNORECASE)
GRAD_RE = re.compile(r"grad(?:ient)?(?:_norm| norm)?[:=]\s*(?P<grad>-?\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class TrainingPoint:
    iteration: int
    avg_tuple_reward: float


@dataclass
class EvalSummary:
    terrain: str
    episodes: int
    avg_dist: float
    success_rate: float


def parse_training_log(log_path: str | Path) -> List[TrainingPoint]:
    points: List[TrainingPoint] = []
    current_iter: int | None = None

    for raw_line in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        iter_match = ITER_RE.match(line)
        if iter_match:
            current_iter = int(iter_match.group("iter"))
            continue

        reward_match = REWARD_RE.match(line)
        if reward_match and current_iter is not None:
            points.append(TrainingPoint(iteration=current_iter, avg_tuple_reward=float(reward_match.group("reward"))))

    return points


def parse_loss_series(log_path: str | Path) -> List[float]:
    losses: List[float] = []
    for raw_line in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LOSS_RE.search(raw_line)
        if match:
            losses.append(float(match.group("loss")))
    return losses


def parse_gradient_norm_series(log_path: str | Path) -> List[float]:
    grads: List[float] = []
    for raw_line in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = GRAD_RE.search(raw_line)
        if match:
            grads.append(float(match.group("grad")))
    return grads


def write_reward_curve(points: Iterable[TrainingPoint], out_csv: str | Path) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "avg_tuple_reward"])
        for p in points:
            writer.writerow([p.iteration, p.avg_tuple_reward])


def read_eval_summary(path: str | Path) -> EvalSummary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalSummary(
        terrain=payload["terrain"],
        episodes=int(payload["episodes"]),
        avg_dist=float(payload["avg_dist"]),
        success_rate=float(payload["success_rate"]),
    )


def build_comparison_report(points: List[TrainingPoint], evals: List[EvalSummary], wall_clock_s: float) -> Dict[str, float]:
    rewards = [p.avg_tuple_reward for p in points]
    successes = [e.success_rate for e in evals]
    return {
        "reward_mean": mean(rewards) if rewards else 0.0,
        "reward_variance": (pstdev(rewards) ** 2) if len(rewards) > 1 else 0.0,
        "reward_final": rewards[-1] if rewards else 0.0,
        "success_rate_mean": mean(successes) if successes else 0.0,
        "wall_clock_s": wall_clock_s,
    }


def write_comparison_report(report: Dict[str, float], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
