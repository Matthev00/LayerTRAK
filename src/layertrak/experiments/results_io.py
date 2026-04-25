import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def make_run_dir(base_dir: Path) -> Path:
    """Create a timestamped directory for a single experiment run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def score_stats(scores: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for a score matrix."""
    stable_scores = scores.astype(np.float64, copy=False)
    percentiles = np.percentile(stable_scores, [1, 5, 50, 95, 99])
    return {
        "min": float(stable_scores.min()),
        "max": float(stable_scores.max()),
        "mean": float(stable_scores.mean()),
        "std": float(stable_scores.std()),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_topk_csv(path: Path, scores: np.ndarray, top_k: int = 10) -> None:
    """Write top-k training samples per target sorted by descending score."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_index", "rank", "train_index", "score"])
        writer.writeheader()
        for target_index, target_scores in enumerate(scores):
            top_indices = np.argsort(target_scores)[::-1][:top_k]
            for rank, train_index in enumerate(top_indices, start=1):
                writer.writerow({
                    "target_index": target_index,
                    "rank": rank,
                    "train_index": int(train_index),
                    "score": float(target_scores[train_index]),
                })


def write_overview_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one summary row per architecture/configuration pair."""
    if not rows:
        return

    fieldnames = [
        "architecture",
        "config_name",
        "num_targets",
        "train_set_size",
        "scores_shape",
        "num_tracked_params",
        "checkpoint_path",
        "prefixes",
        "run_subdir",
        "min",
        "max",
        "mean",
        "std",
        "p1",
        "p5",
        "p50",
        "p95",
        "p99",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_config_artifacts(
    artifact_dir: Path,
    *,
    scores: np.ndarray,
    architecture: str,
    config_name: str,
    prefixes: list[str] | None,
    num_tracked_params: int,
    num_targets: int,
    train_set_size: int,
    checkpoint_path: Path,
) -> dict[str, object]:
    """Save all artifacts for a single architecture/configuration pair."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stats = score_stats(scores)
    summary = {
        "architecture": architecture,
        "config_name": config_name,
        "prefixes": prefixes,
        "num_tracked_params": num_tracked_params,
        "num_targets": num_targets,
        "train_set_size": train_set_size,
        "scores_shape": list(scores.shape),
        "checkpoint_path": str(checkpoint_path),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "score_files": {
            "scores": "scores.npy",
            "stats": "score_stats.json",
            "topk": "topk.csv",
        },
    }

    np.save(artifact_dir / "scores.npy", scores)
    write_json(artifact_dir / "summary.json", summary)
    write_json(artifact_dir / "score_stats.json", stats)
    write_topk_csv(artifact_dir / "topk.csv", scores)

    return {
        "architecture": architecture,
        "config_name": config_name,
        "num_targets": num_targets,
        "train_set_size": train_set_size,
        "scores_shape": "x".join(str(dim) for dim in scores.shape),
        "num_tracked_params": num_tracked_params,
        "checkpoint_path": str(checkpoint_path),
        "prefixes": "" if prefixes is None else ",".join(prefixes),
        "run_subdir": str(artifact_dir.relative_to(artifact_dir.parents[2])),
        **stats,
    }
