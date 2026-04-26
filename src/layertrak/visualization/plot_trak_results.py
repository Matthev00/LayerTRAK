import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from layertrak.settings import settings

CONFIG_ORDER = ["head_only", "late", "mid_late", "early", "full_model"]
OVERVIEW_METRICS = ("mean", "std", "p99_abs", "top1_mean", "top10_mean")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the plotting script."""
    parser = argparse.ArgumentParser(
        description="Generate corrected plots for a saved LayerTRAK experiment run.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a specific run directory. Defaults to the newest directory in trak_results/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated plots. Defaults to <run-dir>/plots.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=25,
        help="Maximum number of target rows shown in score heatmaps.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=100,
        help="Maximum number of training columns shown in score heatmaps.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=80,
        help="Number of bins used in score histograms.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of most influential training samples written per target.",
    )
    parser.add_argument(
        "--num-extreme-targets",
        type=int,
        default=5,
        help="Number of target samples shown as positive/negative image grids per config.",
    )
    parser.add_argument(
        "--extreme-top-k",
        type=int,
        default=5,
        help="Number of positive and negative training samples shown in each image grid.",
    )
    parser.add_argument(
        "--skip-extremes",
        action="store_true",
        help="Skip CIFAR-10 target/top positive/top negative image grids.",
    )
    return parser.parse_args()


def _resolve_run_dir(run_dir: Path | None) -> Path:
    """Return the selected filesystem run directory or the newest available run."""
    if run_dir is not None:
        return run_dir

    results_root = settings.project_root / settings.trak_results_dir
    candidates = sorted(path for path in results_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {results_root}")
    return candidates[-1]


def _load_overview_from_path(path: Path) -> list[dict[str, str]]:
    """Load a run overview CSV file from the filesystem."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_summary(path: Path) -> dict[str, Any]:
    """Load the summary metadata for a single filesystem configuration."""
    return json.loads(path.read_text(encoding="utf-8"))


def _make_output_dir(run_dir: Path, output_dir: Path | None) -> Path:
    """Create and return the directory used for generated plots."""
    resolved_output_dir = output_dir if output_dir is not None else run_dir / "plots"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return resolved_output_dir


def _config_sort_key(config_name: str) -> tuple[int, str]:
    try:
        return CONFIG_ORDER.index(config_name), config_name
    except ValueError:
        return len(CONFIG_ORDER), config_name


def _as_int(value: Any, field_name: str) -> int:
    if value is None:
        raise KeyError(f"Missing {field_name} in summary.json")
    return int(value)


def _normalize_scores(scores: np.ndarray, summary: dict[str, Any]) -> np.ndarray:
    """Return scores in the canonical target x train orientation."""
    num_targets = _as_int(summary.get("num_targets"), "num_targets")
    train_set_size = _as_int(summary.get("train_set_size"), "train_set_size")

    if scores.shape == (num_targets, train_set_size):
        normalized = scores
    elif scores.shape == (train_set_size, num_targets):
        normalized = scores.T
    else:
        raise ValueError(
            "Unexpected score matrix shape "
            f"{scores.shape}; expected ({num_targets}, {train_set_size}) or ({train_set_size}, {num_targets})."
        )

    return normalized.astype(np.float64, copy=False)


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    """Return a finite copy of scores for plotting and derived summaries."""
    finite_mask = np.isfinite(scores)
    if finite_mask.all():
        return scores

    finite_values = scores[finite_mask]
    if finite_values.size == 0:
        return np.zeros_like(scores, dtype=np.float64)

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=float(finite_values.max()),
        neginf=float(finite_values.min()),
    )


def _score_stats(scores: np.ndarray) -> dict[str, float]:
    """Compute stable statistics for target x train scores."""
    scores = _finite_scores(scores)
    values = scores.ravel()
    abs_values = np.abs(values)
    top_sorted = np.sort(scores, axis=1)
    top_k = min(10, scores.shape[1])

    percentiles = np.percentile(values, [1, 5, 50, 95, 99])
    abs_percentiles = np.percentile(abs_values, [95, 99])
    top1 = top_sorted[:, -1]
    top10 = top_sorted[:, -top_k:].mean(axis=1)
    neg1 = top_sorted[:, 0]

    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "p95_abs": float(abs_percentiles[0]),
        "p99_abs": float(abs_percentiles[1]),
        "top1_mean": float(top1.mean()),
        "top1_median": float(np.median(top1)),
        "top10_mean": float(top10.mean()),
        "neg1_mean": float(neg1.mean()),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _top_indices_desc(values: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices of the largest values sorted descending."""
    k = min(top_k, values.shape[0])
    if k <= 0:
        return np.array([], dtype=np.intp)
    indices = np.argpartition(values, -k)[-k:]
    return indices[np.argsort(values[indices])[::-1]]


def _top_indices_asc(values: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices of the smallest values sorted ascending."""
    k = min(top_k, values.shape[0])
    if k <= 0:
        return np.array([], dtype=np.intp)
    indices = np.argpartition(values, k - 1)[:k]
    return indices[np.argsort(values[indices])]


def _write_topk_csv(path: Path, scores: np.ndarray, top_k: int) -> None:
    """Write top-k training samples per target using target x train scores."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_index", "rank", "train_index", "score"])
        writer.writeheader()
        for target_index, target_scores in enumerate(scores):
            top_indices = _top_indices_desc(target_scores, top_k)
            for rank, train_index in enumerate(top_indices, start=1):
                writer.writerow({
                    "target_index": target_index,
                    "rank": rank,
                    "train_index": int(train_index),
                    "score": float(target_scores[train_index]),
                })


def _write_overview_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write corrected summary rows for generated plots."""
    if not rows:
        return

    fieldnames = [
        "architecture",
        "config_name",
        "num_targets",
        "train_set_size",
        "raw_scores_shape",
        "normalized_scores_shape",
        "num_tracked_params",
        "prefixes",
        "min",
        "max",
        "mean",
        "std",
        "p1",
        "p5",
        "p50",
        "p95",
        "p99",
        "p95_abs",
        "p99_abs",
        "top1_mean",
        "top1_median",
        "top10_mean",
        "neg1_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_overview_metric(rows: list[dict[str, object]], metric: str, output_dir: Path) -> None:
    """Plot one corrected summary metric across architectures and layer configs."""
    architectures = sorted({str(row["architecture"]) for row in rows})
    configs = sorted({str(row["config_name"]) for row in rows}, key=_config_sort_key)
    row_by_pair = {(str(row["architecture"]), str(row["config_name"])): row for row in rows}
    x = np.arange(len(architectures))
    width = 0.8 / max(len(configs), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, config_name in enumerate(configs):
        values: list[float] = []
        positions: list[float] = []
        for arch_idx, arch in enumerate(architectures):
            row = row_by_pair.get((arch, config_name))
            if row is None:
                continue
            values.append(float(str(row[metric])))
            positions.append(float(x[arch_idx] + idx * width - (len(configs) - 1) * width / 2))
        ax.bar(positions, values, width=width, label=config_name)

    ax.set_title(f"corrected {metric} by architecture and layer config")
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"overview_corrected_{metric}.png", dpi=180)
    plt.close(fig)


def _plot_num_params(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Plot the number of tracked parameters for each configuration."""
    architectures = sorted({str(row["architecture"]) for row in rows})
    configs = sorted({str(row["config_name"]) for row in rows}, key=_config_sort_key)
    row_by_pair = {(str(row["architecture"]), str(row["config_name"])): row for row in rows}
    x = np.arange(len(architectures))
    width = 0.8 / max(len(configs), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, config_name in enumerate(configs):
        values: list[float] = []
        positions: list[float] = []
        for arch_idx, arch in enumerate(architectures):
            row = row_by_pair.get((arch, config_name))
            if row is None:
                continue
            values.append(float(str(row["num_tracked_params"])))
            positions.append(float(x[arch_idx] + idx * width - (len(configs) - 1) * width / 2))
        ax.bar(positions, values, width=width, label=config_name)

    ax.set_title("Tracked parameters by architecture and layer config")
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.set_ylabel("num_tracked_params")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "overview_num_tracked_params.png", dpi=180)
    plt.close(fig)


def _plot_histogram(scores: np.ndarray, output_path: Path, bins: int, title: str) -> None:
    """Plot the distribution of TRAK scores for one configuration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores.ravel(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel("TRAK score")
    ax.set_ylabel("count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_heatmap(scores: np.ndarray, output_path: Path, title: str, max_targets: int, max_train_samples: int) -> None:
    """Plot a cropped heatmap of the target x train score matrix."""
    cropped = scores[:max_targets, :max_train_samples]
    vmax = float(np.max(np.abs(cropped)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if vmax > 0 else None

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(cropped, aspect="auto", cmap="coolwarm", norm=norm)
    ax.set_title(title)
    ax.set_xlabel("train sample index")
    ax.set_ylabel("target index")
    fig.colorbar(image, ax=ax, label="TRAK score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_topk_curve(scores: np.ndarray, output_path: Path, title: str) -> None:
    """Plot top-score trends across target samples."""
    top_k = min(10, scores.shape[1])
    top1_scores = np.max(scores, axis=1)
    top10_mean = np.partition(scores, -top_k, axis=1)[:, -top_k:].mean(axis=1)
    target_indices = np.arange(scores.shape[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(target_indices, top1_scores, label="top1 score", linewidth=1.5)
    ax.plot(target_indices, top10_mean, label=f"mean top{top_k} score", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("target index")
    ax.set_ylabel("score")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _image_to_numpy(image: object) -> np.ndarray:
    """Convert a torchvision image tensor or PIL image to an HWC array."""
    detach = getattr(image, "detach", None)
    array = detach().cpu().numpy() if callable(detach) else np.asarray(image)

    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    return np.clip(array, 0.0, 1.0)


def _load_raw_cifar10() -> tuple[object, object, list[str]] | None:
    """Load unnormalized CIFAR-10 train/test datasets for image grids."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        print(f"Skipping target extremes: torchvision is unavailable ({exc}).")
        return None

    data_dir = str(settings.project_root / settings.data_dir)
    transform = transforms.ToTensor()
    try:
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    except RuntimeError as exc:
        print(f"Skipping target extremes: CIFAR-10 is not available in {data_dir} ({exc}).")
        return None

    return train_dataset, test_dataset, list(train_dataset.classes)


def _dataset_item(dataset: object, index: int) -> tuple[object, int]:
    image, label = dataset[index]  # type: ignore[index]
    return image, int(label)


def _plot_target_extremes(
    scores: np.ndarray,
    output_path: Path,
    title: str,
    target_index: int,
    top_k: int,
    train_dataset: object,
    target_dataset: object,
    classes: list[str],
) -> None:
    """Plot one target and its strongest positive and negative training influences."""
    target_scores = scores[target_index]
    positive_indices = _top_indices_desc(target_scores, top_k)
    negative_indices = _top_indices_asc(target_scores, top_k)

    fig, axes = plt.subplots(2, top_k + 1, figsize=(2.2 * (top_k + 1), 4.8))
    fig.suptitle(title, fontsize=13)

    target_image, target_label = _dataset_item(target_dataset, target_index)
    target_title = f"target #{target_index}\nlabel={classes[target_label]}"
    for row in range(2):
        axes[row, 0].imshow(_image_to_numpy(target_image))
        axes[row, 0].set_title(target_title)
        axes[row, 0].axis("off")

    for rank, train_index in enumerate(positive_indices, start=1):
        image, label = _dataset_item(train_dataset, int(train_index))
        ax = axes[0, rank]
        ax.imshow(_image_to_numpy(image))
        ax.set_title(
            f"+ rank {rank}\ntrain={int(train_index)}\nscore={target_scores[train_index]:.3f}\nlabel={classes[label]}",
            fontsize=9,
        )
        ax.axis("off")

    for rank, train_index in enumerate(negative_indices, start=1):
        image, label = _dataset_item(train_dataset, int(train_index))
        ax = axes[1, rank]
        ax.imshow(_image_to_numpy(image))
        ax.set_title(
            f"- rank {rank}\ntrain={int(train_index)}\nscore={target_scores[train_index]:.3f}\nlabel={classes[label]}",
            fontsize=9,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _summary_to_row(summary: dict[str, Any], raw_shape: tuple[int, ...], scores: np.ndarray) -> dict[str, object]:
    """Build one corrected overview row from metadata and normalized scores."""
    prefixes = summary.get("prefixes")
    stats = _score_stats(scores)
    return {
        "architecture": str(summary["architecture"]),
        "config_name": str(summary["config_name"]),
        "num_targets": int(scores.shape[0]),
        "train_set_size": int(scores.shape[1]),
        "raw_scores_shape": "x".join(str(dim) for dim in raw_shape),
        "normalized_scores_shape": "x".join(str(dim) for dim in scores.shape),
        "num_tracked_params": int(summary["num_tracked_params"]),
        "prefixes": "" if prefixes is None else ",".join(str(prefix) for prefix in prefixes),
        **stats,
    }


def _generate_config_plots(
    *,
    summary: dict[str, Any],
    raw_scores: np.ndarray,
    output_dir: Path,
    max_targets: int,
    max_train_samples: int,
    hist_bins: int,
    top_k: int,
    num_extreme_targets: int,
    extreme_top_k: int,
    raw_cifar10: tuple[object, object, list[str]] | None,
) -> dict[str, object]:
    """Generate all per-configuration plots and corrected CSV artifacts."""
    architecture = str(summary["architecture"])
    config_name = str(summary["config_name"])
    scores = _finite_scores(_normalize_scores(raw_scores, summary))

    config_output_dir = output_dir / architecture / config_name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = f"{architecture} / {config_name}"
    _plot_histogram(
        scores,
        config_output_dir / "score_histogram.png",
        bins=hist_bins,
        title=f"{title_prefix} corrected score distribution",
    )
    _plot_heatmap(
        scores,
        config_output_dir / "score_heatmap.png",
        title=f"{title_prefix} corrected score heatmap",
        max_targets=max_targets,
        max_train_samples=max_train_samples,
    )
    _plot_topk_curve(
        scores,
        config_output_dir / "topk_curve.png",
        title=f"{title_prefix} corrected top influence by target",
    )

    _write_topk_csv(config_output_dir / "topk.csv", scores, top_k=top_k)
    _write_json(config_output_dir / "score_stats.json", _score_stats(scores))

    if raw_cifar10 is not None and num_extreme_targets > 0 and extreme_top_k > 0:
        train_dataset, target_dataset, classes = raw_cifar10
        for target_index in range(min(num_extreme_targets, scores.shape[0])):
            _plot_target_extremes(
                scores,
                config_output_dir / f"target_{target_index:03d}_extremes.png",
                title=f"{title_prefix} target {target_index}: corrected top positive / negative",
                target_index=target_index,
                top_k=extreme_top_k,
                train_dataset=train_dataset,
                target_dataset=target_dataset,
                classes=classes,
            )

    return _summary_to_row(summary, tuple(raw_scores.shape), scores)


def _iter_filesystem_artifacts(run_dir: Path) -> Iterable[tuple[dict[str, Any], np.ndarray]]:
    """Yield summary and raw scores for each filesystem configuration."""
    for summary_path in sorted(run_dir.glob("*/*/summary.json")):
        config_dir = summary_path.parent
        summary = _load_summary(summary_path)
        raw_scores = np.load(config_dir / "scores.npy")
        yield summary, raw_scores


def _generate_overview_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Generate corrected overview CSV and plots."""
    rows = sorted(rows, key=lambda row: (str(row["architecture"]), _config_sort_key(str(row["config_name"]))))
    _warn_missing_configs(rows)
    _write_overview_csv(output_dir / "overview_corrected.csv", rows)
    _plot_num_params(rows, output_dir)
    for metric in OVERVIEW_METRICS:
        _plot_overview_metric(rows, metric, output_dir)


def _warn_missing_configs(rows: list[dict[str, object]]) -> None:
    """Print architectures that are missing configs present in the same run."""
    configs_by_arch: dict[str, set[str]] = {}
    for row in rows:
        configs_by_arch.setdefault(str(row["architecture"]), set()).add(str(row["config_name"]))

    expected_configs = set().union(*configs_by_arch.values()) if configs_by_arch else set()
    for architecture, configs in sorted(configs_by_arch.items()):
        missing = sorted(expected_configs - configs, key=_config_sort_key)
        if missing:
            print(f"Missing configuration artifact(s) for {architecture}: {', '.join(missing)}")


def generate_plots(
    *,
    run_dir: Path | None = None,
    output_dir: Path | None = None,
    max_targets: int = 25,
    max_train_samples: int = 100,
    hist_bins: int = 80,
    top_k: int = 10,
    num_extreme_targets: int = 5,
    extreme_top_k: int = 5,
    skip_extremes: bool = False,
) -> Path:
    """Generate corrected plots for one saved LayerTRAK run."""
    raw_cifar10 = None if skip_extremes else _load_raw_cifar10()
    corrected_rows: list[dict[str, object]] = []

    resolved_run_dir = _resolve_run_dir(run_dir)
    overview_path = resolved_run_dir / "overview.csv"
    if overview_path.exists():
        _load_overview_from_path(overview_path)
    else:
        print(f"Missing overview.csv in {resolved_run_dir}; using summary.json files instead.")
    resolved_output_dir = _make_output_dir(resolved_run_dir, output_dir)
    artifacts = _iter_filesystem_artifacts(resolved_run_dir)

    for summary, raw_scores in artifacts:
        corrected_rows.append(
            _generate_config_plots(
                summary=summary,
                raw_scores=raw_scores,
                output_dir=resolved_output_dir,
                max_targets=max_targets,
                max_train_samples=max_train_samples,
                hist_bins=hist_bins,
                top_k=top_k,
                num_extreme_targets=num_extreme_targets,
                extreme_top_k=extreme_top_k,
                raw_cifar10=raw_cifar10,
            )
        )

    if not corrected_rows:
        raise FileNotFoundError("No summary.json/scores.npy configuration artifacts found.")

    _generate_overview_plots(corrected_rows, resolved_output_dir)
    return resolved_output_dir


def main() -> None:
    """Run the plotting script from the command line."""
    args = _parse_args()
    output_dir = generate_plots(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        max_targets=args.max_targets,
        max_train_samples=args.max_train_samples,
        hist_bins=args.hist_bins,
        top_k=args.top_k,
        num_extreme_targets=args.num_extreme_targets,
        extreme_top_k=args.extreme_top_k,
        skip_extremes=args.skip_extremes,
    )
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
