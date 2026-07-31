"""Matched-clean quantitative comparison of Sobel and Methods I--III."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from hog_descriptor import describe_hog
from parameter_analysis_pipeline import _load_image
from qualitative_sobel_comparison import (
    DEFAULT_CONFIG,
    METHOD_ORDER,
    array_sha256,
    estimate_method,
    hog_kwargs,
    load_config,
    make_noise,
    make_solvers,
    package_versions,
    resolve_path,
)

SCENARIOS = ("gaussian_20db", "sp5", "sp10")


def relative_error(reference: np.ndarray, observed: np.ndarray) -> float:
    """Compute descriptor relative L2 error."""
    denominator = max(float(np.linalg.norm(reference.ravel())), np.finfo(float).eps)
    return float(np.linalg.norm(observed.ravel() - reference.ravel()) / denominator)


def cosine_similarity(reference: np.ndarray, observed: np.ndarray) -> float:
    """Compute descriptor cosine similarity."""
    left, right = reference.ravel(), observed.ravel()
    denominator = max(
        float(np.linalg.norm(left) * np.linalg.norm(right)), np.finfo(float).eps
    )
    return float(np.dot(left, right) / denominator)


def _format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def write_latex(summary: list[dict[str, Any]], path: Path) -> None:
    """Write the requested LaTeX table, bolding measured block optima only."""
    labels = {
        "gaussian_20db": "Gaussian 20 dB",
        "sp5": "Salt-and-pepper 5\\%",
        "sp10": "Salt-and-pepper 10\\%",
    }
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Noise & Method & HOG Relative Error & Cosine Similarity & Runtime (s) \\",
        r"\midrule",
    ]
    for scenario in SCENARIOS:
        block = [row for row in summary if row["noise"] == scenario]
        best_error = min(row["hog_relative_error_mean"] for row in block)
        best_cosine = max(row["cosine_similarity_mean"] for row in block)
        best_runtime = min(row["runtime_seconds_mean"] for row in block)
        for index, row in enumerate(block):
            values = [
                _format_mean_std(
                    row["hog_relative_error_mean"], row["hog_relative_error_std"]
                ),
                _format_mean_std(
                    row["cosine_similarity_mean"], row["cosine_similarity_std"]
                ),
                _format_mean_std(row["runtime_seconds_mean"], row["runtime_seconds_std"]),
            ]
            optima = (
                row["hog_relative_error_mean"] == best_error,
                row["cosine_similarity_mean"] == best_cosine,
                row["runtime_seconds_mean"] == best_runtime,
            )
            values = [
                rf"\textbf{{{value}}}" if best else value
                for value, best in zip(values, optima)
            ]
            noise_label = labels[scenario] if index == 0 else ""
            lines.append(
                f"{noise_label} & {row['method']} & "
                + " & ".join(values)
                + r" \\"
            )
        if scenario != SCENARIOS[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path = DEFAULT_CONFIG) -> list[Path]:
    """Evaluate every configured test image with paired noise."""
    config, root = load_config(config_path)
    image_paths = [resolve_path(root, item) for item in config["test_images"]]
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Configured test image(s) not found:\n" + "\n".join(map(str, missing))
        )
    if not image_paths:
        raise ValueError("test_images must contain at least one path.")
    output = resolve_path(root, config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    solvers = make_solvers(config)
    rows: list[dict[str, Any]] = []
    noise_records: dict[str, dict[str, str]] = {}
    repeats = int(config.get("runtime_repeats", 1))
    if repeats < 1:
        raise ValueError("runtime_repeats must be at least 1.")

    for image_index, image_path in enumerate(image_paths):
        clean = _load_image(image_path, int(config["max_image_dimension"]))
        if min(clean.shape) <= 2 * int(config["common_roi_pad"]) + 16:
            raise ValueError(
                f"Image {image_path} becomes too small ({clean.shape}) for the ROI/HOG."
            )
        clean_hogs: dict[str, np.ndarray] = {}
        for method in METHOD_ORDER:
            dx, dy = estimate_method(clean, method, config, solvers.get(method))
            clean_hogs[method] = describe_hog(
                dx, dy, **hog_kwargs(config)
            ).descriptor
        image_seed = int(config["seed"]) + image_index
        for scenario in SCENARIOS:
            noisy, noise_detail = make_noise(clean, scenario, config, image_seed)
            noise_records[f"{image_path.name}:{scenario}"] = {
                "sha256": str(noise_detail["noisy_image_sha256"]),
                "seed": str(image_seed),
            }
            for method in METHOD_ORDER:
                timings: list[float] = []
                noisy_descriptor: np.ndarray | None = None
                for _ in range(repeats):
                    start = time.perf_counter()
                    dx, dy = estimate_method(noisy, method, config, solvers.get(method))
                    noisy_descriptor = describe_hog(
                        dx, dy, **hog_kwargs(config)
                    ).descriptor
                    timings.append(time.perf_counter() - start)
                assert noisy_descriptor is not None
                rows.append(
                    {
                        "image": str(image_path.relative_to(root)),
                        "noise": scenario,
                        "method": method,
                        "hog_relative_error": relative_error(
                            clean_hogs[method], noisy_descriptor
                        ),
                        "cosine_similarity": cosine_similarity(
                            clean_hogs[method], noisy_descriptor
                        ),
                        "runtime_seconds": float(statistics.median(timings)),
                        "runtime_trials_seconds": ";".join(f"{x:.9g}" for x in timings),
                        "seed": image_seed,
                        "noisy_image_sha256": array_sha256(noisy),
                    }
                )

    raw_path = output / "sobel_comparison_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for method in METHOD_ORDER:
            group = [
                row for row in rows
                if row["noise"] == scenario and row["method"] == method
            ]
            item: dict[str, Any] = {
                "noise": scenario, "method": method, "image_count": len(group)
            }
            for metric in (
                "hog_relative_error", "cosine_similarity", "runtime_seconds"
            ):
                values = [float(row[metric]) for row in group]
                item[f"{metric}_mean"] = statistics.fmean(values)
                item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.append(item)
    summary_path = output / "sobel_comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    latex_path = output / "sobel_comparison_table.tex"
    write_latex(summary, latex_path)
    metadata_path = output / "quantitative_metadata.json"
    metadata = {
        "status": "COMPLETED",
        "image_count": len(image_paths),
        "images": [str(path.relative_to(root)) for path in image_paths],
        "preprocessing": {
            "grayscale": "cv2.IMREAD_GRAYSCALE",
            "max_image_dimension": int(config["max_image_dimension"]),
            "resize_interpolation": "cv2.INTER_AREA",
        },
        "seed_policy": "base seed + zero-based configured image index; one noisy image reused by all methods",
        "base_seed": int(config["seed"]),
        "noise_realizations": noise_records,
        "common_roi_pad": int(config["common_roi_pad"]),
        "methods": config["methods"],
        "hog": config["hog"],
        "solver": config["solver"],
        "runtime_protocol": (
            f"median of {repeats} run(s); gradient estimation, common ROI crop, "
            "and robust HOG included; loading, resizing, noise generation, fixed "
            "solver/design-matrix construction, and clean reference excluded"
        ),
        "metrics": {
            "hog_relative_error": "L2(noisy-clean)/L2(clean), matched clean per method",
            "cosine_similarity": "cosine(noisy, clean), matched clean per method",
        },
        "package_versions": package_versions(),
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return [raw_path, summary_path, latex_path, metadata_path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quantitatively compare Sobel and fixed Methods I--III."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Shared JSON config path."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    created = run(args.config)
    print("Created:")
    for path in created:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
