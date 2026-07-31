"""Tolerance-based MultiCue boundary evaluation for the four fixed methods."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.io import loadmat
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

from parameter_analysis_pipeline import _load_image
from qualitative_sobel_comparison import (
    DEFAULT_CONFIG,
    METHOD_ORDER,
    estimate_method,
    load_config,
    make_solvers,
    package_versions,
    resolve_path,
)


def _extract_boundary_maps(value: Any) -> Iterable[np.ndarray]:
    """Recursively yield plausible 2-D boundary maps from a MATLAB object."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        for item in value.flat:
            yield from _extract_boundary_maps(item)
    elif isinstance(value, np.ndarray) and value.dtype.names:
        for name in value.dtype.names:
            yield from _extract_boundary_maps(value[name])
    elif isinstance(value, np.ndarray) and value.ndim == 2 and min(value.shape) > 8:
        yield np.asarray(value, dtype=float) > 0


def load_annotations(path: Path) -> list[np.ndarray]:
    """Load all human boundary maps from one MultiCue MATLAB annotation."""
    data = loadmat(path)
    maps: list[np.ndarray] = []
    for key, value in data.items():
        if not key.startswith("__"):
            maps.extend(_extract_boundary_maps(value))
    unique: list[np.ndarray] = []
    for boundary in maps:
        if boundary.any() and not any(
            boundary.shape == old.shape and np.array_equal(boundary, old)
            for old in unique
        ):
            unique.append(boundary)
    if not unique:
        raise ValueError(f"No 2-D human boundary maps found in {path}")
    return unique


def match_counts(
    prediction: np.ndarray, reference: np.ndarray, tolerance: float
) -> tuple[int, int, int]:
    """Maximum-cardinality one-to-one pixel matching within Euclidean tolerance."""
    predicted_points = np.argwhere(prediction)
    reference_points = np.argwhere(reference)
    if not len(predicted_points):
        return 0, 0, int(len(reference_points))
    if not len(reference_points):
        return 0, int(len(predicted_points)), 0
    tree = cKDTree(reference_points)
    neighbours = tree.query_ball_point(predicted_points, tolerance)
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row, columns in enumerate(neighbours):
        row_indices.extend([row] * len(columns))
        column_indices.extend(columns)
    if not row_indices:
        return 0, int(len(predicted_points)), int(len(reference_points))
    graph = sparse.csr_matrix(
        (np.ones(len(row_indices), dtype=np.int8), (row_indices, column_indices)),
        shape=(len(predicted_points), len(reference_points)),
    )
    matching = maximum_bipartite_matching(graph, perm_type="column")
    true_positive = int(np.count_nonzero(matching >= 0))
    return (
        true_positive,
        int(len(predicted_points) - true_positive),
        int(len(reference_points) - true_positive),
    )


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Convert pooled counts to precision, recall, and F1."""
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def normalized_strength(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Convert gradients to a robustly normalized [0,1] edge-strength map."""
    magnitude = np.hypot(dx, dy)
    positive = magnitude[magnitude > 0]
    scale = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    return np.clip(magnitude / max(scale, np.finfo(float).eps), 0.0, 1.0)


def _write_skipped(
    metadata_path: Path, config: dict[str, Any], root: Path, reason: str
) -> list[Path]:
    expected = [
        "<root>/images/<image-id>.jpg",
        "<root>/groundTruth/<image-id>.mat",
    ]
    metadata = {
        "status": "SKIPPED",
        "reason": reason,
        "configured_dataset_root": str(
            resolve_path(root, config["multicue"]["root"])
        ),
        "expected_directory_structure": expected,
        "methods": config["methods"],
        "edge_evaluation": config["multicue"],
        "package_versions": package_versions(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print("MultiCue evaluation SKIPPED.")
    print(reason)
    print("Expected dataset structure:")
    for item in expected:
        print(f"  {item}")
    return [metadata_path]


def run(config_path: str | Path = DEFAULT_CONFIG) -> list[Path]:
    """Run MultiCue when paired local images and annotations are available."""
    config, root = load_config(config_path)
    output = resolve_path(root, config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "multicue_metadata.json"
    multicue = config["multicue"]
    dataset_root = resolve_path(root, multicue["root"])
    images_dir = dataset_root / multicue["images_directory"]
    annotations_dir = dataset_root / multicue["annotations_directory"]
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        return _write_skipped(
            metadata_path,
            config,
            root,
            "Dataset images and/or human annotation directories are not available locally.",
        )
    extensions = {item.lower() for item in multicue["image_extensions"]}
    image_paths = sorted(
        path for path in images_dir.rglob("*") if path.suffix.lower() in extensions
    )
    pairs = [
        (image, annotations_dir / f"{image.stem}{multicue['annotation_extension']}")
        for image in image_paths
    ]
    missing = [annotation for _, annotation in pairs if not annotation.is_file()]
    if not pairs or missing:
        detail = "No paired MultiCue image/annotation files found."
        if missing:
            detail += f" Missing {len(missing)} annotation(s), first: {missing[0]}"
        return _write_skipped(metadata_path, config, root, detail)

    thresholds = [float(value) for value in multicue["thresholds"]]
    if thresholds != sorted(thresholds) or any(x < 0 or x > 1 for x in thresholds):
        raise ValueError("MultiCue thresholds must be sorted values in [0,1].")
    tolerance = float(multicue["edge_tolerance_pixels"])
    solvers = make_solvers(config)
    rows: list[dict[str, Any]] = []
    for image_path, annotation_path in pairs:
        image = _load_image(image_path, int(config["max_image_dimension"]))
        references = load_annotations(annotation_path)
        for method in METHOD_ORDER:
            dx, dy = estimate_method(image, method, config, solvers.get(method))
            strength = normalized_strength(dx, dy)
            for threshold in thresholds:
                counts = [0, 0, 0]
                prediction = strength >= threshold
                for reference in references:
                    pad = int(config["common_roi_pad"])
                    resized = reference
                    if resized.shape != image.shape:
                        resized = (
                            np.asarray(
                                __import__("cv2").resize(
                                    reference.astype(np.uint8),
                                    (image.shape[1], image.shape[0]),
                                    interpolation=__import__("cv2").INTER_NEAREST,
                                )
                            )
                            > 0
                        )
                    reference_roi = resized[pad:-pad, pad:-pad]
                    matched = match_counts(prediction, reference_roi, tolerance)
                    counts = [left + right for left, right in zip(counts, matched)]
                precision, recall, f1 = precision_recall_f1(*counts)
                rows.append(
                    {
                        "image": image_path.stem,
                        "method": method,
                        "threshold": threshold,
                        "annotations": len(references),
                        "tp": counts[0], "fp": counts[1], "fn": counts[2],
                        "precision": precision, "recall": recall, "f1": f1,
                    }
                )
    raw_path = output / "multicue_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, Any]] = []
    curves: dict[str, list[tuple[float, float]]] = {}
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        threshold_results: list[tuple[float, float, float, float]] = []
        for threshold in thresholds:
            group = [row for row in method_rows if row["threshold"] == threshold]
            tp, fp, fn = (
                sum(int(row[key]) for row in group) for key in ("tp", "fp", "fn")
            )
            precision, recall, f1 = precision_recall_f1(tp, fp, fn)
            threshold_results.append((threshold, precision, recall, f1))
        ods = max(threshold_results, key=lambda item: item[3])
        image_best = []
        for image_id in sorted({str(row["image"]) for row in method_rows}):
            image_best.append(
                max(float(row["f1"]) for row in method_rows if row["image"] == image_id)
            )
        pr = sorted((recall, precision) for _, precision, recall, _ in threshold_results)
        recalls = np.asarray([point[0] for point in pr])
        precisions = np.asarray([point[1] for point in pr])
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]
        ap = float(np.trapezoid(precisions, recalls))
        curves[method] = [(float(r), float(p)) for r, p in zip(recalls, precisions)]
        summary.append(
            {
                "method": method, "ods_f1": ods[3], "ods_threshold": ods[0],
                "ods_precision": ods[1], "ods_recall": ods[2],
                "ois_f1": float(np.mean(image_best)), "ap": ap,
                "image_count": len(pairs),
            }
        )
    summary_path = output / "multicue_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    figure_path = output / "multicue_pr_curve.png"
    figure, axis = plt.subplots(figsize=(5.4, 4.5), constrained_layout=True)
    for method in METHOD_ORDER:
        curve = curves[method]
        axis.plot([x for x, _ in curve], [y for _, y in curve], label=method)
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)
    table_path = output / "multicue_table.tex"
    lines = [
        r"\begin{tabular}{lccc}", r"\toprule", r"Method & ODS & OIS & AP \\",
        r"\midrule",
    ]
    lines.extend(
        f"{row['method']} & {row['ods_f1']:.4f} & {row['ois_f1']:.4f} & {row['ap']:.4f} \\\\"
        for row in summary
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {
        "status": "COMPLETED",
        "dataset_root": str(dataset_root),
        "image_count": len(pairs),
        "methods": config["methods"],
        "fixed_before_annotation_evaluation": True,
        "edge_strength": "gradient magnitude divided by its per-image 99th percentile, clipped to [0,1]",
        "matching": (
            "maximum-cardinality one-to-one bipartite matching of predicted and "
            f"reference pixels within Euclidean tolerance {tolerance} pixels"
        ),
        "human_annotation_policy": multicue["human_annotation_policy"],
        "thresholds": thresholds,
        "package_versions": package_versions(),
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return [raw_path, summary_path, figure_path, table_path, metadata_path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fixed Sobel/robust-gradient methods on local MultiCue annotations."
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
