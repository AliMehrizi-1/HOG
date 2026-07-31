from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage

from run_pipeline import (
    add_gaussian_noise_snr,
    add_noise,
    estimate_taylor_gradient,
    evaluate_three_hog_methods,
    show_method_comparison,
)


METHOD_LABELS = {
    "A_sobel_standard": "A: Sobel + Standard HOG",
    "B_taylor_standard": "B: Taylor + Standard HOG",
    "C_taylor_robust": "C: Taylor + Robust HOG",
}
METHOD_COLORS = {
    "A_sobel_standard": "#4C78A8",
    "B_taylor_standard": "#F58518",
    "C_taylor_robust": "#54A24B",
}
NATURAL_IMAGE_DEFAULTS = [
    "Images/cameraman.png",
    "Images/lena.png",
    "Images/home.jpg",
    "Images/girl.png",
    "Images/flower.jpg",
    "Images/clock.png",
]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_image(path: str | Path, max_dimension: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read grayscale image: {path}")
    height, width = image.shape
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(32, round(width * scale)), max(32, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image.astype(float)


def _mean_std_ci(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    count = int(array.size)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(count) if count > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95, "n": count}


def _clustered_image_summary(group: list[dict], metric: str) -> dict:
    """Average seeds within image, then form uncertainty across images."""
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in group:
        by_image[str(row["image"])].append(float(row[metric]))
    image_means = [float(np.mean(values)) for values in by_image.values()]
    result = _mean_std_ci(image_means)
    result["n_images"] = len(image_means)
    result["n_trials"] = len(group)
    return result


def _write_csv(path: Path, records: list[dict]):
    if not records:
        raise ValueError(f"No records available for {path}.")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _condition_sort_key(family: str, level: float):
    family_order = {"saltpepper": 0, "gaussian": 1}
    return family_order[family], float(level)


def _summarize_gradient(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (record["noise_family"], float(record["noise_level"]))
        grouped[key].append(record)

    summary = []
    for (family, level), group in sorted(
        grouped.items(), key=lambda item: _condition_sort_key(*item[0])
    ):
        sobel = _clustered_image_summary(group, "sobel_gradient_rmse")
        taylor = _clustered_image_summary(group, "taylor_gradient_rmse")
        ratio = _clustered_image_summary(group, "taylor_improvement_factor")
        wins = np.mean(
            [
                float(row["taylor_gradient_rmse"])
                < float(row["sobel_gradient_rmse"])
                for row in group
            ]
        )
        summary.append(
            {
                "noise_family": family,
                "noise_level": level,
                "n_images": sobel["n_images"],
                "n_trials": sobel["n_trials"],
                "sobel_rmse_mean": sobel["mean"],
                "sobel_rmse_std": sobel["std"],
                "sobel_rmse_ci95": sobel["ci95"],
                "taylor_rmse_mean": taylor["mean"],
                "taylor_rmse_std": taylor["std"],
                "taylor_rmse_ci95": taylor["ci95"],
                "improvement_factor_mean": ratio["mean"],
                "improvement_factor_std": ratio["std"],
                "taylor_win_rate": float(wins),
            }
        )
    return summary


def _summarize_hog(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            record["noise_family"],
            float(record["noise_level"]),
            record["normalization"],
            record["method"],
        )
        grouped[key].append(record)

    summary = []
    metric_names = [
        "hog_cosine_similarity",
        "hog_relative_error",
        "common_reference_cosine",
        "common_reference_relative_error",
        "clean_fidelity_cosine",
        "clean_fidelity_relative_error",
    ]
    for key, group in sorted(
        grouped.items(),
        key=lambda item: (
            *_condition_sort_key(item[0][0], item[0][1]),
            item[0][2],
            item[0][3],
        ),
    ):
        family, level, normalization, method = key
        output = {
            "noise_family": family,
            "noise_level": level,
            "normalization": normalization,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_images": len({str(row["image"]) for row in group}),
            "n_trials": len(group),
        }
        for metric in metric_names:
            values = _clustered_image_summary(group, metric)
            output[f"{metric}_mean"] = values["mean"]
            output[f"{metric}_std"] = values["std"]
            output[f"{metric}_ci95"] = values["ci95"]
        summary.append(output)
    return summary


def _paired_claims(hog_records: list[dict], gradient_records: list[dict]) -> dict:
    gradient_wins = [
        float(row["taylor_gradient_rmse"]) < float(row["sobel_gradient_rmse"])
        for row in gradient_records
    ]
    by_trial: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in hog_records:
        key = (
            row["image"],
            int(row["seed"]),
            row["noise_family"],
            float(row["noise_level"]),
            row["normalization"],
        )
        by_trial[key][row["method"]] = row

    paired = []
    for key, methods in by_trial.items():
        if "B_taylor_standard" not in methods or "C_taylor_robust" not in methods:
            continue
        standard = methods["B_taylor_standard"]
        robust = methods["C_taylor_robust"]
        paired.append(
            {
                "noise_family": key[2],
                "normalization": key[4],
                "cosine_delta": float(robust["hog_cosine_similarity"])
                - float(standard["hog_cosine_similarity"]),
                "relative_error_delta": float(standard["hog_relative_error"])
                - float(robust["hog_relative_error"]),
                "common_cosine_delta": float(robust["common_reference_cosine"])
                - float(standard["common_reference_cosine"]),
                "common_relative_error_delta": float(
                    standard["common_reference_relative_error"]
                )
                - float(robust["common_reference_relative_error"]),
            }
        )

    strata = {}
    for family in ("saltpepper", "gaussian"):
        for normalization in ("L2", "L2-Hys"):
            values = [
                row
                for row in paired
                if row["noise_family"] == family
                and row["normalization"] == normalization
            ]
            if not values:
                continue
            label = f"{family}_{normalization}"
            strata[label] = {
                "n": len(values),
                "own_clean_cosine_win_rate": float(
                    np.mean([row["cosine_delta"] > 0 for row in values])
                ),
                "own_clean_relative_error_win_rate": float(
                    np.mean([row["relative_error_delta"] > 0 for row in values])
                ),
                "common_reference_cosine_win_rate": float(
                    np.mean([row["common_cosine_delta"] > 0 for row in values])
                ),
                "common_reference_relative_error_win_rate": float(
                    np.mean(
                        [row["common_relative_error_delta"] > 0 for row in values]
                    )
                ),
                "mean_own_clean_cosine_delta": float(
                    np.mean([row["cosine_delta"] for row in values])
                ),
                "mean_own_clean_relative_error_reduction": float(
                    np.mean([row["relative_error_delta"] for row in values])
                ),
                "mean_common_reference_cosine_delta": float(
                    np.mean([row["common_cosine_delta"] for row in values])
                ),
                "mean_common_reference_relative_error_reduction": float(
                    np.mean(
                        [row["common_relative_error_delta"] for row in values]
                    )
                ),
            }

    return {
        "gradient": {
            "n": len(gradient_wins),
            "taylor_lower_rmse_win_rate": float(np.mean(gradient_wins)),
        },
        "robust_hog_vs_taylor_standard": strata,
    }


def _plot_gradient(summary: list[dict], output_path: Path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, family, x_label in [
        (axes[0], "saltpepper", "Impulse density (%)"),
        (axes[1], "gaussian", "Target SNR (dB)"),
    ]:
        subset = [row for row in summary if row["noise_family"] == family]
        x = np.asarray([float(row["noise_level"]) for row in subset])
        for prefix, label, color, marker in [
            ("sobel", "Sobel", "#4C78A8", "o"),
            ("taylor", "Taylor/Maclaurin", "#F58518", "s"),
        ]:
            y = np.asarray([row[f"{prefix}_rmse_mean"] for row in subset])
            error = np.asarray([row[f"{prefix}_rmse_ci95"] for row in subset])
            axis.errorbar(
                x,
                y,
                yerr=error,
                label=label,
                color=color,
                marker=marker,
                linewidth=2,
                capsize=3,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Gradient stability RMSE (lower is better)")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Gradient stability under paired noise realizations")
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def _plot_hog_metric(
    summary: list[dict],
    metric: str,
    ylabel: str,
    output_path: Path,
):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row_idx, family in enumerate(("saltpepper", "gaussian")):
        for col_idx, normalization in enumerate(("L2", "L2-Hys")):
            axis = axes[row_idx, col_idx]
            subset = [
                row
                for row in summary
                if row["noise_family"] == family
                and row["normalization"] == normalization
            ]
            levels = sorted({float(row["noise_level"]) for row in subset})
            for method in METHOD_LABELS:
                method_rows = [
                    row
                    for row in subset
                    if row["method"] == method
                ]
                method_rows.sort(key=lambda row: float(row["noise_level"]))
                y = np.asarray([row[f"{metric}_mean"] for row in method_rows])
                error = np.asarray([row[f"{metric}_ci95"] for row in method_rows])
                axis.errorbar(
                    levels,
                    y,
                    yerr=error,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    marker={"A_sobel_standard": "o", "B_taylor_standard": "s"}.get(
                        method, "^"
                    ),
                    linewidth=2,
                    capsize=3,
                )
            family_title = (
                "Salt-and-pepper density (%)"
                if family == "saltpepper"
                else "Gaussian target SNR (dB)"
            )
            axis.set_title(f"{family_title} — {normalization}")
            axis.set_xlabel(family_title)
            axis.set_ylabel(ylabel)
            axis.set_xticks(levels)
            axis.grid(alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                axis.legend(frameon=False, fontsize=8)
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def _write_markdown_table(
    path: Path,
    gradient_summary: list[dict],
    hog_summary: list[dict],
    claims: dict,
):
    gradient_lookup = {
        (row["noise_family"], float(row["noise_level"])): row
        for row in gradient_summary
    }
    hog_lookup = {
        (
            row["noise_family"],
            float(row["noise_level"]),
            row["normalization"],
            row["method"],
        ): row
        for row in hog_summary
    }
    lines = [
        "# Robust HOG experiment summary",
        "",
        "Values are means over all images and paired seeds. Gradient RMSE is "
        "method-specific noise stability, not absolute derivative accuracy.",
        "",
        "| Noise | Norm | Sobel RMSE | Taylor RMSE | Gain | "
        "A cosine | B cosine | C cosine | A rel.err | B rel.err | C rel.err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, level in sorted(
        gradient_lookup, key=lambda item: _condition_sort_key(*item)
    ):
        gradient = gradient_lookup[(family, level)]
        condition = (
            f"S&P {level:g}%"
            if family == "saltpepper"
            else f"AWGN {level:g} dB"
        )
        for normalization in ("L2", "L2-Hys"):
            method_rows = [
                hog_lookup[(family, level, normalization, method)]
                for method in METHOD_LABELS
            ]
            lines.append(
                f"| {condition} | {normalization} | "
                f"{gradient['sobel_rmse_mean']:.4f} | "
                f"{gradient['taylor_rmse_mean']:.4f} | "
                f"{gradient['improvement_factor_mean']:.2f}x | "
                + " | ".join(
                    f"{row['hog_cosine_similarity_mean']:.4f}"
                    for row in method_rows
                )
                + " | "
                + " | ".join(
                    f"{row['hog_relative_error_mean']:.4f}"
                    for row in method_rows
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Paired win rates",
            "",
            f"- Taylor lower-gradient-RMSE win rate: "
            f"{claims['gradient']['taylor_lower_rmse_win_rate']:.1%} "
            f"(n={claims['gradient']['n']}).",
        ]
    )
    for stratum, values in claims["robust_hog_vs_taylor_standard"].items():
        lines.append(
            f"- {stratum}: Robust HOG own-clean cosine win rate "
            f"{values['own_clean_cosine_win_rate']:.1%}; common-reference "
            f"cosine win rate {values['common_reference_cosine_win_rate']:.1%} "
            f"(n={values['n']})."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiments(args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_parameters = {
        "order": 16,
        "spacing": 1,
        "algorithm": 3,
        "smoothing_sigma": 0.0,
        "tol": 1e-3,
        "max_iter": args.max_iter,
        "init_sigma": None,
        "fuzzy": True,
        "robust_weighting": "huber",
        "robust_delta": 2.0,
        "robust_window": 5,
        "visualization_percentile": 95.0,
    }
    normalizations = ["L2", "L2-Hys"]
    conditions = [
        ("saltpepper", 1.0),
        ("saltpepper", 5.0),
        ("saltpepper", 10.0),
        ("gaussian", 10.0),
        ("gaussian", 20.0),
        ("gaussian", 30.0),
    ]

    gradient_records: list[dict] = []
    hog_records: list[dict] = []
    representative_saved = False
    start = time.perf_counter()

    for image_index, image_path in enumerate(args.images):
        clean = _load_image(image_path, args.max_dimension)
        clean_taylor = estimate_taylor_gradient(
            clean,
            experiment_parameters["order"],
            experiment_parameters["spacing"],
            experiment_parameters["algorithm"],
            smoothing_sigma=experiment_parameters["smoothing_sigma"],
            tol=experiment_parameters["tol"],
            max_iter=experiment_parameters["max_iter"],
            init_sigma=experiment_parameters["init_sigma"],
        )
        for seed in args.seeds:
            for noise_family, noise_level in conditions:
                if noise_family == "saltpepper":
                    noisy = add_noise(clean, "saltpepper", noise_level, seed)
                    achieved_snr = None
                    actual_density = (
                        float(np.count_nonzero(noisy != clean)) / clean.size
                    )
                else:
                    noisy, achieved_snr = add_gaussian_noise_snr(
                        clean, noise_level, seed
                    )
                    actual_density = None

                first_evaluation = None
                noisy_taylor = None
                for normalization in normalizations:
                    evaluation = evaluate_three_hog_methods(
                        clean,
                        noisy,
                        normalization=normalization,
                        clean_taylor=clean_taylor,
                        noisy_taylor=noisy_taylor,
                        **experiment_parameters,
                    )
                    if noisy_taylor is None:
                        noisy_taylor = evaluation["noisy_gradients"]["taylor"]
                    if first_evaluation is None:
                        first_evaluation = evaluation

                    gradient = evaluation["gradient_metrics"]
                    clean_fidelity = evaluation["hog_metrics"][
                        "clean_robust_fidelity_to_taylor_standard"
                    ]
                    for method in METHOD_LABELS:
                        metrics = evaluation["hog_metrics"][method]
                        hog_records.append(
                            {
                                "image": str(image_path),
                                "image_height": clean.shape[0],
                                "image_width": clean.shape[1],
                                "seed": seed,
                                "noise_family": noise_family,
                                "noise_level": noise_level,
                                "achieved_snr_db": (
                                    "" if achieved_snr is None else achieved_snr
                                ),
                                "actual_impulse_density": (
                                    ""
                                    if actual_density is None
                                    else actual_density
                                ),
                                "normalization": normalization,
                                "method": method,
                                "method_label": METHOD_LABELS[method],
                                "hog_cosine_similarity": metrics[
                                    "cosine_similarity"
                                ],
                                "hog_relative_error": metrics["relative_error"],
                                "common_reference_cosine": metrics[
                                    "common_reference_cosine"
                                ],
                                "common_reference_relative_error": metrics[
                                    "common_reference_relative_error"
                                ],
                                "clean_fidelity_cosine": (
                                    clean_fidelity["cosine_similarity"]
                                    if method == "C_taylor_robust"
                                    else 1.0
                                ),
                                "clean_fidelity_relative_error": (
                                    clean_fidelity["relative_error"]
                                    if method == "C_taylor_robust"
                                    else 0.0
                                ),
                            }
                        )

                    if (
                        not representative_saved
                        and image_index == 0
                        and seed == args.seeds[0]
                        and noise_family == "saltpepper"
                        and noise_level == 10.0
                        and normalization == "L2-Hys"
                    ):
                        show_method_comparison(
                            clean,
                            noisy,
                            evaluation,
                            condition_label="salt-and-pepper, density=10%",
                            save_path=str(
                                output_dir / "representative_comparison.png"
                            ),
                            show=False,
                        )
                        representative_saved = True

                gradient = first_evaluation["gradient_metrics"]
                gradient_records.append(
                    {
                        "image": str(image_path),
                        "image_height": clean.shape[0],
                        "image_width": clean.shape[1],
                        "seed": seed,
                        "noise_family": noise_family,
                        "noise_level": noise_level,
                        "achieved_snr_db": (
                            "" if achieved_snr is None else achieved_snr
                        ),
                        "actual_impulse_density": (
                            "" if actual_density is None else actual_density
                        ),
                        "sobel_gradient_rmse": gradient["sobel_rmse"],
                        "taylor_gradient_rmse": gradient["taylor_rmse"],
                        "taylor_improvement_factor": gradient[
                            "taylor_improvement_factor"
                        ],
                    }
                )
                print(
                    f"[{Path(image_path).name}] seed={seed} "
                    f"{noise_family}={noise_level:g} complete"
                )

    gradient_summary = _summarize_gradient(gradient_records)
    hog_summary = _summarize_hog(hog_records)
    claims = _paired_claims(hog_records, gradient_records)

    _write_csv(output_dir / "gradient_trials.csv", gradient_records)
    _write_csv(output_dir / "hog_trials.csv", hog_records)
    _write_csv(output_dir / "gradient_summary.csv", gradient_summary)
    _write_csv(output_dir / "hog_summary.csv", hog_summary)
    _write_markdown_table(
        output_dir / "experiment_table.md",
        gradient_summary,
        hog_summary,
        claims,
    )
    _plot_gradient(
        gradient_summary,
        output_dir / "gradient_rmse.png",
    )
    _plot_hog_metric(
        hog_summary,
        "hog_cosine_similarity",
        "HOG cosine similarity (higher is better)",
        output_dir / "hog_cosine_similarity.png",
    )
    _plot_hog_metric(
        hog_summary,
        "hog_relative_error",
        "HOG relative error (lower is better)",
        output_dir / "hog_relative_error.png",
    )

    elapsed = time.perf_counter() - start
    config = {
        "images": list(args.images),
        "seeds": list(args.seeds),
        "max_dimension": args.max_dimension,
        "conditions": [
            {"noise_family": family, "level": level}
            for family, level in conditions
        ],
        "normalizations": normalizations,
        "parameters": experiment_parameters,
        "metric_definitions": {
            "gradient_stability_rmse": (
                "sqrt(mean((dx_noisy-dx_clean)^2 + "
                "(dy_noisy-dy_clean)^2)); each method uses its own clean output"
            ),
            "hog_cosine_similarity": (
                "dot(H_noisy,H_clean)/(||H_noisy|| ||H_clean||)"
            ),
            "hog_relative_error": (
                "||H_noisy-H_clean||/||H_clean||"
            ),
            "common_reference": (
                "All noisy HOG descriptors compared with clean "
                "Taylor + Standard HOG"
            ),
            "uncertainty": (
                "Noise seeds are averaged within each image; plotted 95% "
                "intervals are 1.96*SD/sqrt(n_images) across images"
            ),
            "gaussian_snr": (
                "10*log10(mean((image-mean(image))^2)/mean(noise^2)); "
                "Gaussian values are not clipped"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "scikit_image": skimage.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "git_sha": _git_sha(),
        "elapsed_seconds": elapsed,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    (output_dir / "claims_summary.json").write_text(
        json.dumps(claims, indent=2),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed,
        "gradient_trials": len(gradient_records),
        "hog_trials": len(hog_records),
        "claims": claims,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired, fair Taylor/Sobel and Standard/Robust HOG experiments."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=NATURAL_IMAGE_DEFAULTS,
        help="Input grayscale-capable images.",
    )
    parser.add_argument(
        "--bsds-test",
        action="store_true",
        help="Use every JPG in BSDS300/images/test instead of --images.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 29, 47],
        help="Paired noise seeds.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=256,
        help="Resize each image so its longest side is at most this value.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=15,
        help="Maximum correntropy gradient iterations.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_results",
        help="Directory for CSV, JSON, Markdown tables and plots.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one image, one seed and max dimension 192 for a smoke test.",
    )
    args = parser.parse_args()
    if args.bsds_test:
        args.images = [
            str(path) for path in sorted(Path("BSDS300/images/test").glob("*.jpg"))
        ]
        if not args.images:
            parser.error("No BSDS test images were found.")
    if args.quick:
        args.images = args.images[:1]
        args.seeds = args.seeds[:1]
        args.max_dimension = min(args.max_dimension, 192)
    return args


if __name__ == "__main__":
    result = run_experiments(parse_args())
    print(json.dumps(result, indent=2))
