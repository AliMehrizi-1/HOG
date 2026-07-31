"""Measure only missing Gaussian rows for the parameter-sensitivity sweeps.

This separate, resumable driver imports the existing estimator and metric
implementation without changing them.  It reuses the measured n=8, h=1 rows,
then fills the remaining n- and h-sweep configurations in a new CSV.
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import numpy as np

from parameter_analysis_pipeline import (
    ALGORITHM_NAMES,
    HOG_NORMALIZATION,
    RANDOM_SEED,
    ROBUST_DELTA,
    ROBUST_HOG,
    ROBUST_WEIGHTING,
    ROBUST_WINDOW,
    VISUALIZATION_PERCENTILE,
    AxisTaylorSolver,
    _apply_common_roi,
    _compute_hog,
    _estimate_separable_gradient,
    _gradient_magnitude_rmse,
    _hog_cosine_similarity,
    _hog_relative_error,
    _load_image,
)
from run_pipeline import add_gaussian_noise_snr


IMAGE_PATH = Path("Images/cameraman.png")
REUSE_PATH = Path(
    "results/final_parameter_analysis/gaussian_three_method_results.csv"
)
OUTPUT_PATH = Path(
    "results/final_parameter_analysis/"
    "gaussian_parameter_sensitivity_results.csv"
)
MAX_DIMENSION = 256
POLYNOMIAL_ORDER = 3
COMMON_PAD = 16
TOL = 1e-3
MAX_ITER = 500
INIT_SIGMA = None
RUNTIME_REPEATS = 3
SNR_LEVELS = (30.0, 20.0, 10.0)
ALGORITHMS = (1, 2, 3)
CONFIGURATIONS = (
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 1),
    (8, 2),
    (8, 3),
    (8, 4),
)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _format(value: float) -> str:
    return format(float(value), ".12g")


def _row_key(row: dict[str, str]) -> tuple[float, int, int, int]:
    return (
        float(row["target_snr_db"]),
        int(row["n"]),
        int(row["h"]),
        int(row["algorithm"]),
    )


def _write_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    level_order = {value: index for index, value in enumerate(SNR_LEVELS)}
    config_order = {
        value: index for index, value in enumerate(CONFIGURATIONS)
    }
    rows.sort(
        key=lambda row: (
            level_order[float(row["target_snr_db"])],
            int(row["algorithm"]),
            config_order[(int(row["n"]), int(row["h"]))],
        )
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(OUTPUT_PATH)


def main() -> None:
    clean = _load_image(IMAGE_PATH, MAX_DIMENSION)
    if clean.shape != (256, 256):
        raise RuntimeError(f"Unexpected image shape: {clean.shape}")
    if clean.shape[0] - 2 * COMMON_PAD != 224:
        raise RuntimeError("Unexpected common ROI geometry.")

    # Each level is generated exactly once.  The same array is then passed to
    # all three methods, ensuring paired noise within a level.
    gaussian_images = {}
    achieved_snrs = {}
    noise_hashes = {}
    for snr_db in SNR_LEVELS:
        noisy, achieved = add_gaussian_noise_snr(clean, snr_db, RANDOM_SEED)
        gaussian_images[snr_db] = noisy
        achieved_snrs[snr_db] = achieved
        noise_hashes[snr_db] = _array_sha256(noisy - clean)

    if not REUSE_PATH.exists():
        raise FileNotFoundError(f"Missing reusable measurements: {REUSE_PATH}")
    with REUSE_PATH.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        reusable_rows = [dict(row) for row in reader]
    if len(reusable_rows) != len(SNR_LEVELS) * len(ALGORITHMS):
        raise RuntimeError("Unexpected reusable Gaussian row count.")

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if list(reader.fieldnames or []) != fieldnames:
                raise RuntimeError("Resume CSV schema mismatch.")
            rows = [dict(row) for row in reader]
    else:
        rows = reusable_rows

    expected_keys = {
        (snr_db, n, h, algorithm)
        for snr_db in SNR_LEVELS
        for n, h in CONFIGURATIONS
        for algorithm in ALGORITHMS
    }
    seen_keys = set()
    for row in rows:
        key = _row_key(row)
        if key not in expected_keys or key in seen_keys:
            raise RuntimeError(f"Unexpected or duplicate cached row: {key}")
        seen_keys.add(key)
        snr_db, _, _, _ = key
        required = {
            "seed": str(RANDOM_SEED),
            "image": str(IMAGE_PATH).replace("\\", "/"),
            "image_height": str(clean.shape[0]),
            "image_width": str(clean.shape[1]),
            "polynomial_order": str(POLYNOMIAL_ORDER),
            "common_pad": str(COMMON_PAD),
            "roi_height": str(clean.shape[0] - 2 * COMMON_PAD),
            "roi_width": str(clean.shape[1] - 2 * COMMON_PAD),
            "runtime_repeats": str(RUNTIME_REPEATS),
            "noise_sha256": noise_hashes[snr_db],
            "hog_normalization": HOG_NORMALIZATION,
            "robust_hog": str(ROBUST_HOG).lower(),
            "robust_weighting": ROBUST_WEIGHTING,
            "robust_delta": _format(ROBUST_DELTA),
            "robust_window": str(ROBUST_WINDOW),
            "tol": _format(TOL),
            "max_iter": str(MAX_ITER),
        }
        for name, value in required.items():
            if row[name] != value:
                raise RuntimeError(
                    f"Cached provenance mismatch for {key}: {name}"
                )
        row["experiment"] = "gaussian_parameter_sensitivity"

    clean_references = {}

    def get_clean_reference(n: int, h: int, algorithm: int):
        key = (n, h, algorithm)
        if key not in clean_references:
            clean_solver = AxisTaylorSolver(n, h, POLYNOMIAL_ORDER)
            clean_estimate = _estimate_separable_gradient(
                clean,
                n_points=n,
                h=h,
                polynomial_order=POLYNOMIAL_ORDER,
                algorithm=algorithm,
                tol=TOL,
                max_iter=MAX_ITER,
                init_sigma=INIT_SIGMA,
                solver=clean_solver,
            )
            clean_estimate = _apply_common_roi(
                clean_estimate, COMMON_PAD
            )
            clean_references[key] = (
                clean_estimate,
                _compute_hog(clean_estimate),
            )
        return clean_references[key]

    completed = {_row_key(row) for row in rows}
    print(
        f"Reusing {len(completed)} measured rows; "
        f"{len(expected_keys - completed)} rows remain.",
        flush=True,
    )
    for snr_db in SNR_LEVELS:
        noisy = gaussian_images[snr_db]
        for algorithm in ALGORITHMS:
            for n, h in CONFIGURATIONS:
                key = (snr_db, n, h, algorithm)
                if key in completed:
                    continue
                solver = AxisTaylorSolver(n, h, POLYNOMIAL_ORDER)
                clean_estimate, clean_hog = get_clean_reference(
                    n, h, algorithm
                )
                runtimes = []
                noisy_estimate = None
                noisy_hog = None
                for _ in range(RUNTIME_REPEATS):
                    start = time.perf_counter()
                    noisy_estimate = _estimate_separable_gradient(
                        noisy,
                        n_points=n,
                        h=h,
                        polynomial_order=POLYNOMIAL_ORDER,
                        algorithm=algorithm,
                        tol=TOL,
                        max_iter=MAX_ITER,
                        init_sigma=INIT_SIGMA,
                        solver=solver,
                    )
                    noisy_estimate = _apply_common_roi(
                        noisy_estimate, COMMON_PAD
                    )
                    noisy_hog = _compute_hog(noisy_estimate)
                    runtimes.append(time.perf_counter() - start)

                rows.append(
                    {
                        "experiment": "gaussian_parameter_sensitivity",
                        "method": ALGORITHM_NAMES[algorithm],
                        "algorithm": algorithm,
                        "noise": f"Gaussian SNR {int(snr_db)} dB",
                        "target_snr_db": _format(snr_db),
                        "achieved_snr_db": _format(
                            achieved_snrs[snr_db]
                        ),
                        "seed": RANDOM_SEED,
                        "image": str(IMAGE_PATH).replace("\\", "/"),
                        "image_height": clean.shape[0],
                        "image_width": clean.shape[1],
                        "polynomial_order": POLYNOMIAL_ORDER,
                        "n": n,
                        "h": h,
                        "common_pad": COMMON_PAD,
                        "roi_height": clean.shape[0] - 2 * COMMON_PAD,
                        "roi_width": clean.shape[1] - 2 * COMMON_PAD,
                        "matched_clean_gradient_RMSE": _format(
                            _gradient_magnitude_rmse(
                                clean_estimate.magnitude,
                                noisy_estimate.magnitude,
                            )
                        ),
                        "hog_relative_error": _format(
                            _hog_relative_error(
                                clean_hog.descriptor,
                                noisy_hog.descriptor,
                            )
                        ),
                        "cosine_similarity": _format(
                            _hog_cosine_similarity(
                                clean_hog.descriptor,
                                noisy_hog.descriptor,
                            )
                        ),
                        "runtime": _format(float(np.median(runtimes))),
                        "runtime_trial_1": _format(runtimes[0]),
                        "runtime_trial_2": _format(runtimes[1]),
                        "runtime_trial_3": _format(runtimes[2]),
                        "runtime_repeats": RUNTIME_REPEATS,
                        "iteration_count": noisy_estimate.iteration_count,
                        "final_residual": _format(
                            noisy_estimate.final_residual
                        ),
                        "converged": str(
                            noisy_estimate.converged
                        ).lower(),
                        "noise_sha256": noise_hashes[snr_db],
                        "hog_normalization": HOG_NORMALIZATION,
                        "robust_hog": str(ROBUST_HOG).lower(),
                        "robust_weighting": ROBUST_WEIGHTING,
                        "robust_delta": _format(ROBUST_DELTA),
                        "robust_window": ROBUST_WINDOW,
                        "visualization_percentile": _format(
                            VISUALIZATION_PERCENTILE
                        ),
                        "tol": _format(TOL),
                        "max_iter": MAX_ITER,
                        "init_sigma": "",
                        "runtime_definition": (
                            "median of 3 noisy gradient-plus-HOG runs; "
                            "solver setup and clean-reference computation "
                            "excluded"
                        ),
                    }
                )
                completed.add(key)
                _write_rows(rows, fieldnames)
                print(
                    f"[Gaussian {snr_db:.0f} dB] n={n}, h={h}, "
                    f"{ALGORITHM_NAMES[algorithm]} complete "
                    f"({len(completed)}/{len(expected_keys)})",
                    flush=True,
                )

    if completed != expected_keys:
        raise RuntimeError("Gaussian sensitivity sweep is incomplete.")
    _write_rows(rows, fieldnames)
    print(f"Saved {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
