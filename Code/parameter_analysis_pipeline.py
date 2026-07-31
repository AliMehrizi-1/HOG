from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv

from gradient_estimation_solver import MaclaurinApproximator
from hog_descriptor import HOGResult, describe_hog, render_hog_group


RANDOM_SEED = 42
N_VALUES = [4, 8, 16, 32]
H_VALUES = [1, 2, 3, 4]
NOISE_SCENARIOS = [
    ("no_noise", 0.0),
    ("saltpepper_5", 5.0),
    ("saltpepper_10", 10.0),
]
ALGORITHMS = [1, 2, 3]
ALGORITHM_NAMES = {
    1: "Direct squared loss",
    2: "Iterative squared loss",
    3: "Correntropy loss",
}

# Fixed HOG settings; these are controlled variables, not sweep parameters.
HOG_NORMALIZATION = "L2-Hys"
ROBUST_HOG = True
ROBUST_WEIGHTING = "huber"
ROBUST_DELTA = 2.0
ROBUST_WINDOW = 5
VISUALIZATION_PERCENTILE = 95.0

SEPARABLE_MODEL_NOTE = (
    "This experiment evaluates a separable 1-D Taylor model applied "
    "independently along x and y. It is not the joint full 2-D Taylor model."
)


@dataclass(frozen=True)
class GradientEstimate:
    magnitude: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    residual_rms: np.ndarray
    iteration_count: int
    final_residual: float
    converged: bool
    native_pad: int


@dataclass(frozen=True)
class AnalysisResult:
    experiment: str
    noise_scenario: str
    noise_density: float
    polynomial_order: int
    n: int
    h: int
    algorithm: int
    gradient_rmse: float
    hog_error: float
    cosine_similarity: float
    runtime: float
    iteration_count: int
    final_residual: float
    converged: bool
    noisy_hog: HOGResult

    def csv_row(self) -> dict:
        return {
            "experiment": self.experiment,
            "noise_scenario": self.noise_scenario,
            "noise_density": self.noise_density,
            "polynomial_order": self.polynomial_order,
            "n": self.n,
            "h": self.h,
            "algorithm": self.algorithm,
            "algorithm_name": ALGORITHM_NAMES[self.algorithm],
            "gradient_RMSE": self.gradient_rmse,
            "hog_error": self.hog_error,
            "cosine_similarity": self.cosine_similarity,
            "runtime": self.runtime,
            "iteration_count": self.iteration_count,
            "final_residual": self.final_residual,
            "converged": self.converged,
        }


class AxisTaylorSolver(MaclaurinApproximator):
    """One overdetermined 1-D Taylor system for a single image axis.

    ``n_points`` is the number of symmetric samples *per axis*.  It is not the
    total number of pixels in a joint 2-D neighbourhood.  For example, with
    polynomial_order=3:

    - n_points=4  -> design matrix 4x3
    - n_points=8  -> design matrix 8x3
    - n_points=16 -> design matrix 16x3
    - n_points=32 -> design matrix 32x3

    The optimization implementations are inherited unchanged from
    ``MaclaurinApproximator``.  This class only supplies the separable 1-D
    design matrix and records the diagnostics exposed by those solvers.
    """

    def __init__(self, n_points: int, h: int, polynomial_order: int):
        if not isinstance(n_points, (int, np.integer)) or n_points < 2:
            raise ValueError("n_points must be an integer of at least 2.")
        if (
            not isinstance(polynomial_order, (int, np.integer))
            or polynomial_order < 1
        ):
            raise ValueError("polynomial_order must be a positive integer.")

        if n_points <= polynomial_order:
            raise ValueError(
                "The separable robust experiment requires "
                "n_points > polynomial_order so each per-axis Taylor system "
                "is overdetermined. Received "
                f"n_points={n_points}, polynomial_order={polynomial_order}."
            )
        # Explicit scientific invariant requested for correntropy and
        # overdetermined least-squares estimation.
        assert n_points > polynomial_order
        if n_points % 2:
            raise ValueError("n_points must be even for symmetric sampling.")
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError("h must be a positive integer.")

        self.order = int(polynomial_order)
        self.sample_count = int(n_points)
        self.spacing = int(h)
        self.degree = int(polynomial_order)
        half = n_points // 2
        self.positions = np.asarray(
            [offset * h for offset in range(-half, 0)]
            + [offset * h for offset in range(1, half + 1)],
            dtype=float,
        )
        self.A = np.asarray(
            [
                [
                    position**degree / math.factorial(degree)
                    for degree in range(1, polynomial_order + 1)
                ]
                for position in self.positions
            ],
            dtype=float,
        )
        if self.A.shape != (n_points, polynomial_order):
            raise RuntimeError("Unexpected separable Taylor matrix shape.")
        matrix_rank = int(np.linalg.matrix_rank(self.A))
        if matrix_rank != polynomial_order:
            raise ValueError(
                f"Per-axis Taylor matrix is rank deficient: rank={matrix_rank}, "
                f"required={polynomial_order}."
            )

        # Pseudoinverse is used only for a verified overdetermined, full-column-
        # rank system. Underdetermined systems have already raised an error.
        self.A_inv = pinv(self.A, rcond=1e-10)
        self.pad = int(np.max(np.abs(self.positions)))
        self.last_solver_diagnostics = {
            "iterations": 0,
            "final_residual": float("nan"),
            "converged": False,
        }

    def solve_batch(
        self,
        differences: np.ndarray,
        algorithm: int,
        *,
        tol: float,
        max_iter: int,
        init_sigma: float | None,
    ) -> tuple[np.ndarray, dict]:
        if differences.ndim != 2 or differences.shape[1] != self.sample_count:
            raise ValueError("Unexpected per-axis difference-matrix shape.")
        if algorithm == 1:
            coefficients = differences @ self.A_inv.T
            residual = differences - coefficients @ self.A.T
            diagnostics = {
                "iterations": 1,
                "final_residual": float(np.sqrt(np.mean(residual**2))),
                "converged": True,
            }
        elif algorithm == 2:
            coefficients = self._solve_iterative_batch(
                differences,
                tol,
                max_iter,
            )
            diagnostics = dict(self.last_solver_diagnostics)
        elif algorithm == 3:
            coefficients = self._solve_correntropy_batch(
                differences,
                init_sigma,
                tol,
                max_iter,
            )
            diagnostics = dict(self.last_solver_diagnostics)
        else:
            raise ValueError("algorithm must be 1, 2 or 3.")
        return coefficients, diagnostics


def _load_image(path: str | Path, max_dimension: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read grayscale image: {path}")
    height, width = image.shape
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(64, round(width * scale)), max(64, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image.astype(float)


def _make_noise_scenarios(clean: np.ndarray) -> dict[str, np.ndarray]:
    """Create every noisy image once from a single deterministic permutation."""
    np.random.seed(RANDOM_SEED)
    permutation = np.random.permutation(clean.size)
    # Alternating values keep every prefix approximately salt/pepper balanced,
    # so the 5% corruption is a deterministic subset of the 10% corruption.
    impulse_values = np.where(
        np.arange(clean.size) % 2 == 0,
        0.0,
        255.0,
    )
    scenarios = {"no_noise": clean.copy()}
    for name, density in NOISE_SCENARIOS[1:]:
        count = int(round(density / 100.0 * clean.size))
        selected = permutation[:count]
        noisy = clean.copy()
        noisy.ravel()[selected] = impulse_values[:count]
        scenarios[name] = noisy
    return scenarios


def _axis_differences(
    image: np.ndarray,
    positions: np.ndarray,
    pad: int,
    *,
    axis: str,
) -> np.ndarray:
    valid_height = image.shape[0] - 2 * pad
    valid_width = image.shape[1] - 2 * pad
    if valid_height <= 0 or valid_width <= 0:
        raise ValueError(
            f"Image shape {image.shape} is too small for pad={pad}."
        )
    centre = image[pad : pad + valid_height, pad : pad + valid_width]
    samples = []
    for position in positions.astype(int):
        if axis == "x":
            sample = image[
                pad : pad + valid_height,
                pad + position : pad + position + valid_width,
            ]
        elif axis == "y":
            sample = image[
                pad + position : pad + position + valid_height,
                pad : pad + valid_width,
            ]
        else:
            raise ValueError("axis must be 'x' or 'y'.")
        samples.append(sample - centre)
    return np.stack(samples, axis=-1).reshape(-1, len(positions))


def _estimate_separable_gradient(
    image: np.ndarray,
    *,
    n_points: int,
    h: int,
    polynomial_order: int,
    algorithm: int,
    tol: float,
    max_iter: int,
    init_sigma: float | None,
    solver: AxisTaylorSolver | None = None,
) -> GradientEstimate:
    if solver is None:
        solver = AxisTaylorSolver(n_points, h, polynomial_order)
    elif (
        solver.sample_count != n_points
        or solver.spacing != h
        or solver.order != polynomial_order
    ):
        raise ValueError("The supplied per-axis solver parameters do not match.")
    horizontal_differences = _axis_differences(
        image,
        solver.positions,
        solver.pad,
        axis="x",
    )
    vertical_differences = _axis_differences(
        image,
        solver.positions,
        solver.pad,
        axis="y",
    )
    horizontal_coefficients, horizontal_diagnostics = solver.solve_batch(
        horizontal_differences,
        algorithm,
        tol=tol,
        max_iter=max_iter,
        init_sigma=init_sigma,
    )
    vertical_coefficients, vertical_diagnostics = solver.solve_batch(
        vertical_differences,
        algorithm,
        tol=tol,
        max_iter=max_iter,
        init_sigma=init_sigma,
    )

    valid_height = image.shape[0] - 2 * solver.pad
    valid_width = image.shape[1] - 2 * solver.pad
    dx = horizontal_coefficients[:, 0].reshape(valid_height, valid_width)
    dy = vertical_coefficients[:, 0].reshape(valid_height, valid_width)

    # One comparable unweighted RMS residual over both per-axis systems.
    horizontal_residual = (
        horizontal_differences - horizontal_coefficients @ solver.A.T
    ).reshape(valid_height, valid_width, solver.sample_count)
    vertical_residual = (
        vertical_differences - vertical_coefficients @ solver.A.T
    ).reshape(valid_height, valid_width, solver.sample_count)
    residual_rms = np.sqrt(
        (
            np.sum(horizontal_residual**2, axis=-1)
            + np.sum(vertical_residual**2, axis=-1)
        )
        / (2 * solver.sample_count)
    )
    final_residual = float(np.sqrt(np.mean(residual_rms**2)))

    return GradientEstimate(
        magnitude=np.hypot(dx, dy),
        dx=dx,
        dy=dy,
        residual_rms=residual_rms,
        iteration_count=max(
            int(horizontal_diagnostics["iterations"]),
            int(vertical_diagnostics["iterations"]),
        ),
        final_residual=final_residual,
        converged=(
            bool(horizontal_diagnostics["converged"])
            and bool(vertical_diagnostics["converged"])
        ),
        native_pad=solver.pad,
    )


def _crop_to_common_roi(
    array: np.ndarray,
    *,
    native_pad: int,
    common_pad: int,
) -> np.ndarray:
    extra = common_pad - native_pad
    if extra < 0:
        raise ValueError("common_pad must not be smaller than native_pad.")
    if extra == 0:
        return array
    if min(array.shape[:2]) <= 2 * extra:
        raise ValueError("Image is too small for the common analysis ROI.")
    return array[extra:-extra, extra:-extra]


def _apply_common_roi(
    estimate: GradientEstimate,
    common_pad: int,
) -> GradientEstimate:
    residual_rms = _crop_to_common_roi(
        estimate.residual_rms,
        native_pad=estimate.native_pad,
        common_pad=common_pad,
    )
    return GradientEstimate(
        magnitude=_crop_to_common_roi(
            estimate.magnitude,
            native_pad=estimate.native_pad,
            common_pad=common_pad,
        ),
        dx=_crop_to_common_roi(
            estimate.dx,
            native_pad=estimate.native_pad,
            common_pad=common_pad,
        ),
        dy=_crop_to_common_roi(
            estimate.dy,
            native_pad=estimate.native_pad,
            common_pad=common_pad,
        ),
        residual_rms=residual_rms,
        iteration_count=estimate.iteration_count,
        final_residual=float(np.sqrt(np.mean(residual_rms**2))),
        converged=estimate.converged,
        native_pad=common_pad,
    )


def _compute_hog(estimate: GradientEstimate) -> HOGResult:
    return describe_hog(
        estimate.dx,
        estimate.dy,
        fuzzy=True,
        normalization=HOG_NORMALIZATION,
        robust=ROBUST_HOG,
        robust_weighting=ROBUST_WEIGHTING,
        robust_delta=ROBUST_DELTA,
        robust_window=ROBUST_WINDOW,
        visualization_percentile=VISUALIZATION_PERCENTILE,
    )


def _gradient_magnitude_rmse(
    clean_magnitude: np.ndarray,
    noisy_magnitude: np.ndarray,
) -> float:
    if clean_magnitude.shape != noisy_magnitude.shape:
        raise ValueError("Clean and noisy magnitudes must have equal shapes.")
    return float(np.sqrt(np.mean((noisy_magnitude - clean_magnitude) ** 2)))


def _hog_relative_error(clean: np.ndarray, noisy: np.ndarray) -> float:
    if clean.shape != noisy.shape:
        raise ValueError("Clean and noisy HOG descriptors must have equal shapes.")
    return float(
        np.linalg.norm(noisy.ravel() - clean.ravel())
        / (np.linalg.norm(clean.ravel()) + np.finfo(float).eps)
    )


def _hog_cosine_similarity(clean: np.ndarray, noisy: np.ndarray) -> float:
    if clean.shape != noisy.shape:
        raise ValueError("Clean and noisy HOG descriptors must have equal shapes.")
    clean_flat = clean.ravel()
    noisy_flat = noisy.ravel()
    denominator = np.linalg.norm(clean_flat) * np.linalg.norm(noisy_flat)
    if denominator <= np.finfo(float).eps:
        return 1.0 if np.allclose(clean_flat, noisy_flat) else 0.0
    return float(np.dot(clean_flat, noisy_flat) / denominator)


def _run_configuration(
    clean: np.ndarray,
    noisy: np.ndarray,
    *,
    experiment: str,
    noise_scenario: str,
    noise_density: float,
    n: int,
    h: int,
    polynomial_order: int,
    algorithm: int,
    tol: float,
    max_iter: int,
    init_sigma: float | None,
    common_pad: int,
    runtime_repeats: int,
    clean_reference: tuple[GradientEstimate, HOGResult] | None = None,
) -> tuple[AnalysisResult, tuple[GradientEstimate, HOGResult]]:
    solver = AxisTaylorSolver(n, h, polynomial_order)
    if clean_reference is None:
        clean_estimate = _estimate_separable_gradient(
            clean,
            n_points=n,
            h=h,
            polynomial_order=polynomial_order,
            algorithm=algorithm,
            tol=tol,
            max_iter=max_iter,
            init_sigma=init_sigma,
            solver=solver,
        )
        clean_estimate = _apply_common_roi(clean_estimate, common_pad)
        clean_hog = _compute_hog(clean_estimate)
        clean_reference = (clean_estimate, clean_hog)
    else:
        clean_estimate, clean_hog = clean_reference

    runtimes = []
    noisy_estimate = None
    noisy_hog = None
    for _ in range(runtime_repeats):
        start = time.perf_counter()
        noisy_estimate = _estimate_separable_gradient(
            noisy,
            n_points=n,
            h=h,
            polynomial_order=polynomial_order,
            algorithm=algorithm,
            tol=tol,
            max_iter=max_iter,
            init_sigma=init_sigma,
            solver=solver,
        )
        noisy_estimate = _apply_common_roi(noisy_estimate, common_pad)
        noisy_hog = _compute_hog(noisy_estimate)
        runtimes.append(time.perf_counter() - start)

    result = AnalysisResult(
        experiment=experiment,
        noise_scenario=noise_scenario,
        noise_density=noise_density,
        polynomial_order=polynomial_order,
        n=n,
        h=h,
        algorithm=algorithm,
        gradient_rmse=_gradient_magnitude_rmse(
            clean_estimate.magnitude,
            noisy_estimate.magnitude,
        ),
        hog_error=_hog_relative_error(
            clean_hog.descriptor,
            noisy_hog.descriptor,
        ),
        cosine_similarity=_hog_cosine_similarity(
            clean_hog.descriptor,
            noisy_hog.descriptor,
        ),
        runtime=float(np.median(runtimes)),
        iteration_count=noisy_estimate.iteration_count,
        final_residual=noisy_estimate.final_residual,
        converged=noisy_estimate.converged,
        noisy_hog=noisy_hog,
    )
    return result, clean_reference


def _plot_analysis(
    results: list[AnalysisResult],
    *,
    experiment: str,
    varying_values: list[int],
    polynomial_order: int,
    output_path: Path,
    show: bool,
):
    lookup = {
        (
            result.noise_scenario,
            result.algorithm,
            result.n if experiment == "n" else result.h,
        ): result
        for result in results
    }
    ordered_results = [
        lookup[(scenario_name, algorithm, value)]
        for scenario_name, _ in NOISE_SCENARIOS
        for algorithm in ALGORITHMS
        for value in varying_values
    ]
    visualizations = render_hog_group(
        [result.noisy_hog for result in ordered_results],
        percentile=VISUALIZATION_PERCENTILE,
    )

    figure, axes = plt.subplots(
        len(NOISE_SCENARIOS) * len(ALGORITHMS),
        len(varying_values),
        figsize=(18, 27),
    )
    for axis, result, visualization in zip(
        axes.flat,
        ordered_results,
        visualizations,
    ):
        convergence_text = (
            f"{result.iteration_count}"
            if result.converged
            else f"{result.iteration_count} (limit)"
        )
        axis.imshow(visualization, cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"{result.noise_scenario} | n={result.n}, h={result.h}\n"
            f"{ALGORITHM_NAMES[result.algorithm]}\n"
            f"RMSE={result.gradient_rmse:.4f} | "
            f"Cosine={result.cosine_similarity:.4f}\n"
            f"HOG error={result.hog_error:.4f} | "
            f"Runtime={result.runtime:.3f} s\n"
            f"Iterations={convergence_text} | "
            f"Residual={result.final_residual:.4f}",
            fontsize=8,
        )
        axis.axis("off")

    varying_name = (
        "symmetric samples per axis n"
        if experiment == "n"
        else "sample spacing h"
    )
    fixed_text = "h=1" if experiment == "n" else "n=8 samples per axis"
    figure.suptitle(
        f"Separable Taylor parameter sensitivity: varying {varying_name}\n"
        f"polynomial_order={polynomial_order}, {fixed_text}, seed={RANDOM_SEED}",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        f"{SEPARABLE_MODEL_NOTE} HOG panels use one pooled "
        "95th-percentile scale.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.99, 0.965))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if show and "agg" not in str(plt.get_backend()).lower():
        plt.show()
    else:
        plt.close(figure)


def _write_results(path: Path, results: list[AnalysisResult]):
    columns = [
        "experiment",
        "noise_scenario",
        "noise_density",
        "polynomial_order",
        "n",
        "h",
        "algorithm",
        "algorithm_name",
        "gradient_RMSE",
        "hog_error",
        "cosine_similarity",
        "runtime",
        "iteration_count",
        "final_residual",
        "converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result.csv_row() for result in results)


def _design_matrix_diagnostics(polynomial_order: int) -> list[dict]:
    diagnostics = []
    for experiment, values in (("n", N_VALUES), ("h", H_VALUES)):
        for value in values:
            n = value if experiment == "n" else 8
            h = 1 if experiment == "n" else value
            solver = AxisTaylorSolver(n, h, polynomial_order)
            diagnostics.append(
                {
                    "experiment": experiment,
                    "n": n,
                    "h": h,
                    "shape": list(solver.A.shape),
                    "rank": int(np.linalg.matrix_rank(solver.A)),
                    "condition_number": float(np.linalg.cond(solver.A)),
                }
            )
    return diagnostics


def _result_summary_table(results: list[AnalysisResult]) -> list[str]:
    lines = [
        "| Sweep | Noise | Algorithm | Min-RMSE setting | RMSE | "
        "Max-cosine setting | Cosine | Min-error setting | Relative error |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in ("n_analysis", "h_analysis"):
        setting_name = "n" if experiment == "n_analysis" else "h"
        for scenario_name, _ in NOISE_SCENARIOS[1:]:
            for algorithm in ALGORITHMS:
                group = [
                    result
                    for result in results
                    if result.experiment == experiment
                    and result.noise_scenario == scenario_name
                    and result.algorithm == algorithm
                ]
                best_rmse = min(group, key=lambda result: result.gradient_rmse)
                best_cosine = min(
                    group,
                    key=lambda result: (
                        -result.cosine_similarity,
                        result.hog_error,
                    ),
                )
                best_error = min(
                    group,
                    key=lambda result: (
                        result.hog_error,
                        -result.cosine_similarity,
                    ),
                )
                rmse_setting = (
                    best_rmse.n if setting_name == "n" else best_rmse.h
                )
                cosine_setting = (
                    best_cosine.n
                    if setting_name == "n"
                    else best_cosine.h
                )
                error_setting = (
                    best_error.n if setting_name == "n" else best_error.h
                )
                lines.append(
                    f"| {setting_name} | {scenario_name} | "
                    f"{ALGORITHM_NAMES[algorithm]} | "
                    f"{setting_name}={rmse_setting} | "
                    f"{best_rmse.gradient_rmse:.4f} | "
                    f"{setting_name}={cosine_setting} | "
                    f"{best_cosine.cosine_similarity:.4f} | "
                    f"{setting_name}={error_setting} | "
                    f"{best_error.hog_error:.4f} |"
                )
    return lines


def _write_report(
    path: Path,
    args,
    common_pad: int,
    results: list[AnalysisResult],
):
    matrix_diagnostics = _design_matrix_diagnostics(args.polynomial_order)
    converged_count = sum(result.converged for result in results)
    iteration_ranges = {
        algorithm: (
            min(
                result.iteration_count
                for result in results
                if result.algorithm == algorithm
            ),
            max(
                result.iteration_count
                for result in results
                if result.algorithm == algorithm
            ),
        )
        for algorithm in ALGORITHMS
    }
    lines = [
        "# Parameter sensitivity analysis",
        "",
        f"> {SEPARABLE_MODEL_NOTE}",
        "",
        "## Experimental controls",
        "",
        f"- Polynomial order: `{args.polynomial_order}`",
        f"- Symmetric samples per axis: `{N_VALUES}`",
        f"- Sample spacing values: `{H_VALUES}`",
        f"- Random seed: `{RANDOM_SEED}`",
        "- Noise scenarios: no noise, 5% salt-and-pepper, and 10% "
        "salt-and-pepper",
        f"- Common valid-image padding: `{common_pad}` pixels",
        f"- Solver tolerance: `{args.tol}`",
        f"- Maximum solver iterations: `{args.max_iter}`",
        f"- Converged rows: `{converged_count}/{len(results)}`; iteration "
        f"ranges: direct `{iteration_ranges[1][0]}–{iteration_ranges[1][1]}`, "
        f"iterative `{iteration_ranges[2][0]}–{iteration_ranges[2][1]}`, "
        f"correntropy `{iteration_ranges[3][0]}–{iteration_ranges[3][1]}`",
        f"- HOG normalization: `{HOG_NORMALIZATION}`",
        f"- Fixed HOG weighting: `{ROBUST_WEIGHTING}` "
        f"(delta={ROBUST_DELTA}, window={ROBUST_WINDOW})",
        f"- Runtime: median of `{args.runtime_repeats}` noisy inference runs; "
        "per-configuration matrix setup and clean-reference computation are "
        "excluded",
        "",
        "Salt-and-pepper density denotes the fraction of selected pixel "
        "locations. A selected endpoint-valued pixel can already equal its "
        "assigned impulse value, so the realized fraction of changed numeric "
        "values can be slightly smaller.",
        "",
        "For every `(n, h, algorithm)` combination, the clean reference uses "
        "the identical estimator and HOG parameters as its noisy counterpart. "
        "The same precomputed noisy image is reused across all parameter "
        "comparisons in a scenario.",
        "",
        "The reported final residual is the unweighted RMS Taylor-fit residual "
        "over the horizontal and vertical per-axis systems on the same common "
        "image ROI used for all metrics. It is a fit diagnostic, not the "
        "correntropy objective, and its absolute scale can change with n and h. "
        "Direct loss reports one closed-form solve; iterative squared loss and "
        "correntropy report the maximum actual batch-iteration count across x "
        "and y. `converged=false` and `(limit)` in a panel mean that at least "
        "one axis reached `max_iter` before the strict all-pixel stopping rule.",
        "",
        "## Per-axis design matrices",
        "",
        "`n_points` counts the symmetric samples along one axis, including both "
        "negative and positive offsets. It does not count a full 2-D "
        "neighbourhood.",
        "",
        "| Sweep | n | h | Matrix shape | Rank | Condition number |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {item['experiment']} | {item['n']} | {item['h']} | "
            f"{item['shape'][0]}x{item['shape'][1]} | {item['rank']} | "
            f"{item['condition_number']:.3f} |"
        )
        for item in matrix_diagnostics
    )
    lines.extend(
        [
            "",
            "## Descriptive results",
            "",
            "The table identifies the setting with the lowest gradient RMSE "
            "and, separately, settings with the highest HOG cosine similarity "
            "and lowest HOG relative error. These optima are descriptive rather "
            "than inferential.",
            "",
        ]
    )
    lines.extend(_result_summary_table(results))
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Gradient RMSE here compares each noisy result with its matched "
            "clean estimator output. It measures noise stability, not absolute "
            "derivative accuracy against analytical ground truth. The no-noise "
            "rows are regression checks and must yield RMSE=0, cosine=1, and "
            "relative error=0.",
            "",
            "This deterministic single-image, single-seed experiment supports "
            "parameter sensitivity analysis only. Population-level claims "
            "require multiple images and seeds; absolute accuracy claims "
            "require an analytical synthetic-gradient reference.",
            "",
            "## Generated artifacts",
            "",
            "- [Fixed h, varying n](n_analysis.png)",
            "- [Fixed n, varying h](h_analysis.png)",
            "- [All numerical measurements](parameter_analysis_results.csv)",
            "- [Reproducibility configuration](parameter_analysis_config.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parameter_analysis(args) -> list[AnalysisResult]:
    if args.runtime_repeats < 1:
        raise ValueError("runtime_repeats must be at least 1.")
    if args.max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if not np.isfinite(args.tol) or args.tol <= 0:
        raise ValueError("tol must be a positive finite number.")
    if args.init_sigma is not None and (
        not np.isfinite(args.init_sigma) or args.init_sigma <= 0
    ):
        raise ValueError("init_sigma must be positive and finite or None.")
    if args.max_dimension < 64:
        raise ValueError("max_dimension must be at least 64.")
    if (
        not isinstance(args.polynomial_order, int)
        or args.polynomial_order < 1
    ):
        raise ValueError("polynomial_order must be a positive integer.")
    if any(n_points <= args.polynomial_order for n_points in N_VALUES):
        raise ValueError(
            "Every separable n condition must satisfy "
            "n_points > polynomial_order. Received "
            f"polynomial_order={args.polynomial_order}, n_values={N_VALUES}."
        )
    assert all(
        n_points > args.polynomial_order for n_points in N_VALUES
    ), "Every per-axis Taylor system must be overdetermined."

    clean = _load_image(args.image, args.max_dimension)
    scenario_images = _make_noise_scenarios(clean)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Largest half-width across both requested sweeps:
    # n=32,h=1 -> 16 pixels; n=8,h=4 -> 16 pixels.
    common_pad = max(max(N_VALUES) // 2, (8 // 2) * max(H_VALUES))
    if min(clean.shape) - 2 * common_pad < 16:
        raise ValueError(
            f"Image shape {clean.shape} is too small for common_pad={common_pad} "
            "and a 2x2-cell HOG descriptor."
        )

    clean_cache: dict[
        tuple[int, int, int],
        tuple[GradientEstimate, HOGResult],
    ] = {}
    result_cache: dict[tuple[str, int, int, int], AnalysisResult] = {}
    all_results: list[AnalysisResult] = []

    def evaluate(
        experiment: str,
        scenario_name: str,
        density: float,
        n: int,
        h: int,
        algorithm: int,
    ):
        clean_key = (n, h, algorithm)
        result_key = (scenario_name, n, h, algorithm)
        if result_key not in result_cache:
            result, clean_reference = _run_configuration(
                clean,
                scenario_images[scenario_name],
                experiment=experiment,
                noise_scenario=scenario_name,
                noise_density=density,
                n=n,
                h=h,
                polynomial_order=args.polynomial_order,
                algorithm=algorithm,
                tol=args.tol,
                max_iter=args.max_iter,
                init_sigma=args.init_sigma,
                common_pad=common_pad,
                runtime_repeats=args.runtime_repeats,
                clean_reference=clean_cache.get(clean_key),
            )
            clean_cache[clean_key] = clean_reference
            result_cache[result_key] = result
        cached = result_cache[result_key]
        all_results.append(
            AnalysisResult(
                experiment=experiment,
                noise_scenario=cached.noise_scenario,
                noise_density=cached.noise_density,
                polynomial_order=cached.polynomial_order,
                n=cached.n,
                h=cached.h,
                algorithm=cached.algorithm,
                gradient_rmse=cached.gradient_rmse,
                hog_error=cached.hog_error,
                cosine_similarity=cached.cosine_similarity,
                runtime=cached.runtime,
                iteration_count=cached.iteration_count,
                final_residual=cached.final_residual,
                converged=cached.converged,
                noisy_hog=cached.noisy_hog,
            )
        )
        print(
            f"[{experiment}/{scenario_name}] n={n}, h={h}, "
            f"algorithm={algorithm} complete"
        )

    n_results = []
    for scenario_name, density in NOISE_SCENARIOS:
        for algorithm in ALGORITHMS:
            for n in N_VALUES:
                evaluate("n_analysis", scenario_name, density, n, 1, algorithm)
                n_results.append(all_results[-1])

    h_results = []
    for scenario_name, density in NOISE_SCENARIOS:
        for algorithm in ALGORITHMS:
            for h in H_VALUES:
                evaluate("h_analysis", scenario_name, density, 8, h, algorithm)
                h_results.append(all_results[-1])

    _plot_analysis(
        n_results,
        experiment="n",
        varying_values=N_VALUES,
        polynomial_order=args.polynomial_order,
        output_path=output_dir / "n_analysis.png",
        show=args.show,
    )
    _plot_analysis(
        h_results,
        experiment="h",
        varying_values=H_VALUES,
        polynomial_order=args.polynomial_order,
        output_path=output_dir / "h_analysis.png",
        show=args.show,
    )
    _write_results(output_dir / "parameter_analysis_results.csv", all_results)
    _write_report(
        output_dir / "parameter_analysis_report.md",
        args,
        common_pad,
        all_results,
    )

    matrix_diagnostics = _design_matrix_diagnostics(args.polynomial_order)
    configuration = {
        "model_note": SEPARABLE_MODEL_NOTE,
        "image": str(args.image),
        "resized_shape": list(clean.shape),
        "random_seed": RANDOM_SEED,
        "polynomial_order": args.polynomial_order,
        "n_definition": (
            "number of symmetric samples per axis; horizontal and vertical "
            "systems are solved independently"
        ),
        "per_axis_matrix_shapes": {
            str(n): [n, args.polynomial_order] for n in N_VALUES
        },
        "per_axis_design_matrix_diagnostics": matrix_diagnostics,
        "n_analysis": {"h": 1, "n_values": N_VALUES},
        "h_analysis": {"n": 8, "h_values": H_VALUES},
        "noise_scenarios": [
            {"name": name, "saltpepper_density_percent": density}
            for name, density in NOISE_SCENARIOS
        ],
        "algorithms": ALGORITHM_NAMES,
        "common_valid_pad": common_pad,
        "gradient_metric": (
            "Matched-clean noise-stability RMSE: "
            "sqrt(mean((magnitude_noisy-magnitude_clean)^2)); "
            "not absolute derivative accuracy"
        ),
        "hog_relative_error": (
            "||H_noisy-H_clean||_2 / ||H_clean||_2"
        ),
        "hog_cosine_similarity": (
            "dot(H_noisy,H_clean)/(||H_noisy||_2 ||H_clean||_2)"
        ),
        "final_residual": (
            "Unweighted RMS Taylor-fit residual over both per-axis systems "
            "on the common analysis ROI; diagnostic, not correntropy loss"
        ),
        "convergence_definition": (
            "converged=true only when both per-axis batch solves satisfy the "
            "strict all-pixel stopping rule before max_iter"
        ),
        "runtime_definition": (
            f"Median of {args.runtime_repeats} noisy gradient-plus-HOG runs; "
            "per-configuration design-matrix setup and clean-reference "
            "computation excluded"
        ),
        "solver_parameters": {
            "tol": args.tol,
            "max_iter": args.max_iter,
            "init_sigma": args.init_sigma,
        },
        "hog_configuration": {
            "normalization": HOG_NORMALIZATION,
            "robust": ROBUST_HOG,
            "robust_weighting": ROBUST_WEIGHTING,
            "robust_delta": ROBUST_DELTA,
            "robust_window": ROBUST_WINDOW,
            "visualization_percentile": VISUALIZATION_PERCENTILE,
        },
    }
    (output_dir / "parameter_analysis_config.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Separable per-axis Taylor/Maclaurin parameter sensitivity "
            "analysis with matched clean references."
        )
    )
    parser.add_argument("--image", default="Images/cameraman.png")
    parser.add_argument("--polynomial-order", type=int, default=3)
    parser.add_argument("--max-dimension", type=int, default=256)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--init-sigma", type=float, default=None)
    parser.add_argument(
        "--runtime-repeats",
        type=int,
        default=3,
        help="Number of noisy inference timings; the median is reported.",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    completed = run_parameter_analysis(arguments)
    results_path = Path(arguments.output_dir) / "parameter_analysis_results.csv"
    print(
        f"Saved {len(completed)} rows to {results_path}. "
        f"{SEPARABLE_MODEL_NOTE}"
    )
