from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# This analysis evaluates a separable 1-D Taylor model applied independently along x and y. It is not the joint full 2-D Taylor model.
MODEL_NOTE = (
    "This analysis evaluates a separable 1-D Taylor model applied independently "
    "along x and y. It is not the joint full 2-D Taylor model."
)

DEFAULT_SEED = 42
POLYNOMIAL_ORDER = 3
N_VALUES = [4, 8, 16, 32]
H_VALUES = [1, 2, 3, 4]
NOISY_SCENARIOS = ["saltpepper_5", "saltpepper_10"]

ALGORITHM_ORDER = [
    "direct_squared",
    "iterative_squared",
    "correntropy",
]
ALGORITHM_LABELS = {
    "direct_squared": "Direct squared loss",
    "iterative_squared": "Iterative squared loss",
    "correntropy": "Correntropy loss",
}
ALGORITHM_COLORS = {
    "direct_squared": "#0072B2",
    "iterative_squared": "#E69F00",
    "correntropy": "#009E73",
}
ALGORITHM_MARKERS = {
    "direct_squared": "o",
    "iterative_squared": "s",
    "correntropy": "^",
}
ALGORITHM_LINESTYLES = {
    "direct_squared": "-",
    "iterative_squared": "--",
    "correntropy": "-.",
}

COMPOSITE_WEIGHTS = {
    "hog_relative_error": 0.45,
    "cosine_similarity": 0.30,
    "gradient_RMSE": 0.20,
    "runtime": 0.05,
}

NUMERIC_METRICS = [
    "gradient_RMSE",
    "hog_relative_error",
    "cosine_similarity",
    "runtime",
    "iterations",
    "residual",
]

COLUMN_ALIASES = {
    "experiment_type": [
        "experiment_type",
        "experiment",
        "sweep",
        "analysis",
    ],
    "noise_type": [
        "noise_type",
        "noise_scenario",
        "scenario",
        "noise",
    ],
    "noise_level": [
        "noise_level",
        "noise_density",
        "density",
        "saltpepper_density",
    ],
    "algorithm": [
        "algorithm_name",
        "algorithm",
        "method",
        "solver",
    ],
    "n": ["n", "n_points", "sample_count", "neighborhood_samples"],
    "h": ["h", "spacing", "sample_spacing"],
    "polynomial_order": [
        "polynomial_order",
        "taylor_order",
        "degree",
        "order",
    ],
    "gradient_RMSE": [
        "gradient_rmse",
        "gradient_RMSE",
        "rmse",
    ],
    "hog_relative_error": [
        "hog_relative_error",
        "hog_error",
        "relative_hog_error",
    ],
    "cosine_similarity": [
        "cosine_similarity",
        "hog_cosine_similarity",
        "cosine",
    ],
    "runtime": ["runtime", "elapsed_time", "time", "runtime_seconds"],
    "iterations": [
        "iterations",
        "iteration_count",
        "num_iterations",
        "n_iterations",
    ],
    "residual": ["residual", "final_residual", "solver_residual"],
    "converged": ["converged", "is_converged"],
    "seed": ["seed", "random_seed"],
}


@dataclass(frozen=True)
class CanonicalRow:
    experiment_type: str
    noise_type: str
    noise_level: float
    algorithm: str
    n: int
    h: int
    polynomial_order: int
    gradient_RMSE: float
    hog_relative_error: float
    cosine_similarity: float
    runtime: float
    iterations: float
    residual: float
    converged: bool
    seed: int
    source_csv: str


@dataclass(frozen=True)
class AggregateRow:
    experiment_type: str
    noise_type: str
    noise_level: float
    algorithm: str
    n: int
    h: int
    polynomial_order: int
    gradient_RMSE: float
    hog_relative_error: float
    cosine_similarity: float
    runtime: float
    iterations: float
    residual: float
    converged: bool
    replicate_count: int
    seed_count: int
    gradient_RMSE_std: float
    hog_relative_error_std: float
    cosine_similarity_std: float
    runtime_std: float


@dataclass(frozen=True)
class PlotSpec:
    attribute: str
    filename_token: str
    title: str
    ylabel: str
    higher_is_better: bool
    log_scale: bool = False


PLOT_SPECS = [
    PlotSpec(
        "gradient_RMSE",
        "gradient_rmse",
        "Gradient-magnitude noise-stability RMSE",
        "Gradient RMSE",
        False,
    ),
    PlotSpec(
        "hog_relative_error",
        "hog_error",
        "HOG relative error",
        r"$\|H_{\mathrm{noisy}}-H_{\mathrm{clean}}\|_2/"
        r"\|H_{\mathrm{clean}}\|_2$",
        False,
    ),
    PlotSpec(
        "cosine_similarity",
        "cosine",
        "HOG cosine similarity",
        "Cosine similarity",
        True,
    ),
    PlotSpec(
        "runtime",
        "runtime",
        "Inference runtime",
        "Runtime (s, logarithmic scale)",
        False,
        True,
    ),
]


def _normalise_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _resolve_columns(fieldnames: Sequence[str]) -> dict[str, str | None]:
    normalised = {_normalise_header(name): name for name in fieldnames}
    resolved: dict[str, str | None] = {}
    required = {
        "experiment_type",
        "noise_type",
        "algorithm",
        "n",
        "h",
        "polynomial_order",
        "gradient_RMSE",
        "hog_relative_error",
        "cosine_similarity",
        "runtime",
        "iterations",
        "residual",
    }
    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            candidate = normalised.get(_normalise_header(alias))
            if candidate is not None:
                found = candidate
                break
        if found is None and canonical in required:
            raise ValueError(
                f"Required result column {canonical!r} is missing. "
                f"Available columns: {list(fieldnames)}"
            )
        resolved[canonical] = found
    return resolved


def _finite_float(value: object, label: str) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; received {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; received {value!r}.")
    return number


def _integer(value: object, label: str) -> int:
    number = _finite_float(value, label)
    rounded = round(number)
    if not math.isclose(number, rounded, abs_tol=1e-9):
        raise ValueError(f"{label} must be integer-valued; received {number}.")
    return int(rounded)


def _boolean(value: object, label: str) -> bool:
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{label} must be boolean; received {value!r}.")


def _canonical_experiment(value: object) -> str:
    text = _normalise_header(str(value))
    if text in {"n", "vary_n", "n_analysis", "n_sweep"} or (
        "vary" in text and text.endswith("_n")
    ):
        return "vary_n"
    if text in {"h", "vary_h", "h_analysis", "h_sweep"} or (
        "vary" in text and text.endswith("_h")
    ):
        return "vary_h"
    raise ValueError(f"Unrecognized experiment type: {value!r}.")


def _canonical_algorithm(value: object) -> str:
    text = _normalise_header(str(value))
    if text in {"1", "algorithm_1", "alg_1"} or "direct" in text:
        return "direct_squared"
    if text in {"2", "algorithm_2", "alg_2"} or "iterative" in text:
        return "iterative_squared"
    if text in {"3", "algorithm_3", "alg_3"} or "correntropy" in text:
        return "correntropy"
    raise ValueError(f"Unrecognized algorithm: {value!r}.")


def _canonical_noise(value: object, level: float | None) -> tuple[str, float]:
    text = _normalise_header(str(value))
    text_level: float | None = None
    if text in {"no_noise", "clean", "none"}:
        text_level = 0.0
    elif "saltpepper_10" in text or text.endswith("_10"):
        text_level = 10.0
    elif "saltpepper_5" in text or text.endswith("_5"):
        text_level = 5.0
    else:
        match = re.search(r"(?:^|_)(\d+(?:\.\d+)?)(?:_|$)", text)
        text_level = float(match.group(1)) if match else None

    if (
        level is not None
        and text_level is not None
        and not math.isclose(level, text_level, abs_tol=1e-9)
    ):
        raise ValueError(
            f"Noise label {value!r} conflicts with numeric level {level}."
        )
    effective_level = level if level is not None else text_level
    if effective_level is not None and math.isclose(
        effective_level,
        0.0,
        abs_tol=1e-9,
    ):
        return "no_noise", 0.0
    if effective_level is not None and math.isclose(
        effective_level,
        5.0,
        abs_tol=1e-9,
    ):
        return "saltpepper_5", 5.0
    if effective_level is not None and math.isclose(
        effective_level,
        10.0,
        abs_tol=1e-9,
    ):
        return "saltpepper_10", 10.0
    raise ValueError(
        f"Unrecognized noise scenario {value!r} with level {level!r}."
    )


def _read_results_csv(path: Path, default_seed: int) -> list[CanonicalRow]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        columns = _resolve_columns(reader.fieldnames)
        normalized_headers = {
            _normalise_header(name): name for name in reader.fieldnames
        }

        def present_columns(canonical: str) -> list[str]:
            found = []
            for alias in COLUMN_ALIASES[canonical]:
                actual = normalized_headers.get(_normalise_header(alias))
                if actual is not None and actual not in found:
                    found.append(actual)
            return found

        algorithm_columns = present_columns("algorithm")
        noise_columns = present_columns("noise_type")
        rows: list[CanonicalRow] = []
        for line_number, raw in enumerate(reader, start=2):
            prefix = f"{path.name}:{line_number}"
            level_column = columns["noise_level"]
            level = (
                _finite_float(raw[level_column], f"{prefix} noise_level")
                if level_column is not None and str(raw[level_column]).strip()
                else None
            )
            noise_candidates = {
                _canonical_noise(raw[column], level)
                for column in noise_columns
                if str(raw[column]).strip()
            }
            if len(noise_candidates) != 1:
                raise ValueError(
                    f"{prefix} contains inconsistent noise columns: "
                    f"{noise_candidates}"
                )
            noise_type, noise_level = next(iter(noise_candidates))
            algorithm_candidates = {
                _canonical_algorithm(raw[column])
                for column in algorithm_columns
                if str(raw[column]).strip()
            }
            if len(algorithm_candidates) != 1:
                raise ValueError(
                    f"{prefix} contains inconsistent algorithm columns: "
                    f"{algorithm_candidates}"
                )
            converged_column = columns["converged"]
            seed_column = columns["seed"]
            row = CanonicalRow(
                experiment_type=_canonical_experiment(
                    raw[columns["experiment_type"]]
                ),
                noise_type=noise_type,
                noise_level=noise_level,
                algorithm=next(iter(algorithm_candidates)),
                n=_integer(raw[columns["n"]], f"{prefix} n"),
                h=_integer(raw[columns["h"]], f"{prefix} h"),
                polynomial_order=_integer(
                    raw[columns["polynomial_order"]],
                    f"{prefix} polynomial_order",
                ),
                gradient_RMSE=_finite_float(
                    raw[columns["gradient_RMSE"]],
                    f"{prefix} gradient_RMSE",
                ),
                hog_relative_error=_finite_float(
                    raw[columns["hog_relative_error"]],
                    f"{prefix} hog_relative_error",
                ),
                cosine_similarity=_finite_float(
                    raw[columns["cosine_similarity"]],
                    f"{prefix} cosine_similarity",
                ),
                runtime=_finite_float(
                    raw[columns["runtime"]],
                    f"{prefix} runtime",
                ),
                iterations=_finite_float(
                    raw[columns["iterations"]],
                    f"{prefix} iterations",
                ),
                residual=_finite_float(
                    raw[columns["residual"]],
                    f"{prefix} residual",
                ),
                converged=(
                    _boolean(raw[converged_column], f"{prefix} converged")
                    if converged_column is not None
                    and str(raw[converged_column]).strip()
                    else True
                ),
                seed=(
                    _integer(raw[seed_column], f"{prefix} seed")
                    if seed_column is not None
                    and str(raw[seed_column]).strip()
                    else default_seed
                ),
                source_csv=str(path.resolve()),
            )
            rows.append(row)
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def _validate_rows(
    rows: Sequence[CanonicalRow],
) -> None:
    seen: set[tuple] = set()
    for row in rows:
        key = (
            row.source_csv,
            row.seed,
            row.experiment_type,
            row.noise_type,
            row.n,
            row.h,
            row.algorithm,
            row.polynomial_order,
        )
        if key in seen:
            raise ValueError(f"Duplicate full experiment row detected: {key}")
        seen.add(key)
        if row.polynomial_order != POLYNOMIAL_ORDER:
            raise ValueError(
                "This report requires polynomial_order=3; received "
                f"{row.polynomial_order}."
            )
        if row.n <= row.polynomial_order:
            raise ValueError(
                "The separable per-axis model requires n_points > "
                f"polynomial_order; received n={row.n}, "
                f"order={row.polynomial_order}."
            )
        if row.experiment_type == "vary_n" and row.h != 1:
            raise ValueError("The n sweep must use fixed h=1.")
        if row.experiment_type == "vary_h" and row.n != 8:
            raise ValueError("The h sweep must use fixed n=8.")
        if row.gradient_RMSE < 0 or row.hog_relative_error < 0:
            raise ValueError("RMSE and HOG relative error must be nonnegative.")
        if row.runtime <= 0:
            raise ValueError("Runtime must be positive for logarithmic plotting.")
        if row.iterations < 0 or row.residual < 0:
            raise ValueError("Iterations and residual must be nonnegative.")
        if not -1.0 - 1e-9 <= row.cosine_similarity <= 1.0 + 1e-9:
            raise ValueError("Cosine similarity is outside its numerical range.")
        if not row.converged:
            raise ValueError(
                "Non-converged rows cannot be used in the publication ranking. "
                "Re-run the experiment with a sufficient iteration limit."
            )

    expected_noises = {"no_noise", *NOISY_SCENARIOS}
    expected_design = {
        (
            experiment,
            noise,
            algorithm,
            n,
            h,
            POLYNOMIAL_ORDER,
        )
        for noise in expected_noises
        for algorithm in ALGORITHM_ORDER
        for experiment, n, h in (
            *[("vary_n", n_value, 1) for n_value in N_VALUES],
            *[("vary_h", 8, h_value) for h_value in H_VALUES],
        )
    }
    source_seed_groups: dict[tuple[str, int], list[CanonicalRow]] = defaultdict(
        list
    )
    for row in rows:
        source_seed_groups[(row.source_csv, row.seed)].append(row)
    for source_seed, group in source_seed_groups.items():
        observed_design = {
            (
                row.experiment_type,
                row.noise_type,
                row.algorithm,
                row.n,
                row.h,
                row.polynomial_order,
            )
            for row in group
        }
        if observed_design != expected_design:
            missing = sorted(expected_design - observed_design)
            extra = sorted(observed_design - expected_design)
            raise ValueError(
                "Every source/seed must contain the complete 72-row design. "
                f"Source/seed={source_seed}; missing={missing[:5]}, "
                f"extra={extra[:5]}."
            )

    if {row.noise_type for row in rows} != expected_noises:
        raise ValueError(
            "Expected noise scenarios no_noise, saltpepper_5 and "
            "saltpepper_10."
        )
    if {row.algorithm for row in rows} != set(ALGORITHM_ORDER):
        raise ValueError("All three requested algorithms must be present.")
    if {row.n for row in rows if row.experiment_type == "vary_n"} != set(
        N_VALUES
    ):
        raise ValueError(f"The n sweep must contain {N_VALUES}.")
    if {row.h for row in rows if row.experiment_type == "vary_h"} != set(
        H_VALUES
    ):
        raise ValueError(f"The h sweep must contain {H_VALUES}.")

    for row in rows:
        if row.noise_type == "no_noise":
            if not math.isclose(row.gradient_RMSE, 0.0, abs_tol=1e-10):
                raise ValueError("No-noise gradient RMSE regression check failed.")
            if not math.isclose(row.hog_relative_error, 0.0, abs_tol=1e-10):
                raise ValueError("No-noise HOG error regression check failed.")
            if not math.isclose(row.cosine_similarity, 1.0, abs_tol=1e-10):
                raise ValueError("No-noise HOG cosine regression check failed.")

    overlap: dict[tuple, list[CanonicalRow]] = defaultdict(list)
    for row in rows:
        if row.n == 8 and row.h == 1:
            overlap[
                (
                    row.source_csv,
                    row.seed,
                    row.noise_type,
                    row.algorithm,
                    row.polynomial_order,
                )
            ].append(row)
    for key, duplicates in overlap.items():
        if len(duplicates) != 2:
            raise ValueError(f"Expected one shared n=8,h=1 row per sweep: {key}")
        for metric in NUMERIC_METRICS:
            values = [getattr(row, metric) for row in duplicates]
            if not math.isclose(values[0], values[1], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(
                    f"Cross-sweep n=8,h=1 mismatch for {key}, metric={metric}."
                )


def _aggregate_group(
    rows: Sequence[CanonicalRow],
    *,
    experiment_type: str,
) -> AggregateRow:
    first = rows[0]

    def mean(attribute: str) -> float:
        return float(fmean(getattr(row, attribute) for row in rows))

    def std(attribute: str) -> float:
        values = [getattr(row, attribute) for row in rows]
        return float(pstdev(values)) if len(values) > 1 else 0.0

    return AggregateRow(
        experiment_type=experiment_type,
        noise_type=first.noise_type,
        noise_level=first.noise_level,
        algorithm=first.algorithm,
        n=first.n,
        h=first.h,
        polynomial_order=first.polynomial_order,
        gradient_RMSE=mean("gradient_RMSE"),
        hog_relative_error=mean("hog_relative_error"),
        cosine_similarity=mean("cosine_similarity"),
        runtime=mean("runtime"),
        iterations=mean("iterations"),
        residual=mean("residual"),
        converged=all(row.converged for row in rows),
        replicate_count=len(rows),
        seed_count=len({row.seed for row in rows}),
        gradient_RMSE_std=std("gradient_RMSE"),
        hog_relative_error_std=std("hog_relative_error"),
        cosine_similarity_std=std("cosine_similarity"),
        runtime_std=std("runtime"),
    )


def _aggregate_sweeps(rows: Sequence[CanonicalRow]) -> list[AggregateRow]:
    grouped: dict[tuple, list[CanonicalRow]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.experiment_type,
                row.noise_type,
                row.algorithm,
                row.n,
                row.h,
                row.polynomial_order,
            )
        ].append(row)
    return [
        _aggregate_group(group, experiment_type=key[0])
        for key, group in sorted(grouped.items())
    ]


def _aggregate_physical_candidates(
    rows: Sequence[CanonicalRow],
) -> list[AggregateRow]:
    # The n=8,h=1 baseline occurs in both OFAT sweeps. Collapse it within
    # each seed/source first so it cannot receive double weight.
    units: dict[tuple, list[CanonicalRow]] = defaultdict(list)
    for row in rows:
        units[
            (
                row.source_csv,
                row.seed,
                row.noise_type,
                row.algorithm,
                row.n,
                row.h,
                row.polynomial_order,
            )
        ].append(row)

    collapsed_units: list[CanonicalRow] = []
    for group in units.values():
        first = group[0]

        def mean(attribute: str) -> float:
            return float(fmean(getattr(row, attribute) for row in group))

        collapsed_units.append(
            CanonicalRow(
                experiment_type="combined",
                noise_type=first.noise_type,
                noise_level=first.noise_level,
                algorithm=first.algorithm,
                n=first.n,
                h=first.h,
                polynomial_order=first.polynomial_order,
                gradient_RMSE=mean("gradient_RMSE"),
                hog_relative_error=mean("hog_relative_error"),
                cosine_similarity=mean("cosine_similarity"),
                runtime=mean("runtime"),
                iterations=mean("iterations"),
                residual=mean("residual"),
                converged=all(row.converged for row in group),
                seed=first.seed,
                source_csv=first.source_csv,
            )
        )

    candidates: dict[tuple, list[CanonicalRow]] = defaultdict(list)
    for row in collapsed_units:
        candidates[
            (
                row.noise_type,
                row.algorithm,
                row.n,
                row.h,
                row.polynomial_order,
            )
        ].append(row)
    return [
        _aggregate_group(group, experiment_type="combined")
        for _, group in sorted(candidates.items())
    ]


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _plot_sweep(
    rows: Sequence[AggregateRow],
    *,
    experiment_type: str,
    x_attribute: str,
    x_values: Sequence[int],
    fixed_parameter: str,
    seed_label: str,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    _configure_plot_style()
    output_paths: list[Path] = []
    x_positions = np.arange(len(x_values), dtype=float)
    for spec in PLOT_SPECS:
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.35),
            sharey=True,
        )
        for panel_index, (axis, noise_type) in enumerate(
            zip(axes, NOISY_SCENARIOS)
        ):
            for algorithm in ALGORITHM_ORDER:
                series = [
                    row
                    for row in rows
                    if row.experiment_type == experiment_type
                    and row.noise_type == noise_type
                    and row.algorithm == algorithm
                ]
                series.sort(key=lambda row: getattr(row, x_attribute))
                observed_x = [getattr(row, x_attribute) for row in series]
                if observed_x != list(x_values):
                    raise ValueError(
                        f"Incomplete {experiment_type}/{noise_type}/{algorithm} "
                        f"series: {observed_x}"
                    )
                y_values = np.asarray(
                    [getattr(row, spec.attribute) for row in series],
                    dtype=float,
                )
                if spec.attribute == "cosine_similarity":
                    y_values = np.clip(y_values, -1.0, 1.0)
                color = ALGORITHM_COLORS[algorithm]
                axis.plot(
                    x_positions,
                    y_values,
                    color=color,
                    linestyle=ALGORITHM_LINESTYLES[algorithm],
                    marker=ALGORITHM_MARKERS[algorithm],
                    markersize=4.8,
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    label=ALGORITHM_LABELS[algorithm],
                    zorder=2,
                )
                std_attribute = f"{spec.attribute}_std"
                standard_deviation = np.asarray(
                    [getattr(row, std_attribute) for row in series],
                    dtype=float,
                )
                if np.any(standard_deviation > 0):
                    axis.errorbar(
                        x_positions,
                        y_values,
                        yerr=standard_deviation,
                        color=color,
                        linewidth=0,
                        elinewidth=0.8,
                        capsize=2,
                        alpha=0.7,
                        zorder=1,
                    )
                best_index = (
                    int(np.argmax(y_values))
                    if spec.higher_is_better
                    else int(np.argmin(y_values))
                )
                axis.scatter(
                    [x_positions[best_index]],
                    [y_values[best_index]],
                    marker="*",
                    s=70,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )

            density = 5 if noise_type == "saltpepper_5" else 10
            axis.set_title(
                f"({chr(97 + panel_index)}) Salt-and-pepper {density}%"
            )
            axis.set_xticks(x_positions, [str(value) for value in x_values])
            axis.set_xlabel(
                "Symmetric samples per axis, $n$"
                if x_attribute == "n"
                else "Sample spacing, $h$ (pixels)"
            )
            axis.grid(axis="y", color="#B8B8B8", linewidth=0.55, alpha=0.55)
            axis.margins(x=0.08, y=0.10)
            if spec.log_scale:
                axis.set_yscale("log")

        axes[0].set_ylabel(spec.ylabel)
        figure.suptitle(
            f"{spec.title}: "
            f"{'n sensitivity' if x_attribute == 'n' else 'h sensitivity'}",
            fontsize=11,
            fontweight="semibold",
            y=0.985,
        )
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=ALGORITHM_COLORS[algorithm],
                linestyle=ALGORITHM_LINESTYLES[algorithm],
                marker=ALGORITHM_MARKERS[algorithm],
                markerfacecolor="white",
                markersize=4.8,
                label=ALGORITHM_LABELS[algorithm],
            )
            for algorithm in ALGORITHM_ORDER
        ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.035),
        )
        figure.text(
            0.5,
            0.008,
            f"$p=3$; {fixed_parameter}; star = best point on each curve; "
            f"{seed_label}.",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#444444",
        )
        figure.tight_layout(rect=(0.02, 0.17, 0.995, 0.93), w_pad=1.8)
        output_path = (
            output_dir / f"{x_attribute}_sweep_{spec.filename_token}.png"
        )
        figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            metadata={
                "Description": MODEL_NOTE,
                "Software": "parameter_sensitivity_report.py",
            },
        )
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[dict],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _format_csv_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-12):
        return str(int(round(value)))
    return f"{value:.12g}"


def _canonical_row_dict(row: CanonicalRow) -> dict:
    return {
        "experiment_type": row.experiment_type,
        "noise_type": row.noise_type,
        "noise_level": _format_csv_number(row.noise_level),
        "algorithm": row.algorithm,
        "n": row.n,
        "h": row.h,
        "polynomial_order": row.polynomial_order,
        "gradient_RMSE": _format_csv_number(row.gradient_RMSE),
        "hog_relative_error": _format_csv_number(row.hog_relative_error),
        "cosine_similarity": _format_csv_number(row.cosine_similarity),
        "runtime": _format_csv_number(row.runtime),
        "iterations": _format_csv_number(row.iterations),
        "residual": _format_csv_number(row.residual),
        "converged": row.converged,
        "seed": row.seed,
        "source_csv": row.source_csv,
    }


def _best_parameter_rows(
    candidates: Sequence[AggregateRow],
) -> list[dict]:
    criteria = [
        ("minimum_gradient_RMSE", "gradient_RMSE", False),
        ("minimum_hog_relative_error", "hog_relative_error", False),
        ("maximum_cosine_similarity", "cosine_similarity", True),
        ("minimum_runtime", "runtime", False),
    ]
    result: list[dict] = []
    for noise_type in NOISY_SCENARIOS:
        for algorithm in ALGORITHM_ORDER:
            group = [
                row
                for row in candidates
                if row.noise_type == noise_type and row.algorithm == algorithm
            ]
            for best_by, attribute, higher_is_better in criteria:
                def key(row: AggregateRow) -> tuple:
                    primary = getattr(row, attribute)
                    if higher_is_better:
                        primary = -primary
                    return (
                        primary,
                        row.hog_relative_error,
                        -row.cosine_similarity,
                        row.gradient_RMSE,
                        row.runtime,
                        row.n,
                        row.h,
                    )

                best = min(group, key=key)
                result.append(
                    {
                        "noise_type": noise_type,
                        "algorithm": algorithm,
                        "best_by": best_by,
                        "best_n": best.n,
                        "best_h": best.h,
                        "polynomial_order": best.polynomial_order,
                        "gradient_RMSE": _format_csv_number(best.gradient_RMSE),
                        "hog_relative_error": _format_csv_number(
                            best.hog_relative_error
                        ),
                        "cosine_similarity": _format_csv_number(
                            best.cosine_similarity
                        ),
                        "runtime": _format_csv_number(best.runtime),
                        "iterations": _format_csv_number(best.iterations),
                        "residual": _format_csv_number(best.residual),
                    }
                )
    return result


def _benefit_normalization(
    values: Sequence[float],
    *,
    higher_is_better: bool,
) -> tuple[np.ndarray, tuple[float, float]]:
    array = np.asarray(values, dtype=float)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    span = maximum - minimum
    if span <= np.finfo(float).eps * max(1.0, abs(minimum), abs(maximum)):
        normalized = np.ones_like(array)
    else:
        normalized = (array - minimum) / span
        if not higher_is_better:
            normalized = 1.0 - normalized
    return normalized, (minimum, maximum)


def _rank_candidates(
    candidates: Sequence[AggregateRow],
) -> tuple[list[dict], dict[str, tuple[float, float]], list[dict]]:
    noisy = [row for row in candidates if row.noise_type in NOISY_SCENARIOS]
    if len(noisy) != 42:
        raise ValueError(
            "Expected 42 unique noisy candidates after deduplicating the "
            f"shared n=8,h=1 baseline; received {len(noisy)}."
        )

    components: dict[str, np.ndarray] = {}
    bounds: dict[str, tuple[float, float]] = {}
    directions = {
        "hog_relative_error": False,
        "cosine_similarity": True,
        "gradient_RMSE": False,
        "runtime": False,
    }
    for attribute, higher_is_better in directions.items():
        values = [getattr(row, attribute) for row in noisy]
        normalized, metric_bounds = _benefit_normalization(
            values,
            higher_is_better=higher_is_better,
        )
        components[attribute] = normalized
        bounds[attribute] = metric_bounds

    scored: list[tuple[AggregateRow, float]] = []
    for index, row in enumerate(noisy):
        score = sum(
            COMPOSITE_WEIGHTS[attribute] * components[attribute][index]
            for attribute in COMPOSITE_WEIGHTS
        )
        scored.append((row, float(score)))
    scored.sort(
        key=lambda item: (
            -item[1],
            item[0].hog_relative_error,
            -item[0].cosine_similarity,
            item[0].gradient_RMSE,
            item[0].runtime,
            item[0].noise_type,
            item[0].algorithm,
            item[0].n,
            item[0].h,
        )
    )

    ranked_rows: list[dict] = []
    for rank, (row, score) in enumerate(scored, start=1):
        ranked_rows.append(
            {
                "rank": rank,
                "noise_type": row.noise_type,
                "algorithm": row.algorithm,
                "n": row.n,
                "h": row.h,
                "polynomial_order": row.polynomial_order,
                "composite_score": f"{score:.10f}",
                "gradient_RMSE": _format_csv_number(row.gradient_RMSE),
                "hog_relative_error": _format_csv_number(
                    row.hog_relative_error
                ),
                "cosine_similarity": _format_csv_number(
                    row.cosine_similarity
                ),
                "runtime": _format_csv_number(row.runtime),
                "iterations": _format_csv_number(row.iterations),
                "residual": _format_csv_number(row.residual),
            }
        )

    scenario_balanced: dict[tuple, list[float]] = defaultdict(list)
    for row, score in scored:
        scenario_balanced[
            (
                row.algorithm,
                row.n,
                row.h,
                row.polynomial_order,
            )
        ].append(score)
    balanced_rows = [
        {
            "algorithm": key[0],
            "n": key[1],
            "h": key[2],
            "polynomial_order": key[3],
            "mean_composite_score": float(fmean(scores)),
            "scenario_count": len(scores),
        }
        for key, scores in scenario_balanced.items()
    ]
    if any(row["scenario_count"] != 2 for row in balanced_rows):
        raise ValueError("Each physical candidate must contain both noisy scenarios.")
    balanced_rows.sort(
        key=lambda row: (
            -row["mean_composite_score"],
            row["algorithm"],
            row["n"],
            row["h"],
        )
    )
    return ranked_rows, bounds, balanced_rows


def _noise_label(noise_type: str) -> str:
    return "S&P 5%" if noise_type == "saltpepper_5" else "S&P 10%"


def _best_by_label(best_by: str) -> str:
    return {
        "minimum_gradient_RMSE": "Minimum gradient RMSE",
        "minimum_hog_relative_error": "Minimum HOG error",
        "maximum_cosine_similarity": "Maximum cosine",
        "minimum_runtime": "Minimum runtime",
    }[best_by]


def _write_best_markdown(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "# Best parameters by algorithm and noise scenario",
        "",
        f"> {MODEL_NOTE}",
        "",
        "The table searches the seven evaluated one-factor-at-a-time pairs "
        "`(4,1), (8,1), (16,1), (32,1), (8,2), (8,3), (8,4)`. It does not "
        "represent a full Cartesian `n x h` grid.",
        "",
        "| Noise | Algorithm | Criterion | n | h | RMSE | HOG error | "
        "Cosine | Runtime (s) | Iterations | Residual |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {_noise_label(row['noise_type'])} | "
            f"{ALGORITHM_LABELS[row['algorithm']]} | "
            f"{_best_by_label(row['best_by'])} | {row['best_n']} | "
            f"{row['best_h']} | {float(row['gradient_RMSE']):.4f} | "
            f"{float(row['hog_relative_error']):.4f} | "
            f"{float(row['cosine_similarity']):.4f} | "
            f"{float(row['runtime']):.3f} | "
            f"{float(row['iterations']):.0f} | "
            f"{float(row['residual']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ranking_markdown(
    path: Path,
    rows: Sequence[dict],
    bounds: dict[str, tuple[float, float]],
) -> None:
    lines = [
        "# Overall noisy-condition parameter ranking",
        "",
        f"> {MODEL_NOTE}",
        "",
        "Only the 42 unique noisy rows are ranked. The shared `n=8,h=1` "
        "baseline is deduplicated and no-noise rows are excluded.",
        "",
        "For a metric `x`, higher-is-better min-max normalization is "
        "`(x-min)/(max-min)`. For RMSE, HOG error, and runtime the benefit is "
        "inverted as `(max-x)/(max-min)`. A zero-range component contributes "
        "1 equally to every candidate. Global bounds are computed across all "
        "42 noisy rows.",
        "",
        "Composite score = `0.45*inverse(HOG error) + 0.30*cosine + "
        "0.20*inverse(RMSE) + 0.05*inverse(runtime)`.",
        "",
        "Because both HOG error and cosine are included, HOG stability has 75% "
        "of the declared weight. The global ranking also compares different "
        "noise severities, so 5% rows tend to outrank otherwise comparable 10% "
        "rows. Scores are candidate-set dependent.",
        "",
        "Normalization bounds:",
        "",
    ]
    for metric in (
        "hog_relative_error",
        "cosine_similarity",
        "gradient_RMSE",
        "runtime",
    ):
        lower, upper = bounds[metric]
        lines.append(f"- `{metric}`: min={lower:.10g}, max={upper:.10g}")
    lines.extend(
        [
            "",
            "| Rank | Noise | Algorithm | n | h | Score | RMSE | HOG error | "
            "Cosine | Runtime (s) | Iterations | Residual |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['rank']} | {_noise_label(row['noise_type'])} | "
            f"{ALGORITHM_LABELS[row['algorithm']]} | {row['n']} | {row['h']} | "
            f"{float(row['composite_score']):.4f} | "
            f"{float(row['gradient_RMSE']):.4f} | "
            f"{float(row['hog_relative_error']):.4f} | "
            f"{float(row['cosine_similarity']):.4f} | "
            f"{float(row['runtime']):.3f} | "
            f"{float(row['iterations']):.0f} | "
            f"{float(row['residual']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative_reduction(old: float, new: float) -> float:
    return 100.0 * (old - new) / old


def _candidate_lookup(
    candidates: Sequence[AggregateRow],
) -> dict[tuple[str, str, int, int], AggregateRow]:
    return {
        (row.noise_type, row.algorithm, row.n, row.h): row
        for row in candidates
    }


def _build_summary_markdown(
    *,
    candidates: Sequence[AggregateRow],
    balanced_rows: Sequence[dict],
    seed_values: Sequence[int],
    source_names: Sequence[str],
    provenance: dict,
) -> str:
    lookup = _candidate_lookup(candidates)
    sp5_n16 = lookup[("saltpepper_5", "correntropy", 16, 1)]
    sp5_n32 = lookup[("saltpepper_5", "correntropy", 32, 1)]
    sp10_n16 = lookup[("saltpepper_10", "correntropy", 16, 1)]
    sp10_n32 = lookup[("saltpepper_10", "correntropy", 32, 1)]
    sp5_h3 = lookup[("saltpepper_5", "correntropy", 8, 3)]
    sp5_h4 = lookup[("saltpepper_5", "correntropy", 8, 4)]
    sp10_h3 = lookup[("saltpepper_10", "correntropy", 8, 3)]
    sp10_h4 = lookup[("saltpepper_10", "correntropy", 8, 4)]

    direct_iterative_pairs = []
    for noise in NOISY_SCENARIOS:
        for n, h in [(4, 1), (8, 1), (16, 1), (32, 1), (8, 2), (8, 3), (8, 4)]:
            direct = lookup[(noise, "direct_squared", n, h)]
            iterative = lookup[(noise, "iterative_squared", n, h)]
            direct_iterative_pairs.append((direct, iterative))
    rmse_improvements = [
        _relative_reduction(direct.gradient_RMSE, iterative.gradient_RMSE)
        for direct, iterative in direct_iterative_pairs
        if direct.gradient_RMSE > 0
    ]
    runtime_ratios = [
        iterative.runtime / direct.runtime
        for direct, iterative in direct_iterative_pairs
        if direct.runtime > 0
    ]
    hog_cosine_differences = [
        abs(iterative.cosine_similarity - direct.cosine_similarity)
        for direct, iterative in direct_iterative_pairs
    ]

    corr_comparisons = []
    for noise in NOISY_SCENARIOS:
        direct = lookup[(noise, "direct_squared", 16, 1)]
        correntropy = lookup[(noise, "correntropy", 16, 1)]
        corr_comparisons.append(
            {
                "noise": _noise_label(noise),
                "rmse_reduction": _relative_reduction(
                    direct.gradient_RMSE,
                    correntropy.gradient_RMSE,
                ),
                "hog_reduction": _relative_reduction(
                    direct.hog_relative_error,
                    correntropy.hog_relative_error,
                ),
                "cosine_gain": (
                    correntropy.cosine_similarity - direct.cosine_similarity
                ),
                "runtime_ratio": correntropy.runtime / direct.runtime,
            }
        )

    balanced_best = balanced_rows[0]
    source_text = ", ".join(source_names)
    seeds_text = ", ".join(str(seed) for seed in sorted(seed_values))
    seed_scope = (
        f"seed={seed_values[0]}"
        if len(seed_values) == 1
        else f"{len(seed_values)} seeds ({seeds_text})"
    )
    lines = [
        "# Parameter sensitivity summary",
        "",
        f"> {MODEL_NOTE}",
        "",
        "## Scope and reporting hygiene",
        "",
        f"The report uses `{source_text}` with matched clean references. "
        f"Available seed(s): `{seeds_text}`. The present conclusions are "
        f"descriptive for the tested image and {seed_scope}. "
        + (
            "No uncertainty interval can be estimated from one seed."
            if len(seed_values) == 1
            else "Curves report configuration means across the supplied seeds."
        ),
        "",
        _provenance_sentence(provenance),
        "",
        "The two sweeps are one-factor-at-a-time: the n sweep fixes h=1, and "
        "the h sweep fixes n=8. Therefore n=32 and h=4 must not be combined "
        "and described as an observed joint optimum. Gradient RMSE measures "
        "matched-clean magnitude stability, not absolute derivative accuracy.",
        "",
        "## Effect of increasing n",
        "",
        "Under salt-and-pepper noise, correntropy improved consistently as n "
        "increased from 16 to 32. At 5% noise, gradient RMSE decreased by "
        f"{_relative_reduction(sp5_n16.gradient_RMSE, sp5_n32.gradient_RMSE):.1f}% "
        f"({sp5_n16.gradient_RMSE:.4f} to {sp5_n32.gradient_RMSE:.4f}) and HOG "
        f"relative error decreased by "
        f"{_relative_reduction(sp5_n16.hog_relative_error, sp5_n32.hog_relative_error):.1f}%, "
        f"while runtime increased by "
        f"{100.0 * (sp5_n32.runtime / sp5_n16.runtime - 1.0):.1f}%. At 10% "
        f"noise the corresponding RMSE reduction was "
        f"{_relative_reduction(sp10_n16.gradient_RMSE, sp10_n32.gradient_RMSE):.1f}% "
        f"and the runtime increase was "
        f"{100.0 * (sp10_n32.runtime / sp10_n16.runtime - 1.0):.1f}%. Thus "
        "n=32 is the robustness-maximizing tested value, whereas n=16 is a "
        "more economical correntropy setting.",
        "",
        "For squared loss, lower gradient RMSE did not always produce a more "
        "stable HOG descriptor. In particular, the descriptor metrics can "
        "flatten or worsen at the largest support. This is why the report keeps "
        "gradient RMSE, HOG error, and cosine as separate outcomes.",
        "",
        "## Effect of increasing h",
        "",
        "For the tested settings with n=8, increasing h improved all three "
        "matched-clean stability metrics. For correntropy, moving from h=3 to "
        f"h=4 reduced RMSE by "
        f"{_relative_reduction(sp5_h3.gradient_RMSE, sp5_h4.gradient_RMSE):.1f}% "
        f"at 5% noise and "
        f"{_relative_reduction(sp10_h3.gradient_RMSE, sp10_h4.gradient_RMSE):.1f}% "
        "at 10% noise. The corresponding HOG gains were smaller: cosine "
        f"increased by {sp5_h4.cosine_similarity - sp5_h3.cosine_similarity:.4f} "
        f"and {sp10_h4.cosine_similarity - sp10_h3.cosine_similarity:.4f}. "
        "Hence h=4 is the best tested value for noise stability, while h=3 is "
        "a cautious alternative when localization and matrix conditioning are "
        "important. These data do not establish an unrestricted optimum beyond "
        "the tested boundary.",
        "",
        "## Direct versus iterative squared loss",
        "",
        "Across the noisy OFAT candidates, iterative squared loss changed "
        f"gradient RMSE by a median improvement of "
        f"{float(np.median(rmse_improvements)):.2f}% and changed HOG cosine by "
        f"at most {max(hog_cosine_differences):.4f}, while requiring a median "
        f"{float(np.median(runtime_ratios)):.1f}x the runtime of direct squared "
        "loss. Direct squared loss is therefore the clearer fast baseline; the "
        "iterative variant offers only a small stability difference for its "
        "additional computation in this experiment.",
        "",
        "## Correntropy robustness-runtime trade-off",
        "",
        f"At n=16,h=1, correntropy reduced RMSE by "
        f"{corr_comparisons[0]['rmse_reduction']:.1f}% and HOG error by "
        f"{corr_comparisons[0]['hog_reduction']:.1f}% at 5% noise, with a "
        f"{corr_comparisons[0]['runtime_ratio']:.1f}x runtime cost relative to "
        "direct squared loss. At 10% noise the reductions were "
        f"{corr_comparisons[1]['rmse_reduction']:.1f}% and "
        f"{corr_comparisons[1]['hog_reduction']:.1f}%, with a "
        f"{corr_comparisons[1]['runtime_ratio']:.1f}x cost. The robustness "
        "improvement is substantial enough to justify correntropy when "
        "impulsive-noise stability is the priority, but not as an unqualified "
        "replacement when runtime dominates.",
        "",
        "## Parameter recommendation",
        "",
        "- Maximum measured robustness: correntropy with n=32,h=1.",
        "- Cost-aware correntropy choice: n=16,h=1.",
        "- Fast squared-loss baseline: direct squared loss.",
        "- Spacing sweep at n=8: h=4 has the best tested stability; h=3 is the "
        "more conservative localization/conditioning compromise.",
        "",
        "Using the requested global min-max composite and then averaging the "
        "two noisy-scenario scores equally, the leading physical candidate is "
        f"{ALGORITHM_LABELS[balanced_best['algorithm']]} with "
        f"n={balanced_best['n']}, h={balanced_best['h']} "
        f"(mean score {balanced_best['mean_composite_score']:.4f}). This score "
        "is descriptive and candidate-set dependent, and it weights two "
        "correlated HOG measures for a combined 75% contribution.",
        "",
        "## Main-paper figures",
        "",
        "- `n_sweep_gradient_rmse.png`, `n_sweep_hog_error.png`, "
        "`n_sweep_cosine.png`, `n_sweep_runtime.png`",
        "- `h_sweep_gradient_rmse.png`, `h_sweep_hog_error.png`, "
        "`h_sweep_cosine.png`, `h_sweep_runtime.png`",
    ]
    return "\n".join(lines) + "\n"


def _build_summary_latex(
    *,
    candidates: Sequence[AggregateRow],
    balanced_rows: Sequence[dict],
    seed_values: Sequence[int],
    provenance: dict,
) -> str:
    lookup = _candidate_lookup(candidates)
    c5_16 = lookup[("saltpepper_5", "correntropy", 16, 1)]
    c5_32 = lookup[("saltpepper_5", "correntropy", 32, 1)]
    c10_16 = lookup[("saltpepper_10", "correntropy", 16, 1)]
    c10_32 = lookup[("saltpepper_10", "correntropy", 32, 1)]
    h5_3 = lookup[("saltpepper_5", "correntropy", 8, 3)]
    h5_4 = lookup[("saltpepper_5", "correntropy", 8, 4)]
    h10_3 = lookup[("saltpepper_10", "correntropy", 8, 3)]
    h10_4 = lookup[("saltpepper_10", "correntropy", 8, 4)]
    balanced_best = balanced_rows[0]
    seed_scope = (
        f"seed {seed_values[0]}"
        if len(seed_values) == 1
        else "seeds " + ", ".join(str(seed) for seed in seed_values)
    )
    lines = [
        "% Generated by parameter_sensitivity_report.py",
        "% " + MODEL_NOTE,
        r"\section{Parameter sensitivity analysis}",
        r"\label{sec:parameter-sensitivity}",
        "",
        r"\paragraph{Scope.}",
        "This analysis evaluates a separable 1-D Taylor model applied "
        "independently along $x$ and $y$. It is not the joint full 2-D Taylor "
        f"model. Results use parameter-matched clean references and {seed_scope}. "
        "They describe matched-clean noise stability for the tested image and "
        "must not be interpreted as absolute gradient accuracy or a "
        "population-level estimate.",
        "",
        _provenance_sentence(provenance),
        "",
        r"\paragraph{Effect of the per-axis sample count.}",
        "With $h=1$, increasing $n$ from 16 to 32 reduced correntropy RMSE by "
        f"{_relative_reduction(c5_16.gradient_RMSE, c5_32.gradient_RMSE):.1f}\\% "
        "under 5\\% salt-and-pepper noise and by "
        f"{_relative_reduction(c10_16.gradient_RMSE, c10_32.gradient_RMSE):.1f}\\% "
        "under 10\\% noise. The corresponding runtime increases were "
        f"{100.0 * (c5_32.runtime / c5_16.runtime - 1.0):.1f}\\% and "
        f"{100.0 * (c10_32.runtime / c10_16.runtime - 1.0):.1f}\\%. "
        "Thus, $n=32$ maximized robustness among the tested values, whereas "
        "$n=16$ provided a more economical correntropy configuration.",
        "",
        r"\paragraph{Effect of sample spacing.}",
        "With $n=8$, increasing $h$ improved all matched-clean stability "
        "metrics. The final step from $h=3$ to $h=4$ reduced correntropy RMSE "
        f"by {_relative_reduction(h5_3.gradient_RMSE, h5_4.gradient_RMSE):.1f}\\% "
        "at 5\\% noise and "
        f"{_relative_reduction(h10_3.gradient_RMSE, h10_4.gradient_RMSE):.1f}\\% "
        "at 10\\% noise, but the cosine gains were only "
        f"{h5_4.cosine_similarity - h5_3.cosine_similarity:.4f} and "
        f"{h10_4.cosine_similarity - h10_3.cosine_similarity:.4f}. "
        "Accordingly, $h=4$ is the best tested stability setting, while $h=3$ "
        "is a more conservative localization and conditioning compromise.",
        "",
        r"\paragraph{Solver trade-off.}",
        "Direct and iterative squared loss produced nearly identical HOG "
        "stability, while the iterative method required several times more "
        "runtime. For the larger-support tested configurations, correntropy "
        "typically produced substantially lower RMSE and HOG error and higher "
        "cosine similarity under impulsive noise, but its runtime was tens of "
        "times larger than direct squared loss. It is therefore justified when "
        "robustness is prioritized, not as an unconditional replacement for "
        "the direct solver.",
        "",
        r"\paragraph{Recommended settings.}",
        "The maximum-robustness tested setting is correntropy with $n=32,h=1$; "
        "a cost-aware alternative is $n=16,h=1$. In the separate spacing sweep "
        "$h=4$ was best among the tested values. The experiment is "
        "one-factor-at-a-time, so it does not support claiming $(n,h)=(32,4)$ "
        "as a joint optimum. The requested scenario-balanced composite also "
        "ranked "
        f"{ALGORITHM_LABELS[balanced_best['algorithm']].lower()} with "
        f"$n={balanced_best['n']},h={balanced_best['h']}$ first "
        f"(score {balanced_best['mean_composite_score']:.4f}), but this score "
        "is candidate-set and weight dependent.",
        "",
    ]
    return "\n".join(lines)


def _write_readme(
    path: Path,
    *,
    source_paths: Sequence[Path],
    output_dir: Path,
    provenance: dict,
) -> None:
    project_root = Path(__file__).resolve().parent
    try:
        relative_output = output_dir.resolve().relative_to(project_root)
    except ValueError:
        relative_output = output_dir.resolve()
    source_args = " ".join(
        f"--input-csv \"{source}\"" for source in source_paths
    )
    lines = [
        "# Parameter sensitivity reporting pipeline",
        "",
        MODEL_NOTE,
        "",
        "Run from the project root:",
        "",
        "```powershell",
        "python -B parameter_sensitivity_report.py",
        "```",
        "",
        "Custom input/output example:",
        "",
        "```powershell",
        f"python -B parameter_sensitivity_report.py {source_args} "
        f"--output-dir \"{relative_output}\"",
        "```",
        "",
        "The script reads the existing experiment CSV. If the default CSV is "
        "missing, it invokes `parameter_analysis_pipeline.py` with "
        "`polynomial_order=3` and seed 42 before reporting. Repeat "
        "`--input-csv` to add future seed-specific result files; every file in "
        "a multi-file run must contain an explicit `seed` column. Plotted and "
        "ranked metrics are then averaged by evaluated configuration after "
        "complete-design validation.",
        "",
        _provenance_sentence(provenance),
        "",
        "Outputs include eight 300-dpi main-paper figures, canonicalized source "
        "data (including no-noise regression rows), best-parameter tables, the "
        "noisy-only composite ranking, and Markdown/LaTeX conclusions.",
        "",
        "The union ranking contains only seven evaluated OFAT pairs, not a "
        "4-by-4 factorial grid. Runtime figures use a logarithmic y-axis. "
        "No-noise rows remain in `canonical_parameter_results.csv` but are "
        "excluded from all noisy-condition rankings.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_input_csv(
    project_root: Path,
    supplied_paths: Sequence[str] | None,
) -> tuple[list[Path], bool]:
    if supplied_paths:
        paths = [Path(path).expanduser().resolve() for path in supplied_paths]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Input CSV file(s) not found: {missing}")
        return paths, False

    default_csv = project_root / "parameter_analysis_results.csv"
    if default_csv.is_file():
        return [default_csv], False

    experiment_script = project_root / "parameter_analysis_pipeline.py"
    if not experiment_script.is_file():
        raise FileNotFoundError(
            "parameter_analysis_results.csv is missing and the existing "
            "experiment pipeline could not be found."
        )
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(experiment_script),
            "--polynomial-order",
            "3",
            "--output-dir",
            str(project_root),
        ],
        cwd=project_root,
        check=True,
    )
    if not default_csv.is_file():
        raise RuntimeError("The fallback experiment did not create its CSV.")
    return [default_csv], True


def _csv_has_seed_column(path: Path) -> bool:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        header = next(reader, [])
    normalized = {_normalise_header(name) for name in header}
    return any(
        _normalise_header(alias) in normalized
        for alias in COLUMN_ALIASES["seed"]
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_experiment_provenance(project_root: Path) -> dict:
    config_path = project_root / "parameter_analysis_config.json"
    if not config_path.is_file():
        return {
            "config_path": None,
            "config_sha256": None,
            "hog_configuration": None,
            "runtime_definition": None,
            "residual_definition": None,
        }
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": _file_sha256(config_path),
        "hog_configuration": configuration.get("hog_configuration"),
        "runtime_definition": configuration.get("runtime_definition"),
        "residual_definition": configuration.get("final_residual"),
    }


def _provenance_sentence(provenance: dict) -> str:
    hog = provenance.get("hog_configuration")
    if not hog:
        return (
            "The experiment configuration file was unavailable; descriptor, "
            "timing, and residual definitions should be supplied before "
            "publication."
        )
    normalization = hog.get("normalization", "unspecified")
    weighting = hog.get("robust_weighting", "unspecified")
    weighting_label = str(weighting).capitalize()
    delta = hog.get("robust_delta", "unspecified")
    window = hog.get("robust_window", "unspecified")
    runtime = provenance.get("runtime_definition") or "runtime definition unavailable"
    residual = (
        provenance.get("residual_definition")
        or "residual definition unavailable"
    )
    return (
        f"All descriptor metrics use fixed Robust HOG with {normalization} "
        f"normalization and {weighting_label} weighting (delta={delta}, "
        f"window={window}). Runtime definition: {runtime}. Residual definition: "
        f"{residual}. Residual and iteration count are diagnostics and are not "
        "components of the composite score."
    )


def generate_report(args: argparse.Namespace) -> list[Path]:
    project_root = Path(__file__).resolve().parent
    provenance = _load_experiment_provenance(project_root)
    source_paths, fallback_ran = _ensure_input_csv(
        project_root,
        args.input_csv,
    )
    if len(source_paths) > 1:
        missing_seed_columns = [
            path for path in source_paths if not _csv_has_seed_column(path)
        ]
        if missing_seed_columns:
            raise ValueError(
                "Every CSV in a multi-file analysis must contain an explicit "
                f"seed column. Missing in: {missing_seed_columns}"
            )
    effective_default_seed = DEFAULT_SEED if fallback_ran else args.default_seed
    source_hashes_before = {str(path): _file_sha256(path) for path in source_paths}
    rows: list[CanonicalRow] = []
    for path in source_paths:
        rows.extend(_read_results_csv(path, effective_default_seed))
    _validate_rows(rows)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_columns = [
        "experiment_type",
        "noise_type",
        "noise_level",
        "algorithm",
        "n",
        "h",
        "polynomial_order",
        "gradient_RMSE",
        "hog_relative_error",
        "cosine_similarity",
        "runtime",
        "iterations",
        "residual",
        "converged",
        "seed",
        "source_csv",
    ]
    canonical_path = output_dir / "canonical_parameter_results.csv"
    _write_csv(
        canonical_path,
        canonical_columns,
        (_canonical_row_dict(row) for row in rows),
    )

    sweep_rows = _aggregate_sweeps(rows)
    physical_candidates = _aggregate_physical_candidates(rows)
    seed_values = sorted({row.seed for row in rows})
    seed_label = (
        f"seed {seed_values[0]}"
        if len(seed_values) == 1
        else "seeds " + ", ".join(str(seed) for seed in seed_values)
    )
    figure_paths = []
    figure_paths.extend(
        _plot_sweep(
            sweep_rows,
            experiment_type="vary_n",
            x_attribute="n",
            x_values=N_VALUES,
            fixed_parameter="$h=1$ pixel",
            seed_label=seed_label,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )
    figure_paths.extend(
        _plot_sweep(
            sweep_rows,
            experiment_type="vary_h",
            x_attribute="h",
            x_values=H_VALUES,
            fixed_parameter="$n=8$ samples per axis",
            seed_label=seed_label,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    best_rows = _best_parameter_rows(
        [
            row
            for row in physical_candidates
            if row.noise_type in NOISY_SCENARIOS
        ]
    )
    best_columns = [
        "noise_type",
        "algorithm",
        "best_by",
        "best_n",
        "best_h",
        "polynomial_order",
        "gradient_RMSE",
        "hog_relative_error",
        "cosine_similarity",
        "runtime",
        "iterations",
        "residual",
    ]
    best_csv = output_dir / "best_params_per_algorithm_and_noise.csv"
    _write_csv(best_csv, best_columns, best_rows)
    best_markdown = output_dir / "best_params_per_algorithm_and_noise.md"
    _write_best_markdown(best_markdown, best_rows)

    ranked_rows, bounds, balanced_rows = _rank_candidates(physical_candidates)
    ranking_columns = [
        "rank",
        "noise_type",
        "algorithm",
        "n",
        "h",
        "polynomial_order",
        "composite_score",
        "gradient_RMSE",
        "hog_relative_error",
        "cosine_similarity",
        "runtime",
        "iterations",
        "residual",
    ]
    ranking_csv = output_dir / "overall_ranked_params.csv"
    _write_csv(ranking_csv, ranking_columns, ranked_rows)
    ranking_markdown = output_dir / "overall_ranked_params.md"
    _write_ranking_markdown(ranking_markdown, ranked_rows, bounds)

    source_names = [path.name for path in source_paths]
    summary_markdown = output_dir / "parameter_sensitivity_summary.md"
    summary_markdown.write_text(
        _build_summary_markdown(
            candidates=physical_candidates,
            balanced_rows=balanced_rows,
            seed_values=seed_values,
            source_names=source_names,
            provenance=provenance,
        ),
        encoding="utf-8",
    )
    summary_latex = output_dir / "parameter_sensitivity_summary.tex"
    summary_latex.write_text(
        _build_summary_latex(
            candidates=physical_candidates,
            balanced_rows=balanced_rows,
            seed_values=seed_values,
            provenance=provenance,
        ),
        encoding="utf-8",
    )
    readme = output_dir / "README.md"
    _write_readme(
        readme,
        source_paths=source_paths,
        output_dir=output_dir,
        provenance=provenance,
    )

    source_hashes_after = {str(path): _file_sha256(path) for path in source_paths}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A source experiment CSV changed during reporting.")

    metadata_path = output_dir / "report_metadata.json"
    metadata = {
        "model_note": MODEL_NOTE,
        "report_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "source_csvs": [
            {
                "path": str(path),
                "sha256": source_hashes_after[str(path)],
            }
            for path in source_paths
        ],
        "experiment_provenance": provenance,
        "seed_values": seed_values,
        "raw_row_count": len(rows),
        "canonical_noisy_candidate_count": len(
            [
                row
                for row in physical_candidates
                if row.noise_type in NOISY_SCENARIOS
            ]
        ),
        "evaluated_parameter_pairs": [
            [4, 1],
            [8, 1],
            [16, 1],
            [32, 1],
            [8, 2],
            [8, 3],
            [8, 4],
        ],
        "ranking_filter": "noise_type in {saltpepper_5, saltpepper_10}",
        "ranking_deduplication_key": [
            "noise_type",
            "algorithm",
            "n",
            "h",
            "polynomial_order",
        ],
        "normalization_scope": "all 42 unique noisy candidate rows",
        "normalization": {
            "higher_is_better": "(x-min)/(max-min)",
            "lower_is_better": "(max-x)/(max-min)",
            "constant_metric": "benefit=1 for every candidate",
            "bounds": {
                metric: {"min": values[0], "max": values[1]}
                for metric, values in bounds.items()
            },
        },
        "composite_weights": COMPOSITE_WEIGHTS,
        "scenario_balanced_ranking": balanced_rows,
        "dpi": args.dpi,
        "nonconverged_rows": sum(not row.converged for row in rows),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    output_paths = [
        *figure_paths,
        canonical_path,
        best_csv,
        best_markdown,
        ranking_csv,
        ranking_markdown,
        summary_markdown,
        summary_latex,
        readme,
        metadata_path,
    ]
    for path in output_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected report artifact is missing: {path}")
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create compact publication-quality reporting artifacts from the "
            "separable Taylor parameter-sensitivity CSV."
        )
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        help=(
            "Experiment CSV path. Repeat for future seed-specific CSV files. "
            "Defaults to parameter_analysis_results.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/parameter_report",
    )
    parser.add_argument("--default-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args()
    if arguments.dpi < 150:
        parser.error("--dpi must be at least 150 for publication output.")
    return arguments


if __name__ == "__main__":
    generated = generate_report(parse_args())
    print(f"Generated {len(generated)} report artifacts:")
    for artifact in generated:
        print(f"  {artifact}")
