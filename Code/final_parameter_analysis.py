"""Build the final, paper-ready parameter-analysis package.

This script is deliberately a reporting/benchmarking layer.  It imports and
uses the existing estimator and HOG implementations without changing either
one.

Model scope
-----------
This analysis evaluates a separable 1-D Taylor model applied independently
along x and y. It is not the joint full 2-D Taylor model.

The matched-clean metrics reported here measure stability to injected noise.
They do not measure absolute derivative accuracy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from parameter_analysis_pipeline import (
    ALGORITHM_NAMES,
    AxisTaylorSolver,
    HOG_NORMALIZATION,
    RANDOM_SEED,
    ROBUST_DELTA,
    ROBUST_HOG,
    ROBUST_WEIGHTING,
    ROBUST_WINDOW,
    SEPARABLE_MODEL_NOTE,
    VISUALIZATION_PERCENTILE,
    _apply_common_roi,
    _compute_hog,
    _estimate_separable_gradient,
    _load_image,
    _make_noise_scenarios,
    _run_configuration,
)


POLYNOMIAL_ORDER = 3
N_VALUES = (4, 8, 16, 32)
H_VALUES = (1, 2, 3, 4)
JOINT_CONFIGURATIONS = (
    (8, 3),
    (8, 4),
    (16, 2),
    (16, 3),
    (16, 4),
    (32, 1),
)
JOINT_NOISES = (
    ("saltpepper_5", 5.0),
    ("saltpepper_10", 10.0),
)
RUNTIME_IMAGE_SIZES = (64, 128, 192, 256)
RUNTIME_ALGORITHMS = (1, 2, 3)
RUNTIME_H = 1
RUNTIME_NOISE = "saltpepper_10"
QUALITATIVE_COMMON_PAD = 16
JOINT_COMMON_PAD = max(n * h // 2 for n, h in JOINT_CONFIGURATIONS)
RUNTIME_COMMON_PAD = max(N_VALUES) // 2

FINAL_NOTE = (
    "separable 1-D Taylor along x/y; matched-clean metrics measure noise "
    "stability, not absolute derivative accuracy."
)

JOINT_COLUMNS = (
    "experiment",
    "noise_scenario",
    "noise_density",
    "seed",
    "image_height",
    "image_width",
    "polynomial_order",
    "n",
    "h",
    "algorithm",
    "algorithm_name",
    "gradient_RMSE",
    "hog_relative_error",
    "cosine_similarity",
    "runtime",
    "iteration_count",
    "final_residual",
    "converged",
    "common_pad",
    "runtime_repeats",
)

RUNTIME_TRIAL_COLUMNS = (
    "image_size",
    "n",
    "h",
    "polynomial_order",
    "algorithm",
    "algorithm_name",
    "noise_scenario",
    "seed",
    "repeat",
    "runtime",
    "iteration_count",
    "final_residual",
    "converged",
    "common_pad",
)


@dataclass(frozen=True)
class JointAggregate:
    n: int
    h: int
    polynomial_order: int
    gradient_rmse: float
    hog_error: float
    cosine_similarity: float
    runtime: float
    robustness_score: float = 0.0
    runtime_benefit: float = 0.0
    balanced_score: float = 0.0


@dataclass(frozen=True)
class JointSelections:
    fastest: JointAggregate
    balanced: JointAggregate
    maximum_robustness: JointAggregate

    def role_items(self) -> tuple[tuple[str, JointAggregate], ...]:
        return (
            ("Fastest", self.fastest),
            ("Balanced", self.balanced),
            ("Maximum weighted robustness", self.maximum_robustness),
        )


def _float(value: str | float | int, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}: {value!r}") from error
    if not np.isfinite(result):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return result


def _int(value: str | float | int, field: str) -> int:
    numeric = _float(value, field)
    rounded = round(numeric)
    if not math.isclose(numeric, rounded, abs_tol=1e-9):
        raise ValueError(f"{field} must be integral; received {value!r}.")
    return int(rounded)


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _format_number(value: float) -> str:
    return format(float(value), ".12g")


def _atomic_write_csv(
    path: Path,
    rows: Iterable[dict],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_optional_float(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-12,
        abs_tol=1e-15,
    )


def validate_cache_provenance(args: argparse.Namespace) -> None:
    """Reject cached measurements whose recorded experiment differs.

    Figures and TeX can be regenerated freely, but joint/runtime measurements
    are reused only when their image, solver settings, repeat counts, design,
    and numerical environment match the package metadata.
    """

    output_dir = args.output_dir
    joint_exists = (output_dir / "joint_correntropy_sweep.csv").exists()
    runtime_exists = (output_dir / "runtime_scaling_trials.csv").exists()
    needs_joint_validation = joint_exists and not args.force_joint
    needs_runtime_validation = runtime_exists and not args.force_runtime
    if not needs_joint_validation and not needs_runtime_validation:
        return

    metadata_path = output_dir / "package_metadata.json"
    if not metadata_path.exists():
        raise ValueError(
            "Cached measurements exist without package_metadata.json. "
            "Use the appropriate --force-joint/--force-runtime flags."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_hash = _sha256_file(args.image)
    common_checks = [
        (
            metadata.get("source_image_sha256") == image_hash,
            "source image hash",
        ),
        (
            int(metadata.get("polynomial_order", -1)) == POLYNOMIAL_ORDER,
            "polynomial order",
        ),
        (
            _same_optional_float(
                metadata.get("solver", {}).get("tol"),
                args.tol,
            ),
            "solver tolerance",
        ),
        (
            int(metadata.get("solver", {}).get("max_iter", -1))
            == args.max_iter,
            "maximum iterations",
        ),
        (
            _same_optional_float(
                metadata.get("solver", {}).get("init_sigma"),
                args.init_sigma,
            ),
            "initial sigma",
        ),
        (
            metadata.get("environment", {}).get("numpy") == np.__version__,
            "NumPy version",
        ),
        (
            metadata.get("environment", {}).get("opencv") == cv2.__version__,
            "OpenCV version",
        ),
    ]
    common_failures = [label for passed, label in common_checks if not passed]

    failures: list[str] = []
    if needs_joint_validation:
        joint_metadata = metadata.get("joint_sweep", {})
        joint_checks = [
            (
                joint_metadata.get("configurations")
                == [list(item) for item in JOINT_CONFIGURATIONS],
                "joint configuration set",
            ),
            (
                joint_metadata.get("noise_scenarios")
                == [name for name, _ in JOINT_NOISES],
                "joint noise scenarios",
            ),
            (
                int(joint_metadata.get("runtime_repeats", -1))
                == args.joint_repeats,
                "joint repeat count",
            ),
            (
                int(joint_metadata.get("common_pad", -1)) == JOINT_COMMON_PAD,
                "joint common ROI",
            ),
        ]
        failures.extend(f"joint: {label}" for label in common_failures)
        failures.extend(
            f"joint: {label}"
            for passed, label in joint_checks
            if not passed
        )
    if needs_runtime_validation:
        runtime_metadata = metadata.get("runtime_scaling", {})
        runtime_checks = [
            (
                runtime_metadata.get("image_sizes")
                == list(RUNTIME_IMAGE_SIZES),
                "runtime image sizes",
            ),
            (
                runtime_metadata.get("n_values") == list(N_VALUES),
                "runtime n values",
            ),
            (
                int(runtime_metadata.get("h", -1)) == RUNTIME_H,
                "runtime h",
            ),
            (
                runtime_metadata.get("noise_scenario") == RUNTIME_NOISE,
                "runtime noise scenario",
            ),
            (
                runtime_metadata.get("algorithms")
                == list(RUNTIME_ALGORITHMS),
                "runtime algorithms",
            ),
            (
                int(runtime_metadata.get("runtime_repeats", -1))
                == args.runtime_repeats,
                "runtime repeat count",
            ),
            (
                int(runtime_metadata.get("common_pad", -1))
                == RUNTIME_COMMON_PAD,
                "runtime common ROI",
            ),
        ]
        failures.extend(f"runtime: {label}" for label in common_failures)
        failures.extend(
            f"runtime: {label}"
            for passed, label in runtime_checks
            if not passed
        )
    if failures:
        unique = ", ".join(dict.fromkeys(failures))
        raise ValueError(
            "Cached measurement provenance does not match this run "
            f"({unique}). Rerun the affected measurements with "
            "--force-joint and/or --force-runtime."
        )


def _load_sensitivity_rows(path: Path) -> list[dict]:
    required = {
        "experiment",
        "noise_scenario",
        "polynomial_order",
        "n",
        "h",
        "algorithm",
        "gradient_RMSE",
        "hog_error",
        "cosine_similarity",
        "runtime",
    }
    raw_rows = _read_csv(path)
    missing = required.difference(raw_rows[0])
    if missing:
        raise ValueError(
            f"Sensitivity CSV is missing columns: {sorted(missing)}"
        )

    rows: list[dict] = []
    for index, raw in enumerate(raw_rows, start=2):
        row = {
            "experiment": raw["experiment"].strip(),
            "noise_scenario": raw["noise_scenario"].strip(),
            "noise_density": _float(raw.get("noise_density", 0.0), "density"),
            "polynomial_order": _int(
                raw["polynomial_order"], "polynomial_order"
            ),
            "n": _int(raw["n"], "n"),
            "h": _int(raw["h"], "h"),
            "algorithm": _int(raw["algorithm"], "algorithm"),
            "algorithm_name": raw.get("algorithm_name", "").strip(),
            "gradient_RMSE": _float(
                raw["gradient_RMSE"], f"row {index} gradient_RMSE"
            ),
            "hog_relative_error": _float(
                raw["hog_error"], f"row {index} hog_error"
            ),
            "cosine_similarity": _float(
                raw["cosine_similarity"],
                f"row {index} cosine_similarity",
            ),
            "runtime": _float(raw["runtime"], f"row {index} runtime"),
            "iteration_count": _int(
                raw.get("iteration_count", 0), "iteration_count"
            ),
            "final_residual": _float(
                raw.get("final_residual", 0.0), "final_residual"
            ),
            "converged": _bool(raw.get("converged", "true")),
        }
        if row["polynomial_order"] != POLYNOMIAL_ORDER:
            raise ValueError(
                "The final report expects polynomial_order=3; "
                f"row {index} contains {row['polynomial_order']}."
            )
        rows.append(row)

    expected = {
        (experiment, noise, algorithm, n, h)
        for experiment, configurations in (
            ("n_analysis", tuple((n, 1) for n in N_VALUES)),
            ("h_analysis", tuple((8, h) for h in H_VALUES)),
        )
        for noise in ("no_noise", "saltpepper_5", "saltpepper_10")
        for algorithm in (1, 2, 3)
        for n, h in configurations
    }
    observed = {
        (
            row["experiment"],
            row["noise_scenario"],
            row["algorithm"],
            row["n"],
            row["h"],
        )
        for row in rows
    }
    missing_rows = expected.difference(observed)
    if missing_rows:
        raise ValueError(
            "Sensitivity CSV does not contain the complete requested design; "
            f"missing {len(missing_rows)} rows."
        )
    return rows


def _detect_hog_panels(source_path: Path) -> list[list[np.ndarray]]:
    """Extract the 9 by 4 HOG canvases from an existing dense sweep figure.

    The source figures were rendered with a single pooled 95th-percentile HOG
    scale.  Cropping their canvases preserves that common mapping exactly.
    """

    image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read source figure: {source_path}")
    _, dark = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        dark,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boxes: list[tuple[int, int, int, int]] = []
    min_side = min(image.shape) * 0.07
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if (
            width >= min_side
            and height >= min_side
            and 0.95 <= width / height <= 1.05
        ):
            boxes.append((x, y, width, height))
    boxes.sort(key=lambda item: (item[1], item[0]))
    if len(boxes) != 36:
        raise ValueError(
            f"Expected 36 HOG canvases in {source_path}, found {len(boxes)}."
        )

    rows: list[list[np.ndarray]] = []
    for row_index in range(9):
        row_boxes = sorted(
            boxes[row_index * 4 : (row_index + 1) * 4],
            key=lambda item: item[0],
        )
        row_images = [
            image[y : y + height, x : x + width]
            for x, y, width, height in row_boxes
        ]
        rows.append(row_images)
    return rows


def _save_qualitative_figure(
    source_path: Path,
    output_path: Path,
    *,
    column_values: Sequence[int],
    parameter_symbol: str,
    fixed_text: str,
) -> None:
    panel_rows = _detect_hog_panels(source_path)
    # Dense source ordering: three scenario blocks, each with Direct,
    # Iterative, and Correntropy rows. SP=10% therefore uses rows 6 and 8.
    selected = (panel_rows[6], panel_rows[8])
    row_names = ("Direct", "Correntropy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "figure.titlesize": 9.5,
        }
    ):
        figure, axes = plt.subplots(
            2,
            4,
            figsize=(7.16, 3.55),
            constrained_layout=False,
        )
        for row_index, (row_name, images) in enumerate(
            zip(row_names, selected)
        ):
            for column_index, (value, image) in enumerate(
                zip(column_values, images)
            ):
                axis = axes[row_index, column_index]
                axis.imshow(
                    image,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                    interpolation="nearest",
                )
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_linewidth(0.45)
                    spine.set_color("#555555")
                if row_index == 0:
                    axis.set_title(
                        rf"${parameter_symbol}={value}$",
                        pad=3.0,
                    )
                if column_index == 0:
                    axis.set_ylabel(
                        row_name,
                        fontsize=8.5,
                        labelpad=5.0,
                        rotation=90,
                    )

        figure.suptitle(
            rf"Robust-HOG visualization under SP=10% ({fixed_text})",
            y=0.995,
            fontweight="semibold",
        )
        figure.text(
            0.5,
            0.012,
            "Common display mapping inherited from the pooled 95th-percentile "
            "scale; order=3, seed=42.",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
        figure.subplots_adjust(
            left=0.075,
            right=0.995,
            bottom=0.075,
            top=0.91,
            wspace=0.06,
            hspace=0.08,
        )
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def generate_qualitative_figures(
    n_source: Path,
    h_source: Path,
    output_dir: Path,
) -> None:
    _save_qualitative_figure(
        n_source,
        output_dir / "n_visual_comparison.png",
        column_values=N_VALUES,
        parameter_symbol="n",
        fixed_text=r"$h=1$",
    )
    _save_qualitative_figure(
        h_source,
        output_dir / "h_visual_comparison.png",
        column_values=H_VALUES,
        parameter_symbol="h",
        fixed_text=r"$n=8$",
    )


def _joint_row_from_result(
    result,
    image_shape: tuple[int, int],
    runtime_repeats: int,
) -> dict:
    return {
        "experiment": "joint_correntropy_sweep",
        "noise_scenario": result.noise_scenario,
        "noise_density": _format_number(result.noise_density),
        "seed": RANDOM_SEED,
        "image_height": image_shape[0],
        "image_width": image_shape[1],
        "polynomial_order": result.polynomial_order,
        "n": result.n,
        "h": result.h,
        "algorithm": result.algorithm,
        "algorithm_name": ALGORITHM_NAMES[result.algorithm],
        "gradient_RMSE": _format_number(result.gradient_rmse),
        "hog_relative_error": _format_number(result.hog_error),
        "cosine_similarity": _format_number(result.cosine_similarity),
        "runtime": _format_number(result.runtime),
        "iteration_count": result.iteration_count,
        "final_residual": _format_number(result.final_residual),
        "converged": str(bool(result.converged)).lower(),
        "common_pad": JOINT_COMMON_PAD,
        "runtime_repeats": runtime_repeats,
    }


def _joint_key(row: dict) -> tuple[int, int, str]:
    return (
        _int(row["n"], "n"),
        _int(row["h"], "h"),
        str(row["noise_scenario"]),
    )


def _validate_joint_rows(
    raw_rows: Sequence[dict],
    runtime_repeats: int,
) -> None:
    expected = {
        (n, h, scenario)
        for n, h in JOINT_CONFIGURATIONS
        for scenario, _ in JOINT_NOISES
    }
    observed = {_joint_key(row) for row in raw_rows}
    if observed != expected or len(raw_rows) != len(expected):
        raise ValueError(
            "Joint sweep CSV is incomplete or duplicated: "
            f"expected {len(expected)} unique rows, found "
            f"{len(observed)} unique rows and {len(raw_rows)} total rows."
        )
    for row in raw_rows:
        if _int(row["polynomial_order"], "polynomial_order") != 3:
            raise ValueError("Joint sweep polynomial_order must be 3.")
        if _int(row["algorithm"], "algorithm") != 3:
            raise ValueError("Joint sweep must contain correntropy only.")
        if _int(row["seed"], "seed") != RANDOM_SEED:
            raise ValueError("Joint sweep seed mismatch.")
        if _int(row["common_pad"], "common_pad") != JOINT_COMMON_PAD:
            raise ValueError("Joint sweep common ROI mismatch.")
        if _int(row["runtime_repeats"], "runtime_repeats") != runtime_repeats:
            raise ValueError("Joint sweep runtime-repeat mismatch.")


def run_joint_correntropy_sweep(
    image_path: Path,
    output_path: Path,
    *,
    runtime_repeats: int,
    tol: float,
    max_iter: int,
    init_sigma: float | None,
    force: bool,
) -> list[dict]:
    rows: list[dict] = []
    completed: set[tuple[int, int, str]] = set()
    if output_path.exists() and not force:
        existing = _read_csv(output_path)
        expected = {
            (n, h, scenario)
            for n, h in JOINT_CONFIGURATIONS
            for scenario, _ in JOINT_NOISES
        }
        for row in existing:
            key = _joint_key(row)
            if key not in expected or key in completed:
                raise ValueError(
                    "Joint sweep CSV contains an unexpected or duplicate row."
                )
            if _int(row["polynomial_order"], "polynomial_order") != 3:
                raise ValueError("Joint sweep polynomial_order mismatch.")
            if _int(row["algorithm"], "algorithm") != 3:
                raise ValueError("Joint sweep algorithm mismatch.")
            if _int(row["seed"], "seed") != RANDOM_SEED:
                raise ValueError("Joint sweep seed mismatch.")
            if _int(row["common_pad"], "common_pad") != JOINT_COMMON_PAD:
                raise ValueError("Joint sweep common ROI mismatch.")
            if (
                _int(row["runtime_repeats"], "runtime_repeats")
                != runtime_repeats
            ):
                raise ValueError("Joint sweep runtime-repeat mismatch.")
            completed.add(key)
            rows.append(row)
        if completed == expected:
            _validate_joint_rows(rows, runtime_repeats)
            return rows

    clean = _load_image(image_path, max_dimension=256)
    scenarios = _make_noise_scenarios(clean)

    for n, h in JOINT_CONFIGURATIONS:
        if n <= POLYNOMIAL_ORDER:
            raise ValueError(
                "The robust separable experiment requires "
                "n_points > polynomial_order."
            )
        assert n > POLYNOMIAL_ORDER
        clean_reference = None
        for scenario_name, density in JOINT_NOISES:
            if (n, h, scenario_name) in completed:
                continue
            result, clean_reference = _run_configuration(
                clean,
                scenarios[scenario_name],
                experiment="joint_correntropy_sweep",
                noise_scenario=scenario_name,
                noise_density=density,
                n=n,
                h=h,
                polynomial_order=POLYNOMIAL_ORDER,
                algorithm=3,
                tol=tol,
                max_iter=max_iter,
                init_sigma=init_sigma,
                common_pad=JOINT_COMMON_PAD,
                runtime_repeats=runtime_repeats,
                clean_reference=clean_reference,
            )
            rows.append(
                _joint_row_from_result(
                    result,
                    clean.shape,
                    runtime_repeats,
                )
            )
            completed.add((n, h, scenario_name))
            rows.sort(
                key=lambda row: (
                    _int(row["n"], "n"),
                    _int(row["h"], "h"),
                    str(row["noise_scenario"]),
                )
            )
            _atomic_write_csv(output_path, rows, JOINT_COLUMNS)

    _validate_joint_rows(rows, runtime_repeats)
    return rows


def _minmax_benefit(
    values: Sequence[float],
    *,
    higher_is_better: bool,
    logarithmic: bool = False,
) -> list[float]:
    transformed = np.asarray(values, dtype=float)
    if logarithmic:
        if np.any(transformed <= 0):
            raise ValueError("Logarithmic normalization requires positive data.")
        transformed = np.log(transformed)
    lower = float(np.min(transformed))
    upper = float(np.max(transformed))
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-15):
        return [1.0] * len(values)
    normalized = (transformed - lower) / (upper - lower)
    if not higher_is_better:
        normalized = 1.0 - normalized
    return [float(value) for value in normalized]


def select_joint_configurations(
    raw_rows: Sequence[dict],
) -> tuple[JointSelections, list[JointAggregate]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for raw in raw_rows:
        grouped[
            (_int(raw["n"], "n"), _int(raw["h"], "h"))
        ].append(raw)
    if set(grouped) != set(JOINT_CONFIGURATIONS):
        raise ValueError("Unexpected joint-sweep configuration set.")
    if any(len(rows) != 2 for rows in grouped.values()):
        raise ValueError("Each joint configuration must have SP5 and SP10 rows.")

    base: list[JointAggregate] = []
    for n, h in JOINT_CONFIGURATIONS:
        rows = grouped[(n, h)]
        base.append(
            JointAggregate(
                n=n,
                h=h,
                polynomial_order=POLYNOMIAL_ORDER,
                gradient_rmse=statistics.fmean(
                    _float(row["gradient_RMSE"], "gradient_RMSE")
                    for row in rows
                ),
                hog_error=statistics.fmean(
                    _float(row["hog_relative_error"], "hog_relative_error")
                    for row in rows
                ),
                cosine_similarity=statistics.fmean(
                    _float(row["cosine_similarity"], "cosine_similarity")
                    for row in rows
                ),
                runtime=statistics.fmean(
                    _float(row["runtime"], "runtime") for row in rows
                ),
            )
        )

    rmse_benefit = _minmax_benefit(
        [row.gradient_rmse for row in base],
        higher_is_better=False,
    )
    hog_benefit = _minmax_benefit(
        [row.hog_error for row in base],
        higher_is_better=False,
    )
    cosine_benefit = _minmax_benefit(
        [row.cosine_similarity for row in base],
        higher_is_better=True,
    )
    runtime_benefit = _minmax_benefit(
        [row.runtime for row in base],
        higher_is_better=False,
        logarithmic=True,
    )

    scored: list[JointAggregate] = []
    quality_weight_sum = 0.20 + 0.45 + 0.30
    for index, row in enumerate(base):
        robustness = (
            0.20 * rmse_benefit[index]
            + 0.45 * hog_benefit[index]
            + 0.30 * cosine_benefit[index]
        ) / quality_weight_sum
        speed = runtime_benefit[index]
        balanced = (
            0.0
            if robustness <= 0.0 or speed <= 0.0
            else 2.0 * robustness * speed / (robustness + speed)
        )
        scored.append(
            JointAggregate(
                n=row.n,
                h=row.h,
                polynomial_order=row.polynomial_order,
                gradient_rmse=row.gradient_rmse,
                hog_error=row.hog_error,
                cosine_similarity=row.cosine_similarity,
                runtime=row.runtime,
                robustness_score=robustness,
                runtime_benefit=speed,
                balanced_score=balanced,
            )
        )

    fastest = min(scored, key=lambda row: (row.runtime, row.n, row.h))
    maximum_robustness = max(
        scored,
        key=lambda row: (
            row.robustness_score,
            -row.runtime,
            -row.n,
            -row.h,
        ),
    )
    balanced = max(
        scored,
        key=lambda row: (
            row.balanced_score,
            row.robustness_score,
            -row.runtime,
        ),
    )
    return (
        JointSelections(
            fastest=fastest,
            balanced=balanced,
            maximum_robustness=maximum_robustness,
        ),
        scored,
    )


def write_joint_selection_csv(
    output_path: Path,
    selections: JointSelections,
    scored: Sequence[JointAggregate],
) -> None:
    role_by_configuration: dict[tuple[int, int], list[str]] = defaultdict(list)
    for role, row in selections.role_items():
        role_by_configuration[(row.n, row.h)].append(role)

    columns = (
        "selection_role",
        "n",
        "h",
        "polynomial_order",
        "mean_gradient_RMSE",
        "mean_hog_relative_error",
        "mean_cosine_similarity",
        "mean_runtime",
        "robustness_score",
        "log_runtime_benefit",
        "balanced_score",
    )
    rows = []
    for row in sorted(scored, key=lambda item: (item.n, item.h)):
        roles = "; ".join(role_by_configuration.get((row.n, row.h), []))
        rows.append(
            {
                "selection_role": roles,
                "n": row.n,
                "h": row.h,
                "polynomial_order": row.polynomial_order,
                "mean_gradient_RMSE": _format_number(row.gradient_rmse),
                "mean_hog_relative_error": _format_number(row.hog_error),
                "mean_cosine_similarity": _format_number(
                    row.cosine_similarity
                ),
                "mean_runtime": _format_number(row.runtime),
                "robustness_score": _format_number(row.robustness_score),
                "log_runtime_benefit": _format_number(row.runtime_benefit),
                "balanced_score": _format_number(row.balanced_score),
            }
        )
    _atomic_write_csv(output_path, rows, columns)


def _runtime_config_key(row: dict) -> tuple[int, int, int]:
    return (
        _int(row["image_size"], "image_size"),
        _int(row["algorithm"], "algorithm"),
        _int(row["n"], "n"),
    )


def _write_runtime_trials(path: Path, rows: Sequence[dict]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            _int(row["algorithm"], "algorithm"),
            _int(row["n"], "n"),
            _int(row["image_size"], "image_size"),
            _int(row["repeat"], "repeat"),
        ),
    )
    _atomic_write_csv(path, ordered, RUNTIME_TRIAL_COLUMNS)


def _benchmark_one_configuration(
    noisy: np.ndarray,
    *,
    image_size: int,
    algorithm: int,
    n: int,
    runtime_repeats: int,
    tol: float,
    max_iter: int,
    init_sigma: float | None,
) -> list[dict]:
    solver = AxisTaylorSolver(n, RUNTIME_H, POLYNOMIAL_ORDER)
    rows = []
    for repeat in range(1, runtime_repeats + 1):
        gc.collect()
        start = time.perf_counter()
        estimate = _estimate_separable_gradient(
            noisy,
            n_points=n,
            h=RUNTIME_H,
            polynomial_order=POLYNOMIAL_ORDER,
            algorithm=algorithm,
            tol=tol,
            max_iter=max_iter,
            init_sigma=init_sigma,
            solver=solver,
        )
        estimate = _apply_common_roi(estimate, RUNTIME_COMMON_PAD)
        _compute_hog(estimate)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "image_size": image_size,
                "n": n,
                "h": RUNTIME_H,
                "polynomial_order": POLYNOMIAL_ORDER,
                "algorithm": algorithm,
                "algorithm_name": ALGORITHM_NAMES[algorithm],
                "noise_scenario": RUNTIME_NOISE,
                "seed": RANDOM_SEED,
                "repeat": repeat,
                "runtime": _format_number(elapsed),
                "iteration_count": estimate.iteration_count,
                "final_residual": _format_number(estimate.final_residual),
                "converged": str(bool(estimate.converged)).lower(),
                "common_pad": RUNTIME_COMMON_PAD,
            }
        )
    return rows


def _runtime_warmup(
    noisy: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    init_sigma: float | None,
) -> None:
    for algorithm in RUNTIME_ALGORITHMS:
        solver = AxisTaylorSolver(4, RUNTIME_H, POLYNOMIAL_ORDER)
        estimate = _estimate_separable_gradient(
            noisy,
            n_points=4,
            h=RUNTIME_H,
            polynomial_order=POLYNOMIAL_ORDER,
            algorithm=algorithm,
            tol=tol,
            max_iter=max_iter,
            init_sigma=init_sigma,
            solver=solver,
        )
        estimate = _apply_common_roi(estimate, RUNTIME_COMMON_PAD)
        _compute_hog(estimate)


def run_runtime_benchmark(
    image_path: Path,
    trials_path: Path,
    summary_path: Path,
    figure_path: Path,
    *,
    runtime_repeats: int,
    tol: float,
    max_iter: int,
    init_sigma: float | None,
    force: bool,
) -> list[dict]:
    existing_rows: list[dict] = []
    if trials_path.exists() and not force:
        existing_rows = _read_csv(trials_path)
        for row in existing_rows:
            if _int(row["polynomial_order"], "polynomial_order") != 3:
                raise ValueError("Runtime-trial polynomial_order mismatch.")
            if _int(row["h"], "h") != RUNTIME_H:
                raise ValueError("Runtime-trial h mismatch.")
            if _int(row["seed"], "seed") != RANDOM_SEED:
                raise ValueError("Runtime-trial seed mismatch.")
            if _int(row["common_pad"], "common_pad") != RUNTIME_COMMON_PAD:
                raise ValueError("Runtime-trial common ROI mismatch.")
            if str(row["noise_scenario"]) != RUNTIME_NOISE:
                raise ValueError("Runtime-trial noise-scenario mismatch.")

    expected_configs = {
        (size, algorithm, n)
        for size in RUNTIME_IMAGE_SIZES
        for algorithm in RUNTIME_ALGORITHMS
        for n in N_VALUES
    }
    by_config: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in existing_rows:
        by_config[_runtime_config_key(row)].append(row)
    completed = {
        key
        for key, rows in by_config.items()
        if len(rows) == runtime_repeats
        and {
            _int(row["repeat"], "repeat") for row in rows
        }
        == set(range(1, runtime_repeats + 1))
    }
    if any(key not in expected_configs for key in by_config):
        raise ValueError("Runtime trial CSV contains unexpected configurations.")

    rows = [
        row
        for row in existing_rows
        if _runtime_config_key(row) in completed
    ]
    missing = list(expected_configs.difference(completed))
    if missing:
        base_clean = _load_image(image_path, max_dimension=256)
        warm_clean = cv2.resize(
            base_clean,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )
        warm_noisy = _make_noise_scenarios(warm_clean)[RUNTIME_NOISE]
        _runtime_warmup(
            warm_noisy,
            tol=tol,
            max_iter=max_iter,
            init_sigma=init_sigma,
        )

        random.Random(RANDOM_SEED).shuffle(missing)
        image_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for image_size, algorithm, n in missing:
            if image_size not in image_cache:
                clean = cv2.resize(
                    base_clean,
                    (image_size, image_size),
                    interpolation=(
                        cv2.INTER_AREA
                        if image_size <= base_clean.shape[0]
                        else cv2.INTER_CUBIC
                    ),
                )
                noisy = _make_noise_scenarios(clean)[RUNTIME_NOISE]
                image_cache[image_size] = (clean, noisy)
            _, noisy = image_cache[image_size]
            rows.extend(
                _benchmark_one_configuration(
                    noisy,
                    image_size=image_size,
                    algorithm=algorithm,
                    n=n,
                    runtime_repeats=runtime_repeats,
                    tol=tol,
                    max_iter=max_iter,
                    init_sigma=init_sigma,
                )
            )
            _write_runtime_trials(trials_path, rows)

    grouped: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_runtime_config_key(row)].append(row)
    if set(grouped) != expected_configs:
        raise ValueError("Runtime benchmark is incomplete.")
    if any(len(group) != runtime_repeats for group in grouped.values()):
        raise ValueError("Runtime benchmark has an invalid repeat count.")

    summary_columns = (
        "image_size",
        "n",
        "h",
        "polynomial_order",
        "algorithm",
        "algorithm_name",
        "noise_scenario",
        "seed",
        "runtime_median",
        "runtime_q1",
        "runtime_q3",
        "runtime_min",
        "runtime_max",
        "runtime_repeats",
        "median_iteration_count",
        "all_converged",
        "common_pad",
    )
    summary_rows = []
    for key in sorted(grouped, key=lambda item: (item[1], item[2], item[0])):
        image_size, algorithm, n = key
        group = grouped[key]
        times = np.asarray(
            [_float(row["runtime"], "runtime") for row in group],
            dtype=float,
        )
        iterations = [
            _int(row["iteration_count"], "iteration_count") for row in group
        ]
        summary_rows.append(
            {
                "image_size": image_size,
                "n": n,
                "h": RUNTIME_H,
                "polynomial_order": POLYNOMIAL_ORDER,
                "algorithm": algorithm,
                "algorithm_name": ALGORITHM_NAMES[algorithm],
                "noise_scenario": RUNTIME_NOISE,
                "seed": RANDOM_SEED,
                "runtime_median": _format_number(float(np.median(times))),
                "runtime_q1": _format_number(
                    float(np.percentile(times, 25.0))
                ),
                "runtime_q3": _format_number(
                    float(np.percentile(times, 75.0))
                ),
                "runtime_min": _format_number(float(np.min(times))),
                "runtime_max": _format_number(float(np.max(times))),
                "runtime_repeats": runtime_repeats,
                "median_iteration_count": _format_number(
                    float(np.median(iterations))
                ),
                "all_converged": str(
                    all(_bool(row["converged"]) for row in group)
                ).lower(),
                "common_pad": RUNTIME_COMMON_PAD,
            }
        )
    _atomic_write_csv(summary_path, summary_rows, summary_columns)
    plot_runtime_scaling(summary_rows, figure_path)
    return summary_rows


def plot_runtime_scaling(rows: Sequence[dict], output_path: Path) -> None:
    colors = {
        4: "#0072B2",
        8: "#D55E00",
        16: "#009E73",
        32: "#CC79A7",
    }
    markers = {4: "o", 8: "s", 16: "^", 32: "D"}
    method_titles = {
        1: "(a) Direct squared",
        2: "(b) Iterative squared",
        3: "(c) Correntropy",
    }
    lookup = {
        (
            _int(row["algorithm"], "algorithm"),
            _int(row["n"], "n"),
            _int(row["image_size"], "image_size"),
        ): row
        for row in rows
    }
    expected = {
        (algorithm, n, size)
        for algorithm in RUNTIME_ALGORITHMS
        for n in N_VALUES
        for size in RUNTIME_IMAGE_SIZES
    }
    if set(lookup) != expected:
        raise ValueError("Runtime summary is incomplete for plotting.")

    all_medians = [
        _float(row["runtime_median"], "runtime_median") for row in rows
    ]
    lower_limit = 10 ** (
        math.floor(math.log10(min(all_medians))) - 0.12
    )
    upper_limit = 10 ** (
        math.ceil(math.log10(max(all_medians))) + 0.12
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
        }
    ):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(7.16, 2.65),
            sharex=True,
            sharey=True,
        )
        for axis, algorithm in zip(axes, RUNTIME_ALGORITHMS):
            for n in N_VALUES:
                selected = [
                    lookup[(algorithm, n, size)]
                    for size in RUNTIME_IMAGE_SIZES
                ]
                medians = np.asarray(
                    [
                        _float(row["runtime_median"], "runtime_median")
                        for row in selected
                    ]
                )
                q1 = np.asarray(
                    [
                        _float(row["runtime_q1"], "runtime_q1")
                        for row in selected
                    ]
                )
                q3 = np.asarray(
                    [
                        _float(row["runtime_q3"], "runtime_q3")
                        for row in selected
                    ]
                )
                axis.plot(
                    RUNTIME_IMAGE_SIZES,
                    medians,
                    color=colors[n],
                    marker=markers[n],
                    markersize=3.3,
                    linewidth=1.15,
                    label=rf"$n={n}$",
                )
                axis.fill_between(
                    RUNTIME_IMAGE_SIZES,
                    q1,
                    q3,
                    color=colors[n],
                    alpha=0.11,
                    linewidth=0,
                )
            axis.set_title(method_titles[algorithm], loc="left", pad=4)
            axis.set_yscale("log")
            axis.set_xlim(min(RUNTIME_IMAGE_SIZES), max(RUNTIME_IMAGE_SIZES))
            axis.set_ylim(lower_limit, upper_limit)
            axis.set_xticks(RUNTIME_IMAGE_SIZES)
            axis.set_xlabel("Square image side (px)")
            axis.grid(
                True,
                which="major",
                axis="both",
                color="#D0D0D0",
                linewidth=0.45,
            )
            axis.grid(
                True,
                which="minor",
                axis="y",
                color="#E8E8E8",
                linewidth=0.3,
            )
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_linewidth(0.55)
        axes[0].set_ylabel("Median runtime (s, log scale)")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 1.025),
        )
        figure.text(
            0.5,
            0.005,
            "SP=10%, h=1, order=3; descriptive IQR across 3 runs. "
            "Pipeline timing; native solve area varies with n.",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
        figure.subplots_adjust(
            left=0.082,
            right=0.995,
            bottom=0.24,
            top=0.80,
            wspace=0.12,
        )
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def _joint_rows_by_key(raw_rows: Sequence[dict]) -> dict[tuple[int, int, str], dict]:
    return {_joint_key(row): row for row in raw_rows}


def write_parameter_summary_tex(
    output_path: Path,
    joint_rows: Sequence[dict],
    selections: JointSelections,
) -> None:
    lookup = _joint_rows_by_key(joint_rows)
    image_heights = {
        _int(row["image_height"], "image_height") for row in joint_rows
    }
    image_widths = {
        _int(row["image_width"], "image_width") for row in joint_rows
    }
    common_pads = {_int(row["common_pad"], "common_pad") for row in joint_rows}
    if (
        len(image_heights) != 1
        or len(image_widths) != 1
        or len(common_pads) != 1
    ):
        raise ValueError("Joint sweep must use one image shape and common ROI.")
    common_pad = next(iter(common_pads))
    roi_height = next(iter(image_heights)) - 2 * common_pad
    roi_width = next(iter(image_widths)) - 2 * common_pad
    lines = [
        "% Generated by final_parameter_analysis.py from "
        "joint_correntropy_sweep.csv.",
        "% Requires the booktabs package.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Representative correntropy configurations from the sparse "
        r"joint sweep. Matched-clean metrics quantify noise stability.}",
        r"\label{tab:parameter-summary}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Role & Noise & $n$ & $h$ & RMSE & HOG rel. error & Cosine & "
        r"Runtime (s) \\",
        r"\midrule",
    ]
    grouped_roles: dict[tuple[int, int], list[str]] = defaultdict(list)
    selected_by_configuration: dict[tuple[int, int], JointAggregate] = {}
    for role, selected in selections.role_items():
        grouped_roles[(selected.n, selected.h)].append(role)
        selected_by_configuration[(selected.n, selected.h)] = selected
    for configuration, roles in grouped_roles.items():
        selected = selected_by_configuration[configuration]
        role_label = " / ".join(roles)
        for scenario, label in (
            ("saltpepper_5", r"SP 5\%"),
            ("saltpepper_10", r"SP 10\%"),
        ):
            row = lookup[(selected.n, selected.h, scenario)]
            lines.append(
                f"{role_label} & {label} & {selected.n} & {selected.h} & "
                f"{_float(row['gradient_RMSE'], 'gradient_RMSE'):.3f} & "
                f"{_float(row['hog_relative_error'], 'hog_relative_error'):.4f} & "
                f"{_float(row['cosine_similarity'], 'cosine_similarity'):.4f} & "
                f"{_float(row['runtime'], 'runtime'):.3f} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\medskip",
            r"\footnotesize All rows use correntropy, polynomial order 3, "
            f"seed 42, and the same {roi_height} by {roi_width} common "
            r"analysis ROI after "
            r"cropping. Fastest minimizes mean runtime. Maximum weighted "
            r"robustness maximizes the normalized matched-clean quality score. "
            r"Balanced "
            r"maximizes the harmonic mean of that score and logarithmic "
            r"runtime benefit across the six measured candidates. Raw SP 5\% "
            r"and SP 10\% metrics are first averaged per configuration, then "
            r"min-max normalized across the six candidates.",
            r"The fastest and balanced criteria may select the same measured "
            r"configuration.",
            r"\end{table*}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _lookup_sensitivity(
    rows: Sequence[dict],
    *,
    experiment: str,
    noise: str,
    algorithm: int,
    n: int,
    h: int,
) -> dict:
    matches = [
        row
        for row in rows
        if row["experiment"] == experiment
        and row["noise_scenario"] == noise
        and row["algorithm"] == algorithm
        and row["n"] == n
        and row["h"] == h
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one sensitivity row for "
            f"{experiment}/{noise}/algorithm={algorithm}/n={n}/h={h}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _percent_reduction(old: float, new: float) -> float:
    if old <= 0:
        raise ValueError("Percentage reduction requires a positive baseline.")
    return 100.0 * (old - new) / old


def write_latex_subsection(
    output_path: Path,
    sensitivity_rows: Sequence[dict],
    joint_rows: Sequence[dict],
    selections: JointSelections,
) -> None:
    n_claims = {}
    h_claims = {}
    for noise in ("saltpepper_5", "saltpepper_10"):
        n16 = _lookup_sensitivity(
            sensitivity_rows,
            experiment="n_analysis",
            noise=noise,
            algorithm=3,
            n=16,
            h=1,
        )
        n32 = _lookup_sensitivity(
            sensitivity_rows,
            experiment="n_analysis",
            noise=noise,
            algorithm=3,
            n=32,
            h=1,
        )
        n_claims[noise] = {
            "rmse_reduction": _percent_reduction(
                n16["gradient_RMSE"], n32["gradient_RMSE"]
            ),
            "runtime_increase": 100.0
            * (n32["runtime"] / n16["runtime"] - 1.0),
        }
        h3 = _lookup_sensitivity(
            sensitivity_rows,
            experiment="h_analysis",
            noise=noise,
            algorithm=3,
            n=8,
            h=3,
        )
        h4 = _lookup_sensitivity(
            sensitivity_rows,
            experiment="h_analysis",
            noise=noise,
            algorithm=3,
            n=8,
            h=4,
        )
        h_claims[noise] = {
            "rmse_reduction": _percent_reduction(
                h3["gradient_RMSE"], h4["gradient_RMSE"]
            ),
            "cosine_gain": h4["cosine_similarity"]
            - h3["cosine_similarity"],
        }

    joint_lookup = _joint_rows_by_key(joint_rows)
    selected_sentences = []
    if (
        selections.fastest.n,
        selections.fastest.h,
    ) == (
        selections.balanced.n,
        selections.balanced.h,
    ):
        selected = selections.fastest
        row5 = joint_lookup[(selected.n, selected.h, "saltpepper_5")]
        row10 = joint_lookup[(selected.n, selected.h, "saltpepper_10")]
        mean_runtime = statistics.fmean(
            [
                _float(row5["runtime"], "runtime"),
                _float(row10["runtime"], "runtime"),
            ]
        )
        selected_sentences.append(
            "The fastest and balanced criteria both selected "
            f"$(n,h)=({selected.n},{selected.h})$, with mean runtime "
            f"{mean_runtime:.3f}~s over SP 5\\% and SP 10\\%."
        )
    else:
        for role, selected in (
            ("fastest", selections.fastest),
            ("balanced", selections.balanced),
        ):
            row5 = joint_lookup[(selected.n, selected.h, "saltpepper_5")]
            row10 = joint_lookup[(selected.n, selected.h, "saltpepper_10")]
            mean_runtime = statistics.fmean(
                [
                    _float(row5["runtime"], "runtime"),
                    _float(row10["runtime"], "runtime"),
                ]
            )
            selected_sentences.append(
                f"The {role} selection was $(n,h)=({selected.n},"
                f"{selected.h})$, with mean runtime "
                f"{mean_runtime:.3f}~s over SP 5\\% and SP 10\\%."
            )
    robust = selections.maximum_robustness
    robust5 = joint_lookup[(robust.n, robust.h, "saltpepper_5")]
    robust10 = joint_lookup[(robust.n, robust.h, "saltpepper_10")]
    robust_runtime = statistics.fmean(
        [
            _float(robust5["runtime"], "runtime"),
            _float(robust10["runtime"], "runtime"),
        ]
    )
    selected_sentences.append(
        "The maximum-weighted-robustness selection was "
        f"$(n,h)=({robust.n},{robust.h})$, with mean runtime "
        f"{robust_runtime:.3f}~s over SP 5\\% and SP 10\\%."
    )
    package_directory = output_path.parent.as_posix().rstrip("/")
    graphic_path_line = (
        rf"\graphicspath{{{{{package_directory}/}}{{./}}}}"
    )
    summary_input_line = (
        r"\IfFileExists{parameter_summary.tex}"
        r"{\input{parameter_summary.tex}}"
        rf"{{\input{{{package_directory}/parameter_summary.tex}}}}"
    )

    lines = [
        "% Generated by final_parameter_analysis.py.",
        f"% {SEPARABLE_MODEL_NOTE}",
        f"% {FINAL_NOTE}",
        "% Requires graphicx and booktabs.",
        r"\subsection{Parameter sensitivity and sparse joint selection}",
        r"\label{sec:final-parameter-analysis}",
        graphic_path_line,
        "",
        r"\paragraph{Scope and metric interpretation.}",
        "This experiment evaluates a separable 1-D Taylor model applied "
        "independently along $x$ and $y$. It is not the joint full 2-D "
        "Taylor model. The matched-clean gradient RMSE and HOG metrics "
        "measure stability to injected noise for seed 42; they do not "
        "measure absolute derivative accuracy. All comparisons use "
        "parameter-matched clean references.",
        r"\emph{Model note: separable 1-D Taylor along $x/y$; "
        r"matched-clean metrics measure noise stability, not absolute "
        r"derivative accuracy.}",
        "",
        r"\paragraph{One-factor sensitivity.}",
        "At fixed $h=1$, changing the per-axis sample count from $n=16$ "
        "to $n=32$ reduced correntropy gradient RMSE by "
        f"{n_claims['saltpepper_5']['rmse_reduction']:.1f}\\% under SP 5\\% "
        f"and {n_claims['saltpepper_10']['rmse_reduction']:.1f}\\% under "
        "SP 10\\%. The corresponding runtime changes were "
        f"{n_claims['saltpepper_5']['runtime_increase']:.1f}\\% and "
        f"{n_claims['saltpepper_10']['runtime_increase']:.1f}\\%, "
        "respectively. At fixed $n=8$, changing $h$ from 3 to 4 reduced "
        "correntropy RMSE by "
        f"{h_claims['saltpepper_5']['rmse_reduction']:.1f}\\% and "
        f"{h_claims['saltpepper_10']['rmse_reduction']:.1f}\\%, while "
        "the cosine gains were only "
        f"{h_claims['saltpepper_5']['cosine_gain']:.4f} and "
        f"{h_claims['saltpepper_10']['cosine_gain']:.4f}. These statements "
        "apply only to the measured one-factor sweeps.",
        "",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{n_visual_comparison.png}",
        r"\caption{Qualitative Robust-HOG comparison for Direct and "
        r"Correntropy gradients at SP 10\% while varying the per-axis sample "
        r"count. All panels share the pooled 95th-percentile display scale.}",
        r"\label{fig:n-visual-comparison}",
        r"\end{figure*}",
        "",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{h_visual_comparison.png}",
        r"\caption{Qualitative Robust-HOG comparison for Direct and "
        r"Correntropy gradients at SP 10\% while varying sample spacing. "
        r"All panels share the pooled 95th-percentile display scale.}",
        r"\label{fig:h-visual-comparison}",
        r"\end{figure*}",
        "",
        r"\paragraph{Sparse joint correntropy sweep.}",
        "The joint experiment evaluated only "
        "$(8,3)$, $(8,4)$, $(16,2)$, $(16,3)$, $(16,4)$, and $(32,1)$ "
        "under SP 5\\% and SP 10\\%; it is not a full factorial search. "
        + " ".join(selected_sentences)
        + " Selection labels are candidate-set dependent and do not imply "
        "a universal optimum.",
        "",
        summary_input_line,
        "",
        r"\paragraph{Runtime scaling.}",
        "Figure~\\ref{fig:runtime-scaling} reports newly measured runtime "
        "curves rather than extrapolating from the single-size sensitivity "
        "CSV. Each panel uses the same image sizes and $n$ values, and the "
        "shared logarithmic vertical axis permits direct scale comparison. "
        "The timer includes noisy gradient estimation, common-ROI cropping, "
        "and Robust HOG, while image loading, resizing, noise generation, "
        "and design-matrix construction are excluded. For fixed $h=1$, the "
        "estimator first solves $(S-n)^2$ native gradient centres and then "
        "crops every output to the common $(S-32)^2$ ROI. The curves "
        "therefore report implementation-level pipeline runtime rather than "
        "fixed-output-pixel complexity. "
        "Correntropy timing is data and convergence dependent; monotonic "
        "runtime in $n$ is therefore not assumed. Each interquartile band is "
        "a descriptive spread over three repeats, not a confidence interval.",
        "",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{runtime_scaling_2d.png}",
        r"\caption{Median gradient-plus-HOG inference runtime versus square "
        r"image side under SP 10\%, with fixed $h=1$ and order 3. Shaded "
        r"bands show the descriptive interquartile range of three timing "
        r"repeats; native solve area varies with $n$ before common cropping.}",
        r"\label{fig:runtime-scaling}",
        r"\end{figure*}",
        "",
        r"\paragraph{Reproducibility limitation.}",
        "The noise realization and benchmark order use seed 42. Because "
        "only one noise seed and one source image are included, the reported "
        "values are deterministic case-study measurements rather than "
        "population estimates or confidence intervals.",
        "",
    ]
    text = "\n".join(lines)
    if "--" in text:
        raise RuntimeError("Unresolved double-hyphen placeholder in LaTeX.")
    output_path.write_text(text, encoding="utf-8")


def write_readme(
    output_path: Path,
    *,
    joint_repeats: int,
    runtime_repeats: int,
    selections: JointSelections,
) -> None:
    lines = [
        "# Final parameter-analysis package",
        "",
        SEPARABLE_MODEL_NOTE,
        "",
        FINAL_NOTE,
        "",
        "## Contents",
        "",
        "- `n_visual_comparison.png`: Direct and Correntropy Robust-HOG "
        "panels for n = 4, 8, 16, 32 at h = 1 and SP = 10%.",
        "- `h_visual_comparison.png`: Direct and Correntropy Robust-HOG "
        "panels for h = 1, 2, 3, 4 at n = 8 and SP = 10%.",
        "- `joint_correntropy_sweep.csv`: the 12 measured sparse joint rows.",
        "- `joint_correntropy_selection.csv`: averaged scores and the "
        "fastest, balanced, and maximum-weighted-robustness labels.",
        "- `parameter_summary.tex`: compact numerical table.",
        "- `runtime_scaling_trials.csv` and "
        "`runtime_scaling_summary.csv`: measured timing provenance.",
        "- `runtime_scaling_2d.png`: shared-axis, log-runtime line plots.",
        "- `parameter_analysis_subsection.tex`: manuscript-ready subsection "
        "with no unresolved numerical placeholders.",
        "- `package_metadata.json`: experiment and environment metadata.",
        "",
        "## Selected sparse-joint configurations",
        "",
        f"- Fastest: n = {selections.fastest.n}, h = {selections.fastest.h}.",
        f"- Balanced: n = {selections.balanced.n}, h = "
        f"{selections.balanced.h}.",
        "- Maximum weighted robustness: n = "
        f"{selections.maximum_robustness.n}, h = "
        f"{selections.maximum_robustness.h}.",
        "",
        "Selections are restricted to the six measured correntropy pairs. "
        "Raw SP 5% and SP 10% metrics are first averaged per configuration; "
        "the six configuration means are then min-max benefit scaled. "
        "Weighted robustness uses relative weights 0.20 RMSE, 0.45 HOG "
        "relative error, and 0.30 cosine. Balanced is the harmonic mean of "
        "weighted robustness and min-max log-runtime benefit.",
        "",
        "Runtime scaling reports implementation-level pipeline time. At h = "
        "1, the native solve area is (S - n)^2 before every result is cropped "
        "to the common (S - 32)^2 ROI; timings are not normalized per output "
        f"pixel. The IQR from {runtime_repeats} repeats is descriptive, not "
        "a confidence interval. Image loading/resizing, noise generation, "
        "and design-matrix construction are outside the timer.",
        "",
        "## Reproduce",
        "",
        "Run from the repository root:",
        "",
        "```powershell",
        "python -B final_parameter_analysis.py",
        "```",
        "",
        "Existing complete benchmark CSVs are reused only when the recorded "
        "image hash, solver settings, design, repeats, and core library "
        "versions match. Add `--force-joint` or `--force-runtime` to repeat "
        "the affected measurements.",
        "",
        f"Joint runtimes use the median of {joint_repeats} runs. Runtime "
        f"scaling uses {runtime_repeats} trials per point.",
        "",
        "No legacy 3-D image-size runtime dataset was present, so the 2-D "
        "runtime plot is based on newly measured trials rather than values "
        "inferred from the one-size sensitivity CSV.",
        "",
        "The original one-factor sensitivity CSV does not contain a seed "
        "column; its seed 42 provenance is recorded in "
        "`parameter_analysis_config.json` and the generating source. The "
        "joint and runtime CSVs record seed 42 explicitly.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_metadata(
    output_path: Path,
    args: argparse.Namespace,
    selections: JointSelections,
) -> None:
    metadata = {
        "model_note": SEPARABLE_MODEL_NOTE,
        "metric_note": (
            "Matched-clean metrics measure noise stability, not absolute "
            "derivative accuracy."
        ),
        "seed": RANDOM_SEED,
        "source_image": str(args.image),
        "source_image_sha256": _sha256_file(args.image),
        "sensitivity_csv": str(args.sensitivity_csv),
        "polynomial_order": POLYNOMIAL_ORDER,
        "n_definition": (
            "number of symmetric samples per axis; x and y are solved "
            "independently"
        ),
        "qualitative": {
            "noise": "saltpepper_10",
            "algorithms": [1, 3],
            "n_values": list(N_VALUES),
            "h_values": list(H_VALUES),
            "source_scaling": (
                "cropped from existing dense figures rendered with one pooled "
                "95th-percentile HOG scale"
            ),
            "common_pad": QUALITATIVE_COMMON_PAD,
        },
        "joint_sweep": {
            "configurations": [list(item) for item in JOINT_CONFIGURATIONS],
            "noise_scenarios": [name for name, _ in JOINT_NOISES],
            "runtime_repeats": args.joint_repeats,
            "common_pad": JOINT_COMMON_PAD,
            "selection": {
                "fastest": [
                    selections.fastest.n,
                    selections.fastest.h,
                ],
                "balanced": [
                    selections.balanced.n,
                    selections.balanced.h,
                ],
                "maximum_weighted_robustness": [
                    selections.maximum_robustness.n,
                    selections.maximum_robustness.h,
                ],
                "robustness_score": (
                    "raw SP5/SP10 metrics are averaged per configuration "
                    "before min-max benefit scaling across configurations; "
                    "(0.20 inverse RMSE + 0.45 inverse HOG error + "
                    "0.30 cosine) / 0.95"
                ),
                "balanced_score": (
                    "harmonic mean of robustness_score and inverse min-max "
                    "log-runtime benefit"
                ),
            },
        },
        "runtime_scaling": {
            "image_sizes": list(RUNTIME_IMAGE_SIZES),
            "n_values": list(N_VALUES),
            "h": RUNTIME_H,
            "noise_scenario": RUNTIME_NOISE,
            "algorithms": list(RUNTIME_ALGORITHMS),
            "runtime_repeats": args.runtime_repeats,
            "common_pad": RUNTIME_COMMON_PAD,
            "benchmark_order_seed": RANDOM_SEED,
            "runtime_definition": (
                "timer includes noisy gradient estimation, common-ROI crop, "
                "and robust HOG; image loading/resizing, noise generation, "
                "and solver/design-matrix construction are excluded; median "
                "and descriptive IQR across repeated trials"
            ),
            "native_solve_area": (
                "(image_side - n)^2 centres for h=1, before common-ROI "
                "cropping; timings are not normalized per output pixel"
            ),
            "spread_interpretation": (
                "IQR over three timing repeats is descriptive, not a "
                "confidence interval"
            ),
        },
        "solver": {
            "tol": args.tol,
            "max_iter": args.max_iter,
            "init_sigma": args.init_sigma,
        },
        "hog": {
            "normalization": HOG_NORMALIZATION,
            "robust": ROBUST_HOG,
            "weighting": ROBUST_WEIGHTING,
            "delta": ROBUST_DELTA,
            "window": ROBUST_WINDOW,
            "visualization_percentile": VISUALIZATION_PERCENTILE,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "opencv_threads": cv2.getNumThreads(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _verify_png(path: Path, *, minimum_width: int, minimum_height: int) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unreadable PNG: {path}")
    if image.shape[1] < minimum_width or image.shape[0] < minimum_height:
        raise ValueError(
            f"PNG is unexpectedly small: {path} has shape {image.shape}."
        )
    if float(np.std(image)) < 1.0:
        raise ValueError(f"PNG appears blank: {path}")


def verify_package(
    output_dir: Path,
    *,
    joint_repeats: int,
    runtime_repeats: int,
) -> None:
    required = (
        "n_visual_comparison.png",
        "h_visual_comparison.png",
        "parameter_summary.tex",
        "runtime_scaling_trials.csv",
        "runtime_scaling_summary.csv",
        "runtime_scaling_2d.png",
        "joint_correntropy_sweep.csv",
        "joint_correntropy_selection.csv",
        "parameter_analysis_subsection.tex",
        "README.md",
        "package_metadata.json",
    )
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise ValueError(f"Final package is missing files: {missing}")

    _verify_png(
        output_dir / "n_visual_comparison.png",
        minimum_width=1600,
        minimum_height=700,
    )
    _verify_png(
        output_dir / "h_visual_comparison.png",
        minimum_width=1600,
        minimum_height=700,
    )
    _verify_png(
        output_dir / "runtime_scaling_2d.png",
        minimum_width=1600,
        minimum_height=500,
    )

    joint_rows = _read_csv(output_dir / "joint_correntropy_sweep.csv")
    _validate_joint_rows(joint_rows, joint_repeats)
    runtime_trials = _read_csv(output_dir / "runtime_scaling_trials.csv")
    expected_trial_count = (
        len(RUNTIME_IMAGE_SIZES)
        * len(N_VALUES)
        * len(RUNTIME_ALGORITHMS)
        * runtime_repeats
    )
    if len(runtime_trials) != expected_trial_count:
        raise ValueError(
            f"Expected {expected_trial_count} runtime trials, found "
            f"{len(runtime_trials)}."
        )
    expected_trial_keys = {
        (size, algorithm, n, repeat)
        for size in RUNTIME_IMAGE_SIZES
        for algorithm in RUNTIME_ALGORITHMS
        for n in N_VALUES
        for repeat in range(1, runtime_repeats + 1)
    }
    observed_trial_keys = {
        (
            _int(row["image_size"], "image_size"),
            _int(row["algorithm"], "algorithm"),
            _int(row["n"], "n"),
            _int(row["repeat"], "repeat"),
        )
        for row in runtime_trials
    }
    if observed_trial_keys != expected_trial_keys:
        raise ValueError("Runtime trial keys are incomplete or duplicated.")
    for row in runtime_trials:
        if (
            _int(row["h"], "h") != RUNTIME_H
            or _int(row["polynomial_order"], "polynomial_order")
            != POLYNOMIAL_ORDER
            or _int(row["seed"], "seed") != RANDOM_SEED
            or _int(row["common_pad"], "common_pad")
            != RUNTIME_COMMON_PAD
            or str(row["noise_scenario"]) != RUNTIME_NOISE
        ):
            raise ValueError("Runtime trial provenance mismatch.")
    runtime_summary = _read_csv(output_dir / "runtime_scaling_summary.csv")
    if len(runtime_summary) != (
        len(RUNTIME_IMAGE_SIZES) * len(N_VALUES) * len(RUNTIME_ALGORITHMS)
    ):
        raise ValueError("Runtime summary row count mismatch.")
    summary_lookup = {
        (
            _int(row["image_size"], "image_size"),
            _int(row["algorithm"], "algorithm"),
            _int(row["n"], "n"),
        ): row
        for row in runtime_summary
    }
    expected_summary_keys = {
        (size, algorithm, n)
        for size in RUNTIME_IMAGE_SIZES
        for algorithm in RUNTIME_ALGORITHMS
        for n in N_VALUES
    }
    if set(summary_lookup) != expected_summary_keys:
        raise ValueError("Runtime summary keys are incomplete or duplicated.")
    grouped_trials: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in runtime_trials:
        key = (
            _int(row["image_size"], "image_size"),
            _int(row["algorithm"], "algorithm"),
            _int(row["n"], "n"),
        )
        grouped_trials[key].append(_float(row["runtime"], "runtime"))
    for key, times in grouped_trials.items():
        summary = summary_lookup[key]
        checks = (
            (
                float(np.median(times)),
                _float(summary["runtime_median"], "runtime_median"),
            ),
            (
                float(np.percentile(times, 25.0)),
                _float(summary["runtime_q1"], "runtime_q1"),
            ),
            (
                float(np.percentile(times, 75.0)),
                _float(summary["runtime_q3"], "runtime_q3"),
            ),
        )
        if any(
            not math.isclose(
                expected_value,
                observed_value,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
            for expected_value, observed_value in checks
        ):
            raise ValueError(f"Runtime summary statistic mismatch for {key}.")

    for tex_name in (
        "parameter_summary.tex",
        "parameter_analysis_subsection.tex",
    ):
        text = (output_dir / tex_name).read_text(encoding="utf-8")
        if "--" in text:
            raise ValueError(f"Unresolved placeholder in {tex_name}.")
        if SEPARABLE_MODEL_NOTE not in text and tex_name.endswith(
            "subsection.tex"
        ):
            raise ValueError("Required separable-model note is absent.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the final paper-ready parameter-analysis package."
    )
    parser.add_argument(
        "--sensitivity-csv",
        type=Path,
        default=Path("parameter_analysis_results.csv"),
    )
    parser.add_argument(
        "--n-source-figure",
        type=Path,
        default=Path("n_analysis.png"),
    )
    parser.add_argument(
        "--h-source-figure",
        type=Path,
        default=Path("h_analysis.png"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("Images/cameraman.png"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_parameter_analysis"),
    )
    parser.add_argument("--joint-repeats", type=int, default=3)
    parser.add_argument("--runtime-repeats", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--init-sigma", type=float, default=None)
    parser.add_argument("--force-joint", action="store_true")
    parser.add_argument("--force-runtime", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.joint_repeats < 1 or args.runtime_repeats < 1:
        raise ValueError("Repeat counts must be positive.")
    if args.max_iter < 1:
        raise ValueError("max_iter must be positive.")
    if args.tol <= 0:
        raise ValueError("tol must be positive.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_cache_provenance(args)
    sensitivity_rows = _load_sensitivity_rows(args.sensitivity_csv)

    generate_qualitative_figures(
        args.n_source_figure,
        args.h_source_figure,
        output_dir,
    )

    joint_path = output_dir / "joint_correntropy_sweep.csv"
    joint_rows = run_joint_correntropy_sweep(
        args.image,
        joint_path,
        runtime_repeats=args.joint_repeats,
        tol=args.tol,
        max_iter=args.max_iter,
        init_sigma=args.init_sigma,
        force=args.force_joint,
    )
    selections, scored = select_joint_configurations(joint_rows)
    write_joint_selection_csv(
        output_dir / "joint_correntropy_selection.csv",
        selections,
        scored,
    )

    run_runtime_benchmark(
        args.image,
        output_dir / "runtime_scaling_trials.csv",
        output_dir / "runtime_scaling_summary.csv",
        output_dir / "runtime_scaling_2d.png",
        runtime_repeats=args.runtime_repeats,
        tol=args.tol,
        max_iter=args.max_iter,
        init_sigma=args.init_sigma,
        force=args.force_runtime,
    )

    write_parameter_summary_tex(
        output_dir / "parameter_summary.tex",
        joint_rows,
        selections,
    )
    write_latex_subsection(
        output_dir / "parameter_analysis_subsection.tex",
        sensitivity_rows,
        joint_rows,
        selections,
    )
    write_readme(
        output_dir / "README.md",
        joint_repeats=args.joint_repeats,
        runtime_repeats=args.runtime_repeats,
        selections=selections,
    )
    write_metadata(output_dir / "package_metadata.json", args, selections)
    verify_package(
        output_dir,
        joint_repeats=args.joint_repeats,
        runtime_repeats=args.runtime_repeats,
    )

    print(f"Final package: {output_dir.resolve()}")
    for role, selection in selections.role_items():
        print(
            f"{role}: n={selection.n}, h={selection.h}, "
            f"mean runtime={selection.runtime:.3f} s"
        )


if __name__ == "__main__":
    main()
