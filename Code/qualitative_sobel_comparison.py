"""Publication-ready qualitative Sobel comparison and shared experiment utilities."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hog_descriptor import HOGResult, describe_hog, render_hog_group
from parameter_analysis_pipeline import (
    AxisTaylorSolver,
    GradientEstimate,
    _crop_to_common_roi,
    _estimate_separable_gradient,
    _load_image,
)
from run_pipeline import add_gaussian_noise_snr, add_noise, sobel_gradient

METHOD_ORDER = ("Sobel", "Method I", "Method II", "Method III")
DEFAULT_CONFIG = Path("configs/sobel_comparison.json")


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and validate the shared JSON configuration."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = json.load(stream)
    root = config_path.parent.parent
    required = {"seed", "test_images", "methods", "hog", "solver", "common_roi_pad"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Configuration is missing keys: {', '.join(missing)}")
    if int(config["seed"]) < 0:
        raise ValueError("seed must be non-negative.")
    for name in METHOD_ORDER:
        if name not in config["methods"]:
            raise ValueError(f"Missing method configuration: {name}")
    pad = int(config["common_roi_pad"])
    for name in METHOD_ORDER[1:]:
        params = config["methods"][name]
        native_pad = int(params["n"]) // 2 * int(params["h"])
        if pad < native_pad:
            raise ValueError(
                f"common_roi_pad={pad} is smaller than {name}'s native pad={native_pad}."
            )
    return config, root


def resolve_path(root: Path, value: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def package_versions() -> dict[str, str]:
    """Return exact runtime package versions."""
    names = ("numpy", "opencv-python", "scipy", "scikit-image", "matplotlib")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    versions["python"] = sys.version
    versions["platform"] = platform.platform()
    return versions


def array_sha256(array: np.ndarray) -> str:
    """Hash an array including its shape and dtype."""
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def hog_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Translate shared HOG configuration to the existing HOG API."""
    hog = config["hog"]
    return {
        "fuzzy": bool(hog["fuzzy"]),
        "normalization": str(hog["normalization"]),
        "robust": bool(hog["robust"]),
        "robust_weighting": str(hog["robust_weighting"]),
        "robust_delta": float(hog["robust_delta"]),
        "robust_window": int(hog["robust_window"]),
        "min_weight": float(hog.get("min_weight", 0.05)),
        "visualization_percentile": float(hog["visualization_percentile"]),
    }


def crop_sobel(array: np.ndarray, common_pad: int) -> np.ndarray:
    """Crop a full-size Sobel array to the shared valid ROI."""
    if min(array.shape) <= 2 * common_pad:
        raise ValueError(f"Image shape {array.shape} is too small for pad={common_pad}.")
    return array[common_pad:-common_pad, common_pad:-common_pad]


def estimate_method(
    image: np.ndarray,
    method: str,
    config: dict[str, Any],
    solver: AxisTaylorSolver | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one gradient using the existing Sobel or Taylor implementation."""
    pad = int(config["common_roi_pad"])
    if method == "Sobel":
        gradient = sobel_gradient(image)
        return crop_sobel(gradient.dx, pad), crop_sobel(gradient.dy, pad)
    params = config["methods"][method]
    estimate: GradientEstimate = _estimate_separable_gradient(
        image,
        n_points=int(params["n"]),
        h=int(params["h"]),
        polynomial_order=int(params["polynomial_order"]),
        algorithm=int(params["algorithm"]),
        tol=float(config["solver"]["tol"]),
        max_iter=int(config["solver"]["max_iter"]),
        init_sigma=config["solver"]["init_sigma"],
        solver=solver,
    )
    return (
        _crop_to_common_roi(estimate.dx, native_pad=estimate.native_pad, common_pad=pad),
        _crop_to_common_roi(estimate.dy, native_pad=estimate.native_pad, common_pad=pad),
    )


def make_solvers(config: dict[str, Any]) -> dict[str, AxisTaylorSolver]:
    """Construct fixed design matrices outside measured runtime regions."""
    solvers: dict[str, AxisTaylorSolver] = {}
    for method in METHOD_ORDER[1:]:
        params = config["methods"][method]
        solvers[method] = AxisTaylorSolver(
            int(params["n"]), int(params["h"]), int(params["polynomial_order"])
        )
    return solvers


def make_noise(
    clean: np.ndarray, scenario: str, config: dict[str, Any], seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a deterministic scenario shared by all methods."""
    if scenario == "gaussian_20db":
        target = float(config["noise"]["gaussian_snr_db"])
        noisy, achieved = add_gaussian_noise_snr(clean, target, seed)
        detail: dict[str, Any] = {
            "kind": "Gaussian", "target_snr_db": target, "achieved_snr_db": achieved
        }
    elif scenario.startswith("sp"):
        density = float(scenario[2:])
        noisy = add_noise(clean, "saltpepper", density, seed)
        detail = {"kind": "salt-and-pepper", "density_percent": density}
    else:
        raise ValueError(f"Unknown noise scenario: {scenario}")
    detail["seed"] = seed
    detail["noisy_image_sha256"] = array_sha256(noisy)
    return noisy, detail


def method_title(method: str, config: dict[str, Any]) -> str:
    """Format a panel label containing the fixed parameters."""
    if method == "Sobel":
        return "Sobel\n3 x 3 kernel"
    params = config["methods"][method]
    return (
        f"{method}: {params['name']}\n"
        f"n={params['n']}, h={params['h']}, p={params['polynomial_order']}"
    )


def run(config_path: str | Path = DEFAULT_CONFIG) -> list[Path]:
    """Generate the two configured qualitative figures and metadata."""
    config, root = load_config(config_path)
    image_path = resolve_path(root, config["representative_image"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Representative image not found: {image_path}")
    clean = _load_image(image_path, int(config["max_image_dimension"]))
    if min(clean.shape) <= 2 * int(config["common_roi_pad"]) + 16:
        raise ValueError(
            f"Representative image {image_path} becomes too small ({clean.shape}) "
            "for the common ROI and HOG blocks."
        )
    output = resolve_path(root, config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    solvers = make_solvers(config)
    scenario_outputs = {
        "gaussian_20db": output / "qualitative_gaussian_20db.png",
        "sp10": output / "qualitative_sp10.png",
    }
    metadata: dict[str, Any] = {
        "status": "COMPLETED",
        "representative_image": str(image_path.relative_to(root)),
        "image_shape_after_preprocessing": list(clean.shape),
        "seed": int(config["seed"]),
        "common_roi_pad": int(config["common_roi_pad"]),
        "methods": config["methods"],
        "hog": config["hog"],
        "solver": config["solver"],
        "display_scale": "one pooled positive-pixel percentile across all four HOG panels",
        "package_versions": package_versions(),
        "scenarios": {},
    }
    for scenario, destination in scenario_outputs.items():
        noisy, noise_detail = make_noise(clean, scenario, config, int(config["seed"]))
        hog_results: list[HOGResult] = []
        for method in METHOD_ORDER:
            dx, dy = estimate_method(noisy, method, config, solvers.get(method))
            hog_results.append(describe_hog(dx, dy, **hog_kwargs(config)))
        panels = render_hog_group(
            hog_results, percentile=float(config["hog"]["visualization_percentile"])
        )
        figure, axes = plt.subplots(1, 5, figsize=(15.5, 3.45), constrained_layout=True)
        noisy_roi = crop_sobel(noisy, int(config["common_roi_pad"]))
        axes[0].imshow(noisy_roi, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Noisy input")
        for axis, method, panel in zip(axes[1:], METHOD_ORDER, panels):
            axis.imshow(panel, cmap="gray", vmin=0, vmax=255)
            axis.set_title(method_title(method, config), fontsize=8.5)
        for axis in axes:
            axis.axis("off")
        figure.savefig(destination, dpi=int(config["qualitative_dpi"]), bbox_inches="tight")
        plt.close(figure)
        metadata["scenarios"][scenario] = {
            **noise_detail,
            "displayed_input_roi_shape": list(noisy_roi.shape),
            "output": str(destination.relative_to(root)),
        }
    metadata_path = output / "qualitative_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return [*scenario_outputs.values(), metadata_path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready Gaussian and impulsive Sobel comparisons."
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
