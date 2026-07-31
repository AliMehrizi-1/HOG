from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from gradient_estimation_solver import MaclaurinApproximator
from hog_descriptor import HOGResult, describe_hog, render_hog_group


@dataclass(frozen=True)
class GradientResult:
    magnitude: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    pad: int
    parameters: dict


def print_params(header: str, params: dict):
    print(f"\n{header}")
    for key, value in params.items():
        print(f"  {key:<30}= {value}")
    print()


def estimate_taylor_gradient(
    data: np.ndarray,
    order: int,
    spacing: int,
    algorithm: int = 3,
    *,
    smoothing_sigma: float = 0.0,
    tol: float = 1e-3,
    max_iter: int = 15,
    init_sigma: float | None = None,
    verbose: bool = False,
) -> GradientResult:
    """Estimate gradients only; HOG choices do not affect this result."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("data must be a non-empty grayscale image.")
    if not np.isfinite(data).all():
        raise ValueError("data contains NaN or infinite values.")
    if smoothing_sigma < 0 or not np.isfinite(smoothing_sigma):
        raise ValueError("smoothing_sigma must be a non-negative finite number.")

    parameters = {
        "neighbour_samples_n": order,
        "taylor_degree": 2,
        "spacing_h": spacing,
        "algorithm": algorithm,
        "smoothing_sigma": smoothing_sigma,
        "tol": tol,
        "max_iter": max_iter,
        "correntropy_sigma": (
            init_sigma if init_sigma is not None else "automatic (MAD)"
        )
        if algorithm == 3
        else None,
    }
    if verbose:
        print_params("[Taylor/Maclaurin gradient parameters]", parameters)

    image = (
        cv2.GaussianBlur(data, (3, 3), smoothing_sigma)
        if smoothing_sigma > 0
        else data
    )
    approximator = MaclaurinApproximator(order, spacing)
    field = approximator.approximate_initial_field(
        image,
        algorithm,
        tol,
        max_iter,
        init_sigma=init_sigma,
    )
    dx = field[..., 0]
    dy = field[..., 1]
    return GradientResult(
        magnitude=np.hypot(dx, dy),
        dx=dx,
        dy=dy,
        pad=approximator.pad,
        parameters=parameters,
    )


def main_function(
    data: np.ndarray,
    order: int,
    spacing: int,
    algorithm: int = 3,
    *,
    smoothing_sigma: float = 0.0,
    tol: float = 1e-3,
    max_iter: int = 15,
    fuzzy: bool = True,
    init_sigma: float | None = None,
    normalization: str = "L2-Hys",
    robust_hog: bool = True,
    robust_weighting: str = "huber",
    robust_delta: float = 2.0,
    robust_window: int = 5,
    visualization_percentile: float = 95.0,
    verbose: bool = True,
):
    """Backward-compatible proposed Taylor-gradient + HOG pipeline."""
    gradient = estimate_taylor_gradient(
        data,
        order,
        spacing,
        algorithm,
        smoothing_sigma=smoothing_sigma,
        tol=tol,
        max_iter=max_iter,
        init_sigma=init_sigma,
        verbose=verbose,
    )
    hog = describe_hog(
        gradient.dx,
        gradient.dy,
        fuzzy=fuzzy,
        normalization=normalization,
        robust=robust_hog,
        robust_weighting=robust_weighting,
        robust_delta=robust_delta,
        robust_window=robust_window,
        visualization_percentile=visualization_percentile,
    )
    params = {
        **gradient.parameters,
        "fuzzy_hog": fuzzy,
        "hog_normalization": normalization,
        "robust_hog": robust_hog,
        "robust_weighting": robust_weighting if robust_hog else "none",
        "robust_delta": robust_delta if robust_hog else None,
        "robust_window": robust_window if robust_hog else None,
        "visualization_percentile": visualization_percentile,
    }
    return (
        hog.visualization,
        gradient.magnitude,
        gradient.dx,
        gradient.dy,
        gradient.pad,
        params,
    )


def sobel_gradient(image: np.ndarray) -> GradientResult:
    """Return Sobel derivatives in per-pixel derivative units."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise ValueError("image must be a finite, non-empty grayscale array.")
    # The 3x3 Sobel derivative has a gain of 8 on a linear ramp.
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    return GradientResult(
        magnitude=np.hypot(dx, dy),
        dx=dx,
        dy=dy,
        pad=0,
        parameters={"method": "3x3 Sobel", "gain_correction": 8.0},
    )


def gradient_rmse(
    dx: np.ndarray,
    dy: np.ndarray,
    reference_dx: np.ndarray,
    reference_dy: np.ndarray,
) -> float:
    """sqrt(mean((dx-dx_ref)^2 + (dy-dy_ref)^2))."""
    arrays = [
        np.asarray(value, dtype=float)
        for value in (dx, dy, reference_dx, reference_dy)
    ]
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("All gradient arrays must have identical shapes.")
    finite = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    if not finite.any():
        raise ValueError("No finite pixels are available for RMSE.")
    squared_error = (arrays[0] - arrays[2]) ** 2 + (
        arrays[1] - arrays[3]
    ) ** 2
    return float(np.sqrt(np.mean(squared_error[finite])))


def hog_cosine_similarity(
    descriptor: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Cosine similarity of two flattened, identically shaped HOG tensors."""
    first = np.asarray(descriptor, dtype=float)
    second = np.asarray(reference, dtype=float)
    if first.shape != second.shape:
        raise ValueError("HOG descriptors must have identical shapes.")
    first = first.ravel()
    second = second.ravel()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= np.finfo(float).eps:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.dot(first, second) / denominator)


def hog_relative_error(
    descriptor: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Relative L2 error ||H-H_clean||_2 / (||H_clean||_2 + eps)."""
    first = np.asarray(descriptor, dtype=float)
    second = np.asarray(reference, dtype=float)
    if first.shape != second.shape:
        raise ValueError("HOG descriptors must have identical shapes.")
    return float(
        np.linalg.norm(first.ravel() - second.ravel())
        / (np.linalg.norm(second.ravel()) + np.finfo(float).eps)
    )


def _crop_by_pad(image: np.ndarray, pad: int) -> np.ndarray:
    return image[pad:-pad, pad:-pad] if pad else image


def add_noise(
    image: np.ndarray,
    noise_type: str | None,
    noise_level: float,
    seed: int,
) -> np.ndarray:
    """Compatibility helper for Gaussian-standard-deviation or impulse noise."""
    rng = np.random.default_rng(seed)
    noisy = np.asarray(image, dtype=float).copy()
    kind = None if noise_type is None else str(noise_type).strip().lower()

    if kind in (None, "", "none"):
        return noisy
    if kind == "gaussian":
        if noise_level < 0:
            raise ValueError("Gaussian noise level must be non-negative.")
        return noisy + rng.normal(0.0, noise_level, noisy.shape)
    if kind == "saltpepper":
        if not 0 <= noise_level <= 100:
            raise ValueError("Salt-and-pepper density must be in [0, 100].")
        count = int(round(noise_level / 100.0 * noisy.size))
        selected = rng.choice(noisy.size, size=count, replace=False)
        split = count // 2
        flat = noisy.ravel()
        flat[selected[:split]] = 0.0
        flat[selected[split:]] = 255.0
        return noisy
    raise ValueError("noise_type must be None, 'gaussian' or 'saltpepper'.")


def add_gaussian_noise_snr(
    image: np.ndarray,
    snr_db: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Add zero-mean Gaussian noise at a target contrast SNR in dB.

    Signal power is computed after subtracting the image mean.  Values are not
    clipped, so the requested SNR is preserved for the numerical experiment.
    """
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")
    image = np.asarray(image, dtype=float)
    centred = image - float(np.mean(image))
    signal_power = float(np.mean(centred**2))
    if signal_power <= np.finfo(float).eps:
        raise ValueError("Cannot define contrast SNR for a constant image.")
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), image.shape)
    noisy = image + noise
    achieved = 10.0 * np.log10(signal_power / float(np.mean(noise**2)))
    return noisy, float(achieved)


def evaluate_three_hog_methods(
    clean: np.ndarray,
    noisy: np.ndarray,
    *,
    order: int = 16,
    spacing: int = 1,
    algorithm: int = 3,
    smoothing_sigma: float = 0.0,
    tol: float = 1e-3,
    max_iter: int = 15,
    init_sigma: float | None = None,
    fuzzy: bool = True,
    normalization: str = "L2-Hys",
    robust_weighting: str = "huber",
    robust_delta: float = 2.0,
    robust_window: int = 5,
    visualization_percentile: float = 95.0,
    clean_taylor: GradientResult | None = None,
    noisy_taylor: GradientResult | None = None,
) -> dict:
    """Fair A/B/C evaluation using one shared HOG implementation.

    A: Sobel gradient + standard HOG
    B: Taylor/Maclaurin gradient + standard HOG
    C: Taylor/Maclaurin gradient + robust HOG
    """
    clean = np.asarray(clean, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    if clean.shape != noisy.shape:
        raise ValueError("clean and noisy images must have identical shapes.")

    if clean_taylor is None:
        clean_taylor = estimate_taylor_gradient(
            clean,
            order,
            spacing,
            algorithm,
            smoothing_sigma=smoothing_sigma,
            tol=tol,
            max_iter=max_iter,
            init_sigma=init_sigma,
        )
    if noisy_taylor is None:
        noisy_taylor = estimate_taylor_gradient(
            noisy,
            order,
            spacing,
            algorithm,
            smoothing_sigma=smoothing_sigma,
            tol=tol,
            max_iter=max_iter,
            init_sigma=init_sigma,
        )
    if clean_taylor.pad != noisy_taylor.pad:
        raise RuntimeError("Clean and noisy Taylor gradients have different pads.")

    clean_sobel_full = sobel_gradient(clean)
    noisy_sobel_full = sobel_gradient(noisy)
    pad = clean_taylor.pad
    clean_sobel = GradientResult(
        magnitude=_crop_by_pad(clean_sobel_full.magnitude, pad),
        dx=_crop_by_pad(clean_sobel_full.dx, pad),
        dy=_crop_by_pad(clean_sobel_full.dy, pad),
        pad=pad,
        parameters=clean_sobel_full.parameters,
    )
    noisy_sobel = GradientResult(
        magnitude=_crop_by_pad(noisy_sobel_full.magnitude, pad),
        dx=_crop_by_pad(noisy_sobel_full.dx, pad),
        dy=_crop_by_pad(noisy_sobel_full.dy, pad),
        pad=pad,
        parameters=noisy_sobel_full.parameters,
    )

    common_hog = {
        "fuzzy": fuzzy,
        "normalization": normalization,
        "visualization_percentile": visualization_percentile,
    }
    clean_hogs = {
        "A_sobel_standard": describe_hog(
            clean_sobel.dx, clean_sobel.dy, robust=False, **common_hog
        ),
        "B_taylor_standard": describe_hog(
            clean_taylor.dx, clean_taylor.dy, robust=False, **common_hog
        ),
        "C_taylor_robust": describe_hog(
            clean_taylor.dx,
            clean_taylor.dy,
            robust=True,
            robust_weighting=robust_weighting,
            robust_delta=robust_delta,
            robust_window=robust_window,
            **common_hog,
        ),
    }
    noisy_hogs = {
        "A_sobel_standard": describe_hog(
            noisy_sobel.dx, noisy_sobel.dy, robust=False, **common_hog
        ),
        "B_taylor_standard": describe_hog(
            noisy_taylor.dx, noisy_taylor.dy, robust=False, **common_hog
        ),
        "C_taylor_robust": describe_hog(
            noisy_taylor.dx,
            noisy_taylor.dy,
            robust=True,
            robust_weighting=robust_weighting,
            robust_delta=robust_delta,
            robust_window=robust_window,
            **common_hog,
        ),
    }

    gradient_metrics = {
        "sobel_rmse": gradient_rmse(
            noisy_sobel.dx,
            noisy_sobel.dy,
            clean_sobel.dx,
            clean_sobel.dy,
        ),
        "taylor_rmse": gradient_rmse(
            noisy_taylor.dx,
            noisy_taylor.dy,
            clean_taylor.dx,
            clean_taylor.dy,
        ),
    }
    gradient_metrics["taylor_improvement_factor"] = (
        gradient_metrics["sobel_rmse"]
        / max(gradient_metrics["taylor_rmse"], np.finfo(float).eps)
    )

    hog_metrics = {}
    for method in clean_hogs:
        hog_metrics[method] = {
            "cosine_similarity": hog_cosine_similarity(
                noisy_hogs[method].descriptor,
                clean_hogs[method].descriptor,
            ),
            "relative_error": hog_relative_error(
                noisy_hogs[method].descriptor,
                clean_hogs[method].descriptor,
            ),
            "common_reference_cosine": hog_cosine_similarity(
                noisy_hogs[method].descriptor,
                clean_hogs["B_taylor_standard"].descriptor,
            ),
            "common_reference_relative_error": hog_relative_error(
                noisy_hogs[method].descriptor,
                clean_hogs["B_taylor_standard"].descriptor,
            ),
        }

    robust_standard = hog_metrics["B_taylor_standard"]
    robust_method = hog_metrics["C_taylor_robust"]
    hog_metrics["robust_gain_over_taylor_standard"] = {
        "cosine_similarity_delta": (
            robust_method["cosine_similarity"]
            - robust_standard["cosine_similarity"]
        ),
        "relative_error_reduction": (
            robust_standard["relative_error"] - robust_method["relative_error"]
        )
        / max(robust_standard["relative_error"], np.finfo(float).eps),
    }
    hog_metrics["clean_robust_fidelity_to_taylor_standard"] = {
        "cosine_similarity": hog_cosine_similarity(
            clean_hogs["C_taylor_robust"].descriptor,
            clean_hogs["B_taylor_standard"].descriptor,
        ),
        "relative_error": hog_relative_error(
            clean_hogs["C_taylor_robust"].descriptor,
            clean_hogs["B_taylor_standard"].descriptor,
        ),
    }

    return {
        "gradient_metrics": gradient_metrics,
        "hog_metrics": hog_metrics,
        "clean_gradients": {
            "sobel": clean_sobel,
            "taylor": clean_taylor,
        },
        "noisy_gradients": {
            "sobel": noisy_sobel,
            "taylor": noisy_taylor,
        },
        "clean_hogs": clean_hogs,
        "noisy_hogs": noisy_hogs,
        "parameters": {
            **clean_taylor.parameters,
            "hog_normalization": normalization,
            "fuzzy_hog": fuzzy,
            "robust_weighting": robust_weighting,
            "robust_delta": robust_delta,
            "robust_window": robust_window,
            "visualization_percentile": visualization_percentile,
        },
    }


def _display_scale(image: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=float)
    low, high = np.percentile(finite, [100.0 - percentile, percentile])
    if high <= low:
        return np.zeros(image.shape, dtype=float)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def show_method_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    evaluation: dict,
    *,
    condition_label: str,
    save_path: str | None = None,
    show: bool = True,
):
    """Publication-oriented 3x3 comparison using common HOG rendering."""
    gradient_metrics = evaluation["gradient_metrics"]
    hog_metrics = evaluation["hog_metrics"]
    noisy_gradients = evaluation["noisy_gradients"]
    noisy_hogs = evaluation["noisy_hogs"]
    method_order = [
        "A_sobel_standard",
        "B_taylor_standard",
        "C_taylor_robust",
    ]
    shared_hog_visualizations = dict(
        zip(
            method_order,
            render_hog_group(
                [noisy_hogs[method] for method in method_order],
                percentile=evaluation["parameters"]["visualization_percentile"],
            ),
        )
    )

    figure, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    robust_weights = noisy_hogs["C_taylor_robust"].pixel_weights
    panels = [
        (_display_scale(clean), "Clean image", {}),
        (_display_scale(noisy), f"Noisy image\n{condition_label}", {}),
        (
            robust_weights,
            "Robust pixel weights\n(dark = attenuated)",
            {"vmin": 0.0, "vmax": 1.0},
        ),
        (
            _display_scale(noisy_gradients["sobel"].magnitude),
            f"Sobel magnitude\nRMSE = {gradient_metrics['sobel_rmse']:.4f}",
            {},
        ),
        (
            _display_scale(noisy_gradients["taylor"].magnitude),
            f"Taylor/Maclaurin magnitude\nRMSE = {gradient_metrics['taylor_rmse']:.4f}",
            {},
        ),
        (
            _display_scale(
                np.abs(
                    noisy_gradients["sobel"].magnitude
                    - noisy_gradients["taylor"].magnitude
                )
            ),
            "Absolute magnitude difference",
            {},
        ),
    ]
    for method, title in [
        ("A_sobel_standard", "A. Sobel + Standard HOG"),
        ("B_taylor_standard", "B. Taylor + Standard HOG"),
        ("C_taylor_robust", "C. Taylor + Robust HOG"),
    ]:
        values = hog_metrics[method]
        panels.append(
            (
                shared_hog_visualizations[method],
                f"{title}\ncos = {values['cosine_similarity']:.4f}, "
                f"rel.err = {values['relative_error']:.4f}",
                {},
            )
        )

    for axis, (image, title, image_kwargs) in zip(axes.flat, panels):
        axis.imshow(image, cmap="gray", **image_kwargs)
        axis.set_title(title, fontsize=10)
        axis.axis("off")

    gain = hog_metrics["robust_gain_over_taylor_standard"]
    figure.suptitle(
        "Fair gradient/HOG comparison\n"
        f"Taylor gradient gain: "
        f"{gradient_metrics['taylor_improvement_factor']:.2f}x | "
        f"Robust HOG cosine delta: {gain['cosine_similarity_delta']:+.4f} | "
        f"relative-error reduction: {gain['relative_error_reduction']:+.1%}",
        fontsize=14,
        fontweight="bold",
    )
    if save_path:
        figure.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"Comparison figure saved to: {save_path}")
    if show and "agg" not in str(plt.get_backend()).lower():
        plt.show()
    else:
        plt.close(figure)


def save_metrics(path: str | Path, metadata: dict):
    with open(path, "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, ensure_ascii=False)
    print(f"Reproducible metrics saved to: {path}")


def _serializable_evaluation(evaluation: dict) -> dict:
    return {
        "gradient_metrics": evaluation["gradient_metrics"],
        "hog_metrics": evaluation["hog_metrics"],
        "parameters": evaluation["parameters"],
        "metric_definitions": {
            "gradient_rmse": (
                "sqrt(mean((dx_noisy-dx_clean)^2 + "
                "(dy_noisy-dy_clean)^2))"
            ),
            "hog_cosine_similarity": (
                "dot(H_noisy,H_clean)/(||H_noisy||_2 ||H_clean||_2)"
            ),
            "hog_relative_error": (
                "||H_noisy-H_clean||_2 / ||H_clean||_2"
            ),
        },
    }


if __name__ == "__main__":
    IMAGE_PATH = "Images/cameraman.png"
    NOISE_TYPE = "saltpepper"
    NOISE_LEVEL = 10
    RANDOM_SEED = 42

    ALGORITHM = 3
    NEIGHBOUR_SAMPLES = 16
    SPACING = 1
    SMOOTHING_SIGMA = 0.0
    TOL = 1e-3
    MAX_ITER = 15
    INIT_SIGMA = None

    HOG_NORMALIZATION = "L2-Hys"
    ROBUST_WEIGHTING = "huber"
    ROBUST_DELTA = 2.0
    ROBUST_WINDOW = 5
    VISUALIZATION_PERCENTILE = 95.0

    FIGURE_PATH = "pipeline_comparison.png"
    METRICS_PATH = "pipeline_metrics.json"

    image_path = Path(IMAGE_PATH)
    original_uint8 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if original_uint8 is None:
        raise FileNotFoundError(f"Could not read grayscale image: {image_path}")
    original = original_uint8.astype(float)
    noisy = add_noise(original, NOISE_TYPE, NOISE_LEVEL, RANDOM_SEED)

    start = time.perf_counter()
    evaluation = evaluate_three_hog_methods(
        original,
        noisy,
        order=NEIGHBOUR_SAMPLES,
        spacing=SPACING,
        algorithm=ALGORITHM,
        smoothing_sigma=SMOOTHING_SIGMA,
        tol=TOL,
        max_iter=MAX_ITER,
        init_sigma=INIT_SIGMA,
        normalization=HOG_NORMALIZATION,
        robust_weighting=ROBUST_WEIGHTING,
        robust_delta=ROBUST_DELTA,
        robust_window=ROBUST_WINDOW,
        visualization_percentile=VISUALIZATION_PERCENTILE,
    )
    elapsed = time.perf_counter() - start

    report = {
        "image_path": IMAGE_PATH,
        "noise_type": NOISE_TYPE,
        "noise_level": NOISE_LEVEL,
        "random_seed": RANDOM_SEED,
        **_serializable_evaluation(evaluation),
        "elapsed_seconds": elapsed,
    }
    print_params("[Gradient metrics]", evaluation["gradient_metrics"])
    for method, values in evaluation["hog_metrics"].items():
        print_params(f"[HOG metrics: {method}]", values)
    save_metrics(METRICS_PATH, report)

    show_method_comparison(
        original,
        noisy,
        evaluation,
        condition_label=f"{NOISE_TYPE}, density={NOISE_LEVEL}%",
        save_path=FIGURE_PATH,
    )
