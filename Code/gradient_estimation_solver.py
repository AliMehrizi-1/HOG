from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from numpy.linalg import pinv


def der_column(matrix: List[List[np.ndarray]] | np.ndarray, idx: int, pad: int = 0) -> np.ndarray:
    """Extract one coefficient from a rectangular derivative field.

    ``pad`` is retained for backward compatibility.  The field is already
    cropped by the estimator, so no additional padding is applied here.
    """
    del pad
    field = np.asarray(matrix, dtype=float)
    if field.ndim != 3 or field.shape[0] == 0 or field.shape[1] == 0:
        raise ValueError("The derivative field must be a non-empty HxWxC array.")
    if not 0 <= idx < field.shape[2]:
        raise IndexError(f"Derivative coefficient index {idx} is out of range.")
    return field[..., idx]


def normalise_0_255(img: np.ndarray) -> np.ndarray:
    """Safely map a finite numeric image to uint8."""
    img = np.asarray(img, dtype=float)
    if img.size == 0:
        raise ValueError("Cannot normalise an empty image.")
    if not np.isfinite(img).all():
        raise ValueError("Image contains NaN or infinite values.")

    mi, ma = float(img.min()), float(img.max())
    if ma - mi <= np.finfo(float).eps:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.rint((img - mi) / (ma - mi) * 255.0).astype(np.uint8)


def solve_alg1(d_vec: np.ndarray, A_inv: np.ndarray) -> np.ndarray:
    """Algorithm I: direct least squares."""
    return A_inv @ d_vec


def solve_alg2(
    d_vec: np.ndarray,
    A: np.ndarray,
    A_inv: np.ndarray | None = None,
    C: np.ndarray | None = None,
    C_inv: np.ndarray | None = None,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> np.ndarray:
    """Algorithm II: iterative least squares with Nesterov acceleration.

    Column scaling avoids the poor conditioning caused by mixing first- and
    second-order Taylor terms.  Early stopping also provides mild
    regularisation on noisy data.
    """
    del A_inv, C, C_inv
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    column_scale = np.linalg.norm(A, axis=0)
    column_scale[column_scale == 0] = 1.0
    design = A / column_scale

    theta = np.zeros(A.shape[1], dtype=float)
    momentum_point = theta.copy()
    momentum_weight = 1.0
    lipschitz = max(float(np.linalg.norm(design, ord=2) ** 2), 1e-12)

    for _ in range(max_iter):
        gradient = design.T @ (design @ momentum_point - d_vec)
        theta_new = momentum_point - gradient / lipschitz

        if np.linalg.norm(theta_new - theta) <= tol * (1.0 + np.linalg.norm(theta)):
            theta = theta_new
            break

        weight_new = (1.0 + math.sqrt(1.0 + 4.0 * momentum_weight**2)) / 2.0
        momentum_point = theta_new + (
            (momentum_weight - 1.0) / weight_new
        ) * (theta_new - theta)
        theta = theta_new
        momentum_weight = weight_new

    return theta / column_scale


def _robust_bandwidth(residual: np.ndarray, data: np.ndarray) -> float:
    """Estimate a robust correntropy bandwidth in image-intensity units."""
    centred = residual - np.median(residual)
    mad_scale = 1.4826 * float(np.median(np.abs(centred)))
    numeric_floor = 1e-3 * max(float(np.ptp(data)), 1.0)
    return max(2.5 * mad_scale, numeric_floor, 1e-6)


def solve_alg3(
    d_vec: np.ndarray,
    A: np.ndarray,
    A_inv: np.ndarray,
    sigma: float | None,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> Tuple[np.ndarray, float]:
    """Algorithm III: maximum-correntropy IRLS.

    The bandwidth is fixed during one local solve to prevent it from
    collapsing to zero.  When ``sigma`` is None it is estimated from the
    initial least-squares residual using MAD.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    beta = A_inv @ d_vec
    initial_residual = d_vec - A @ beta
    if sigma is None:
        sigma_used = _robust_bandwidth(initial_residual, d_vec)
    else:
        sigma_used = float(sigma)
        if not np.isfinite(sigma_used) or sigma_used <= 0:
            raise ValueError("sigma must be a positive finite number or None.")

    for _ in range(max_iter):
        residual = d_vec - A @ beta
        exponent = np.clip(-0.5 * (residual / sigma_used) ** 2, -700.0, 0.0)
        weights = np.maximum(np.exp(exponent), 1e-8)

        sqrt_weights = np.sqrt(weights)
        weighted_A = sqrt_weights[:, None] * A
        weighted_d = sqrt_weights * d_vec
        beta_new = pinv(weighted_A, rcond=1e-10) @ weighted_d

        if np.linalg.norm(beta_new - beta) <= tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new

    return beta, sigma_used


class MaclaurinApproximator:
    """Estimate a 2-D local polynomial from symmetric neighbour samples.

    ``order`` is kept as the public parameter name for compatibility, but it
    denotes the number of neighbour samples (n), not the Taylor degree.
    Values 8, 16 and 32 correspond to one, two and four symmetric rings.
    A quadratic expansion has five unknown coefficients and therefore makes
    the usual configurations overdetermined.
    """

    def __init__(self, order: int, spacing: int, degree: int = 2):
        if not isinstance(order, (int, np.integer)) or order < 8 or order % 8:
            raise ValueError("order must be a positive multiple of 8 (for example 8, 16, 32).")
        if not isinstance(spacing, (int, np.integer)) or spacing < 1:
            raise ValueError("spacing must be a positive integer.")
        if degree != 2:
            raise ValueError("This estimator currently supports a quadratic Taylor expansion only.")

        self.order = int(order)
        self.sample_count = int(order)
        self.spacing = int(spacing)
        self.degree = degree

        self.pts_fwd = self._generate_offsets()
        # Retained as aliases for compatibility with older callers.
        self.pts_rev = list(reversed(self.pts_fwd))
        self.A = self._build_matrix(self.pts_fwd)
        self.C = self._build_matrix(self.pts_rev)
        self.A_inv = pinv(self.A, rcond=1e-10)
        self.C_inv = pinv(self.C, rcond=1e-10)
        self.pad = max(max(abs(x), abs(y)) for x, y in self.pts_fwd)

    def _generate_offsets(self) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        rings = self.sample_count // 8
        for radius in range(1, rings + 1):
            offset = radius * self.spacing
            points.extend(
                [
                    (offset, 0),
                    (-offset, 0),
                    (0, offset),
                    (0, -offset),
                    (offset, offset),
                    (-offset, -offset),
                    (offset, -offset),
                    (-offset, offset),
                ]
            )
        return points

    @staticmethod
    def _maclaurin_terms_row(x: float, y: float) -> List[float]:
        # [fx, fy, fxx, fxy, fyy], including Taylor factorial factors.
        return [x, y, x * x / 2.0, x * y, y * y / 2.0]

    def _build_matrix(self, points: List[Tuple[int, int]]) -> np.ndarray:
        return np.asarray(
            [self._maclaurin_terms_row(x, y) for x, y in points],
            dtype=float,
        )

    @staticmethod
    def _difference_vector(
        data: np.ndarray,
        row: int,
        col: int,
        points: List[Tuple[int, int]],
    ) -> np.ndarray:
        centre = float(data[row, col])
        # Points use Cartesian (x, y); array indexing uses [row + y, col + x].
        return np.asarray(
            [float(data[row + y, col + x]) - centre for x, y in points],
            dtype=float,
        )

    def approximate_initial_field(
        self,
        data: np.ndarray,
        algorithm: int,
        tol: float,
        max_iter: int,
        init_sigma: float | None = None,
    ) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must be a two-dimensional grayscale image.")
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or infinite values.")
        if algorithm not in (1, 2, 3):
            raise ValueError("algorithm must be 1, 2 or 3.")
        if tol <= 0 or not np.isfinite(tol):
            raise ValueError("tol must be a positive finite number.")
        if min(data.shape) <= 2 * self.pad:
            raise ValueError(
                f"Image shape {data.shape} is too small for pad={self.pad}."
            )

        valid_height = data.shape[0] - 2 * self.pad
        valid_width = data.shape[1] - 2 * self.pad
        centre = data[
            self.pad : data.shape[0] - self.pad,
            self.pad : data.shape[1] - self.pad,
        ]
        differences = np.stack(
            [
                data[
                    self.pad + y : self.pad + y + valid_height,
                    self.pad + x : self.pad + x + valid_width,
                ]
                - centre
                for x, y in self.pts_fwd
            ],
            axis=-1,
        )
        flat_differences = differences.reshape(-1, self.sample_count)

        if algorithm == 1:
            flat_beta = flat_differences @ self.A_inv.T
        elif algorithm == 2:
            flat_beta = self._solve_iterative_batch(
                flat_differences, tol, max_iter
            )
        else:
            flat_beta = self._solve_correntropy_batch(
                flat_differences, init_sigma, tol, max_iter
            )

        return flat_beta.reshape(valid_height, valid_width, self.A.shape[1])

    def _solve_iterative_batch(
        self,
        differences: np.ndarray,
        tol: float,
        max_iter: int,
    ) -> np.ndarray:
        """Vectorised counterpart of Algorithm II for an entire image."""
        column_scale = np.linalg.norm(self.A, axis=0)
        column_scale[column_scale == 0] = 1.0
        design = self.A / column_scale
        lipschitz = max(float(np.linalg.norm(design, ord=2) ** 2), 1e-12)

        theta = np.zeros((differences.shape[0], self.A.shape[1]), dtype=float)
        momentum_point = theta.copy()
        momentum_weight = 1.0

        iterations = 0
        converged = False
        for iteration in range(max_iter):
            iterations = iteration + 1
            residual = momentum_point @ design.T - differences
            theta_new = momentum_point - (residual @ design) / lipschitz
            delta = np.linalg.norm(theta_new - theta, axis=1)
            threshold = tol * (1.0 + np.linalg.norm(theta, axis=1))
            if np.all(delta <= threshold):
                theta = theta_new
                converged = True
                break

            weight_new = (
                1.0 + math.sqrt(1.0 + 4.0 * momentum_weight**2)
            ) / 2.0
            momentum_point = theta_new + (
                (momentum_weight - 1.0) / weight_new
            ) * (theta_new - theta)
            theta = theta_new
            momentum_weight = weight_new

        beta = theta / column_scale
        final_residual = differences - beta @ self.A.T
        self.last_solver_diagnostics = {
            "iterations": iterations,
            "final_residual": float(np.sqrt(np.mean(final_residual**2))),
            "converged": converged,
        }
        return beta

    def _solve_correntropy_batch(
        self,
        differences: np.ndarray,
        sigma: float | None,
        tol: float,
        max_iter: int,
    ) -> np.ndarray:
        """Vectorised maximum-correntropy IRLS for all valid pixels."""
        beta = differences @ self.A_inv.T
        initial_residual = differences - beta @ self.A.T

        if sigma is None:
            residual_median = np.median(initial_residual, axis=1)
            mad = 1.4826 * np.median(
                np.abs(initial_residual - residual_median[:, None]),
                axis=1,
            )
            numeric_floor = 1e-3 * np.maximum(np.ptp(differences, axis=1), 1.0)
            bandwidth = np.maximum.reduce(
                [2.5 * mad, numeric_floor, np.full_like(mad, 1e-6)]
            )
        else:
            sigma_value = float(sigma)
            if not np.isfinite(sigma_value) or sigma_value <= 0:
                raise ValueError("sigma must be a positive finite number or None.")
            bandwidth = np.full(differences.shape[0], sigma_value, dtype=float)

        identity = np.eye(self.A.shape[1], dtype=float)
        regularisation = 1e-10
        iterations = 0
        converged = False
        for iteration in range(max_iter):
            iterations = iteration + 1
            residual = differences - beta @ self.A.T
            exponent = np.clip(
                -0.5 * (residual / bandwidth[:, None]) ** 2,
                -700.0,
                0.0,
            )
            weights = np.maximum(np.exp(exponent), 1e-8)
            normal_matrix = np.einsum(
                "pn,ni,nj->pij", weights, self.A, self.A, optimize=True
            )
            right_hand_side = np.einsum(
                "pn,ni,pn->pi",
                weights,
                self.A,
                differences,
                optimize=True,
            )
            beta_new = np.linalg.solve(
                normal_matrix + regularisation * identity,
                right_hand_side[..., None],
            )[..., 0]

            delta = np.linalg.norm(beta_new - beta, axis=1)
            threshold = tol * (1.0 + np.linalg.norm(beta, axis=1))
            beta = beta_new
            if np.all(delta <= threshold):
                converged = True
                break

        final_residual = differences - beta @ self.A.T
        self.last_solver_diagnostics = {
            "iterations": iterations,
            "final_residual": float(np.sqrt(np.mean(final_residual**2))),
            "converged": converged,
        }
        return beta
