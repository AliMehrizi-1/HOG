from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np
from numpy.linalg import pinv


# ───────────────────────────── Utility helpers ───────────────────────────

def der_column(matrix: List[List[np.ndarray]], idx: int, pad: int) -> np.ndarray:
    flat = [pixel[idx] for row in matrix for pixel in row]
    return np.array(flat, float).reshape(len(matrix), len(matrix[0]) if matrix else 0)


def normalise_0_255(img: np.ndarray) -> np.ndarray:
    mi, ma = img.min(), img.max()
    return ((img - mi) / (ma - mi + 1e-9) * 255).astype(np.uint8)


# ─────────────────────────────── Solvers ─────────────────────────────────

def solve_alg1(d_vec: np.ndarray, A_inv: np.ndarray) -> np.ndarray:
    # Algorithm I – direct least-squares
    return A_inv @ d_vec


def solve_alg2(
    d_vec: np.ndarray,
    A: np.ndarray,
    A_inv: np.ndarray,
    C: np.ndarray,           
    C_inv: np.ndarray,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> np.ndarray:
    # Algorithm II – iterative least-squares
    n_cols = A.shape[1]
    first_idx = [0, 1]  # first-order derivatives
    second_idx = [2, 3, 4] if n_cols >= 5 else []
    alpha_idx = list(range(5, n_cols))

    A_fs = A[:, first_idx + second_idx] 
    A_fs_inv = pinv(A_fs)
    A_alpha = A[:, alpha_idx] if alpha_idx else None
    A_alpha_inv = pinv(A_alpha) if alpha_idx else None

    f_vec = np.zeros(A_fs.shape[1])
    alpha = np.zeros(len(alpha_idx)) if alpha_idx else np.empty(0)

    prev_full = np.concatenate([f_vec, alpha])
    for t in range(max_iter):
        # residual excluding current estimates
        res = d_vec - (A_fs @ f_vec) - (A_alpha @ alpha if alpha_idx else 0.0)

        # update low-order part (Eq. 22–23)
        f_new = A_fs_inv @ (d_vec - (A_alpha @ alpha if alpha_idx else 0.0))

        # update high-order part (Eq. 24)
        alpha_new = A_alpha_inv @ (d_vec - A_fs @ f_new) if alpha_idx else alpha

        cur_full = np.concatenate([f_new, alpha_new])

        # Nesterov-style momentum
        if t > 0:
            m = (t - 1) / (t + 2)
            cur_full += m * (cur_full - prev_full)

        # convergence check
        if np.linalg.norm(cur_full - prev_full) < tol:
            f_vec, alpha = cur_full[: len(f_vec)], cur_full[len(f_vec) :]
            break

        f_vec, alpha = cur_full[: len(f_vec)], cur_full[len(f_vec) :]
        prev_full = cur_full.copy()

    beta = np.zeros(n_cols)
    beta[first_idx + second_idx] = f_vec
    if alpha_idx:
        beta[alpha_idx] = alpha
    return beta


# ───────────────────── Algorithm III – single-kernel correntropy ─────────

def solve_alg3(
    d_vec: np.ndarray,
    A: np.ndarray,
    A_inv: np.ndarray,
    sigma: float,             # initial kernel bandwidth σ
    tol: float = 1e-3,
    max_iter: int = 50,
) -> Tuple[np.ndarray, float]:
    """Algorithm III 

    Implements half-quadratic optimization:
      1. Compute residual e = d - A β
      2. Compute weights π_i = exp(-e_i^2 / (2 σ^2))
      3. Solve weighted LS: (Aᵀ Π A) β = Aᵀ Π d
      4. Update σ² ← Σ π_i e_i² / Σ π_i  (Eq. 31)
      5. Optional Nesterov momentum
    """
    sigma = max(sigma, 1e-12)
    beta = A_inv @ d_vec  
    prev = beta.copy()

    for t in range(max_iter):
        res = d_vec - A @ beta
        denom = 2.0 * sigma ** 2
        pi = np.exp(np.clip(-(res ** 2) / denom, -700, 700))  # robustness weights

        # Weighted least-squares step
        Aw = (pi ** 0.5)[:, None] * A
        dw = (pi ** 0.5) * d_vec
        bn = pinv(Aw) @ dw

        # Update bandwidth σ using HQ rule (Eq. 31)
        sigma = math.sqrt(
            max(np.sum(pi * (res ** 2)) / (np.sum(pi) + 1e-12), 1e-12)
        )

        # Nesterov momentum
        if t > 0:
            m = (t - 1) / (t + 2)
            bn += m * (bn - prev)

        # Convergence
        if np.linalg.norm(bn - beta) < tol:
            return bn, sigma

        prev, beta = beta, bn

    return beta, sigma


# ─────────────────────── Maclaurin Approximator ─────────────────────────

class MaclaurinApproximator:
    # Build coefficient matrices 

    def __init__(self, order: int, spacing: int):
        self.order = order
        self.spacing = spacing
        self.pts_fwd = self._generate_offsets(True)
        self.pts_rev = self._generate_offsets(False)
        self.A = self._build_matrix(self.pts_fwd)
        self.C = self._build_matrix(self.pts_rev)
        self.A_inv = np.linalg.inv(self.A) if self.A.shape[0] == self.A.shape[1] else pinv(self.A)
        self.C_inv = np.linalg.inv(self.C) if self.C.shape[0] == self.C.shape[1] else pinv(self.C)

    def _generate_offsets(self, forward: bool) -> List[Tuple[int, int]]:
        # Generate forward/reversed radial+diagonal offsets used for differences.
        mr = max(1, self.order // 8)
        pts: List[Tuple[int, int]] = []
        for r in range(1, mr + 1):
            pts += [
                (r * self.spacing, 0),
                (-r * self.spacing, 0),
                (0, r * self.spacing),
                (0, -r * self.spacing),
            ]
        for r in range(1, mr + 1):
            pts += [
                (r * self.spacing, r * self.spacing),
                (-r * self.spacing, -r * self.spacing),
                (r * self.spacing, -r * self.spacing),
                (-r * self.spacing, r * self.spacing),
            ]
        return pts if forward else list(reversed(pts))

    def _maclaurin_terms_row(self, x: float, y: float) -> List[float]:
        # Construct one row of the Maclaurin coefficient matrix for offset (x, y).
        row: List[float] = []
        for k in range(1, self.order + 1):
            for i in range(k + 1):
                row.append(
                    x ** (k - i)
                    * y ** i
                    / (math.factorial(k - i) * math.factorial(i))
                )
        return row

    def _build_matrix(self, pts: List[Tuple[int, int]]) -> np.ndarray:
        # Build full coefficient matrix from list of offsets.
        return np.array([self._maclaurin_terms_row(x, y) for x, y in pts], float)

    def _difference_vector(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        pts: List[Tuple[int, int]],
    ) -> np.ndarray:
        # Compute difference vector d_vec for pixel (i, j) relative to neighbors.
        c = float(data[i, j])
        return np.array([float(data[i + dx, j + dy]) - c for dx, dy in pts], float)

    def approximate_initial_field(
        self,
        data: np.ndarray,
        algorithm: int,
        tol: float,
        max_iter: int,
        init_sigma: float | None = None,
    ) -> List[List[np.ndarray]]:
        """
        For each valid pixel, estimate derivative coefficients (β) using selected algorithm.

        algorithm: 1,2,3 corresponding to the three methods.
        init_sigma: initial σ for algorithm 3; if None, uses tol * 0.5.
        """
        pad = 2 * self.spacing * self.order
        sigma = init_sigma if init_sigma is not None else tol * 0.5

        field: List[List[np.ndarray]] = []
        for i in range(pad, data.shape[0] - pad):
            row: List[np.ndarray] = []
            for j in range(pad, data.shape[1] - pad):
                d_vec = self._difference_vector(data, i, j, self.pts_rev)
                if algorithm == 1:
                    beta = solve_alg1(d_vec, self.A_inv)
                elif algorithm == 2:
                    beta = solve_alg2(
                        d_vec, self.A, self.A_inv, self.C, self.C_inv, tol, max_iter
                    )
                else:  # algorithm == 3
                    beta, sigma = solve_alg3(
                        d_vec, self.A, self.A_inv, sigma, tol, max_iter
                    )
                row.append(beta / self.spacing)
            field.append(row)
        return field
