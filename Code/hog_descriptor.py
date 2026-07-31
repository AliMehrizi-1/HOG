from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from skimage.draw import line as _draw_line

from gradient_estimation_solver import der_column


@dataclass(frozen=True)
class HOGResult:
    """All products derived from one pair of gradient components."""

    descriptor: np.ndarray
    cell_histogram: np.ndarray
    visualization: np.ndarray
    magnitude: np.ndarray
    weighted_magnitude: np.ndarray
    pixel_weights: np.ndarray
    normalization: str
    robust_weighting: str


class HistogramImage:
    """HOG computed from externally supplied gradients.

    Standard and robust variants share the same cell histogram, orientation
    interpolation, block geometry and normalization.  The robust variant only
    changes the magnitude vote through an orientation-preserving pixel weight.
    """

    CELL = 8
    ORI = 9
    BLOCK_SIZE = 2
    BIN_EDGES = np.linspace(0.0, np.pi, ORI + 1)
    BIN_CENTRE = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
    BIN_WIDTH = np.pi / ORI
    L2_HYS_CLIP = 0.2
    EPSILON = 1e-5

    @staticmethod
    def _validate_gradients(dx: np.ndarray, dy: np.ndarray):
        dx = np.asarray(dx, dtype=float)
        dy = np.asarray(dy, dtype=float)
        if dx.ndim != 2 or dy.ndim != 2 or dx.shape != dy.shape:
            raise ValueError("dx and dy must be same-size two-dimensional arrays.")
        if dx.size == 0:
            raise ValueError("Gradient arrays must not be empty.")
        if not np.isfinite(dx).all() or not np.isfinite(dy).all():
            raise ValueError("Gradient arrays contain NaN or infinite values.")
        return dx, dy

    @staticmethod
    def _crop_to_cells(mag: np.ndarray, ang: np.ndarray):
        if mag.ndim != 2 or ang.ndim != 2 or mag.shape != ang.shape:
            raise ValueError("Magnitude and angle must be same-size 2-D arrays.")
        cell_rows = mag.shape[0] // HistogramImage.CELL
        cell_cols = mag.shape[1] // HistogramImage.CELL
        if cell_rows < HistogramImage.BLOCK_SIZE or cell_cols < HistogramImage.BLOCK_SIZE:
            minimum = HistogramImage.CELL * HistogramImage.BLOCK_SIZE
            raise ValueError(
                f"Gradient image must be at least {minimum}x{minimum} pixels "
                "for a 2x2-cell HOG block."
            )
        height = cell_rows * HistogramImage.CELL
        width = cell_cols * HistogramImage.CELL
        return (
            mag[:height, :width],
            ang[:height, :width],
            cell_rows,
            cell_cols,
        )

    @staticmethod
    def _sum_cells(votes: np.ndarray, cell_rows: int, cell_cols: int) -> np.ndarray:
        return votes.reshape(
            cell_rows,
            HistogramImage.CELL,
            cell_cols,
            HistogramImage.CELL,
        ).sum(axis=(1, 3))

    @staticmethod
    def _hog_cells_hard(mag: np.ndarray, ang: np.ndarray):
        mag, ang, cell_rows, cell_cols = HistogramImage._crop_to_cells(mag, ang)
        histogram = np.zeros(
            (cell_rows, cell_cols, HistogramImage.ORI),
            dtype=float,
        )
        bins = np.floor(ang / HistogramImage.BIN_WIDTH).astype(int)
        bins %= HistogramImage.ORI

        for bin_idx in range(HistogramImage.ORI):
            votes = np.where(bins == bin_idx, mag, 0.0)
            histogram[..., bin_idx] = HistogramImage._sum_cells(
                votes, cell_rows, cell_cols
            )
        return histogram

    @staticmethod
    def _hog_cells_fuzzy(mag: np.ndarray, ang: np.ndarray):
        mag, ang, cell_rows, cell_cols = HistogramImage._crop_to_cells(mag, ang)
        histogram = np.zeros(
            (cell_rows, cell_cols, HistogramImage.ORI),
            dtype=float,
        )

        # Linear interpolation around bin centres with unsigned 0/pi wrap.
        position = ang / HistogramImage.BIN_WIDTH - 0.5
        lower_unwrapped = np.floor(position).astype(int)
        right_weight = position - lower_unwrapped
        left_weight = 1.0 - right_weight
        left = lower_unwrapped % HistogramImage.ORI
        right = (left + 1) % HistogramImage.ORI

        cell_y = np.arange(mag.shape[0]) // HistogramImage.CELL
        cell_x = np.arange(mag.shape[1]) // HistogramImage.CELL
        grid_y, grid_x = np.meshgrid(cell_y, cell_x, indexing="ij")
        flat_cell = (grid_y * cell_cols + grid_x).ravel()
        flat_histogram = histogram.ravel()

        np.add.at(
            flat_histogram,
            flat_cell * HistogramImage.ORI + left.ravel(),
            mag.ravel() * left_weight.ravel(),
        )
        np.add.at(
            flat_histogram,
            flat_cell * HistogramImage.ORI + right.ravel(),
            mag.ravel() * right_weight.ravel(),
        )
        return histogram

    @staticmethod
    def robust_pixel_weights(
        magnitude: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        *,
        method: str = "huber",
        delta: float = 2.0,
        window_size: int = 5,
        min_weight: float = 0.05,
    ) -> np.ndarray:
        """Return orientation-preserving weights for impulsive gradients.

        A local component-wise median provides the expected gradient.  Huber
        or correntropy weights are then computed from the vector deviation.
        Small and locally coherent gradients receive unit weight; only
        inconsistent pixels are attenuated.
        """
        method_key = str(method).strip().lower()
        if method_key not in {"huber", "correntropy"}:
            raise ValueError("robust weighting must be 'huber' or 'correntropy'.")
        if delta <= 0 or not np.isfinite(delta):
            raise ValueError("robust_delta must be a positive finite number.")
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("robust_window must be an odd integer of at least 3.")
        if not 0 <= min_weight <= 1:
            raise ValueError("min_weight must be in [0, 1].")

        local_dx = median_filter(dx, size=window_size, mode="reflect")
        local_dy = median_filter(dy, size=window_size, mode="reflect")
        deviation = np.hypot(dx - local_dx, dy - local_dy)

        local_deviation = median_filter(
            deviation, size=window_size, mode="reflect"
        )
        local_mad = 1.4826 * median_filter(
            np.abs(deviation - local_deviation),
            size=window_size,
            mode="reflect",
        )
        positive_magnitude = magnitude[magnitude > 0]
        reference = (
            float(np.percentile(positive_magnitude, 75.0))
            if positive_magnitude.size
            else 1.0
        )
        scale_floor = max(1e-3 * reference, 1e-6)
        scale = np.maximum(local_mad, scale_floor)
        normalized_deviation = deviation / (delta * scale + 1e-12)

        if method_key == "huber":
            weights = np.ones_like(magnitude)
            outliers = normalized_deviation > 1.0
            weights[outliers] = 1.0 / normalized_deviation[outliers]
        else:
            weights = np.exp(
                np.clip(-0.5 * normalized_deviation**2, -50.0, 0.0)
            )

        # A low-magnitude pixel cannot be an impulsive high-energy vote.  Keep
        # it intact even when its orientation differs from the local median.
        locally_small = magnitude <= np.hypot(local_dx, local_dy) + delta * scale
        weights[locally_small] = 1.0
        return np.clip(weights, min_weight, 1.0)

    @staticmethod
    def _normalise_blocks(
        histogram: np.ndarray,
        normalization: str = "L2-Hys",
    ) -> np.ndarray:
        """Return the conventional block tensor (rows, cols, 2, 2, bins)."""
        normalization_key = str(normalization).strip().upper().replace("_", "-")
        if normalization_key == "L2HYS":
            normalization_key = "L2-HYS"
        if normalization_key not in {"L2", "L2-HYS"}:
            raise ValueError("normalization must be 'L2' or 'L2-Hys'.")

        cell_rows, cell_cols, bins = histogram.shape
        block = HistogramImage.BLOCK_SIZE
        descriptor = np.empty(
            (
                cell_rows - block + 1,
                cell_cols - block + 1,
                block,
                block,
                bins,
            ),
            dtype=float,
        )

        for row in range(cell_rows - block + 1):
            for col in range(cell_cols - block + 1):
                values = histogram[row : row + block, col : col + block].copy()
                values /= np.sqrt(
                    np.sum(values**2) + HistogramImage.EPSILON**2
                )
                if normalization_key == "L2-HYS":
                    values = np.minimum(values, HistogramImage.L2_HYS_CLIP)
                    values /= np.sqrt(
                        np.sum(values**2) + HistogramImage.EPSILON**2
                    )
                descriptor[row, col] = values
        return descriptor

    @staticmethod
    def _hog_canvas(histogram: np.ndarray) -> np.ndarray:
        cell_rows, cell_cols, bin_count = histogram.shape
        height = cell_rows * HistogramImage.CELL
        width = cell_cols * HistogramImage.CELL
        canvas = np.zeros((height, width), dtype=float)
        radius = HistogramImage.CELL // 2 - 1

        for cell_row in range(cell_rows):
            for cell_col in range(cell_cols):
                y0 = cell_row * HistogramImage.CELL + HistogramImage.CELL / 2
                x0 = cell_col * HistogramImage.CELL + HistogramImage.CELL / 2
                for bin_idx in range(bin_count):
                    theta = HistogramImage.BIN_CENTRE[bin_idx]
                    line_dx = radius * np.cos(theta)
                    line_dy = radius * np.sin(theta)
                    y1 = int(round(y0 - line_dy))
                    x1 = int(round(x0 - line_dx))
                    y2 = int(round(y0 + line_dy))
                    x2 = int(round(x0 + line_dx))
                    rr, cc = _draw_line(y1, x1, y2, x2)
                    canvas[rr, cc] += histogram[cell_row, cell_col, bin_idx]
        return canvas

    @staticmethod
    def _scale_canvas(canvas: np.ndarray, scale: float) -> np.ndarray:
        if scale <= np.finfo(float).eps:
            return np.zeros(canvas.shape, dtype=np.uint8)
        return np.rint(np.clip(canvas / scale, 0.0, 1.0) * 255.0).astype(
            np.uint8
        )

    @staticmethod
    def _render_hog(
        histogram: np.ndarray,
        *,
        percentile: float = 95.0,
    ) -> np.ndarray:
        """Render raw cell votes using robust percentile intensity scaling."""
        if not 0 < percentile <= 100:
            raise ValueError("visualization_percentile must be in (0, 100].")

        canvas = HistogramImage._hog_canvas(histogram)
        positive = canvas[canvas > 0]
        if positive.size == 0:
            return np.zeros(canvas.shape, dtype=np.uint8)
        scale = float(np.percentile(positive, percentile))
        return HistogramImage._scale_canvas(canvas, scale)

    def describe(
        self,
        dx: np.ndarray,
        dy: np.ndarray,
        *,
        fuzzy: bool = True,
        normalization: str = "L2-Hys",
        robust: bool = False,
        robust_weighting: str = "huber",
        robust_delta: float = 2.0,
        robust_window: int = 5,
        min_weight: float = 0.05,
        visualization_percentile: float = 95.0,
    ) -> HOGResult:
        dx, dy = self._validate_gradients(dx, dy)
        magnitude = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx) % np.pi

        if robust:
            pixel_weights = self.robust_pixel_weights(
                magnitude,
                dx,
                dy,
                method=robust_weighting,
                delta=robust_delta,
                window_size=robust_window,
                min_weight=min_weight,
            )
        else:
            pixel_weights = np.ones_like(magnitude)

        weighted_magnitude = magnitude * pixel_weights
        histogram = (
            self._hog_cells_fuzzy if fuzzy else self._hog_cells_hard
        )(weighted_magnitude, angle)
        descriptor = self._normalise_blocks(histogram, normalization)
        visualization = self._render_hog(
            histogram,
            percentile=visualization_percentile,
        )
        return HOGResult(
            descriptor=descriptor,
            cell_histogram=histogram,
            visualization=visualization,
            magnitude=magnitude,
            weighted_magnitude=weighted_magnitude,
            pixel_weights=pixel_weights,
            normalization=normalization,
            robust_weighting=robust_weighting if robust else "none",
        )

    def calculate(
        self,
        mag: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        *,
        fuzzy: bool = True,
        normalization: str = "L2-Hys",
        robust: bool = False,
        robust_weighting: str = "huber",
        robust_delta: float = 2.0,
        robust_window: int = 5,
        visualization_percentile: float = 95.0,
    ):
        """Backward-compatible wrapper returning descriptor and visualization."""
        supplied_magnitude = np.asarray(mag, dtype=float)
        result = self.describe(
            dx,
            dy,
            fuzzy=fuzzy,
            normalization=normalization,
            robust=robust,
            robust_weighting=robust_weighting,
            robust_delta=robust_delta,
            robust_window=robust_window,
            visualization_percentile=visualization_percentile,
        )
        if supplied_magnitude.shape != result.magnitude.shape:
            raise ValueError("mag, dx and dy must have identical shapes.")
        return result.descriptor, result.visualization


_histogram_image = HistogramImage()


def describe_hog(
    dx: np.ndarray,
    dy: np.ndarray,
    **kwargs,
) -> HOGResult:
    return _histogram_image.describe(dx, dy, **kwargs)


def render_hog_group(
    results: list[HOGResult],
    *,
    percentile: float = 95.0,
) -> list[np.ndarray]:
    """Render several HOGs with one pooled percentile scale for fair panels."""
    if not results:
        return []
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100].")
    canvases = [
        HistogramImage._hog_canvas(result.cell_histogram) for result in results
    ]
    positive_parts = [canvas[canvas > 0] for canvas in canvases]
    positive_parts = [part for part in positive_parts if part.size]
    if not positive_parts:
        return [np.zeros(canvas.shape, dtype=np.uint8) for canvas in canvases]
    scale = float(np.percentile(np.concatenate(positive_parts), percentile))
    return [HistogramImage._scale_canvas(canvas, scale) for canvas in canvases]


def get_hog(
    grad: list[list[np.ndarray]] | np.ndarray,
    pad: int = 0,
    *,
    fuzzy: bool = True,
    normalization: str = "L2-Hys",
    robust: bool = False,
    robust_weighting: str = "huber",
    robust_delta: float = 2.0,
    robust_window: int = 5,
    visualization_percentile: float = 95.0,
):
    dx = der_column(grad, 0, pad)
    dy = der_column(grad, 1, pad)
    result = _histogram_image.describe(
        dx,
        dy,
        fuzzy=fuzzy,
        normalization=normalization,
        robust=robust,
        robust_weighting=robust_weighting,
        robust_delta=robust_delta,
        robust_window=robust_window,
        visualization_percentile=visualization_percentile,
    )
    return result.visualization, result.magnitude, dx, dy
