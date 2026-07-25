import numpy as np
from scipy.ndimage import uniform_filter
from skimage.draw import line as _draw_line
from gradient_estimation_solver import der_column  

class HistogramImage:
    CELL = 8
    ORI  = 9
    BIN_EDGES  = np.linspace(0, np.pi, ORI + 1)
    BIN_CENTRE = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
    BIN_WIDTH  = np.pi / ORI

    @staticmethod
    def _hog_cells_hard(mag: np.ndarray, ang: np.ndarray):
        h, w = mag.shape
        ny, nx = h // HistogramImage.CELL, w // HistogramImage.CELL
        hist = np.zeros((ny, nx, HistogramImage.ORI), float)
        start = HistogramImage.CELL // 2
        for b in range(HistogramImage.ORI):
            mask = (ang >= HistogramImage.BIN_EDGES[b]) & (ang < HistogramImage.BIN_EDGES[b+1])
            vote = np.where(mask, mag, 0)
            sm = uniform_filter(vote, size=HistogramImage.CELL, mode="nearest")
            hist[...,b] = sm[start::HistogramImage.CELL, start::HistogramImage.CELL]
        return hist

    @staticmethod
    def _hog_cells_fuzzy(mag: np.ndarray, ang: np.ndarray):
        h, w = mag.shape
        ny, nx = h // HistogramImage.CELL, w // HistogramImage.CELL
        hist = np.zeros((ny, nx, HistogramImage.ORI), float)

        ang_idx = ang / HistogramImage.BIN_WIDTH
        left  = np.floor(ang_idx).astype(int) % HistogramImage.ORI
        right = (left + 1) % HistogramImage.ORI
        w_r = ang_idx - left
        w_l = 1.0 - w_r

        cell_y = np.arange(h) // HistogramImage.CELL
        cell_x = np.arange(w) // HistogramImage.CELL
        CY, CX  = np.meshgrid(cell_y, cell_x, indexing="ij")

        flat_cell  = (CY * nx + CX).ravel()
        flat_mag   = mag.ravel()
        flat_left  = left.ravel()
        flat_right = right.ravel()

        np.add.at(hist.ravel(),
                  flat_cell*HistogramImage.ORI + flat_left,
                  flat_mag * w_l.ravel())
        np.add.at(hist.ravel(),
                  flat_cell*HistogramImage.ORI + flat_right,
                  flat_mag * w_r.ravel())
        return hist

    @staticmethod
    def _render_hog(hist: np.ndarray):
        ny, nx, nb = hist.shape
        H, W = ny * HistogramImage.CELL, nx * HistogramImage.CELL
        canvas = np.zeros((H, W), float)
        radius = HistogramImage.CELL//2 - 1
        maxm = hist.max() + 1e-9

        for cy in range(ny):
            for cx in range(nx):
                y0 = cy*HistogramImage.CELL + HistogramImage.CELL/2
                x0 = cx*HistogramImage.CELL + HistogramImage.CELL/2
                for b in range(nb):
                    θ = HistogramImage.BIN_CENTRE[b]
                    dx = radius * np.cos(θ)
                    dy = radius * np.sin(θ)
                    y1, x1 = int(round(y0-dy)), int(round(x0-dx))
                    y2, x2 = int(round(y0+dy)), int(round(x0+dx))
                    rr, cc = _draw_line(y1, x1, y2, x2)
                    canvas[rr, cc] += hist[cy, cx, b] / maxm

        canvas = canvas / (canvas.max() + 1e-9) * 255
        return canvas.astype(np.uint8)

    def calculate(self, mag: np.ndarray, dx: np.ndarray, dy: np.ndarray, *, fuzzy: bool=False):
        ang = (np.arctan2(dy, dx) % np.pi)
        hist = (self._hog_cells_fuzzy if fuzzy else self._hog_cells_hard)(mag, ang)
        img  = self._render_hog(hist)
        return hist, img

_histogram_image = HistogramImage()

def get_hog(grad: list[list[np.ndarray]], pad: int, *, fuzzy: bool=False):
    dx = der_column(grad, 0, pad)
    dy = der_column(grad, 1, pad)
    mag = np.hypot(dx, dy)
    _, img = _histogram_image.calculate(mag, dx, dy, fuzzy=fuzzy)
    return img, mag, dx, dy
