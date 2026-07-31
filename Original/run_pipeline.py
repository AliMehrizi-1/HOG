import time
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog as skimage_hog
import matplotlib.pyplot as plt

from gradient_estimation_solver import (
    MaclaurinApproximator, normalise_0_255
)
from hog_descriptor import get_hog

def print_params(header: str, params: dict):
    print(f"\n{header}")
    for k, v in params.items():
        print(f"  {k:<16}= {v}")
    print()

def main_function(
    data: np.ndarray,
    order: int,
    spacing: int,
    algorithm: int = 1,
    *,
    smoothing_sigma: float = 1.0,
    tol: float = 1e-3,
    max_iter: int = 50,
    fuzzy: bool = False,
    init_sigma: float | None = None,
):
    params = {
        "order": order,
        "spacing": spacing,
        "algorithm": algorithm,
        "smoothing_sigma": smoothing_sigma,
        "tol": tol,
        "max_iter": max_iter,
        "fuzzy": fuzzy,
    }
    if algorithm == 3:
        params["init_sigma"] = init_sigma if init_sigma is not None else tol * 0.5

    print_params("[main_function parameters]", params)

    img = cv2.GaussianBlur(data, (3, 3), smoothing_sigma) if smoothing_sigma > 0 else data
    approx = MaclaurinApproximator(order, spacing)
    if algorithm == 3:
        grad_field = approx.approximate_initial_field(
            img,
            algorithm,
            tol,
            max_iter,
            init_sigma=init_sigma,
        )
    else:
        grad_field = approx.approximate_initial_field(
            img,
            algorithm,
            tol,
            max_iter,
        )
    pad = 2 * spacing * order

    hog_img, mac_mag, dx, dy = get_hog(grad_field, pad, fuzzy=fuzzy)
    return hog_img, mac_mag, dx, dy, pad, params

def sobel_gradient(img: np.ndarray):
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = normalise_0_255(np.hypot(dx, dy))
    return mag, dx, dy

def show_diagnostics(original, noisy, mac_mag, sobel_mag, robust_hog, sk_hog, panels):
    titles = {1: "Original", 2: "With noise", 3: "Maclaurin |∇f|",
              4: "Sobel |∇f|", 5: "Robust HOG", 6: "skimage HOG"}
    imgs = {1: original, 2: noisy, 3: mac_mag, 4: sobel_mag, 5: robust_hog, 6: sk_hog}
    cols, rows = 3, int(np.ceil(len(panels)/3))
    plt.figure(figsize=(4*cols,4*rows))
    for idx, key in enumerate(panels, 1):
        plt.subplot(rows, cols, idx)
        plt.title(titles[key])
        plt.imshow(imgs[key], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    IMAGE_PATH       = "Images/cameraman.png"
    NOISE_TYPE       = "None"           # "gaussian", "saltpepper", or None
    NOISE_LEVEL      = 25                    # for gaussian: std; for saltpepper: percent
    algorithm        = 3                     # 1, 2 or 3
    order, spacing   = 8, 1
    smoothing_sigma  = 0
    tol              = 1e-3
    max_iter         = 100
    fuzzy_mode       = True                  
    init_sigma       = 0.01                   
    PANELS           = [1,2,3,4,5,6]

    if Path(IMAGE_PATH).exists():
        original = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE).astype(float)
    else:
        original = np.fromfunction(lambda i,j: np.exp(i+j), (200,200), float)

    noisy = original.copy()
    if NOISE_TYPE=="gaussian":
        noisy += np.random.normal(0, NOISE_LEVEL, noisy.shape)
    elif NOISE_TYPE=="saltpepper":
        p   = NOISE_LEVEL/100/2
        rnd = np.random.choice([0,255,-1], noisy.shape, p=[p,p,1-2*p])
        noisy = np.where(rnd==-1, noisy, rnd)

    t0 = time.time()
    hog_img, mac_mag, _, _, pad, params = main_function(
        noisy, order, spacing, algorithm,
        smoothing_sigma=smoothing_sigma,
        tol=tol,
        max_iter=max_iter,
        fuzzy=fuzzy_mode,
        init_sigma=init_sigma
    )
    print(f"Finished in {time.time()-t0:.2f}s  |  algorithm={algorithm}  |  fuzzy={fuzzy_mode}")

    sobel_mag, _, _ = sobel_gradient(noisy)
    _, sk_hog_img   = skimage_hog(
        noisy.astype("uint8"), orientations=9,
        pixels_per_cell=(8,8), cells_per_block=(2,2),
        visualize=True, feature_vector=False
    )

    show_diagnostics(
        normalise_0_255(original),
        normalise_0_255(noisy),
        normalise_0_255(mac_mag),
        sobel_mag,
        hog_img,
        sk_hog_img,
        PANELS
    )

    print_params("[final parameters]", {
        "IMAGE_PATH": IMAGE_PATH,
        "NOISE_TYPE": NOISE_TYPE,
        "NOISE_LEVEL": NOISE_LEVEL,
        "algorithm": algorithm,
        "order": order,
        "spacing": spacing,
        "smoothing_sigma": smoothing_sigma,
        "tol": tol,
        "max_iter": max_iter,
        "fuzzy_mode": fuzzy_mode,
        "init_sigma": init_sigma,
        "PANELS": PANELS,
    })
