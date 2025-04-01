# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np
MAX = 255

def MSE(i1: np.ndarray, i2: np.ndarray) -> float:
    return (np.square(i1-i2)).mean()

def PSNR(i1: np.ndarray, i2: np.ndarray) -> float:
    return 20 * np.log10(MAX) - 10 * np.log10(MSE(i1, i2))

def SSIM(i1: np.ndarray, i2: np.ndarray) -> float:
    mx = i1.mean()
    my = i2.mean()
    vx = i1.var()
    vy = i2.var()
    covMatrix = np.cov(i1.flatten(),i2.flatten())
    cov = covMatrix[0][1]
    k1 = 0.01
    k2 = 0.03
    L = 255
    c1 = np.square(k1*L)
    c2 = np.square(k2*L)
    result = ((2*mx*my + c1)*(2*cov+c2))/((np.square(mx) + np.square(my) + c1)*(vx + vy + c2))
    return result

def NPCC(i1: np.ndarray, i2: np.ndarray) -> float:
    covMatrix = np.cov(i1.flatten(),i2.flatten())
    cov = covMatrix[0][1]
    stdx = np.std(i1)
    stdy = np.std(i2)
    return cov/(stdx*stdy)

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    # Your implementation here
    args = (i1, i2)
    result = {}
    result["mse"] = MSE(args)
    result["psnr"] = PSNR(args)
    result["ssim"] = SSIM(args)
    result["npcc"] = NPCC(args)
    return result



