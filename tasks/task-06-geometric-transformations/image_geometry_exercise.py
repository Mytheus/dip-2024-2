# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np
# import cv2

def translate(img: np.ndarray, number_of_shifts: int = 1) -> np.ndarray:
    H,W = img.shape
    translated_img = np.roll(img, number_of_shifts, axis=1)
    translated_img = np.roll(translated_img, number_of_shifts, axis=0)
    
    translated_img[:number_of_shifts] = np.zeros_like(translated_img[number_of_shifts])
    translated_img[:,:number_of_shifts] = np.zeros((translated_img.shape[0], number_of_shifts))

    return translated_img
def rotate(img: np.ndarray) -> np.ndarray:
    rotated_img = np.rot90(img, k=1, axes=(1,0))
    return rotated_img

def stretch(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    H,W = img.shape
    new_W = int(W * scale)
    
    x_indices = np.linspace(0, W - 1, new_W)
    x_indices = np.round(x_indices).astype(int)
    
    stretched_img = img[:, x_indices]
    return stretched_img

def mirror(img: np.ndarray) -> np.ndarray:
    # mImgP1 = img[:,:int(W/2)]
    # mImgP2 = img[:,int(W/2):W]
    mirrored_img = np.flip(img, axis=1)
    return mirrored_img

def distort(img: np.ndarray, k: float = -0.02) -> np.ndarray:
    H,W = img.shape
    distorted_img = np.zeros_like(img, dtype=np.float32)  
    
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xv, yv = np.meshgrid(x, y)  

    k=-0.2
    
    r = np.sqrt(xv**2 + yv**2)

    
    r_prime = r * (1 + k * r**2)

    
    mask = r > 0
    xv[mask] = (xv[mask] / r[mask]) * r_prime[mask]  
    yv[mask] = (yv[mask] / r[mask]) * r_prime[mask]  

    
    xv = ((xv + 1) * 0.5 * (W - 1)).clip(0, W - 1)
    yv = ((yv + 1) * 0.5 * (H - 1)).clip(0, H - 1)

    
    x0 = np.floor(xv).astype(int)  
    y0 = np.floor(yv).astype(int)  
    x1 = np.clip(x0 + 1, 0, W - 1)  
    y1 = np.clip(y0 + 1, 0, H - 1)  

    
    Ia = img[y0, x0]  
    Ib = img[y1, x0]  
    Ic = img[y0, x1]  
    Id = img[y1, x1]  

    
    wa = (x1 - xv) * (y1 - yv)  
    wb = (x1 - xv) * (yv - y0)  
    wc = (xv - x0) * (y1 - yv)  
    wd = (xv - x0) * (yv - y0)  

    
    distorted_img = (wa * Ia + wb * Ib + wc * Ic + wd * Id).astype(img.dtype)
    return distorted_img

def apply_geometric_transformations(img: np.ndarray) -> dict:
    # Your implementation here
    
    return {
        "translated": translate(img),
        "rotated": rotate(img),
        "stretched": stretch(img),
        "mirrored": mirror(img),
        "distorted": distort(img),
    }

# i1 = cv2.imread('../../img/lena.png', cv2.IMREAD_GRAYSCALE)
# images = apply_geometric_transformations(i1)
# translatedImg = images["translated"]
# rotatedImg = images["rotated"]
# mirroredImg = images["mirrored"]
# stretchedImg = images["stretched"]
# distortedImg = images["distorted"]

# cv2.imshow('image', distortedImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()