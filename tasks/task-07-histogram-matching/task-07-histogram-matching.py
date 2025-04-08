# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    # Your implementation here
    return ski.exposure.match_histograms(source_img, reference_img, channel_axis=-1)

def plot_images(source_img: np.ndarray, reference_img: np.ndarray, matched_img: np.ndarray):
    fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source_img)
    ax1.set_title('Source')
    ax2.imshow(reference_img)
    ax2.set_title('Reference')
    ax3.imshow(matched_img)
    ax3.set_title('Matched')

    plt.tight_layout()
    plt.show()

def plot_histograms(source_img: np.ndarray, reference_img: np.ndarray, matched_img: np.ndarray):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    for i, img in enumerate((source_img, reference_img, matched_img)):
        for c, c_color in enumerate(('red', 'green', 'blue')):
            img_hist, bins = ski.exposure.histogram(img[..., c], source_range='dtype')
            axes[c, i].plot(bins, img_hist / img_hist.max())
            axes[c, 0].set_ylabel(c_color)
    axes[0, 0].set_title('Source')
    axes[0, 1].set_title('Reference')
    axes[0, 2].set_title('Matched')

    plt.tight_layout()
    plt.show()

def histogram_matching():
    source = cv.imread("source.jpg")
    reference = cv.imread("reference.jpg")
    matched = match_histograms_rgb(source, reference)

    plot_images(source, reference, matched)
    plot_histograms(source, reference, matched)