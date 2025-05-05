import cv2
import numpy as np
import matplotlib.pyplot as plt

imgs = [
    "flowers.jpg",
    "gecko.png", 
    "rice.tif",
    "beans.png",
    "blobs.png",
    "chips.png",
    "coffee.png",
    "dowels.tif"
]
path = "../../img/"
images = []
for i in imgs:
    read = cv2.imread(path + i)
    images.append(read)

def segmentImage(image: np.ndarray, subPos = None) -> tuple:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    adaptive = cv2.adaptiveThreshold(gray, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    
    highlight = np.zeros_like(image)
    highlight[adaptive == 255] = (255, 255, 255)
    
    segmented = cv2.addWeighted(image, 1, highlight, 0.5, 0)
    segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)

    return adaptive, segmented
    
plt.figure(figsize=(12, 7))  
plt.suptitle("Image Segmentation", fontsize=16, y=1.05)


for i, img in enumerate(images):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    adaptive, segment = segmentImage(img)
    
    plt.subplot(3, len(images), i + 1)
    plt.imshow(img_rgb)
    plt.title("Original", pad=10)
    plt.axis('off')
    
    plt.subplot(3, len(images), len(images) + i + 1)
    plt.imshow(adaptive, cmap='gray')
    plt.title("Adaptive", pad=10)
    plt.axis('off')
    
    plt.subplot(3, len(images), 2*len(images) + i + 1)
    plt.imshow(segment)
    plt.title("Segmented", pad=10)
    plt.axis('off')

plt.tight_layout(pad=2.0)
plt.show()