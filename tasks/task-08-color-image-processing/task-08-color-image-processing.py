import numpy as np
import cv2
from matplotlib import pyplot as plt

# 1. Display Color Histograms for RGB Images

# Objective: Calculate and display separate histograms for the R, G, and B channels of a color image.

# Topics: Color histograms, channel separation.

# Challenge: Compare histograms of different images (e.g., nature vs. synthetic images).


def DisplayColorHistograms(source_img: np.ndarray):
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    colors = ('red', 'green', 'blue')
    plt.figure(figsize=(12, 6))
    plt.subplot(2,2,1)
    plt.imshow(source_img)
    plt.title('Original Image')
    for i, color in enumerate(colors):
        plt.subplot(2, 2, i + 2)
        hist = cv2.calcHist([source_img], [i], None, [256], [0, 256])
        # parameters: source image, channel number, mask, number of bins, pixel range
        plt.plot(hist, color=color)
        plt.title(f'{color.capitalize()} Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()

def compareHistograms(source_img1: np.ndarray, source_img2: np.ndarray):
    source_img1 = cv2.cvtColor(source_img1, cv2.COLOR_BGR2RGB)
    source_img2 = cv2.cvtColor(source_img2, cv2.COLOR_BGR2RGB)
    colors = ('red', 'green', 'blue')
    images = [source_img1, source_img2]
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(source_img1)
    plt.title('Image 1')
    plt.subplot(2, 2, 2)
    plt.imshow(source_img2)
    plt.title('Image 2')
    for u in range(0,2,1):
        plt.subplot(2, 2, u + 3)
        for i, color in enumerate(colors):
            hist = cv2.calcHist([images[u]], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=color)
            plt.title(f'Image {u+1}')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.xlim([0, 256])
            plt.legend()   
    plt.tight_layout()
    plt.show()
# Examples images
source_img1 = cv2.imread('../../img/baboon.png')
source_img2 = cv2.imread('../../img/monkey.jpeg')
source_img3 = cv2.imread('../../img/strawberries.tif')
source_img4 = cv2.imread('../../img/lena.png')
source_img5 = cv2.imread('../../img/rgbcube_kBKG.png')
source_img6 = cv2.imread('../../img/rgb.png')
# Display the color histograms
# DisplayColorHistograms(source_img1)
# #Compare two histograms
# compareHistograms(source_img1, source_img2)

# ---

# 2. Visualize Individual Color Channels

# Objective: Extract and display the Red, Green, and Blue channels of a color image as grayscale and pseudo-colored images.

# Topics: Channel separation and visualization.

# Bonus: Reconstruct the original image using the separated channels.

def displayChannels(source_img: np.ndarray):
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    #Separate into 3 channels
    channels = cv2.split(source_img)
    zero_channel = np.zeros(source_img.shape[:2], dtype=np.uint8)
    pseudo_colored_images = []
    colors = ['Red', 'Green', 'Blue']
    #Merge the channels
    pseudo_colored_images.append(cv2.merge([channels[0], zero_channel, zero_channel]))
    pseudo_colored_images.append(cv2.merge([zero_channel, channels[1], zero_channel]))
    pseudo_colored_images.append(cv2.merge([zero_channel, zero_channel, channels[2]]))
    plt.figure(figsize=(10, 6))
    for i, channel in enumerate(channels):
        plt.subplot(4, 2, 2*i + 1)
        plt.imshow(channel, cmap='gray')
        plt.title(f'{colors[i]} channel (Grayscale)')
        plt.axis('off')
        
        plt.subplot(4, 2, 2*i + 2)
        plt.imshow(pseudo_colored_images[i])
        plt.title(f'{colors[i]} channel (Pseudo-colored)')
        plt.axis('off')
    plt.subplot(4, 2, 7)
    plt.imshow(source_img)
    plt.title('Original Image')
    plt.axis('off')
    reconstructed_image = cv2.merge([channels[0], channels[1], channels[2]])
    plt.subplot(4, 2, 8)
    plt.imshow(reconstructed_image)
    plt.title('Reconstructed Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#displayChannels(source_img1)

# ---

# 3. Convert Between Color Spaces (RGB ↔ HSV, LAB, YCrCb, CMYK)

# Objective: Convert an RGB image to other color spaces and display the result.

# Topics: Color space conversion.

# Challenge: Display individual channels from each converted space.

def rgb_to_cmyk(rgb_img):
    #Convert RGB image (0-255) to CMYK (0-1)
    rgb_normalized = rgb_img.astype(np.float32) / 255.0
    
    # Calculate CMY components
    k = 1 - np.max(rgb_normalized, axis=2)
    c = (1 - rgb_normalized[..., 2] - k) / (1 - k + 1e-5)  # Avoid division by zero
    m = (1 - rgb_normalized[..., 1] - k) / (1 - k + 1e-5)
    y = (1 - rgb_normalized[..., 0] - k) / (1 - k + 1e-5)
    
    # Stack channels and clip values
    cmyk = np.stack([c, m, y, k], axis=2)
    return np.clip(cmyk, 0, 1)

def displayEachChannel(source_img: np.ndarray, name:str):
    plt.figure(name,figsize=(12, 6))
    for i in range(source_img.shape[2]):
        plt.subplot(1, source_img.shape[2], i+1)
        plt.imshow(source_img[:, :, i], cmap='gray')
        plt.title(f'Channel {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def convertColorSpaces(source_img:np.ndarray):
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    # Convert 
    hsv_image = cv2.cvtColor(source_img, cv2.COLOR_RGB2HSV)
    lab_image = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB)
    ycrcb_image = cv2.cvtColor(source_img, cv2.COLOR_RGB2YCrCb)
    cmyk_image = rgb_to_cmyk(source_img)

    plt.figure(figsize=(12, 6))
    images = [source_img, hsv_image, lab_image, ycrcb_image, cmyk_image]
    titles = ['Original Image (RGB)', 'Converted Image (HSV)', 'Converted Image (LAB)', 
              'Converted Image (YCrCb)', 'Converted Image (CMYK)']
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(3, 2, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    for i, (img, title) in enumerate(zip(images, titles)):
        displayEachChannel(img, title)

#convertColorSpaces(source_img4)

# ---

# 4. Compare Effects of Blurring in RGB vs HSV

# Objective: Apply Gaussian blur in both RGB and HSV color spaces and compare results.

# Topics: Color space effect on filtering.

# Discussion: Why HSV might preserve color better in some operations.

def blurRGBvsHSV(source_img: np.ndarray, kernel_size=(15,15), sigma= 5):
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    blurred_rgb = cv2.GaussianBlur(source_img, kernel_size, sigma)
    
    hsv_image = cv2.cvtColor(source_img, cv2.COLOR_RGB2HSV)
    hsvChannels = cv2.split(hsv_image)
    blurred_hsv = []
    #It is necessary to blur each channel separately
    for i in range(len(hsvChannels)):
        blurred_hsv.append(cv2.GaussianBlur(hsvChannels[i], kernel_size, sigma))
    blurred_hsv = cv2.merge(blurred_hsv)
    blurred_hsv_to_rgb = cv2.cvtColor(blurred_hsv, cv2.COLOR_HSV2RGB)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 5, 1)
    plt.imshow(source_img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.imshow(blurred_rgb)
    plt.title(f'RGB Blur (σ={sigma})')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(hsv_image)
    plt.title('HSV Image')
    plt.axis('off')

    
    plt.subplot(1, 5, 4)
    plt.imshow(blurred_hsv)
    plt.title(f'HSV Blur (σ={sigma})')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(blurred_hsv_to_rgb)
    plt.title(f'Blurred HSV to RGB (σ={sigma})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#blurRGBvsHSV(source_img3)
# ---

# 5. Apply Edge Detection Filters (Sobel, Laplacian) on Color Images

# Objective: Apply Sobel and Laplacian filters on individual channels and on the grayscale version of the image.

# Topics: Edge detection, spatial filtering.

# Bonus: Merge edge maps from all channels to form a combined result.

def EdgeDetectionFilter(source_img: np.ndarray):
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)

    # Apply Sobel
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_gray = np.sqrt(sobel_x**2 + sobel_y**2)
    laplacian_gray = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)

    channels = cv2.split(source_img)
    channels_names = ['Red', 'Green', 'Blue']

    plt.figure(figsize=(12, 6))
    plt.subplot(4, 3, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.subplot(4, 3, 2)
    plt.imshow(sobel_gray, cmap='gray')
    plt.title('Sobel Edge Detection (Gray)')
    plt.axis('off')

    plt.subplot(4, 3, 3)
    plt.imshow(laplacian_gray, cmap='gray')
    plt.title('Laplacian Edge Detection (Gray)')
    plt.axis('off')

    sobel_channels = []
    laplacian_channels = []

    for i, (channel, name) in enumerate(zip(channels, channels_names)):
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=5)
        sobel_channel = np.sqrt(sobel_x**2 + sobel_y**2)
        laplacian_channel = cv2.Laplacian(channel, cv2.CV_64F, ksize=3)
        sobel_channels.append(sobel_channel)
        laplacian_channels.append(laplacian_channel)
        plt.subplot(4, 3, 4 + i * 3)
        plt.imshow(channel, cmap='gray')
        plt.title(f'{name} Channel')
        plt.axis('off')

        plt.subplot(4, 3, 5 + i * 3)
        plt.imshow(sobel_channel, cmap='gray')
        plt.title(f'Sobel Edge ({name})')
        plt.axis('off')

        plt.subplot(4, 3, 6 + i * 3)
        plt.imshow(laplacian_channel, cmap='gray')
        plt.title(f'Laplacian Edge ({name})')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Bonus
    # Merge Sobel and Laplacian channels
    sobel_combined = cv2.merge(sobel_channels)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    laplacian_combined = cv2.merge(laplacian_channels)
    laplacian_combined = cv2.normalize(laplacian_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(sobel_combined)
    plt.title('Combined Sobel Edges')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(laplacian_combined)
    plt.title('Combined Laplacian Edges')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(np.sum(sobel_combined, axis=2), cmap='gray')
    plt.title('Sum of Sobel Edges')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(np.sum(laplacian_combined, axis=2), cmap='gray')
    plt.title('Sum of Laplacian Edges')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
#EdgeDetectionFilter(source_img6)

# ---

# 6. High-pass and Low-pass Filtering in the Frequency Domain

# Objective: Perform DFT on each channel of a color image, apply high-pass and low-pass masks, then reconstruct the image.

# Topics: Frequency domain filtering, Fourier Transform.

# Tools: cv2.dft(), cv2.idft(), numpy.fft.

def frequencyFiltering(source_img: np.ndarray):
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    channels = cv2.split(source_img)
    filtered_images = []
    titles = ['Original Image', 'Low-pass Filtered', 'High-pass Filtered']

    for channel in channels:
        # DFT
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # masks
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        mask_low = np.zeros((rows, cols, 2), np.uint8)
        mask_high = np.ones((rows, cols, 2), np.uint8)
        radius = 50  # Radius for low-pass filter
        mask_low[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1
        mask_high[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

        dft_low = dft_shift * mask_low
        dft_high = dft_shift * mask_high

        # inverse DFT
        low_pass = cv2.idft(np.fft.ifftshift(dft_low))
        low_pass = cv2.magnitude(low_pass[:, :, 0], low_pass[:, :, 1])
        high_pass = cv2.idft(np.fft.ifftshift(dft_high))
        high_pass = cv2.magnitude(high_pass[:, :, 0], high_pass[:, :, 1])

        low_pass = cv2.normalize(low_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        filtered_images.append((channel, low_pass, high_pass))

    # Reconstruct
    low_pass_reconstructed = cv2.merge([img[1] for img in filtered_images])
    high_pass_reconstructed = cv2.merge([img[2] for img in filtered_images])

    plt.figure(figsize=(12, 6))
    for i, (original, low, high) in enumerate(filtered_images):
        plt.subplot(4, 3, i * 3 + 1)
        plt.imshow(original, cmap='gray')
        plt.title(f'Channel {i + 1} - Original')
        plt.axis('off')

        plt.subplot(4, 3, i * 3 + 2)
        plt.imshow(low, cmap='gray')
        plt.title(f'Channel {i + 1} - Low-pass')
        plt.axis('off')

        plt.subplot(4, 3, i * 3 + 3)
        plt.imshow(high, cmap='gray')
        plt.title(f'Channel {i + 1} - High-pass')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Reconstructed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(source_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(low_pass_reconstructed)
    plt.title('Low-pass Filtered Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(high_pass_reconstructed)
    plt.title('High-pass Filtered Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


#frequencyFiltering(source_img3)

# ---

# 7. Visualize and Manipulate Bit Planes of Color Images

# Objective: Extract bit planes (especially MSB and LSB) from each color channel and display them.

# Topics: Bit slicing, data visualization.

# Challenge: Reconstruct the image using only the top 4 bits of each channel.

def visualizeBitPlanes(source_img: np.ndarray):
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    channels = cv2.split(source_img)
    colors = ['Red', 'Green', 'Blue']

    plt.figure(figsize=(12, 6))
    for c_idx, (channel, color) in enumerate(zip(channels, colors)):
        for bit in range(8):
            # element wise bit operation, shifting left and right to get the bit-position
            # results will be either one or zero
            bit_plane = (channel & (1 << bit)) >> bit
            plt.subplot(len(channels), 8, c_idx * 8 + bit + 1)
            plt.imshow(bit_plane * 255, cmap='gray')
            plt.title(f'{color} - Bit {bit}')
            plt.axis('off')
            
    plt.tight_layout()
    plt.show()

    # Reconstruct using top 4 bits
    reconstructed_channels_top4 = []
    reconstructed_channels_bottom4 = []
    for channel in channels:
        reconstructed_top4 = np.zeros_like(channel, dtype=np.uint8)
        reconstructed_bottom4 = np.zeros_like(channel, dtype=np.uint8)
        for bit in range(8):
            if (bit < 4):
                reconstructed_bottom4 |= (channel & (1 << bit))
            else:
                reconstructed_top4 |= (channel & (1 << bit))
        reconstructed_channels_top4.append(reconstructed_top4)
        reconstructed_channels_bottom4.append(reconstructed_bottom4)

    reconstructed_image_top4 = cv2.merge(reconstructed_channels_top4)
    reconstructed_image_bottom4 = cv2.merge(reconstructed_channels_bottom4)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(reconstructed_image_top4)
    plt.title('Reconstructed Image (Top 4 bits)')
    plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(reconstructed_image_bottom4)
    # plt.title('Reconstructed Image (Bottom 4 bits)')
    # plt.axis('off')
    plt.tight_layout()
    plt.show()


#visualizeBitPlanes(source_img3)

# ---

# 8. Color-based Object Segmentation using HSV Thresholding

# Objective: Convert image to HSV, apply thresholding to extract objects of a certain color (e.g., red apples).

# Topics: Color segmentation, HSV masking.

# Bonus: Use trackbars to adjust thresholds dynamically.


def colorSegmentationWithTrackbars(source_img: np.ndarray):
    def nothing(x):
        pass

    hsv_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow('Segmentation')
    cv2.createTrackbar('H Min', 'Segmentation', 0, 179, nothing)
    cv2.createTrackbar('H Max', 'Segmentation', 179, 179, nothing)
    cv2.createTrackbar('S Min', 'Segmentation', 0, 255, nothing)
    cv2.createTrackbar('S Max', 'Segmentation', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Segmentation', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'Segmentation', 255, 255, nothing)

    while True:
        h_min = cv2.getTrackbarPos('H Min', 'Segmentation')
        h_max = cv2.getTrackbarPos('H Max', 'Segmentation')
        s_min = cv2.getTrackbarPos('S Min', 'Segmentation')
        s_max = cv2.getTrackbarPos('S Max', 'Segmentation')
        v_min = cv2.getTrackbarPos('V Min', 'Segmentation')
        v_max = cv2.getTrackbarPos('V Max', 'Segmentation')

        # Define range
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        result = cv2.bitwise_and(source_img, source_img, mask=mask)

        combined = np.hstack((source_img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result))

        cv2.imshow('Segmentation', combined)
        if cv2.waitKey(1) > -1:
            break

    cv2.destroyAllWindows()

#colorSegmentationWithTrackbars(source_img1)

# ---

# 9. Convert and Visualize Images in the NTSC (YIQ) Color Space

# Objective: Manually convert an RGB image to NTSC (YIQ) and visualize the Y, I, and Q channels.

# Topics: Color space math, visualization.

# Note: OpenCV doesn’t support YIQ directly, so students can implement the conversion using matrices.

def convertToYIQ(source_img: np.ndarray):
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    # Conversion matrix from RGB to YIQ
    rgb_to_yiq_matrix = np.array([[0.299, 0.587, 0.114],
                                   [0.596, -0.274, -0.322],
                                   [0.211, -0.523, 0.312]])
    normalized_img = source_img.astype(np.float64) / 255.0
    yiq_img = np.dot(normalized_img, rgb_to_yiq_matrix.T)

    channels = []
    channels_names = ['Y Channel (Luminance)', 'I Channel (In-phase)', 'Q Channel (Quadrature)']
    for i in range(len(channels_names)):
        channels.append(yiq_img[:, :, i])

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(source_img)
    plt.title('Original Image (RGB)')
    plt.axis('off')

    for i, (channel, title) in enumerate(zip(channels, channels_names)):
        plt.subplot(2, 2, i + 2)
        plt.imshow(channel, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

#convertToYIQ(source_img3)


# ---

# 10. Color Image Enhancement with Histogram Equalization

# Objective: Apply histogram equalization on individual channels in different color spaces (e.g., Y in YCrCb, L in LAB).

# Topics: Contrast enhancement, color models.

# Discussion: Explain why histogram equalization should not be directly applied to RGB.

# Answer: Histogram equalization can't be applied directly to a RGB image because the channels are interdependent,
# modifying them separately can lead to the loss of color balance and introduce artifacts.

def histogramEqualization(source_img: np.ndarray):
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    ycrcb_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2YCrCb)
    lab_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB)

    # Y channel in YCrCb
    ycrcb_channels = list(cv2.split(ycrcb_img))
    ycrcb_channels[0] = cv2.equalizeHist(ycrcb_channels[0])
    equalized_ycrcb = cv2.merge(ycrcb_channels)
    equalized_ycrcb_rgb = cv2.cvtColor(equalized_ycrcb, cv2.COLOR_YCrCb2RGB)

    # L channel in LAB
    lab_channels = list(cv2.split(lab_img))
    lab_channels[0] = cv2.equalizeHist(lab_channels[0])
    equalized_lab = cv2.merge(lab_channels)
    equalized_lab_rgb = cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(source_img)
    plt.title('Original Image (RGB)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(equalized_ycrcb_rgb)
    plt.title('Equalized Y Channel (YCrCb)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(equalized_lab_rgb)
    plt.title('Equalized L Channel (LAB)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

#histogramEqualization(source_img4)

