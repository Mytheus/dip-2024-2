import numpy as np

def get_image_info(image):
    """
    Extracts metadata and statistical information from an image.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - dict: Dictionary containing image metadata and statistics.
    """
    
    ### START CODE HERE ###
    img_shp = image.shape
    height = img_shp[0]
    width = img_shp[1]
    dtype = image.dtype
    min_val = np.min(image)
    max_val = np.max(image)
    mean_val = np.mean(image)
    std_val = np.std(image)
    
    num_channels = 1 if image.ndim == 2 else image.shape[2]
    bits_per_channel = image.dtype.itemsize * 8
    depth = num_channels * bits_per_channel
    ### END CODE HERE ###

    return {
        "width": width,
        "height": height,
        "dtype": dtype,
        "depth": depth,
        "min_value": min_val,
        "max_value": max_val,
        "mean": mean_val,
        "std_dev": std_val
    }

# Example Usage:
sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
info = get_image_info(sample_image)

# Print results
for key, value in info.items():
    print(f"{key}: {value}")