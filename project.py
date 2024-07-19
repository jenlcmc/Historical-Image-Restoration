# libs need for project
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import cv2

from numpy.fft import fft2, fftshift, ifft2

def frequency_analysis(image):
    # Apply Fourier Transform
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)

    # Magnitude Spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    return magnitude_spectrum

def local_std_deviation(image, kernel_size=3): # use kernel 3 for detail and 7 for smooth

    # Calculate local standard deviation
    local_mean = cv2.blur(image, (kernel_size, kernel_size))
    local_mean_sq = cv2.blur(np.square(image), (kernel_size, kernel_size))
    local_std = np.sqrt(local_mean_sq - np.square(local_mean))

    return local_std

def estimate_noise(image):
    # Ensure the image is grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

    # Compute the Laplacian (2nd derivative) of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # The variance of the Laplacian is a good estimator for noise
    return np.sqrt(np.var(laplacian))

def quantify_noise(noise_map):
    total_pixels = noise_map.size
    noise_pixels = np.sum(noise_map)
    noise_percentage = (noise_pixels / total_pixels) * 100
    return noise_percentage

from skimage.metrics import structural_similarity

def calculate_metrics(original, denoised):
    # Calculate PSNR, specifying the range of pixel values
    data_range = original.max() - original.min()  # If range is not standard, calculate from images
    # Calculate SSIM
    ssim = structural_similarity(original, denoised, data_range=data_range)
    # the higehr it is the better the denoised images is but we need to be aware of over blurring

    return ssim

def adaptive_threshold_detection(image, scale_factor=0.5):
    # Convert to grayscale if not already
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

    # Calculate local standard deviation and adaptive threshold
    local_std = local_std_deviation(image)
    adaptive_threshold = local_std / np.mean(local_std) * scale_factor

    # Perform frequency analysis and calculate noise
    magnitude_spectrum = frequency_analysis(image)
    # we do the division to normalize the magnitude_spectrum
    noise = np.divide(magnitude_spectrum, local_std, out=np.zeros_like(magnitude_spectrum), where=local_std!=0)
    noise = (noise > adaptive_threshold).astype(int) # Thresholding that we set it to 1 if the noise is greater than the threshold
    # and 0 otherwise

    return noise

# from the result of the noise level assessment (noise map), we can determine the level of noise in the image and apply the appropriate noise reduction technique
# function to denoising the image

def denoise_image(image, noise_map, sigma_color=75, sigma_space=75):
    # Compute the mean noise of the image
    mean_noise = np.mean(noise_map)

    # Adjust the filter parameters based on mean noise
    adjusted_sigma_color = sigma_color * mean_noise
    adjusted_sigma_space = sigma_space * mean_noise

    # Apply bilateral filter to the entire image
    denoised_image = cv2.bilateralFilter(image, d=-1, sigmaColor=adjusted_sigma_color, sigmaSpace=adjusted_sigma_space)

    return denoised_image

def denoise_color_image(color_image, noise_map, sigma_color=75, sigma_space=75):
    # Split the color image into its component channels
    b, g, r = cv2.split(color_image)

    # Denoise each channel using the noise map
    denoised_b = denoise_image(b, noise_map, sigma_color, sigma_space)
    denoised_g = denoise_image(g, noise_map, sigma_color, sigma_space)
    denoised_r = denoise_image(r, noise_map, sigma_color, sigma_space)

    # Merge the channels back together
    denoised_color_image = cv2.merge((denoised_b, denoised_g, denoised_r))

    return denoised_color_image

# do adaptive histogram equalization to improve the contrast and brightness of the image

def adaptive_histogram_equalization(image, contrastThreshold, gridSize):

    # Convert to grayscale if not already
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    assert img is not None, "file could not be read, check with os.path.exists()"

    # Create a CLAHE object (higher contrastThreshold means higher contrast variability)
    clahe = cv2.createCLAHE(contrastThreshold, gridSize)
    cl1 = clahe.apply(img)
    return cl1

# do edge detection to detect any scratches or defects on the image
def edge_detection(image, lowerThreshold, upperThreshold):
  # Convert to grayscale if not already
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    assert img is not None, "file could not be read, check with os.path.exists()"

  # Get edges using Canny edge detection, adjust threshold bounds as needed
    edges = cv2.Canny(img,lowerThreshold,upperThreshold)
    return edges


# Example usage:
# Assuming 'edges' is your edge mask obtained from Canny edge detection
# filled_and_dilated_edges = edge_fill_and_dilate(edges)

#Uses the edge map to detect scratches in the image. After research, we need a neural network to be trained to do this for different images.
def detect_scratches(image, lowerThreshold, upperThreshold, texture_threshold=60, minScratchLength=10, minContourArea=100, maxContourArea=3000):
    # Ensure the image is in BGR format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Get edges using Canny edge detection
    edges = cv2.Canny(image, lowerThreshold, upperThreshold)

    # Calculate texture
    texture = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 3, 3, ksize=5)

    # Combine edge and texture information
    combined = edges * (np.abs(texture) > texture_threshold)

    # Apply color-based segmentation to detect white areas
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 185], dtype=np.uint8)  # Adjusted lower threshold
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # Combine edge and texture information with white mask
    combined = combined + white_mask

    # Find contours
    contours, _ = cv2.findContours(combined.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an empty mask to mark scratches
    scratch_mask = np.zeros_like(image[:, :, 0])

    # Iterate over contours
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)

        # Filter contours based on area
        if area < minContourArea or area > maxContourArea:
            continue

        # Approximate the contour to simplify it
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the contour length is less than minScratchLength, ignore it
        if cv2.arcLength(approx, True) < minScratchLength:
            continue

        # Draw the contour on the mask
        cv2.drawContours(scratch_mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Morphological closing to fill in gaps within scratches
    kernel = np.ones((40, 40), np.uint8)
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((7, 7), np.uint8)
    dilated_mask = cv2.dilate(scratch_mask, kernel, iterations=1)

    return dilated_mask

def inpaint_scratches(image, scratch_mask):
    # Calculate radius for inpainting, not based on scratch density but on image resolution
    radius = max(50, min(image.shape[0], image.shape[1]) // 100)

    # Perform inpainting
    inpainted_image = cv2.inpaint(image, scratch_mask, radius, cv2.INPAINT_TELEA)

    return inpainted_image

def gamma_correction(image, gamma=1.0):
    # Gamma correction is a process for adjusting the brightness of an image
    # by using a non-linear transformation between the input values and the mapped output values

    # The 'gamma' value is a parameter of this transformation, and it controls the level of correction

    # If gamma < 1, the image will be darker; if gamma > 1, the image will be brighter

    # Calculate the inverse of gamma
    inv_gamma = 1.0 / gamma

    # Create a lookup table for the gamma correction
    # This table maps each pixel value [0, 255] to a new value based on the gamma correction formula
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction to the image using the lookup table
    # cv2.LUT is a function that maps an array of pixel values (image) to another array of values (table)
    corrected_image = cv2.LUT(image, table)

    # Return the gamma corrected image
    return corrected_image

def histogram_stretching(image):
    # Split into color channels
    b, g, r = cv2.split(image)

    # Apply histogram stretching to each channel
    b_stretched = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    g_stretched = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    r_stretched = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)

    # Merge channels back
    img_stretched = cv2.merge((b_stretched, g_stretched, r_stretched))

    return img_stretched

def unsharp_mask(image, sigma=1.0, strength=1.5):
    # Use a Gaussian filter to blur the image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Calculate the sharpened image
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    return sharpened

# Image Sharpening to improve the sharpness of the image or reduce the blurriness of the image
def sharpen(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image and then return the variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # If the variance is below a threshold, the image is likely blurry
    if laplacian_var < 10:  # Threshold value, may need to be adjusted or we need new algo taht can detect the blurriness
        # Apply unsharp mask
        sharpened = unsharp_mask(image, sigma=2.0, strength=3.5)
        return sharpened
    else:
        # Image is already sharp or the blurriness is within acceptable range
        return image

# Color Restoration, Correction and Enhancement to improve the color of the image if it is faded or discolored
def color_restoration(image):
    # 1st, use gamme correction to adjust the brightness of the image
    corrected_image = gamma_correction(image, gamma=1.5)

    # then apply histogram stretching to improve the contrast of the image
    stretched_image = histogram_stretching(corrected_image)

    return stretched_image

def process_image(image_path):
    color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    if color_img is None:
        raise FileNotFoundError(f"The image at {image_path} does not exist.")

    scale_factors = [5, 10, 60, 100, 200]
    for scale in scale_factors:
        # Perform processing on the grayscale image
        noise_map = adaptive_threshold_detection(gray_img, scale_factor=scale)
        denoised_gray = denoise_image(gray_img, noise_map)

        equalized_gray = adaptive_histogram_equalization(denoised_gray, 2.0, (8, 8))

        # Initialize scratch_mask and inpainted for the case when image_path is not "/img/scratch-photo.jpg"
        scratch_mask = None
        inpainted = None

        # Construct the path to the scratch image relative to the current directory
        scratch_img = os.path.join(current_dir, 'img', 'scratch-photo.jpg')

        if image_path == scratch_img:
            edges = edge_detection(equalized_gray, 150, 200)
            scratch_mask = detect_scratches(equalized_gray, 150, 200)

        # Convert the grayscale results back to BGR for color processing (if necessary)
        equalized_color = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)

        # Apply the same transformations or use the insights on the color image
        denoised_color = denoise_color_image(color_img, noise_map, sigma_color=75, sigma_space=75)
        # Call the color restoration function
        tonemapped = color_restoration(denoised_color)
        # Sharpen the image
        sharpened = sharpen(tonemapped)

        if image_path == scratch_img:
            # Inpaint the scratches
            inpainted = inpaint_scratches(sharpened, scratch_mask)

        # Calculate metrics based on the grayscale processing
        original = gray_img.astype(np.float32)
        denoised = denoised_gray.astype(np.float32)
        ssim = calculate_metrics(original, denoised)
        print(f"Scale Factor: {scale}, SSIM: {ssim}")

        # Visualization
        fig, axes = plt.subplots(1, 7 if scratch_mask is not None else 4, figsize=(35, 5))
        axes[0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))  # Original color image
        axes[1].imshow(denoised_gray, cmap='gray')  # Denoised grayscale image
        axes[2].imshow(cv2.cvtColor(equalized_color, cv2.COLOR_BGR2RGB))  # Equalized color image

        if scratch_mask is not None:
            axes[3].imshow(edges, cmap='gray')  # Edge detection result
            axes[4].imshow(scratch_mask, cmap='gray')  # Scratch mask
            axes[5].imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))  # Restored color image
            axes[6].imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))  # Inpainted color image
            axes[3].set_title("Edge Detection")
            axes[4].set_title("Scratch Mask")
            axes[5].set_title("Restored Image")
            axes[6].set_title("Inpainted Image")

        else:
            axes[3].imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))  # Restored color image
            axes[3].set_title("Restored Image")

        # Set the titles of the images
        axes[0].set_title("Original Image")
        axes[1].set_title("Denoised Image")
        axes[2].set_title("Equalized Image")

        plt.show()

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # check for image files
            image_path = os.path.join(directory, filename)
            process_image(image_path)

def process_path(path):
    if os.path.isfile(path):
        process_image(path)
    elif os.path.isdir(path):
        process_directory(path)
    else:
        print("It is not a valid file or directory")


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the img directory relative to the current directory
img_dir = os.path.join(current_dir, 'img')

process_path(img_dir)