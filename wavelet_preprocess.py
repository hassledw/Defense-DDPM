import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def apply_wavelet_transform(image):
    # Apply a 2-level discrete wavelet transform using the Haar wavelet
    coeffs = [pywt.wavedec2(image[:, :, i], 'haar', level=2) for i in range(3)]
    return coeffs

def threshold(coeffs, threshold_value):
    """Apply soft thresholding to the wavelet coefficients of each channel."""
    thresholded_coeffs = []
    for channel in coeffs:
        cA, details = channel[0], channel[1:]
        thresholded_details = []
        for level in details:
            cH, cV, cD = level
            cH = pywt.threshold(cH, threshold_value, mode='soft')
            cV = pywt.threshold(cV, threshold_value, mode='soft')
            cD = pywt.threshold(cD, threshold_value, mode='soft')
            thresholded_details.append((cH, cV, cD))
        thresholded_coeffs.append([cA] + thresholded_details)
    return thresholded_coeffs

def reconstruct_image(coeffs):
    # Reconstruct each channel from its coefficients
    channels = [pywt.waverec2(c, 'haar') for c in coeffs]
    # Normalize each channel
    channels = [cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX) for c in channels]
    channels = [np.uint8(c) for c in channels]
    return cv2.merge(channels)


def display_images(original, denoised):
    """Display the original and denoised images side by side."""
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_rgb)
    plt.title('Denoised Image')
    plt.show() 


def process_image(file_path, output_path):
    # Read the image
    image = load_image(file_path) 
    
    # Apply the preprocessing algorithm
    coeffs = apply_wavelet_transform(image)
    threshold_value = np.median([np.abs(coeffs[c][-1][-1]) for c in range(3)]) / 0.6745
    coeffs_denoised = threshold(coeffs, threshold_value)
    image_denoised = reconstruct_image(coeffs_denoised)
    
    # Save the denoised image
    cv2.imwrite(output_path, image_denoised)

def process_directory(input_dir, output_dir):
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            process_image(input_path, output_path)


if __name__ == "__main__":
    directory_path = './bird-data/FGSM025-test'
    output_path    = './bird-data/FGSM025-wavelet-test'
    process_directory(directory_path, output_path)

    
