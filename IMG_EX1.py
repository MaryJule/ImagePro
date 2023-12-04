#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Maria Melanie Julia Federer
Date: 04.12.2023
Description: Exercises in Photogrammetry, Remote Sensing, and Image Processing

License:
This script is provided by the author, Maria Melanie Julia Federer, for educational purposes.
You are free to use and modify this script as long as you agree
    -If any issues arise with the professor regarding the script, 
    it will be understood that the script was created independently by me.

 
"""


# %% TASK : Load Packages
import numpy as np
import matplotlib.pyplot as plt


# %% TASK 1 

###1.1
imagebmp = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/image.bmp") # Load the image
rows,cols = imagebmp.shape

###1.2
#Write youre own funktion

def calculate_histogram(image):
    """
    Calculate a histogram for a given grayscale image with 8 bits.
    Parameters:
    - image: numpy.ndarray
      A grayscale image represented as a 2D NumPy array with values in the range [0, 255].

    Returns:
    - histogram: numpy.ndarray
      The computed histogram with 256 bins.
      
      
    Example usage:
      Load your grayscale image (imagebmp) here
      Calculate the histogram
      histogram = calculate_histogram(imagebmp)
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image should have data type uint8 (8 bits).")

    # Initialize an array to store the histogram
    histogram = np.zeros(256, dtype=np.int32)

    # Iterate through the image and accumulate pixel counts in the histogram
    for pixel_value in image.flatten():
        histogram[pixel_value] += 1

    return histogram

histogram = calculate_histogram(imagebmp)

plt.plot(range(256), histogram, color='gray')
plt.title('Grayscale Histogram (Own Calculation)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

#Check with matplotlib.pyplot.stem()
def plot_histogram(image):
    
    """
    Assuming the image is already in 8-bit grayscale
    Calculate histogram
    """
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Plot histogram
    plt.stem(bins[:-1], histogram, '-', use_line_collection=True)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()
    

###1.3
def mean_value(image):
    """
    Calculating the mean value of an image
    """
    total_pixels = image.shape[0] * image.shape[1]
    sum_pixels = np.sum(image)
    mean_val = sum_pixels / total_pixels
    return mean_val

def variance(image, mean_val):
    """
    Calculating the variance of an image
    """
    total_pixels = image.shape[0] * image.shape[1]
    sum_squared_diff = np.sum((image - mean_val) ** 2)
    variance_val = sum_squared_diff / total_pixels
    return variance_val

def standard_deviation(variance_val):
    """
    Calculating the standard deviation of an image
    """
    std_dev = np.sqrt(variance_val)
    return std_dev

# Calculate own
mean_val = mean_value(imagebmp)
variance_val = variance(imagebmp, mean_val)
std_dev_val = standard_deviation(variance_val)
print(f"Mean: {mean_val}, Variance: {variance_val}, Standard Deviation: {std_dev_val}")

# Validate with numpy
np_mean = np.mean(imagebmp)
np_variance = np.var(imagebmp)
np_std_dev = np.std(imagebmp)
print(f"NumPy Mean: {np_mean}, NumPy Variance: {np_variance}, NumPy Standard Deviation: {np_std_dev}")
    
    
#%% TASK 2

# Load the images
imageContrast = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/image_a.bmp")
imageInvert = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/image_b.bmp")
imageNoise = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/image_c.bmp")
imageBlurred = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/image_d.bmp")

# Calculate covariance and correlation coefficients for each variation
def calculate_covariance_and_correlation(image1, image2):
    # Flatten the images
    flat_image1 = image1.flatten()
    flat_image2 = image2.flatten()
    
    # Calculate covariance matrix
    covariance_matrix = np.cov(flat_image1, flat_image2)
    
    # Calculate correlation coefficient matrix
    correlation_coefficient_matrix = np.corrcoef(flat_image1, flat_image2)
    
    # Extract covariance and correlation coefficient from the matrices
    covariance = covariance_matrix[0, 1]
    correlation_coefficient = correlation_coefficient_matrix[0, 1]
    
    return covariance, correlation_coefficient

# Perform analysis for each variation and print the results
variations = {
    'Contrast': imageContrast,
    'Invert': imageInvert,
    'Noise': imageNoise,
    'Blurred': imageBlurred
}

for variation_name, variation_image in variations.items():
    covariance, correlation_coefficient = calculate_covariance_and_correlation(imageContrast, variation_image)
    
    # Display the original and variation images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(imageContrast, cmap='gray')
    ax1.set_title(' ')
    
    ax2.imshow(variation_image, cmap='gray')
    ax2.set_title(' ')
    
    plt.suptitle(f"Analysis for {variation_name} Variation\nCovariance: {covariance}, Correlation Coefficient: {correlation_coefficient}")
    plt.show()


# %% TASK 3

#load images
imageContrast = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/image_a.bmp") # Load Car Image
imageInvert = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/image_b.bmp") # Load Letter
imageNoise = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/image_c.bmp") # Load Car Image
imageBlurred = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/image_d.bmp") # Load Letter

# Example for one image
image_flat = imageContrast.flatten()

# Histogram
histogram, bins = np.histogram(image_flat, bins=256, range=[0, 256])

# Mean, Variance, Standard Deviation
mean_val = np.mean(image_flat)
variance_val = np.var(image_flat)
std_dev_val = np.std(image_flat)

# List of images
images = [imageContrast, imageInvert, imageNoise, imageBlurred]
image_names = ["Contrast", "Invert", "Noise", "Blurred"]

# Plot histograms for all images
for image, name in zip(images, image_names):
    # Flatten the image
    image_flat = image.flatten()

    # Calculate the histogram
    histogram, bins = np.histogram(image_flat, bins=256, range=[0, 256])

    # Calculate Mean, Variance, Standard Deviation
    mean_val = np.mean(image_flat)
    variance_val = np.var(image_flat)
    std_dev_val = np.std(image_flat)

    # Plot the histogram
    plt.figure()
    plt.stem(bins[:-1], histogram, '-', use_line_collection=True)
    plt.title(f'Histogram of {name} Variation')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Display mean, variance, and standard deviation as text on the plot
    plt.text(0.7, 0.9, f'Mean: {mean_val:.2f}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.85, f'Variance: {variance_val:.2f}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.8, f'Standard Deviation: {std_dev_val:.2f}', transform=plt.gca().transAxes)

    plt.show()


# Function to calculate covariance and correlation
def calculate_stats(image1, image2):
    covariance = np.cov(image1, image2)[0, 1]
    correlation = np.corrcoef(image1, image2)[0, 1]
    return covariance, correlation

# Initialize lists to store results
histograms = []
means = []
variances = []
std_devs = []
cov_corr_results = []

# Create a list of image names for reference
image_names = ["Contrast", "Invert", "Noise", "Blurred"]

# Iterate over all images
for image in [imageContrast, imageInvert, imageNoise, imageBlurred]:
    # Flatten the image
    image_flat = image.flatten()
    
    # Calculate histogram
    histogram, bins = np.histogram(image_flat, bins=256, range=[0, 256])
    histograms.append(histogram)
    
    # Calculate mean, variance, and standard deviation
    mean_val = np.mean(image_flat)
    variance_val = np.var(image_flat)
    std_dev_val = np.std(image_flat)
    
    means.append(mean_val)
    variances.append(variance_val)
    std_devs.append(std_dev_val)

    # Calculate covariance and correlation with the original image
    covariance, correlation = calculate_stats(imagebmp.flatten(), image_flat)
    cov_corr_results.append((covariance, correlation))

# Print and describe the results
for i, image_name in enumerate(image_names):
    print(f"Results for {image_name} Image:")
    print(f"Mean: {means[i]}, Variance: {variances[i]}, Standard Deviation: {std_devs[i]}")
    print(f"Covariance with Original Image: {cov_corr_results[i][0]}")
    print(f"Correlation with Original Image: {cov_corr_results[i][1]}")
    print()
    

#%% TASK 4

# Load the template images and functions from IP01_function.py

from IP01_function import correlation, getMaximumCorrPoint

query_image = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/query.bmp")
templateA = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/templateA.bmp")
templateG = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/templateG.bmp")
templateP = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/templateP.bmp")
templateV = plt.imread("/home/maria/Documents/TUM/Module/IMP_Ü/Session_2/images/templateV.bmp")

#4.2

def templateSearch(image, template):
    correlation_image = correlation(image, template)
    r, c = getMaximumCorrPoint(correlation_image)
    return r, c

# Find positions of templates in the query image
r_A, c_A = templateSearch(query_image, templateA)
r_G, c_G = templateSearch(query_image, templateG)
r_P, c_P = templateSearch(query_image, templateP)
r_V, c_V = templateSearch(query_image, templateV)

# 4.3 Plot result template search
fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(query_image, cmap=plt.get_cmap("gray"))
axes[0, 0].scatter(x=[c_A], y=[r_A], c='r', s=10)
axes[0, 0].set_title("Template A")

axes[0, 1].imshow(query_image, cmap=plt.get_cmap("gray"))
axes[0, 1].scatter(x=[c_G], y=[r_G], c='r', s=10)
axes[0, 1].set_title("Template G")

axes[1, 0].imshow(query_image, cmap=plt.get_cmap("gray"))
axes[1, 0].scatter(x=[c_P], y=[r_P], c='r', s=10)
axes[1, 0].set_title("Template P")

axes[1, 1].imshow(query_image, cmap=plt.get_cmap("gray"))
axes[1, 1].scatter(x=[c_V], y=[r_V], c='r', s=10)
axes[1, 1].set_title("Template V")

plt.tight_layout()
plt.show()

