import numpy as np 
from skimage import io, img_as_ubyte, filters, img_as_float, feature
import matplotlib.pyplot as plt

import matplotlib
# matplotlib.use("TkAgg")

# part 2
from skimage.filters import median
from skimage.morphology import disk

# part 4
from scipy.signal import convolve2d

# part 5
from scipy.spatial.distance import cosine

def apply_filter(image, filter):
    image = image.astype('float64')

    h, w = filter.shape
    pad_h, pad_w = h // 2, w // 2

    # Pad image
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    filtered_image = np.zeros_like(image)

    # Apply the filter
    for i in np.arange(0,image.shape[0],1):
        for j in np.arange(0,image.shape[1],1):
            for m in np.arange(0,filter.shape[0],1):
                for n in np.arange(0,filter.shape[1],1):
                    filtered_image[i, j] += padded_img[i+m, j+n] * filter[m, n]

    return filtered_image

def part1():
    # Read the image
    image = io.imread('moon.png', as_gray=True)
    image = img_as_ubyte(image)

    # Define the filters
    laplacian_kernel = np.array([[0, 1, 0], 
                                 [1, -4, 1], 
                                 [0, 1, 0]])
    
    gaussian_kernel = np.array([[1, 4, 7, 4, 1], 
                                [4, 16, 26, 16, 4], 
                                [7, 26, 41, 26, 7], 
                                [4, 16, 26, 16, 4], 
                                [1, 4, 7, 4, 1]]) / 273
    
    filter3 = np.array([[0, 0, 0, 0, 0], 
                        [0, 1, 0, 1, 0], 
                        [0, 0, 0, 1, 0]])
    
    filter4 = np.array([[0, 0, 0], 
                        [6, 0, 6], 
                        [0, 0, 0]])

    # Apply the filters
    filtered_laplacian = apply_filter(image, laplacian_kernel)
    filtered_gaussian = apply_filter(image, gaussian_kernel)
    filtered_3 = apply_filter(image, filter3)
    filtered_4 = apply_filter(image, filter4)

    # Display the images
    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(filtered_laplacian, cmap='gray'), plt.title('Laplace Filtered Image')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(filtered_gaussian, cmap='gray'), plt.title('Gaussian Filtered Image')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(filtered_3, cmap='gray'), plt.title(' Filtered Image')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(filtered_4, cmap='gray'), plt.title('Filtered Image')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow((image-filtered_laplacian).astype(int), cmap='gray', vmin=0, vmax=255), plt.title('Enhanced Image')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow((image+(image-filtered_gaussian)*3).astype(int), cmap='gray', vmin=0, vmax=255), plt.title('gaussian enhanced Image')
    plt.show()

def part2():
    # Read the image
    image = io.imread('noisy.jpg', as_gray=True)
    image = img_as_ubyte(image)
    
    # Apply the filters
    filtered_median = median(image, disk(3))  # by convention in book
    filtered_gaussian = filters.gaussian(image, sigma=1)

    # Display the images
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original'), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(filtered_median, cmap='gray'), plt.title('median'), plt.axis('off')
    plt.subplot(1, 3, 3), plt.imshow(filtered_gaussian, cmap='gray'), plt.title('gaussian'), plt.axis('off')
    plt.show()

def part3():
    # load the images
    D = img_as_float(io.imread("damage_cameraman.png"))
    I = img_as_float(io.imread("damage_cameraman.png"))
    M = img_as_float(io.imread("damage_mask.png"))

    # Ensure the mask is boolean
    M = M.astype(bool)

    # impaint
    for i in range(50):
        # Step (a): Apply Gaussian smoothing to damaged image
        blurred_image = filters.gaussian(I, sigma=1)
        
        # Step (b): Replace undamaged pixels with the originals 
        # convention search on google:
        I[M == False] = blurred_image[M == False]

    # Display the images
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1), plt.imshow(D, cmap='gray'), plt.title('damaged image'), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(I, cmap='gray'), plt.title('restored image'), plt.axis('off')
    plt.show()

def part4():
    # load the image
    image = io.imread('ex2.jpg', as_gray=True)
    image = img_as_ubyte(image)

    # define Sobel filters for derivatives
    sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # compute derivatives using the Sobel filters
    horizontal_derivative = convolve2d(image, sobel_horizontal, mode='same', boundary='symm')
    vertical_derivative = convolve2d(image, sobel_vertical, mode='same', boundary='symm')

    # original image
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray'), plt.title('Image')
    plt.show()

    # horizontal derivative image
    plt.figure(figsize=(10, 8))
    plt.colorbar(plt.imshow(horizontal_derivative, cmap='gray')), plt.title('Horizontal gradient')
    plt.show()

    # vertical derivative image
    plt.figure(figsize=(10, 8))
    plt.colorbar(plt.imshow(vertical_derivative, cmap='gray')), plt.title('Vertical gradient')
    plt.show()

    # Gradient magnitude calculation
    E = np.sqrt(horizontal_derivative**2 + vertical_derivative**2)

    # gradient magnitude image
    plt.figure(figsize=(10, 8))
    plt.colorbar(plt.imshow(E, cmap='gray')),plt.title('Gradient magnitude image')
    plt.show()

def part5():
    # Load image
    original = io.imread('ex2.jpg', as_gray=True)
    # original = img_as_ubyte(original)

    target = io.imread('canny_target.jpg', as_gray=True)
    # target = img_as_ubyte(target)

    # original image
    plt.figure(figsize=(10, 8))
    plt.imshow(original, cmap='gray'), plt.title('Image')
    plt.show()

    # Apply gaussian filter
    smoothed_image = filters.gaussian(original, sigma=1)  # sigma can be adjusted

    # blurred image
    plt.figure(figsize=(10, 8))
    plt.imshow(smoothed_image, cmap='gray'), plt.title('Gaussian filter')
    plt.show()

    # initialize variables
    best_distance = 1e10
    best_params = [0, 0, 0]

    # iterate thru the parameter ranges
    for low_threshold in [50, 70, 90]:
        for high_threshold in [150, 170, 190]:
            for sigma in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]:

                # Apply the Canny method with the parameters to the image
                canny_output = feature.canny(original, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

                # Compute cosine distance
                this_dist = cosine(canny_output.flatten(), target.flatten())

                # Check if this combination of parameters is better
                if (this_dist < best_distance) and np.sum(canny_output > 0.0) > 0:
                    best_distance = this_dist
                    best_params = [low_threshold, high_threshold, sigma]

    # the minimum cosine distance will be around 0.075, but any cosine distance below 0.1 is acceptable.

    # apply the canny method with the best parameters
    my_image = feature.canny(original, sigma=best_params[2], low_threshold=best_params[0], high_threshold=best_params[1])

    # Plot the target and my image
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1), plt.imshow(target, cmap='gray'), plt.title('Target Image')
    plt.subplot(1, 2, 2), plt.imshow(my_image, cmap='gray'), plt.title('My Image')
    plt.show()

    print(f"Best Cosine Distance: {best_distance}")

if __name__ == '__main__':
    part1()
    part2()
    part3()
    # COLOUR BAR NOT RIGHT SIZE
    part4()
    part5()
