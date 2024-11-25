"""Include your imports here
Some example imports are below"""

import numpy as np 
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def part1_histogram_compute():
    filename = r'test.jpg'
    image = io.imread(filename, as_gray=True)
    img = img_as_ubyte(image)

    """add your code here"""
    n = 64

    # Histogram computed by your code (cannot use in-built functions!)
    h,w = img.shape
    hist = np.zeros(n, dtype=int)
    bin_width = 256 / n

    # Iterate through pixels and update histogram
    for i in np.arange(0,h,1):
        for j in np.arange(0,w,1):
            pixel_value = img[i,j]
            bin_index = int(pixel_value / bin_width)
            hist[bin_index] += 1

    # Histogram computed by numpy
    hist_np, _ = np.histogram(img, bins=64, range=[0, 256])

    """Plotting code provided here
    Make sure to match the variable names in your code!"""

    plt.figure(figsize=(8, 10))
    plt.subplot(121), plt.plot(hist), plt.title('My Histogram')
    plt.xlim([0, n])
    plt.subplot(122), plt.plot(hist_np), plt.title('Numpy Histogram')
    plt.xlim([0, n])

    plt.show()

def part2_histogram_equalization():
    filename = 'test.jpg'
    image = io.imread(filename, as_gray=True)
    img = img_as_ubyte(image)
    
    """add your code here"""
    n_bins = 64

    # 64-bin Histogram computed by your code (cannot use in-built functions!)
    h,w = img.shape
    hist = np.zeros(n_bins, dtype=int)
    bin_width = 256 / n_bins

    # Iterate through pixels and update histogram
    for i in np.arange(0,h,1):
        for j in np.arange(0,w,1):
            pixel_value = img[i,j]
            bin_index = int(pixel_value / bin_width)
            hist[bin_index] += 1

    ## HINT: Initialize another image (you can use np.zeros) and update the pixel intensities in every location

    # Equalized image computed by your code
    img_eq1 = np.zeros_like(img)
    
    # Compute cumulative sum
    cumsum = np.cumsum(hist)
    cumsum_normalized = cumsum / cumsum[-1]

    # Update img pixel intensities based on the equalization
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            pixel_value = img[i, j]
            img_eq1[i, j] = int(cumsum_normalized[int(pixel_value / bin_width)] * 255)

    # Histogram of equalized image
    hist_eq = np.zeros(n_bins, dtype=int)

    # Update the equalized image histogram
    for i in np.arange(0, h, 1):
        for j in np.arange(0, w, 1):
            pixel_value = img_eq1[i, j]
            bin_index = int(pixel_value / bin_width)
            hist_eq[bin_index] += 1

    """Plotting code provided here
    Make sure to match the variable names in your code!"""

    plt.figure(figsize=(8, 10))
    plt.subplot(221), plt.imshow(image, 'gray'), plt.title('Original Image')
    plt.subplot(222), plt.plot(hist), plt.title('Histogram')
    plt.xlim([0, n_bins])
    plt.subplot(223), plt.imshow(img_eq1, 'gray'), plt.title('New Image')
    plt.subplot(224), plt.plot(hist_eq), plt.title('Histogram After Equalization')
    plt.xlim([0, n_bins])
    
    plt.show()   

def part3_histogram_comparing():

    filename1 = 'day.jpg'
    filename2 = 'night.jpg'

    # Read in the image
    image1 = io.imread(filename1, as_gray=True)
    # Read in another image
    image2 = io.imread(filename2, as_gray=True)

    img1 = img_as_ubyte(image1)
    img2 = img_as_ubyte(image2)
    
    """add your code here"""

    # Calculate the histograms for img1 and img2 (you can use skimage or numpy)
    hist1, _ = np.histogram(img1, bins=256, range=[0, 256])
    hist2, _ = np.histogram(img2, bins=256, range=[0, 256])

    # Normalize the histograms for img1 and img2
    hist1_norm = hist1 / np.sum(hist1)
    hist2_norm = hist2 / np.sum(hist2)

    # Calculate the Bhattacharya coefficient (check the wikipedia page linked on eclass for formula)
    # Value must be close to 0.87
    bc = np.sum(np.sqrt(hist1_norm * hist2_norm))

    print("Bhattacharyya Coefficient: ", bc)

def part4_histogram_matching():
    filename1 = 'day.jpg'
    filename2 = 'night.jpg'

    #============Grayscale============

    # Read in the image
    source_gs = io.imread(filename1, as_gray=True)
    source_gs = img_as_ubyte(source_gs)

    # Read in another image
    template_gs = io.imread(filename2, as_gray=True)
    template_gs = img_as_ubyte(template_gs)
    
    """add your code here"""

     # Calculate histograms for images
    hist_source, _ = np.histogram(source_gs, bins=256, range=[0, 256])
    hist_template, _ = np.histogram(template_gs, bins=256, range=[0, 256])

    # Calculate cumulative sum for the images
    cumsum_source = np.cumsum(hist_source) / np.sum(hist_source)
    cumsum_template = np.cumsum(hist_template) / np.sum(hist_template)

    Arr = np.zeros(256, dtype=int)
    a_prime = 0
    for a in range(256):
        while cumsum_source[a] > cumsum_template[a_prime]:
            a_prime +=1

        Arr[a] = a_prime

    h,w = source_gs.shape
    matched_gs = np.zeros_like(source_gs)
    for i in np.arange(0,h,1):
        for j in np.arange(0,w,1):
            a = source_gs[i,j]
            matched_gs[i,j] = Arr[a]

    fig = plt.figure()
    gs = plt.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source_gs, cmap=plt.cm.gray)
    ax1.set_title('source_gs')
    ax2.imshow(template_gs, cmap=plt.cm.gray)
    ax2.set_title('template_gs')
    ax3.imshow(matched_gs, cmap=plt.cm.gray)
    ax3.set_title('matched_gs')
    plt.show()


    #============RGB============
    # Read in the image
    source_rgb = io.imread(filename1)
    source_rgb = img_as_ubyte(source_rgb)

    # Read in another image
    template_rgb = io.imread(filename2)
    template_rgb = img_as_ubyte(template_rgb)
    
    """add your code here"""
    ## HINT: Repeat what you did for grayscale for each channel of the RGB image.
        
    matched_rgb = np.zeros_like(source_rgb)

    for channel in range(3):

        # Calculate histograms for images
        rgb_hist_source, _ = np.histogram(source_rgb[:, :, channel], bins=256, range=[0, 256])
        rgb_hist_template, _ = np.histogram(template_rgb[:, :, channel], bins=256, range=[0, 256])

        # Calculate cumulative sum for the images
        rgb_cumsum_source = np.cumsum(rgb_hist_source) / np.sum(rgb_hist_source)
        rgb_cumsum_template = np.cumsum(rgb_hist_template) / np.sum(rgb_hist_template)

        rgb_arr = np.zeros(256, dtype=int)
        a_prime = 0

        for a in range(256):
            while rgb_cumsum_source[a] > rgb_cumsum_template[a_prime]:
                a_prime +=1

            rgb_arr[a] = a_prime

        # h,w = source_rgb.shape
        # matched_rgb = np.zeros_like(source_rgb)
        # for i in np.arange(0,h,1):
        #     for j in np.arange(0,w,1):
        #         a = source_rgb[i,j]
        #         matched_rgb[i,j] = rgb_arr[a]
            
        matched_rgb[:, :, channel] = rgb_arr[source_rgb[:, :, channel]]
    
    fig = plt.figure()
    gs = plt.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source_rgb)
    ax1.set_title('source_rgb')
    ax2.imshow(template_rgb)
    ax2.set_title('template_rgb')
    ax3.imshow(matched_rgb)
    ax3.set_title('matched_rgb')
    plt.show()

if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
