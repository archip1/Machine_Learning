# Import libraries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp

def read_image():
    original_img = io.imread('bird.jpeg')
    return original_img

def calculate_trans_mat(image):
    """
    return translation matrix that shifts center of image to the origin and its inverse
    """

    # TODO: implement this function (overwrite the two lines above)
    h, w = image.shape[:2]

    # shift center to origin
    trans_mat = np.array([[1, 0, -w/2],
                          [0, 1, -h/2],
                          [0, 0, 1]])

    # shift center back to the original location
    trans_mat_inv = np.array([[1, 0, w/2],
                              [0, 1, h/2],
                              [0, 0, 1]])
    
    return trans_mat, trans_mat_inv

def rotate_image(image):
    ''' rotate and return image '''
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    # TODO: determine angle and create Tr
    angle = -75
    angle_rad = np.radians(angle)

    Tr = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                   [np.sin(angle_rad), np.cos(angle_rad), 0],
                   [0, 0, 1]])

    # TODO: compute inverse transformation to go from output to input pixel locations
    Tr_inv =  np.linalg.inv(Tr)
    combined_inv = np.dot(trans_mat_inv, np.dot(Tr_inv, trans_mat))

    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel location and inverse transform matrix, copy over value from input location to output location
            in_location = np.dot(combined_inv, [out_x, out_y, 1])
            input_x = int(in_location[0])
            input_y = int(in_location[1])

            if (input_x >= 0 and input_x < w and input_y >= 0 and input_y < h):
                out_img[out_y, out_x] = image[input_y, input_x]

    return out_img, Tr

def scale_image(image):
    ''' scale image and return '''
    # TODO: implement this function, similar to above
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

# create Ts
    Ts = np.array([[1.5, 0, 0],
                   [0, 2.5, 0],
                   [0, 0, 1]])
    
    # TODO: compute inv erse transformation to go from output to input pixel locations
    Ts_inv = np.linalg.inv(Ts)
    combined_inv = np.dot(trans_mat_inv, np.dot(Ts_inv, trans_mat))

    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel location and inverse transform matrix, copy over value from input location to output location
            in_location = np.dot(combined_inv, [out_x, out_y, 1])
            input_x = int(in_location[0])
            input_y = int(in_location[1])

            if (input_x >= 0 and input_x < w and input_y >= 0 and input_y < h):
                out_img[out_y, out_x] = image[input_y, input_x]

    return out_img, Ts

def skew_image(image):
    ''' Skew image and return '''
    # TODO: implement this function like above
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    # create Tskew
    Tskew = np.array([[1, 0.2, 0],
                      [0.2, 1, 0],
                      [0, 0, 1]])
    
    # TODO: compute inv erse transformation to go from output to input pixel locations
    Tskew_inv = np.linalg.inv(Tskew)
    combined_inv = np.dot(trans_mat_inv, np.dot(Tskew_inv, trans_mat))

    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel location and inverse transform matrix, copy over value from input location to output location
            in_location = np.dot(combined_inv, [out_x, out_y, 1])
            input_x = int(in_location[0])
            input_y = int(in_location[1])

            if (input_x >= 0 and input_x < w and input_y >= 0 and input_y < h):
                out_img[out_y, out_x] = image[input_y, input_x]

    return out_img, Tskew

def combined_warp(image):
    ''' implement your own code to perform the combined warp of rotate, scale, skew and return image + transformation matrix  '''
    # TODO: implement combined warp on your own. 
    # You need to combine the transformation matrices before performing the warp
    # (you may want to use the above functions to get the transformation matrices)
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    # rotate, scale, skew
    rotated_img, Tr = rotate_image(image)
    scaled_img, Ts = scale_image(image)
    skewed_img, Tskew = skew_image(image)

    # combine transformations
    temp = np.dot(Ts, Tr)
    Tc = np.dot(temp, Tskew)

    # TODO: compute inverse transformation to go from output to input pixel locations
    Tc_inv = np.linalg.inv(Tc)
    combined_inv = np.dot(np.dot(trans_mat_inv, Tc_inv), trans_mat)

    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel location and inverse transform matrix, copy over value from input location to output location
            in_location = np.dot(combined_inv, [out_x, out_y, 1])
            input_x = int(in_location[0])
            input_y = int(in_location[1])

            if (input_x >= 0 and input_x < w and input_y >= 0 and input_y < h):
                out_img[out_y, out_x] = image[input_y, input_x]

    return out_img, Tc
    
def combined_warp_bilinear(image):
    ''' perform the combined warp with bilinear interpolation (just show image) '''
    # TODO: implement combined warp -- you can use skimage.trasnform functions for this part (import if needed)
    # (you may want to use the above functions (above combined) to get the combined transformation matrix)
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    combined_warp_img, Tc = combined_warp(image)

    Tc_inv = np.linalg.inv(Tc)
    combined_inv = np.dot(np.dot(trans_mat_inv, Tc_inv), trans_mat)

    out_img = np.zeros_like(image)
    out_img = warp(image.astype(np.float64), combined_inv.astype(np.float64), order=1)

    return out_img

if __name__ == "__main__":
    image = read_image()
    plt.imshow(image), plt.title("Oiginal Image"), plt.show()

    rotated_img, _ = rotate_image(image)
    plt.figure(figsize=(15,5))
    plt.subplot(131),plt.imshow(rotated_img), plt.title("Rotated Image")

    scaled_img, _ = scale_image(image)
    plt.subplot(132),plt.imshow(scaled_img), plt.title("Scaled Image")

    skewed_img, _ = skew_image(image)
    plt.subplot(133),plt.imshow(skewed_img), plt.title("Skewed Image"), plt.show()

    combined_warp_img, _ = combined_warp(image)
    plt.figure(figsize=(10,5))
    plt.subplot(121),plt.imshow(combined_warp_img), plt.title("Combined Warp Image")

    combined_warp_biliear_img = combined_warp_bilinear(image)
    plt.subplot(122),plt.imshow(combined_warp_biliear_img.astype(np.uint8)), plt.title("Combined Warp Image with Bilinear Interpolation"),plt.show()
