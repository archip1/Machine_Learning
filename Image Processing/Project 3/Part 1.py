# import statements
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import io

def part1():
    """ BasicBayer: reconstruct RGB image using GRGB pattern"""
    filename_Grayimage = 'PeppersBayerGray.bmp'

    # read image
    img = io.imread(filename_Grayimage, as_gray =True)
    h,w = img.shape

    # our final image will be a 3 dimentional image with 3 channels
    rgb = np.zeros((h,w,3),np.uint8)

    # reconstruction of the green channel IG
    IG = np.copy(img) # copy the image into each channel
    for row in range(0,h,4): # loop step is 4 since our mask size is 4.
        for col in range(0,w,4): # loop step is 4 since our mask size is 4.
            # TODO: compute pixel value for each location where mask is unshaded (0) 
            
            # interpolate each pixel using its every valid (shaded) neighbour
            A = int(img[row,col])
            B = int(img[row,col+1])
            C = int(img[row,col+2])
            D = int(img[row,col+3])
            E = int(img[row+1,col])
            F = int(img[row+1,col+1])
            G = int(img[row+1,col+2])
            H = int(img[row+1,col+3])
            I = int(img[row+2,col])
            J = int(img[row+2,col+1])
            K = int(img[row+2,col+2])
            L = int(img[row+2,col+3])
            M = int(img[row+3,col])
            N = int(img[row+3,col+1])
            O = int(img[row+3,col+2])
            P = int(img[row+3,col+3])

            IG[row,col]= A  # A
            IG[row,col+2]= C  # C
            IG[row+1,col+1]= F  # F
            IG[row+1,col+3]= H  # H
            IG[row+2,col]= I  # I
            IG[row+2,col+2] = K  # K
            IG[row+3,col+1]= N  # N
            IG[row+3,col+3]= P  # P
        
            IG[row,col+1]= (A+C+F)/3  # B 
            
            IG[row,col+3]= (C+H)/2  # D
            
            IG[row+1,col]= (A+F+I)/3  # E
            
            IG[row+1,col+2]= (C+F+H+K)/4  # G

            IG[row+2,col+1]= (F+I+K+N)/4  # J 

            IG[row+2,col+3]= (H+K+P)/3  # L 

            IG[row+3,col]= (I+N)/2  # M
            
            IG[row+3,col+2]= (N+K+P)/3  # O

    # TODO: reconstruction of the red channel IR (simiar to loops above)
    IR = np.copy(img)
    for row in range(0,h,4):
        for col in range(0,w,4):
            # TODO: compute pixel value for each location where mask is unshaded (0) 
            
            # interpolate each pixel using its every valid (shaded) neighbour
            A = int(img[row,col])
            B = int(img[row,col+1])
            C = int(img[row,col+2])
            D = int(img[row,col+3])

            E = int(img[row+1,col])
            F = int(img[row+1,col+1])
            G = int(img[row+1,col+2])
            H = int(img[row+1,col+3])

            I = int(img[row+2,col])
            J = int(img[row+2,col+1])
            K = int(img[row+2,col+2])
            L = int(img[row+2,col+3])

            M = int(img[row+3,col])
            N = int(img[row+3,col+1])
            O = int(img[row+3,col+2])
            P = int(img[row+3,col+3])

            IR[row,col+1] = B  # B
            IR[row,col+3] = D  # D
            IR[row+2,col+1] = J  # J
            IR[row+2,col+3] = L  # L

            IR[row,col+2]= (B+D)/2  # C            
            IR[row+1,col+1]= (B+J)/2  # F
            IR[row+1,col+2]= (B+D+L+J)/4  # G
            IR[row+1,col+3]= (D+L)/2  # H
            IR[row+2,col+2]= (J+L)/2  # K

            IR[row,col]= B  # A
            IR[row+1,col]= IR[row+1,col+1]  # E
            IR[row+2,col]= J  # I
            IR[row+3,col]= J  # M
            IR[row+3,col+1]= J  # N
            IR[row+3,col+2]= IR[row+2,col+2]  # O
            IR[row+3,col+3]= L  # P


    # TODO: reconstruction of the blue channel IB (similar to loops above)
    IB = np.copy(img)
    for row in range(0,h,4):
        for col in range(0,w,4):
            # TODO: compute pixel value for each location where mask is unshaded (0) 
            
            # interpolate each pixel using its every valid (shaded) neighbour
            A = int(img[row,col])
            B = int(img[row,col+1])
            C = int(img[row,col+2])
            D = int(img[row,col+3])
            E = int(img[row+1,col])
            F = int(img[row+1,col+1])
            G = int(img[row+1,col+2])
            H = int(img[row+1,col+3])
            I = int(img[row+2,col])
            J = int(img[row+2,col+1])
            K = int(img[row+2,col+2])
            L = int(img[row+2,col+3])
            M = int(img[row+3,col])
            N = int(img[row+3,col+1])
            O = int(img[row+3,col+2])
            P = int(img[row+3,col+3])

            IB[row+1,col]= E  # E
            IB[row+1,col+2]= G  # G
            IB[row+3,col]= M  # M
            IB[row+3,col+2]= O  # O

            IB[row+1,col+1]= (E+G)/2  # F
            IB[row+2,col]= (E+M)/2  # I
            IB[row+2,col+1] = (E+G+M+O)/4  # J
            IB[row+2,col+2]= (G+O)/2  # K
            IB[row+3,col+1]= (M+O)/2  # N

            IB[row,col]= E  # A
            IB[row,col+1] = IB[row+1,col+1] # B
            IB[row,col+2]= G  # C
            IB[row,col+3]= G  # D
            IB[row+1,col+3]= G  # H
            IB[row+2,col+3] = IB[row+2,col+2]  # L
            IB[row+3,col+3]= O  # P
            
    # TODO: merge the three channels IG, IB, IR in the correct order
    rgb[:,:,0]=IR
    rgb[:,:,1]=IG
    rgb[:,:,2]=IB

    # display the channels and the image
    plt.figure(figsize=(10, 8))

    #  show green (IR) in first subplot (221) and add title
    plt.subplot(221)
    plt.imshow(IG, cmap='gray'), plt.title('IG')

    #  show IR in second subplot (223) and title
    plt.subplot(222)
    plt.imshow(IR, cmap='gray'), plt.title('IR')

    # show IB in third subplot (224) and title
    plt.subplot(223)
    plt.imshow(IB, cmap='gray'), plt.title('IB')

    # show rgb image in final subplot (224) and add title
    plt.subplot(224)
    plt.imshow(rgb),plt.title('rgb')
    plt.show()

if __name__  == "__main__":
    part1()