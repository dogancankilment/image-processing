import matplotlib.pyplot as plt
import numpy as np
import scipy


def rgb2gray(rgb):
    return rgb[0]/3 + rgb[1]/3 + rgb[2]/3
    # gray_level = gray_level / 3


def rgb2gray_image(input_image):
    using_image = plt.imread(input_image)

    # two_d_image = np.zeros(three_d_image.shape[0:2])
    gray_image = np.zeros(using_image.shape[0], using_image.shape[1])
    binary_image = np.zeros(using_image.shape[0], using_image.shape[1])

    threshold = 200

    for i in range(using_image.shape[0]):
        for j in range(using_image.shape[1]):
            n = using_image[i, j, 0] / 3 + using_image[i, j, 1] / 3 + using_image[i, j, 2] / 3
            gray_image[i, j] = n

            # if n>100:
            if n > threshold:
                binary_image[i,j] = 255
                # threshold_image[i, j] = 255

            else:
                binary_image[i, j] = 0
                # threshold_image[i, j] = 0

    scipy.misc.imsave(using_image + "_gray.jpg", gray_image)
    scipy.misc.imsave(binary_image + "_binary.jpg", binary_image)

    # plt.imshow(two_d_image, plt.cm.gray)
    # plt.imshow(two_d_image, plt.cm.binary)

    plt.subplot(1,3,1),plt.imshow(input_image)
    plt.subplot(1,3,2),plt.imshow(gray_image, plt.cm.gray)
    plt.subplot(1,3,3),plt.imshow(binary_image, plt.cm.binary)

img_test = plt.imread("test.jpg")
rgb2gray_image(img_test)