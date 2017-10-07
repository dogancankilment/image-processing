import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import timeit


def image_transpose(img):
    image = np.zeros((500,375,3))
    # transpose_image.shape()

    for i in range(375):
        for j in range(500):
            # print img[i,j,:]
            image[j,i,:]=img[i,j,:]

    return image

def image_negative(img):
    image = np.zeros((500, 375, 3))
    # transpose_image.shape()

    for i in range(375):
        for j in range(500):
            # print img[i,j,:]
            image[j, i, :] = 1 - img[i, j, :]

    return image

def mirroring_image(img):
    image = np.zeros((375, 500, 3))
    # transpose_image.shape()

    for i in range(375):
        for j in range(500):
            # x axis mirroring
            image[375-i-1, j, :] = img[i, j, :]
            # y axis mirroring
            image[i, 500-j-1, :] = img[i, j, :]
            # origin mirroring
            image[375-i-1, 500 - j - 1, :] = img[i, j, :]

    return image

def read_image():
    # show future.jpg image on your local
    img = mpimg.imread("future.jpg")
    img.shape()

    plt.imshow(img)
    plt.show()

    return img

def inverse_image(img):
    img_inverse = img[1:375:2, :, :]

    return img_inverse

def max_image(img):
    m_image = img[:,100,:].max()

    return m_image

def ince_image(img):
    img_ince = img[:,1:500:2,:]

    return img_ince

def draw_subplot():
    plt.subplot(1,2,1).plt.imshow(img_inverse)
    plt.subplot(1,2,1).plt.imshow(img_ince)
    plt.show()

    return True

def show_image_shape(img):
    shape = img.ndim,img.shape()

    return shape
    # (3, (375, 500, 3))

def numpy_test():
    # starting time
    start_time = timeit.default_timer()

    # real function
    np.arrange(10)

    # ending time
    print(timeit.default_timer() - start_time)