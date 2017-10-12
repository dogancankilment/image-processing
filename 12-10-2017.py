import matplotlib.pyplot as plt
import numpy as np
import scipy

plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()

img_test = plt.imread("test.jpg")

img_test.ndim()
img_test.shape()

plt.imshow(img_test)

# red
plt.imshow(img_test[:,:,0,+10])
# green
plt.imshow(img_test[:,:,1,+10])
# blue
plt.imshow(img_test[:,:,2,+10])

rgb_pixel = [0,0,0]
gray_level_pixel = 0

# rgb_pixel = [10,0,0]
# gray_level_pixel = 10

# rgb_pixel = [0,10,0]
# gray_level_pixel = 10

# rgb_pixel = [10,10,10]
# gray_level_pixel = 12

# rgb pixel to gray level
def rgb2gray(rgb):
    return rgb[0]/3 + rgb[1]/3 + rgb[2]/3
    # gray_level = gray_level / 3


def rgb2gray_image(input_image):
    three_d_image = plt.imread(input_image)
    two_d_image = np.zeros(three_d_image.shape[0], three_d_image.shape[1])
    for i in range(three_d_image.shape[0]):
        for j in range(three_d_image.shape[1]):
            two_d_image[i, j]= three_d_image[0]/3 + three_d_image[1]/3 + three_d_image[2]/3

    scipy.misc.imsave(three_d_image + "_gray.jpg", two_d_image)

    plt.subplot(1, 2, 1)
    plt.imshow(three_d_image)
    plt.subplot(1, 2, 2)
    plt.imshow(img_2, cmap='gray')
    plt.show()

rgb2gray_image(img_test)

rgb2gray([2,5,7])
img_1 = plt.imread("test.jpg")
img_2 = np.zeros((img_1.shape[0],img_1.shape[1]))

for i in range(img_1.shape[0]):
    for j in range(img_2.shape[1]):
        img_2[i, j] = rgb2gray(img_1[i, j, :])

plt.subplot(1, 2, 1)
plt.imshow(img_1)
plt.subplot(1, 2, 2)
plt.imshow(img_2, cmap='gray')
plt.show()


scipy.misc.imread()

