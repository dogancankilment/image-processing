#test_image = plt.imread("test.jpg")

def count_mask(img_1, mask):
        counter = 0
        control_a = False
        control_b = False
        control_c = False
        control_d = False
        
        m,n = img_1.shape
        
        for i in range(m-1):
            for j in range(n-1):
                if (img_1[i,j]==mask[0][0]):
                    control_a = True
                if (img_1[i,j+1]==mask[0][1]):
                    control_b = True
                if (img_1[i+1,j]==mask[1][0]):
                    control_c = True
                if (img_1[i+1,j+1]==mask[1][1]):
                    control_d = True
                if (control_a and control_b and control_c and control_d):
                    counter = counter + 1
        return counter

i_m = [internal_mask_1, internal_mask_2, internal_mask_3, internal_mask_4]
e_m = [external_mask_1, external_mask_2, external_mask_3, external_mask_4]

import math

def count_internal_mask(image):
    counter_internal = 0
    for mask in i_m:
        counter_internal = counter_internal + count_mask(img_1,mask)
    return counter_internal

def count_external_mask(image):
    counter_external = 0
    for mask in e_m:
        counter_external = counter_external + count_mask(img_1,mask)
    return counter_external

c1=count_internal_mask(img_1)
c2=count_external_mask(img_1)
math.abs(c1-c2)/4

import matplotlib.pyplot as plt
plt.imshow(img_1, cmap='Greys', interpolation='nearest')
plt.show()

import numpy as np
# img_1 = np.random.randint(10)
size = 2
# img_1 = np.random.randint(0,2,(size,size))
img_1 = np.matrix('0,0,0,0,0 ; 0,1,1,1,0 ; 0,0,1,0,0 ; 0,0,1,0,0; 0,0,0,0,0')
img_1 # random test image
