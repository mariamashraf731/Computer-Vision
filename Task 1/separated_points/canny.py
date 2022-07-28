import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def GaussianBlur(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def SobelFilter(image):
   
  #  initialization of the output image array(sobel x, sobel y , gradient) (all elements are 0)
    H_filtered_image= np.zeros(image.shape)
    V_filtered_image  = np.zeros(image.shape)
    G_filtered_image = np.zeros(image.shape)
    size=image.shape
 # Sobel filters
    vertical_grad_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # kernel Gx
    horizontal_grad_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # kernel Gy
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            H_filtered_image[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2],   vertical_grad_filter))
            V_filtered_image[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2],horizontal_grad_filter))
    
    G_filtered_image= np.sqrt(np.square(H_filtered_image) + np.square(V_filtered_image))
    G_filtered_image = np.multiply( G_filtered_image, 255.0 /  G_filtered_image.max())
   

    angles = np.rad2deg(np.arctan2(V_filtered_image, H_filtered_image))
    angles[angles < 0] += 180
    G_filtered_image = G_filtered_image.astype('uint8')
    return G_filtered_image, angles


def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
            
            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed

    
def double_threshold_hysteresis(image, low, high):
    weak = 50
    strong = 255
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape
    
    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result


image = np.array(Image.open('jaguar.jpg')).astype(np.uint8)
# Convert image to gray scale(Weighted (luminosity))
Gray_img = (0.299 * image[:, :, 0] +0.587 * image[:, :, 1] +0.114 * image[:, :, 2]).astype(np.uint8)
image=GaussianBlur(Gray_img)
image, angles = SobelFilter(Gray_img)
image = non_maximum_suppression(image, angles)
gradient = np.copy(image)
image = double_threshold_hysteresis(image, 0, 50)

plt.imsave('canny.jpg', image, cmap='gray', format='jpg')