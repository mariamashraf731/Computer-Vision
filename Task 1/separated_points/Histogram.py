import cv2
import numpy as np
import matplotlib.pyplot as plt

# create our own histogram function


def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)

    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[int(pixel)] += 1

    # return our final result
    return histogram


img = cv2.imread('jaguar.jpg', 0)

# To ascertain total numbers of rows and
# columns of the image, size of the image
# m, n = img.shape
pixles = np.linspace(0, 255, 256)
freqs = get_histogram(img.flatten(), 256)  # hist_plot(img)

# plotting the histogram
plt.stem(pixles, freqs)
plt.xlabel('intensity value')
plt.ylabel('number of pixels')
plt.title('Histogram of the original image')
plt.show()
