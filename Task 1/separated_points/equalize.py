import cv2
import numpy as np
import matplotlib.pyplot as plt

# create our own histogram function


def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)

    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1

    # return our final result
    return histogram


def acumulativeSum(histogram):
    accumulativeList = []
    j = 0
    for i in range(0, len(histogram)):
        j += histogram[i]
        accumulativeList.append(j)
    return accumulativeList


def normalization(accArr):
    # Formula sum of sk = sum(nj/N)
    nj = (accArr-min(accArr))*255
    N = max(accArr) - min(accArr)
    # re-normalize the cumsum
    accArr = nj/N
    # cast it back to uint8 since we can't use floating point values in images
    accArr = accArr.astype('uint8')
    return accArr


def mappingNewImage(arr):
    # put array back into original shape since we flattened it
    img_new = np.reshape(arr, img.shape)
    return img_new


def equalize(flat):
    # execute our histogram function
    hist = get_histogram(flat, 256)
    # get the acumulative sum of histogram
    cs = acumulativeSum(hist)
    # normalize cs from 0 to 255
    normalizedCs = normalization(cs)
    # get the value from cumulative sum for every index in flat, and set that as img_new
    imgNew = mappingNewImage(normalizedCs[flat])
    return imgNew


# image = cv2.imread('./apple.jpg')
# # convert our image into a numpy array
# img = np.asarray(image)
# # put pixels in a 1D array by flattening out img array
# flat = img.flatten()

# imgNew = equalize(flat)
# # set up side-by-side image display
# fig = plt.figure()
# fig.set_figheight(15)
# fig.set_figwidth(15)
# fig.add_subplot(1, 2, 1)
# plt.imshow(img)

# # display the new image
# fig.add_subplot(1, 2, 2)
# plt.imshow(imgNew)
# plt.show(block=True)
