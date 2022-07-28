from cProfile import label
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import Histogram as histo
import equalize as eq

# Read Images
img = mpimg.imread('jaguar.jpg')
img = np.asarray(img)
pixles = np.linspace(0, 255, 256)
# get R,G,B channels
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]


def RGBHistogram(r, g, b):
    Rflat = r.flatten()
    Gflat = g.flatten()
    Bflat = b.flatten()
    Rhisto = histo.get_histogram(Rflat, 256)
    Ghisto = histo.get_histogram(Gflat, 256)
    Bhisto = histo.get_histogram(Bflat, 256)
    return Rhisto, Ghisto, Bhisto

# convert RGB to GreyScale


def RGBtoGrey(rgbimg):
    R, G, B = rgbimg[:, :, 0], rgbimg[:, :, 1], rgbimg[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return imgGray


imgGray = RGBtoGrey(img)
# get R G B histograms
Rhisto, Ghisto, Bhisto = RGBHistogram(R, G, B)
# get R g B Distribution curves
Rcs = eq.acumulativeSum(Rhisto)
Gcs = eq.acumulativeSum(Ghisto)
Bcs = eq.acumulativeSum(Bhisto)
# # set up side-by-side image display img gray and histo and Cum Sum


fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)
fig.add_subplot(2, 2, 1)
plt.title('The RGB original image')
plt.imshow(img)
fig.add_subplot(2, 2, 2)
plt.title(' Grey Scale Image')
plt.imshow(imgGray, cmap='gray')
plt.savefig('grey.png')
fig.add_subplot(2, 2, 3)
plt.title(' RGB Histograms')
plt.plot(pixles, Bhisto)
plt.plot(pixles, Rhisto)
plt.plot(pixles, Ghisto)

# display the new image
fig.add_subplot(2, 2, 4)
plt.title(' RGB Distribution curves')
plt.plot(Gcs, label="Green")
plt.plot(Rcs, label="Red")
plt.plot(Bcs, label='Blue')

plt.show(block=True)
