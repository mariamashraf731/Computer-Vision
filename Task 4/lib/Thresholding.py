import cv2
import grayScale
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def histogram(image):
    h, w = image.shape
    grayscale_array = []
    for px in range(0, h):
        for py in range(0, w):
            intensity = image[px][py]
            grayscale_array.append(intensity)
    bins = range(0, 255)
    img_histogram = np.histogram(grayscale_array, bins)
    return img_histogram


def Global_thresholding(image, threshold):
    h, w = np.shape(image)
    # pixel threshold
    for px in range(0, h):
        for py in range(0, w):
            intensity = image[px][py]
            if (intensity <= threshold):
                intensity = 0
            else:
                intensity = 255
            image[px][py] = intensity

    return image


def localThresholding(img, nx, ny, thresholdingOption):

    grayImage = np.copy(img)

    if len(img.shape) > 2:
        grayImage = grayScale.grayImg(grayImage)
    else:
        pass

    YMax, XMax = grayImage.shape
    subImage = np.zeros((YMax, XMax))
    yWindowSize = YMax // ny
    xWindowSize = XMax // nx
    xWindow = []
    yWindow = []
    for i in range(0, nx):
        xWindow.append(xWindowSize * i)

    for i in range(0, ny):
        yWindow.append(yWindowSize * i)

    xWindow.append(XMax)
    yWindow.append(YMax)
    for x in range(0, nx):
        for y in range(0, ny):
            if thresholdingOption == getOptimmalThreshold:

                newimg = grayImage[yWindow[y]
                    :yWindow[y + 1], xWindow[x]:xWindow[x + 1]]
                im, bins = np.histogram(newimg, range(257))
                subImage[yWindow[y]:yWindow[y + 1], xWindow[x]:xWindow[x + 1]] = getOptimmalThreshold(im, 200
                                                                                                      )
            else:
                subImage[yWindow[y]:yWindow[y + 1], xWindow[x]:xWindow[x + 1]] = thresholdingOption(
                    grayImage[yWindow[y]:yWindow[y + 1], xWindow[x]:xWindow[x + 1]])

    return subImage


def otsuThresholding(img):
    grayImage = np.copy(img)
    if len(img.shape) > 2:
        grayImage = grayScale.grayImg(img)
    else:
        pass
    yWindowSize, xWindowSize = grayImage.shape
    # get pixels values probabilities using histogram
    HistValues = plt.hist(grayImage.ravel(), 256)
    # get the total number of pixels
    total_pixels = xWindowSize*yWindowSize
    current_max, threshold = 0, 0
    sumT, sumF, sumB = 0, 0, 0
    for i in range(0, 256):
        # getting the sum of probabilities of all pixels values
        sumT += i * HistValues[0][i]
    weightB, weightF = 0, 0
    varBetween, meanB, meanF = 0, 0, 0

    # Iterating on all pixels' values to get the best threshold
    for i in range(0, 256):
        weightB += HistValues[0][i]
        weightF = total_pixels - weightB
        # Check if the pixels values represented in one value
        if weightF == 0:
            break
        sumB += i*HistValues[0][i]
        sumF = sumT - sumB
        meanB = sumB/weightB
        meanF = sumF/weightF
        varBetween = weightB * weightF
        varBetween *= (meanB-meanF)*(meanB-meanF)
        if varBetween > current_max:
            current_max = varBetween
            threshold = i

    # print("threshold is:", threshold)
    ostu_image = Global_thresholding(grayImage, threshold)
    return ostu_image


def spectralThresholding(img):

    grayImage = np.copy(img)
    if len(img.shape) > 2:
        grayImage = grayScale.grayImg(grayImage)
    else:
        pass
    # Get Image Dimensions
    yWindowSize, xWindowSize = grayImage.shape
    # Get The Values of The Histogram Bins
    HistValues = plt.hist(grayImage .ravel(), 256)[0]
    # Calculate The Probability Density Function
    PDF = HistValues / (yWindowSize * xWindowSize)
    # Calculate The Cumulative Density Function
    CDF = np.cumsum(PDF)
    OptimalLow = 1
    OptimalHigh = 1
    MaxVariance = 0
    # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
    Global = np.arange(0, 256)
    GMean = sum(Global * PDF) / CDF[-1]
    for LowT in range(1, 254):
        for HighT in range(LowT + 1, 255):
            # Background Intensities Array
            Back = np.arange(0, LowT)
            # Low Intensities Array
            Low = np.arange(LowT, HighT)
            # High Intensities Array
            High = np.arange(HighT, 256)
            # Get Low Intensities CDF
            CDFL = np.sum(PDF[LowT:HighT])
            # Get Low Intensities CDF
            CDFH = np.sum(PDF[HighT:256])
            # Calculation Mean of Background & The Object (Foreground), Based on CDF & PDF
            BackMean = sum(Back * PDF[0:LowT]) / CDF[LowT]
            LowMean = sum(Low * PDF[LowT:HighT]) / CDFL
            HighMean = sum(High * PDF[HighT:256]) / CDFH
            # Calculate Cross-Class Variance
            Variance = (CDF[LowT] * (BackMean - GMean) ** 2 + (CDFL * (LowMean - GMean) ** 2) + (
                CDFH * (HighMean - GMean) ** 2))
            # Filter Out Max Variance & It's Threshold
            if Variance > MaxVariance:
                MaxVariance = Variance
                OptimalLow = LowT
                OptimalHigh = HighT
        """
       Apply Double Thresholding To Image to get the Lowest Allowed Value using Low Threshold Ratio/Intensity and the Minimum Value To Be Boosted using High Threshold Ratio/Intensity
       """
    # Create Empty Array
    ThresholdedImage = np.zeros(grayImage.shape)

    HighPixel = 255
    LowPixel = 128

    # Find Position of Strong & Weak Pixels
    HighRow,  HighCol = np.where(grayImage >= OptimalHigh)
    LowRow, LowCol = np.where(
        (grayImage <= OptimalHigh) & (grayImage >= OptimalLow))

    # Apply Thresholding
    ThresholdedImage[HighRow, HighCol] = HighPixel
    ThresholdedImage[LowRow, LowCol] = LowPixel

    return ThresholdedImage


def globaloptimalThreshold(grayimg):
    im, bins = np.histogram(grayimg, range(257))
    optimalthreshold = getOptimmalThreshold(im, 200)
    # print(optimalthreshold)
    # Map threshold with the original image
    segImg = grayimg > optimalthreshold
    return segImg


def getOptimmalThreshold(im, threshold):
    # devide image into two sections "Background & foreGround"
    # im, bins = np.histogram(im, range(257))
    back = im[:threshold]
    fore = im[threshold:]
    # Compute the centroids or Mean
    mBack = (back*np.arange(0, threshold)).sum()/back.sum()
    mFore = (fore*np.arange(threshold, len(im))).sum()/fore.sum()
    # New threshold
    NewThreshold = int(np.round((mBack+mFore)/2))
    # print(mBack, mFore, NewThreshold)
    # Recursion with the newthreshold till the threshold is the same "const"
    if(NewThreshold != threshold):
        return getOptimmalThreshold(im, NewThreshold)
    return NewThreshold


imgGray = cv2.imread("./images/ct_scan.png", cv2.IMREAD_GRAYSCALE)
npImg = np.array(imgGray)
# get global threshold

# globalThresholdImg = globaloptimalThreshold(npImg)
# fig = plt.figure(1)
# plt.imshow(globalThresholdImg, cmap=plt.cm.gray)


# get localThreshold

# localOptThreshold = localThresholding(npImg, 1, 1, getOptimmalThreshold)
# print(localOptThreshold)
# localOptimalImg = npImg > localOptThreshold
# plt.imshow(localOptimalImg, cmap=plt.cm.gray)
# plt.savefig("./Output/ctlocaloptimal.png", bbox_inches='tight')
# plt.show()

image = cv2.imread('./images/jaguar.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
locImg = localThresholding(npImg, 10, 10, otsuThresholding)
# globalImg = spectralThresholding(image)

# fig, ax = plt.subplots(figsize=(15, 15))
plt.imshow(locImg, cmap=plt.cm.gray)
plt.show()
# plt.savefig("jaguarotsuImage.png")
# cv2.imwrite('./Output/localjaguarotsuImage.png', locImg)
