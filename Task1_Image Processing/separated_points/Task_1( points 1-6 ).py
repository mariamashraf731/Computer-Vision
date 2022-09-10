from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt



#============================================1.additive noise Functions======================================================


def Gaussian(img, mean, std):
    
    noise_dist = np.random.normal(mean, std, img.shape)       # normal distribuation of the noisee
    noise_matrix = noise_dist.reshape(img.shape)              # noise matrix in shape of the image (1411, 1400, 3)
    noisy_image = img + noise_matrix                          # image with added noise
    noisy_image = np.clip(noisy_image, 0, 255)                # limit the result values in range of 0(min) : 255(max)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def Uniform(img, mean, std):

    noise_dist = np.random.uniform(mean, std, img.shape)        
    noise_matrix = noise_dist.reshape(img.shape)             # noise matrix in shape of the image (1411, 1400, 3)
    noisy_image = img + noise_matrix                         # image with added noise
    noisy_image = np.clip(noisy_image, 0, 255)               # limit the result values in range of 0(min) : 255(max)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def Salt_Pepper(image,prob):

    SP_noisy = np.zeros(image.shape,np.uint8) 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            random_pixs = np.random.random()
            if random_pixs < prob:
                SP_noisy[i][j] = 0
            elif random_pixs> (1 - prob):
               SP_noisy[i][j] = 255
            else:
               SP_noisy[i][j] = image[i][j]
    return SP_noisy

#========================================Filtering noise Functions================================================================
def convolve(image,filter="Gaussian"):
    m, n = image.shape
    filtered_img=np.zeros((m,n))
    if filter=="Median":
        for i in range(m-2):  #rows
            for j in range(n-2):    #columns
                filtered_img[i:i+3,j:j+3]=image[i:i+3,j:j+3]
                window=image[i:i+3,j:j+3]
                list=window.flatten()
                median=np.sort(list)[4]
                filtered_img[i+1,j+1]=median
    #  padding is added in upcomming filters to generate the same size of input image
    else:           
        if filter=="Gaussian":
            kernel=(1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]])

        elif filter=="Average":
            kernel=(1/9)*np.ones((3,3))

        for i in range(m-2):  #rows
            for j in range(n-2):    #columns
                filtered_img[i+1,j+1]=np.sum((image[i:i+3,j:j+3])*(kernel))
 
    return filtered_img 

#=======================================================Detect edges in the image==============================================================

def Prewitt(img):  
  
    # Prewitt Operator
    prewitt_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    prewitt_y = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]

   
    w,h = img.shape
    newgradientImage = np.zeros((w, h))

    #Applying gradient for each pixel in image 
    for row in range(w-len(prewitt_x)):
    
            for col in range(h-len(prewitt_x)):
                Gx = 0
                Gy = 0
                for i in range(len(prewitt_x)):
                    for j in range(len(prewitt_y)):
                        pixelVal = img[row+i, col+j] 
                        Gx += prewitt_x[i][j] * pixelVal 
                        Gy += prewitt_y[i][j] * pixelVal 

                newgradientImage[row+1,col+1]= int(math.sqrt(Gx*Gx + Gy*Gy))
    return newgradientImage
    #plt.imsave('jaguar-prewitt.png', newgradientImage, cmap='gray', format='png')


def Roberts(img):

    roberts_x = [[1,0],[0,-1]]
    roberts_y = [[0,1],[-1,0]]
    w,h = img.shape
    newgradientImage = np.zeros((w, h))
    #Applying gradient for each pixel in image 
    for row in range(w-len(roberts_x)):
        for col in range(h-len(roberts_y)):
            Gx = 0
            Gy = 0
            for i in range(len(roberts_x)):
                for j in range(len(roberts_y)):
                    val = img[row+i, col+j] 
                    Gx += roberts_x[i][j] * val
                    Gy += roberts_y[i][j] * val
            newgradientImage[row+1,col+1] = int(math.sqrt(Gx*Gx + Gy*Gy))
    return newgradientImage
    #plt.imsave('jaguar-roberts.png', newgradientImage, cmap='gray', format='png')

def sobel (image):


    # image shape
    H, V = image.shape   

    # Sobel filters
    vertical_grad_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # kernel Gx
    horizontal_grad_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # kernel Gy

    #  initialization of the output image array(sobel x, sobel y , gradient) (all elements are 0)
    H_filtered_image= np.zeros((H,V))
    V_filtered_image  = np.zeros((H,V))
    G_filtered_image = np.zeros((H,V))


    # Scan the image in both x and y directions
    for i in range(H - 2):
        for j in range(V - 2):

            gx = np.sum(np.multiply(vertical_grad_filter, image[i:i + 3, j:j + 3]))  # x direction
            H_filtered_image[i + 1, j + 1] = abs(gx)
            gy = np.sum(np.multiply(horizontal_grad_filter, image[i:i + 3, j:j + 3]))  # y direction
            V_filtered_image[i + 1, j + 1]=abs(gy)
            G_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  

    return  np.uint8(abs(H_filtered_image)),np.uint8(abs(V_filtered_image)), np.uint8(abs(G_filtered_image))
    #return  H_filtered_image,V_filtered_image, G_filtered_image

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

#====================================================histogram and distribution curve==========================


#===================================================Equalize the image.============================

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


#=================Read Image in BGR and Gray scale

# Open the Original image

img = np.array(Image.open('jaguar.jpg')).astype(np.uint8)

# Apply gray scale
gray_img = (0.299 * img[:, :, 0] +0.587 * img[:, :, 1] +0.114 * img[:, :, 2]).astype(np.uint8) 

#===============1.Add noies(Uniform & Gaussian & Salt and pepper ) on both Gray and BGR 

Gaussian_noise_BGR= Gaussian(img, 0, 100)
Gaussian_noise_Gray= Gaussian(gray_img, 0, 100)
uniform_noise_BGR=Uniform(img,0,100)
uniform_noise_Gray=Uniform(gray_img,0,100)
Salt_Pepper_noise_BGR=Salt_Pepper(img,0.05)
Salt_Pepper_noise_Gray=Salt_Pepper(gray_img,0.05)

plt.imsave("Gaussian_BGR.jpg", Gaussian_noise_BGR, cmap='gray', format='jpg')
plt.imsave("Gaussian_Gray.jpg", Gaussian_noise_Gray, cmap='gray', format='jpg')
plt.imsave("uniform_BGR.jpg", uniform_noise_BGR, cmap='gray', format='jpg')
plt.imsave("uniform_Gray.jpg", uniform_noise_Gray, cmap='gray', format='jpg')
plt.imsave("SP_BGR.jpg",Salt_Pepper_noise_BGR, cmap='gray', format='jpg')
plt.imsave("SP_Gray.jpg", Salt_Pepper_noise_Gray, cmap='gray', format='jpg')

#============2.Filter the noisy image using the following low pass filters.-Average, Gaussian and median filters

image_s_p = np.array(Image.open('SP_Gray.jpg')).astype(np.uint8)
image_s_p = (0.299 * image_s_p[:, :, 0] +0.587 * image_s_p[:, :, 1] +0.114 * image_s_p[:, :, 2]).astype(np.uint8) 
image_gauss = np.array(Image.open('Gaussian_Gray.jpg')).astype(np.uint8)
image_gauss = (0.299 * image_gauss [:, :, 0] +0.587 * image_gauss [:, :, 1] +0.114 * image_gauss [:, :, 2]).astype(np.uint8)
image_uni = np.array(Image.open('uniform_Gray.jpg')).astype(np.uint8)
image_uni = (0.299 * image_uni [:, :, 0] +0.587 * image_uni [:, :, 1] +0.114 * image_uni [:, :, 2]).astype(np.uint8)

#choose your filter (default is gaussian) and convolve it with image
filtered_median=convolve(image_s_p,"Median")        #median best suited for salt and paper
filtered_gaussian=convolve(image_gauss,"Gaussian")  #gaussian best suited for gaussian
filtered_average=convolve(image_uni ,"Average")     #average best suited for uniform

plt.imsave("filtered_median.jpg",filtered_median, cmap='gray', format='jpg')
plt.imsave("filtered_gaussian.jpg",filtered_gaussian, cmap='gray', format='jpg')
plt.imsave("filtered_average.jpg", filtered_average, cmap='gray', format='jpg')

#================3.Detect edges in the image using the following masks-Sobel , Roberts , Prewitt and Canny edge detectors.

sobel_x,sobel_y,gradient=sobel(gray_img)
Prewitt_Gray= Prewitt(gray_img)
Roberts_Gray=Roberts(gray_img)
image, angles = SobelFilter(gray_img)
image = non_maximum_suppression(image, angles)
gradient = np.copy(image)
canny = double_threshold_hysteresis(image, 0, 50)

plt.imsave('sobelx.jpg', sobel_x, cmap='gray', format='jpg')
plt.imsave('sobely.jpg', sobel_y, cmap='gray', format='jpg')
plt.imsave('sobel.jpg', gradient, cmap='gray', format='jpg')

plt.imsave('Prewitt_Gray.jpg', Prewitt_Gray, cmap='gray', format='jpg')
plt.imsave('Roberts_Gray.jpg', Roberts_Gray, cmap='gray', format='jpg')

plt.imsave('canny.jpg', canny, cmap='gray', format='jpg')

#=======================4.

#=======================5.Equalize the image.

# convert our image into a numpy array
img = np.asarray(gray_img)
# put pixels in a 1D array by flattening out img array
flat = img.flatten()
imgNew = equalize(flat)

plt.imsave('equalize.jpg', imgNew, cmap='gray', format='jpg')




















