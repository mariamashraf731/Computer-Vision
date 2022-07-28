
import cv2
import numpy as np

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

#read image
image_s_p = cv2.imread('SP_BGR.jpg',0)
image_gauss = cv2.imread('Gaussian_BGR.jpg',0)
image_uni = cv2.imread('uniform_BGR.jpg',0)

#choose your filter (default is gaussian) and convolve it with image
filtered_median=convolve(image_s_p,"Median")   #median best suited for salt and paper
filtered_gaussian=convolve(image_gauss,"Gaussian")  #gaussian best suited for gaussian
filtered_average=convolve(image_uni ,"Average")   #average best suited for uniform

#save images in RGB
filtered_median = filtered_median.astype(np.uint8)
filtered_gaussian= filtered_gaussian.astype(np.uint8)
filtered_average = filtered_average.astype(np.uint8)
filtered_median = cv2.cvtColor(filtered_median,cv2.COLOR_GRAY2BGR)
filtered_gaussian= cv2.cvtColor(filtered_gaussian,cv2.COLOR_GRAY2RGB)
filtered_average  = cv2.cvtColor(filtered_average,cv2.COLOR_GRAY2RGB)
cv2.imwrite('filtered_median.jpg', filtered_median)
cv2.imwrite('filtered_gaussian.jpg', filtered_gaussian)
cv2.imwrite('filtered_average.jpg', filtered_average)
