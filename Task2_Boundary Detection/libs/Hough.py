import os 
import cv2
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import math

def grayImg(img):
    # Apply gray scale
    gray_img = np.round(0.299 * img[:, :, 0] +
                    0.587 * img[:, :, 1] +
                    0.114 * img[:, :, 2]).astype(np.uint8)  

    return gray_img 


def canny_edge_detection(img,lower=0.4,upper=1.3):
  
    # Using OTSU thresholding - bimodal image
    otsu_threshold_val, ret_matrix = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    lower_threshold = otsu_threshold_val * lower
    upper_threshold = otsu_threshold_val * upper
    
    
    edges = cv2.Canny(img, lower_threshold, upper_threshold)
    return edges

#===================================================Line detection=========================================

def hough_line(edge,added):
    #1.finite number of possible values of Rho and Theta 
    theta = np.deg2rad(np.arange(0, 180,1))
    cos = np.cos(theta)
    sin = np.sin(theta)
    num_angles = len(theta)
    
    width, height = edge.shape
    distance= round(math.sqrt(width **2 + height **2))

    #2. 2D array (accumulator) for the Hough Space of n_rhos(2*d) and n_thetas(180).
    accumulator = np.zeros((2 * distance, num_angles), dtype=np.uint8)
  
    # Threshold to get edges pixel location (x,y)
    edge_pixels = np.where(edge != 0)
    x_idxs= edge_pixels[0]
    y_idxs= edge_pixels[1]
    # loop through all possible angles, calculate rho , find theta and rho index , and increment the accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t in range(len(theta)):
            rho = int(round(y * cos[t] + x * sin[t]))
            accumulator[rho, t] += added

    return accumulator

def superimpose(image,acc,count):

     pixels = np.where(acc > count)
     rhos = pixels[0]
     angles =  pixels[1] 
     for i in range(len(pixels[0])):
        a = np.cos(np.deg2rad(angles [i]))
        b = np.sin(np.deg2rad(angles [i]))
        x0 = a*rhos[i]
        y0 = b*rhos[i]
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(image, pt1,pt2,(0,255,0),1)

    # show result
     plt.subplot(133), plt.imshow(image)
     plt.subplot(132), plt.imshow(acc,cmap="gray")
     plt.show()

#============================================================================================================
def HoughCircles(img,rMin, rMax,binTh,accTh,deltaR=1):

    #Edge detection on the input image
    grayImage=grayImg(img)
    edgeImage=canny_edge_detection(grayImage)

    #cv2.imshow('Edge Image',edgeImage)
    #cv2.waitKey(0)

    imageGradX, imageGradY = np.where(edgeImage== 255)

    ## Creating Thetas from 0 to 360 degree 
    thetaRange = np.arange(0, 360)
    ## Creating rRange from rMin to rMax degree with step deltaR , default of deltaR=1 
    rRange = np.arange(rMin,rMax,step=deltaR)  
    # Initializing up the accumulator
    accumulator = np.array([[[0]*len(edgeImage[0])]*len(edgeImage)]*len(rRange))
    
    # Performing the transform and filling the accumulator
    for theta in thetaRange:
        for r in rRange:
            rcosTheta = int(np.round(r*np.cos(np.deg2rad(theta))))
            rsinTheta = int(np.round(r*np.sin(np.deg2rad(theta))))
            for x, y in zip(imageGradX, imageGradY):
                i = x-rcosTheta
                j = y-rsinTheta
                if (i>0 and j>0 and i<len(edgeImage) and j<len(edgeImage[0])):
                    accumulator[r-binTh,i,j]+=1
    
    imageCopy = img.copy()
    # Get only circles with atleast binTh hits in the accumulator cells
    R, I, J = np.where(accumulator>accTh)
    
    circles = set({})
    for r, i, j in zip(R, I, J):
    
       """ Considering lines within a bin of r-binTh and r+binTh and I-binTh and I+binTh and J-binTh and J+binTh 
       # to be one circle to avoid labelling the same line as multiple lines. """
       ind = np.where((R>=r-binTh) & (R<r+binTh) & (I<i+binTh) & (I>=i-binTh) & (J<j+binTh) & (J>=j-binTh))
       if R[ind].size > 0 or I[ind].size > 0 or J[ind].size>0:
        r = int(np.round(np.mean(R[ind])))
        i = int(np.round(np.mean(I[ind])))
        j = int(np.round(np.mean(J[ind])))
       else:
          continue
        
       circles.add((i,j,r))
       cv2.circle(imageCopy, (j,i), r+binTh, [0,255,0], thickness=2, lineType=8, shift=0)        

    print("Number of detected Circles: ", len(circles))
    cv2.imwrite('NewDetectedCircles.jpg',imageCopy)
    return imageCopy
    
   
