import cv2
import numpy as np




def canny_edge_detection(img):
  
    # Using OTSU thresholding - bimodal image
    otsu_threshold_val, ret_matrix = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    lower_threshold = otsu_threshold_val * 0.4
    upper_threshold = otsu_threshold_val * 1.3
    
    print(lower_threshold,upper_threshold)
    
    edges = cv2.Canny(img, lower_threshold, upper_threshold)
    return edges