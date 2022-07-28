import numpy as np




def grayImg(img):
    # Apply gray scale
    gray_img = np.round(0.299 * img[:, :, 0] +
                    0.587 * img[:, :, 1] +
                    0.114 * img[:, :, 2]).astype(np.uint8) 

    return gray_img 