import matplotlib.pyplot as plt
import numpy as np
from filters import High_pass_filter , low_pass_filter , inv_FFT_all_channel

def Hybrid(img1,img2,padding_type):
    """
    Parameters:
        img1: first image
        img2:secong image
        padding_type : "constant" , "edge" , "linear_ramp" , "maximum" ..etc   ->see np.pad() documentation for all padding modes

        This function takes two images ,filters one by LPF and the other by HPF then adds them pixel-wise 
        Note:
            Padding is applied if the 2 images have different sizes to make them of equal dimensions
    """
    size1=img1.shape
    size2=img2.shape
    try:
        return(np.abs(inv_FFT_all_channel(High_pass_filter(img1))) + np.abs(inv_FFT_all_channel(low_pass_filter(img2))))
    except:
        height_diff=int(np.ceil(np.abs(size1[0]-size2[0])/2))
        width_diff=int(np.ceil(np.abs(size1[1]-size2[1])/2))
        if (size1[0]-size2[0]>0):
            if (size1[1]-size2[1]>0):
                img2=np.pad(img2,((height_diff,np.abs(size1[0]-size2[0])-height_diff),(width_diff,np.abs(size1[1]-size2[1])-width_diff),(0,0)),padding_type)  #((top, bottom), (left, right))
            else:
                img2=np.pad(img2,((height_diff,np.abs(size1[0]-size2[0])-height_diff),(0,0),(0,0)),padding_type)
                img1=np.pad(img1,((0,0),(width_diff,np.abs(size1[1]-size2[1])-width_diff),(0,0)),padding_type) 

        else:
            if (size1[1]-size2[1]<0):
                img1=np.pad(img1,((height_diff,np.abs(size1[0]-size2[0])-height_diff),(width_diff,np.abs(size1[1]-size2[1])-width_diff),(0,0)),padding_type)  #((top, bottom), (left, right))
            else:
                img2=np.pad(img2,((0,0),(width_diff,np.abs(size1[1]-size2[1])-width_diff),(0,0)),padding_type) 
                img1=np.pad(img1,((height_diff,np.abs(size1[0]-size2[0])-height_diff),(0,0),(0,0)),padding_type)
        return(np.abs(inv_FFT_all_channel(High_pass_filter(img1))) + np.abs(inv_FFT_all_channel(low_pass_filter(img2))))


img1 = plt.imread('cat.jpg')
img2 = plt.imread('dog.jpg')
Hybrid_img = Hybrid(img1,img2,padding_type="edge") 
plt.imshow(Hybrid_img)
plt.axis("off")
plt.show()