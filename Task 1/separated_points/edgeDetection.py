
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

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
    plt.imsave('jaguar-prewitt.png', newgradientImage, cmap='gray', format='png')




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
    plt.imsave('jaguar-roberts.png', newgradientImage, cmap='gray', format='png')




def main():
# Open the image
    img = np.array(Image.open('jaguar.jpg')).astype(np.uint8)

# Apply gray scale
    gray_img = np.round(0.299 * img[:, :, 0] +
                    0.587 * img[:, :, 1] +
                    0.114 * img[:, :, 2]).astype(np.uint8) 
#Prewitt 
    Prewitt(gray_img)

#Roberts
    Roberts(gray_img)

if __name__ == '__main__':
    while True:
        main()