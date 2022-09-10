import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def histogram(image):
  h,w=image.shape
  grayscale_array = []
  for px in range(0,h):
    for py in range(0,w):
      intensity = image[px][py] 
      grayscale_array.append(intensity)

  total_pixels = w*h
  bins = range(0,257)
  img_histogram = np.histogram(grayscale_array, bins)
  return img_histogram



def Global_thresholding(image,threshold):
   h,w = np.shape(image)
   # pixel threshold 
   for px in range(0,h):
    for py in range(0,w):
      intensity = image[px][py] 
      if (intensity <= threshold):
        intensity = 0
      else:
        intensity = 255
      image[px][py]=intensity
    
   return image 




def otsu(image):
  h,w=image.shape
  # get pixels values probabilities using histogram 
  hist = histogram(image)
  # get the total number of pixels 
  total_pixels = w*h
  current_max, threshold = 0, 0
  sumT, sumF, sumB = 0, 0, 0
  for i in range(0,256):
    # getting the sum of probabilities of all pixels values 
    sumT += i * hist[0][i]
  weightB, weightF = 0, 0
  varBetween, meanB, meanF = 0, 0, 0

  #Iterating on all pixels' values to get the best threshold 
  for i in range(0,256):
    weightB += hist[0][i]
    weightF = total_pixels - weightB
    #Check if the pixels values represented in one value 
    if weightF == 0:
      break
    sumB += i*hist[0][i]
    sumF = sumT - sumB
    meanB = sumB/weightB
    meanF = sumF/weightF
    varBetween = weightB * weightF
    varBetween *= (meanB-meanF)*(meanB-meanF)
    if varBetween > current_max:
      current_max = varBetween
      threshold = i 
   
  print ("threshold is:", threshold)
  ostu_image=Global_thresholding(image,threshold) 
  return ostu_image



def Bradly_local_thresholding(image):


    h, w = image.shape

    S = int(w//8)
    s2 = int(S//2)
    T = 4

    #integral img
    int_img = np.zeros_like(image, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row,col] = image[0:row,0:col].sum()

    
    Bradly_img = np.zeros_like(image)    

    for col in range(w):
        for row in range(h):
            
            y0 = max(row-s2, 0)
            y1 = min(row+s2, h-1)
            x0 = max(col-s2, 0)
            x1 = min(col+s2, w-1)

            count = (y1-y0)*(x1-x0)

            sum_ =int( int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0])

            if image[row, col]*count < sum_*(100.-T)/100.:
                Bradly_img[row,col] = 0
            else:
                Bradly_img[row,col] = 255
    return Bradly_img




#def ostu_local_threshold():







def main():
# Open the image
    img = np.array(Image.open("jaguar.jpg")).astype(np.uint8)

# Apply gray scale
    gray_img = np.round(0.299 * img[:, :, 0] +
                    0.587 * img[:, :, 1] +
                   0.114 * img[:, :, 2]).astype(np.uint8) 

#Global thresholding
    #threshold=4
    #globalThreshold_img=Global_thresholding(gray_img,threshold)
    ##plt.imshow(globalThreshold_img)
    #plt.set_cmap('gray') 
    #plt.show() 
    #plt.imsave('global_threshold.png', globalThreshold_img, cmap='gray', format='png')


#Local thresholding 
## 1.Bradly threshold 
    #BradlyThreshold_img=Bradly_local_thresholding(gray_img)
    #plt.imshow(BradlyThreshold_img)
    #plt.set_cmap('gray') 
    #plt.show() 
    #plt.imsave('Bradly_local_threshold.png', BradlyThreshold_img, cmap='gray', format='png')

## 2.Ostu threshold 
    BradlyThreshold_img=otsu(gray_img)
    plt.imshow(BradlyThreshold_img)
    plt.set_cmap('gray') 
    plt.show() 
    plt.imsave('Bradly_local_threshold.png', BradlyThreshold_img, cmap='gray', format='png')
      
    

    

if __name__ == '__main__':
    while True:
        main()