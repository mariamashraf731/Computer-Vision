import cv2
from cv2 import COLOR_BGR2GRAY
import numpy as np



def Gaussian(img, mean, std):
    
    noise_dist = np.random.normal(mean, std, img.shape)         # normal distribuation of the noisee
    #random=np.zeros((img.shape[0], img.shape[1],img.shape[2]),dtype=np.uint8)
    #random=np.random.randn(*random.shape)
    #noise_dist = (np.pi*std) * np.exp(-0.5*((random-mean)/std)**2)
    #print( noise_dist.shape)                                 # noise shape (1411, 1400, 3)
    noise_matrix = noise_dist.reshape(img.shape)              # noise matrix in shape of the image (1411, 1400, 3)
    noisy_image = img + noise_matrix                          # image with added noise
    noisy_image = np.clip(noisy_image, 0, 255)                # limit the result values in range of 0(min) : 255(max)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def Uniform(img, mean, std):

    noise_dist = np.random.uniform(mean, std, img.shape)        
    #random=np.zeros((img.shape[0], img.shape[1],img.shape[2]),dtype=np.uint8)
    #random=np.random.randn(*random.shape)
    #noise_dist = (np.pi*std) * np.exp(-0.5*((random-mean)/std)**2)
    #print( noise_dist.shape)                                # noise shape (1411, 1400, 3)
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



#=================Read Image

img=cv2.imread("apple.jpg")
imgGray=cv2.cvtColor(img,COLOR_BGR2GRAY)

#print(img.shape)

#============Add noies(Uniform & Gaussian & Salt and pepper ) on both Gray and BGR 
Gaussian_noise_BGR= Gaussian(img, 0, 100)
Gaussian_noise_Gray= Gaussian(imgGray, 0, 100)
uniform_noise_BGR=Uniform(img,0,100)
uniform_noise_Gray=Uniform(imgGray,0,100)
Salt_Pepper_noise_BGR=Salt_Pepper(img,0.05)
Salt_Pepper_noise_Gray=Salt_Pepper(imgGray,0.05)

#=================Show images

cv2.imshow("original_image",img)

cv2.imshow("Gaussian_BGR",Gaussian_noise_BGR)
cv2.imshow("Gaussian_Gray",Gaussian_noise_Gray)
cv2.imshow("uniform_BGR",uniform_noise_BGR)
cv2.imshow("uniform_Gray",uniform_noise_Gray)
cv2.imshow("SP_BGR",Salt_Pepper_noise_BGR)
cv2.imshow("SP_Gray",Salt_Pepper_noise_Gray)

#======================Save images

cv2.imwrite("Gaussian_BGR.jpg",Gaussian_noise_BGR)
cv2.imwrite("Gaussian_Gray.jpg",Gaussian_noise_Gray)
cv2.imwrite("uniform_BGR.jpg",uniform_noise_BGR)
cv2.imwrite("uniform_Gray.jpg",uniform_noise_Gray)
cv2.imwrite("SP_BGR.jpg",Salt_Pepper_noise_BGR)
cv2.imwrite("SP_Gray.jpg",Salt_Pepper_noise_Gray)


cv2.waitKey(0)




