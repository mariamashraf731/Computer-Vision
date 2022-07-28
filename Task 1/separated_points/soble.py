import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#==========================# Sobel on gray scale

def sobel (image):

   # Convert image to gray scale(Weighted (luminosity))
    Gray_img = (0.299 * image[:, :, 0] +0.587 * image[:, :, 1] +0.114 * image[:, :, 2]).astype(np.uint8)

    # image shape
    H, V = Gray_img.shape   

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

            gx = np.sum(np.multiply(vertical_grad_filter, Gray_img[i:i + 3, j:j + 3]))  # x direction
            H_filtered_image[i + 1, j + 1] = abs(gx)
            gy = np.sum(np.multiply(horizontal_grad_filter, Gray_img[i:i + 3, j:j + 3]))  # y direction
            V_filtered_image[i + 1, j + 1]=abs(gy)
            G_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  

    #return  np.uint8(abs(H_filtered_image)),np.uint8(abs(V_filtered_image)), np.uint8(abs(G_filtered_image))
    return  H_filtered_image,V_filtered_image, G_filtered_image

#============================================Run================================    

# Open the image
image = np.array(Image.open('jaguar.jpg')).astype(np.uint8)


sobel_x,sobel_y,gradient=sobel(image)

# plt.figure()
# plt.title('Apple.png')
plt.imsave('sobelx.jpg', sobel_x, cmap='gray', format='jpg')
plt.imsave('sobely.jpg', sobel_y, cmap='gray', format='jpg')
plt.imsave('sobel.jpg', gradient, cmap='gray', format='jpg')

# plt.imshow(abs(H_filtered_image), cmap='gray')
# plt.imshow( abs(V_filtered_image), cmap='gray')
# plt.imshow(abs(G_filtered_image), cmap='gray')
# plt.show()

#==========================================================================================


# import cv2 as cv
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# #load birds image
# image = np.array(Image.open('lena.jpg')).astype(np.uint8)

# #convert to gray image
# gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# #detect sobel gradients
# sobel_x_edges = cv.Sobel(gray_image, cv.CV_64F,1, 0)
# sobel_y_edges = cv.Sobel(gray_image, cv.CV_64F,0, 1)

# #convert all -ve pixels to positives
# sobel_x_edges = np.uint8(np.absolute(sobel_x_edges))
# sobel_y_edges = np.uint8(np.absolute(sobel_y_edges))

# # #show images
# # plt.figure()
# # plt.title('sobel.png')
# #plt.imsave('sobelx.png', sobel_x_edges, cmap='gray', format='png')
# #plt.imsave('sobely.png', sobel_y_edges, cmap='gray', format='png')
# #plt.imshow(sobel_x_edges, cmap='gray')
# # plt.show()
# cv.imshow("Sobel X Edges", sobel_x_edges)
# cv.imshow("Sobel y Edges", sobel_y_edges)

# cv.waitKey(0)


