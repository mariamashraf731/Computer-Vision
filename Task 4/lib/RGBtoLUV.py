import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL

def RGBtoLUV(Image):
    Image_LUV = np.zeros(Image.shape)
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            color = Image[i,j]
            r = float(color[0])/255
            g = float(color[1])/255
            b = float(color[2])/255

            x = r * 0.412453 + g * 0.357580 + b * 0.180423
            y = r * 0.212671 + g * 0.715160 + b * 0.072169
            z = r * 0.019334 + g * 0.119193 + b * 0.950227
            if y > 0.008856 :
                l_val = 255.0 / 100.0 * (116 * pow(y, 1.0/3.0)-16)
            else:
                l_val = 255.0 / 100.0 * (903.3 * y)
            u = 4 * x / (x + 15 * y + 3 * z)
            v = 9 * y / (x + 15 * y + 3 * z)
            u_val = 255 / 354 * (13 * l_val * (u - 0.19793943) + 134)
            v_val = 255 / 262 * (13 * l_val*(v - 0.46831096)+140)
            Image_LUV[i,j][0] = l_val
            Image_LUV[i,j][1] = u_val
            Image_LUV[i,j][2] = v_val
            print(Image_LUV[i,j])
    return Image_LUV





img = cv2.imread(r'D:\eng mariam\4th\2nd\CV\Computer-Vision-Tasks-main\resources\Images\faces\Lenna_512.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_luv = cv2.cvtColor(np.array(img_rgb).astype('float32')/255, cv2.COLOR_RGB2Luv)
img_luv2 = RGBtoLUV(img_rgb)
img_luv2 = np.array(img_luv2,np.int32)


plt.figure()

plt.subplot(1, 4, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')

plt.subplot(1, 4, 2)
plt.imshow(img_rgb)
plt.axis('off')
plt.title('RGB image')

plt.subplot(1, 4, 3)
plt.imshow(img_luv)
plt.axis('off')
plt.title('CV2_LUV image')

plt.subplot(1, 4, 4)
plt.imshow(img_luv2)
plt.axis('off')
plt.title('LUV image')

plt.show()