# Computer Vision SBE404B

# Assignment1- Image filtering, processing, edge detection, hybrid images

#####

| Submitted by:            | Sec. | B.N. |
| ------------------------ | ---- | ---- |
| Ashar Seif el Nasr Saleh | 1    | 9    |
| Alaa Allah Essam Abdrabo | 1    | 13   |
| Razan Salah El-said      | 1    | 32   |
| Sohila Mohamed Maher     | 1    | 38   |
| Mariam Ashraf Mohamed    | 2    | 24   |




#Final Submission is on Task_1 jupyter notebook
# 1.Add additive noise to the image

## 1.1 Uniform

- ## Original Image

![Image](jaguar.jpg)

- ## Uniform noise added to BGR Image

![Image](uniform_BGR.jpg)

- ## Uniform noise added to Gray Image

![Image](Uniform_Gray.jpg)

## 1.2 Gaussian

- ## Original Image

![Image](jaguar.jpg)

- ## Gaussian noise added to BGR Image

![Image](Gaussian_BGR.jpg)

- ## Gaussian Image added to Gray Image

![Image](Gaussian_Gray.jpg)

## 1.3 salt & pepper

- ## Original Image

![Image](jaguar.jpg)

- ## salt & pepper noise added to BGR Image

![Image](SP_BGR.jpg)

- ## salt & pepper noise added to Gray Image

![Image](SP_Gray.jpg)

# 2.Filter the noisy image using the following low pass filters

## 2.1 Average

- ## Noisy Image

![Image](Uniform_Gray.jpg)

- ## Filtered Image

![Image](filtered_average.jpg)

## 2.2 Gaussian

- ## Noisy Image

![Image](Gaussian_Gray.jpg)

- ## Filtered Image

![Image](filtered_gaussian.jpg)

## 2.3 Median

- ## Noisy Image

![Image](SP_Gray.jpg)

- ## Filtered Image

![Image](filtered_median.jpg)

# 3.Detect edges in the image using the following masks

## 3.1 Sobel

- ## Original Image

![Image](jaguar.jpg)

- ## Sobel_X

![Image](sobelx.jpg)

- ## Sobel_y

![Image](sobely.jpg)

- ## Gradient

![Image](sobel.jpg)

## 3.2 Roberts

- ## Original Image

![Image](jaguar.jpg)

- ## Roberts edge detection

![Image](Roberts_Gray.jpg)

## 3.3 Prewitt

- ## Original Image

![Image](jaguar.jpg)

- ## Prewitt edge detection

![Image](Prewitt_Gray.jpg)

## 3.4 Canny

- ## Original Image

![Image](jaguar.jpg)

- ## Canny edge detection

![Image](canny.jpg)

# 4.Draw histogram and distribution curve

![Image](Histo.png)

# 5.Equalize the image.

- ## Original Image

![Image](jaguar.jpg)

- ## Equalized Image

### Steps:

### 1.Get histogram

### 2.Accumulative Sum

### 3.Normalize this sum from 0 to 255

### 4. Mapping from histogram

![Image](equalize.jpg)

# 8. RGB Histo & Distirbution , Convert to GreyScale

- ## Original Image

![Image](jaguar.jpg)

- ## Grey Scale Image

![Image](grey.png)

- ## R G B Histograms

![Image](RGBhisto.png)

- ## R G B Distribution

![Image](RGBdist.png)

# 9. Frequency domain Filters(Low-pass and High-pass filters)

- ## Original Image

![Image](jaguar.jpg)

- ## Low-pass filtered Image
![Image](lowpass.png)


- ## High-pass filtered Image
![Image](highpass.png)


# 10. Hybrid image
![Image](hybrid.png)
