# Computer Vision SBE404B

# Assignment_3 " Feature point detection, features descriptors (SIFT) and image matching (SSD and normalized cross correlation)"

#####

| Submitted by:            | Sec. | B.N. | E-mail                   |
| ------------------------ | ---- | ---- | ------------------------ |
| Ashar Seif el Nasr Saleh | 1    | 9    | asharzanqour@gmail.com
| Alaa Allah Essam Abdrabo | 1    | 13   | alaaessammirah@gmail.com |
| Razan Salah El-sayed      | 1    | 32   | razansalah022@gmail.com  |
| Sohila Mohamed Maher     | 1    | 38   | sohilamohamed583@gmail.com
| Mariam Ashraf Mohamed    | 2    | 24   | mariamashraf731@outlook.com |

# 1. Code Architecture

# 1.1 Extract the unique features in all images using Harris operator and λ-

![Image](./output/harrisRes.jpg)

1- Get Harris response Function

```python
    # Calculation of Sobelx
    sobelx = cv2.Sobel(imggray, cv2.CV_64F, 1, 0, ksize=5)
    # Calculation of Sobely
    sobely = cv2.Sobel(imggray, cv2.CV_64F, 0, 1, ksize=5)
    # Apply GaussianBlue for noise cancellation
    Ixx = cv2.GaussianBlur(src=sobelx ** 2, ksize=(5, 5), sigmaX=0)
    Ixy = cv2.GaussianBlur(src=sobely*sobelx,ksize=(5, 5),  sigmaX=0)
    Iyy = cv2.GaussianBlur(src=sobely ** 2, ksize=(5, 5), sigmaX=0)
    k = 0.05
    # determinant
    detA = Ixx * Iyy - Ixy ** 2
    # trace
    traceA = Ixx + Iyy
    # get r
    harris_response = detA - k * traceA ** 2

```

2- Get Harris indices

```python
# find edges and corners using r
    #Edge : r < 0
    #Corner : r > 0
    #Flat: r = 0
    threshold = 0.01
    harrisRecsopy = np.copy(harrisRes)
    rMatrix = cv2.dilate(harrisRecsopy, None)
    rMax = np.max(rMatrix)
    corner = np.array(harrisRes > (rMax*threshold), dtype="int8")
```

3- Mapping harris indices to the original image "corner with Red point"

```python
cornerImg[corners == 1] = [0, 0, 255]
```

# 1.2 Generate feature descriptors using scale invariant features (SIFT)

*  The alghorithm composed of nine seperated function , as shown in the main function computeKeypointsAndDescriptors(), which gives you a clear overview of the different components involved in SIFT, the return of the function are the keypoints and descriptors of the image.
```python
def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image
    """
    image = image.astype('float32')
    base_image = generateBaseImage(image, sigma, assumed_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors
```
1- generateBaseImage() : To appropriately blur and double the input image to produce the base image of our “image pyramid”, a set of successively blurred and downsampled images that form our scale space.
```python
def generateBaseImage(image, sigma, assumed_blur):
    """Generate base image from input image by upsampling by 2 in both directions and blurring
    """
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff) 
```

2- computeNumberOfOctaves() : to compute the number of layers (octaves) in image pyramid.
```python
def computeNumberOfOctaves(image_shape):
    """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
    """
    return int(round(np.log(min(image_shape)) / np.log(2) - 1))
```

3- generateGaussianKernels(): to create a list of scales (gaussian kernel sizes).
```python
def generateGaussianKernels(sigma, num_intervals):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
    """
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels
```
4- generateGaussianImages(): blurs and downsamples the base image.
```python
def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """Generate scale-space pyramid of Gaussian images
    """
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image) 
        for gaussian_kernel in gaussian_kernels[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
    return array(gaussian_images, dtype=object)
```

5- generateDoGImages(): Subtracts adjacent pairs of gaussian images to form a pyramid of difference-of-Gaussian (“DoG”) images.
```python
def generateDoGImages(gaussian_images):
    """Generate Difference-of-Gaussians image pyramid
    """
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return array(dog_images, dtype=object)
```

6- findScaleSpaceExtrema(): identify keypoints.
```python
def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    """Find pixel positions of all scale-space extrema in the image pyramid
    """
    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints
```

7-removeDuplicateKeypoints():  clean up these keypoints by removing duplicates and converting them to the input image size.
```python
def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints
```
8-convertKeypointsToInputImageSize():Convertes keypoints to the input image size.
```python
def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints
```

9-generateDescriptors():Convertes keypoints to the input image size.
```python
def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint
    """
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
    
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')
```

# 1.4 Match the image set features using sum of squared differences (SSD) and normalized cross correlations (NCC)
* Taking the first 30 matchings between the input images.

![Image](./output/image_matching1.png)

![Image](./output/image_matching2.png)

1- Calculate SSD

```python
    sum_square = 0

    # Get SSD between the 2 vectors
    for m in range(len(desc1)):
        sum_square += (desc1[m] - desc2[m]) ** 2

    # The (-) sign here because the condition we applied after this function call is reversed
    sum_square = - (np.sqrt(sum_square))
```

2- Calculate NCC

```python
    # Normalize the 2 vectors
    out1_norm = (desc1 - np.mean(desc1)) / (np.std(desc1))
    out2_norm = (desc2 - np.mean(desc2)) / (np.std(desc2))

    # Apply similarity product between the 2 normalized vectors
    corr_vector = np.multiply(out1_norm, out2_norm)

    # Get mean of the result vector
    corr = float(np.mean(corr_vector))
```

3- Apply feature matching
```python

    # sift
    num_key_points1, descriptors_1 = _sift.computeKeypointsAndDescriptors(img1)
    num_key_points2, descriptors_2 = _sift.computeKeypointsAndDescriptors(img2)

    # List to store the matches scores
    matches = []

    # Loop over each key point in image1
    # We need to calculate similarity with each key point in image2
    for kp1 in range(num_key_points1):
        # Initial variables which will be updated in the loop
        distance = -np.inf
        y_index = -1
        # Loop over each key point in image2
        for kp2 in range(num_key_points2):

            # Match features between the 2 vectors
            value = match_calculator(desc1[kp1], desc2[kp2])
            if value > distance:
                distance = value
                y_index = kp2

        # Create a cv2.DMatch object for each match and set 
        cur = cv2.DMatch() 
        cur.queryIdx = kp1 #The index of the feature in the first image
        cur.trainIdx = y_index #The index of the feature in the second image
        cur.distance = distance #The distance between the two features
        matches.append(cur)
```
# 1.4 GUI

To apply Harris : choose input image file->open image1 , then choose Image_operation->harris 


![Image](./output/harrisui.png)


To apply SIFT : choose input image file->open image1 , then choose Image_operation->sift


![Image](./output/siftui.png)


To apply Image matching :choose input image1 the input image 2 file->open image1 , file->open image2 ,then choose Image_operation->matching
user can then choose to use SSD or NCC from the dropbox (enabled only with matching)


![Image](./output/matchingui.png)

![Image](./output/matchingui2.png)


Note : user should wait some time in sift and image matching to get the output due to creating image descriptors(in case no sift was applied before matching) ,try using the cat images because they have small size to speed up the process .

# 1. Libraries Versions
  #### Python 3.7.2
  #### OpenCv 4.5.5
  #### Numpy  1.20.3
  #### Matplotlib 3.5.1
  

# Results

### 1.1 Extract the unique features in all images using Harris operator and λ-

|    Before corner detection    |      After corner detection      |
| :---------------------------: | :------------------------------: |
| ![Image](./images/harris.jpg) | ![Image](./output/harrisRes.jpg) |

### 1.2 Generate feature descriptors using scale invariant features (SIFT)
|    Original Image     |     Image Keypoints with SIFT       |
| :---------------------------: | :------------------------------: |
| ![Image](./images/book.jpg) | ![Image](./output/BookSIFTDescriptor.png)|
| ![Image](./images/box.png) | ![Image](./output/BoxSIFTDescriptor.png)|
| ![Image](./images/query.jpg) | ![Image](./output/masjidSIFTDescriptor.png)|
| ![Image](./images/leuveninlight.png) | ![Image](./output/leuvenSIFTDescriptor.png)|

### 1.3.1 Comparison between SIFT Keypoints in this algorithm and Open Cv built in algorithm 
|    Algorithm SIFT Descriptors VS OpenCv SIFT Descriptors |
| :---------------------------: | 
| ![Image](./output/BookSIFTComparison.png) |
| ![Image](./output/BoxSIFTComparison.png) | 
| ![Image](./output/masjidSIFTComparison.png) | 
| ![Image](./output/leuvenSIFTComparison.png)| 

### 1.4 Match the image set features using sum of squared differences (SSD) and normalized cross correlations (NCC)

|    Before image matching    |      After image matching      |
| :---------------------------: | :------------------------------: |
| ![Image](./images/book.jpg) | ![Image](./output/ssd.png) SSD|
| ![Image](./images/bookintable.jpg) | ![Image](./output/ncc_2.png) NCC|
| ![Image](./images/box.png) | ![Image](./output/ssd_2.png) SSD|
| ![Image](./images/box_in_scene.png) | ![Image](./output/ncc.png) NCC|

# Discussion

### 1.2 Generate feature descriptors using scale invariant features (SIFT):
* The computations time of the  algorithm ranges between 80 to 160 s with average one minute and half for different images.
* The Comparison table shown in results section shows that the algorithm output and OpenCV Built-in SIFT function are very similar and compute almost the same descriptors.



### 1.3 Match the image set features using sum of squared differences (SSD) and normalized cross correlations (NCC):
* The computations in this algorithm are heavily and extreme, and it took around 10 minutes or more to finish the whole process in both algorithms.

* The Sum Square Distance is calculated between two feature vectors.SSD values examples: (50, 200, 70), we need to minimize SSD (Choose 50) , in case of SSD matching: (value is returned as a "negative" number) (-50, -200, -70) so we compare it with -np.inf. (The sorting is reversed later)

* The Normalized Cross Correlation is calculated between two feature vectors.NCC values examples: (0.58, 0.87, 0.42), we need to maximize NCC (Choose 0.87) , in case of NCC matching: (value is returned as a "positive" number) (-58, -0.87, -0.42) so we compare it with -np.inf. (The sorting is reversed later)



