import numpy as np
import cv2
from libs import sift , Image_matching
import time
from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 10

img1 = cv2.imread('images/book.jpg', 0)           
img2 = cv2.imread('images/bookintable.jpg', 0) 

# Compute SIFT keypoints and descriptors
t1 = time.time()
kp1, des1 = _sift.computeKeypointsAndDescriptors(img1)
t2 = time.time()
print(f"Computation time: {t2 - t1}s")
kp2, des2 = _sift.computeKeypointsAndDescriptors(img2)


# Matching Images after SIFT DETECTION
# Initialize and use FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
acceptedMatches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        acceptedMatches.append(m)

if len(acceptedMatches) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in acceptedMatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in acceptedMatches]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    siftimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        siftimg[hdif:hdif + h1, :w1, i] = img1
        siftimg[:h2, w1:w1 + w2, i] = img2

    # Draw SIFT keypoint matches
    for m in acceptedMatches:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(siftimg, pt1, pt2, (255, 0, 0))

    cv2.imwrite('siftimg.jpg', siftimg)
    #cv2.imshow(siftimg)
else:
    print("Not enough matches are found - %d/%d" % (len(acceptedMatches), MIN_MATCH_COUNT))



# Compering between OpenCv SIFT and this algorithm 
sift = cv2.SIFT_create()
kps, dcs = sift.detectAndCompute(img1, None)

fig, ax = plt.subplots(1, 2, figsize=(20, 20))

ax[0].set_title("Algorithm SIFT Descriptor",  fontsize='medium')
ax[0].imshow(img1, 'gray')
for pnt in kp1:
        ax[0].scatter(pnt.pt[0], pnt.pt[1], s=pnt.size, c="red")

ax[1].set_title("OpenCv SIFT Descriptor", fontsize='medium')
ax[1].imshow(img1, 'gray')
for pnt in kps:
        ax[1].scatter(pnt.pt[0], pnt.pt[1], s=pnt.size, c="red")

plt.savefig("SIFTComparison.png")
print(len(kps), len(kp1))


fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(img1,'gray')
for pnt in kp1:
        ax.scatter(pnt.pt[0], pnt.pt[1], s=pnt.size, c="red")

plt.savefig("SIFTDescriptor.png")

# feature matching using ssd and ncc
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# sift
keypoints_1, descriptors_1 = _sift.computeKeypointsAndDescriptors(img1)
keypoints_2, descriptors_2 = _sift.computeKeypointsAndDescriptors(img2)


matches = apply_feature_matching(descriptors_1, descriptors_2, Image_matching.calculate_ssd)
matches = sorted(matches, key=lambda x: x.distance, reverse=True)

matched_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                matches[:30], img2, flags=2)

plt.imshow(matched_image)
plt.show()

matches = apply_feature_matching(descriptors_1, descriptors_2, Image_matching.calculate_ncc)
matches = sorted(matches, key=lambda x: x.distance, reverse=True)


matched_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                matches[:30], img2, flags=2)

plt.imshow(matched_image)
plt.show()