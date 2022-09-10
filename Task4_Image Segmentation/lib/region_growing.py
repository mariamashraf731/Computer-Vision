import numpy as np
import matplotlib.pyplot as plt
import cv2


np.random.seed(42)

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))

def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1),
                    Point(1, 0), Point(1, 1), Point(0, 1),
                    Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]

    return connects


def Region_growing(img, seeds, thresh, p = 1):

    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []

    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)

    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label

        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y

            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue

            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))

            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))

    return seedMark




if __name__ == "__main__":

    image = cv2.imread('regional-growing.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    img = cv2.imread('regional-growing.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    seed_points = []
    for i in range(3):
        x = np.random.randint(0, img.shape[0])
        y = np.random.randint(0, img.shape[1])
        seed_points.append(Point(x, y))
    
    binaryImg = Region_growing(img_gray, seed_points, 10)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.imshow(binaryImg, cmap='gray')
    plt.axis('off')
    plt.title(f'Segmented image')
    plt.show()