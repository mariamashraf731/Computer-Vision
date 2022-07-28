import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog , QLabel , QMessageBox ,QComboBox
from PyQt5.QtGui import QIcon, QPixmap
from mainGui import Ui_MainWindow
from PIL import Image, ImageStat
import sys 
import qdarkstyle

np.random.seed(42)

def euclidean_distance(point1, point2):
    """
    Computes euclidean distance of point1 and point2.
    
    point1 and point2 are lists.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def clusters_distance(cluster1, cluster2):
    """
    Computes distance between two clusters.
    
    cluster1 and cluster2 are lists of lists of points
    """
    return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])
  
def clusters_distance_2(cluster1, cluster2):
    """
    Computes distance between two centroids of the two clusters
    
    cluster1 and cluster2 are lists of lists of points
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)

class Mean_Shift:

    def __init__(self, radius=30):
        self.radius = radius
    
    def create_feature_space(self,image):

        self.width=image.shape[1]
        self.height=image.shape[0]

        self.data=np.zeros([self.width*self.height,3])
        self.indices=np.zeros([self.width*self.height,2])
        i=0

        for r in range(self.height) :
            for c in range(self.width):
                self.indices[i][0]=r
                self.indices[i][1]=c
                self.data[i]=image[r][c]
                i=i+1
    
    def euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self):

        self.centroids = []
        self.clusters=[]

        self.output=np.zeros([self.height,self.width,3])
        originaldata=self.data.copy()

        while len(self.data) > 0:
            centroid = self.data[0]

            while True:
                points_within_radius = []

                for feature in self.data:
                    if (np.linalg.norm(feature-centroid) <= self.radius).all() :
                        points_within_radius.append(feature)

                indices_within_radius=np.array([i for i, b in enumerate(originaldata) for s in points_within_radius if all(s == b)])
                indices_within_radius=np.unique(indices_within_radius,axis=0)
            #save old centroid
                old_centroid = centroid    
            #update new centroid
                if (len(points_within_radius) > 0):
                    centroid = np.mean(points_within_radius, axis=0)
            #check convergence
                if self.euclidean_distance(old_centroid, centroid) < 0.1:
                    break

            self.centroids.append(centroid)
            self.clusters.append(indices_within_radius)

            data_cpy=self.data.copy()
            self.data=np.array([i for i in data_cpy if not (i==points_within_radius).all(1).any()])

        for i in range(len(self.centroids)):
            for pixel_idx in self.clusters[i]:
                temp=self.indices[pixel_idx][0]
                self.output[int(self.indices[pixel_idx][0])][int(self.indices[pixel_idx][1])]=self.centroids[i]

class AgglomerativeClustering:
    
    def __init__(self, k=2, initial_k=25):
        self.k = k
        self.initial_k = initial_k
        
    def initial_clusters(self, points):
        """
        partition pixels into self.initial_k groups based on color similarity
        """
        groups = {}
        d = int(256 / (self.initial_k))
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []
        for i, p in enumerate(points):
            if i%100000 == 0:
                print('processing pixel:', i)
            go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))  
            groups[go].append(p)
        return [g for g in groups.values() if len(g) > 0]
        
    def fit(self, points):

        # initially, assign each point to a distinct cluster
        print('Computing initial clusters ...')
        self.clusters_list = self.initial_clusters(points)
        print('number of initial clusters:', len(self.clusters_list))
        print('merging clusters ...')

        while len(self.clusters_list) > self.k:

            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min([(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                 key=lambda c: clusters_distance_2(c[0], c[1]))

            # Remove the two clusters from the clusters list

            self.clusters_list = [c for c in self.clusters_list if np.all(np.array(c) != np.array(cluster1)) and np.all(np.array(c) != np.array(cluster2))]
            

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            # Add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)

            print('number of clusters:', len(self.clusters_list))
        
        print('assigning cluster num to each point ...')
        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num
                
        print('Computing cluster centers ...')
        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)
                    


    def predict_cluster(self, point):
        """
        Find cluster number of point
        """
        # assuming point belongs to clusters that were computed by fit functions
        return self.cluster[tuple(point)]

    def predict_center(self, point):
        """
        Find center of the cluster that point belongs to
        """
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.ui.actionrgb_luv.triggered.connect(lambda: self.RGBtoLUV_process())
        self.ui.actionsegmentaion.triggered.connect(lambda: self.Segmentation_process())
        self.ui.actionthresholding.triggered.connect(lambda: self.Thresholding_process())
        self.ui.actionOpen_Image.triggered.connect(lambda: self.open_Image())
        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(347, 260, 111, 71)
        self.comboBox.addItem("Select")
        self.comboBox.setDisabled(True)

        self.comboBox_W = QComboBox(self)
        self.comboBox_W.setGeometry(550, 400, 111, 71)
        self.comboBox_W.setDisabled(True)
        self.comboBox_W.setVisible(False)
        options=["None","1","2","4","6","8","10"]
        self.comboBox_W.addItems(options)

        self.comboBox_H = QComboBox(self)
        self.comboBox_H.setGeometry(90, 400, 111, 71)
        self.comboBox_H.setDisabled(True)
        self.comboBox_H.setVisible(False)
        options=["None","1","2","4","6","8","10"]
        self.comboBox_H.addItems(options)

        self.comboBox_C = QComboBox(self)
        self.comboBox_C.setGeometry(340, 460, 111, 71)
        self.comboBox_C.setDisabled(True)
        self.comboBox_C.setVisible(False)
        options=["None","1","2","4","6","8","10"]
        self.comboBox_C.addItems(options)

        self.comboBox.currentTextChanged.connect(lambda: self.choose_process())
        self.comboBox_H.currentTextChanged.connect(lambda: self.choose_process())
        self.comboBox_W.currentTextChanged.connect(lambda: self.choose_process())
        self.comboBox_C.currentTextChanged.connect(lambda: self.choose_process())


    def choose_process(self):
        if self.ui.operation.text() == "Segmentation":
            self.Segmentation()
        if self.ui.operation.text() == "Thresholding":
            self.Thresholding()

    def RGBtoLUV_process(self):
        self.comboBox.clear()
        self.comboBox.addItem("Select")
        self.comboBox.setDisabled(True)
        self.ui.input1.setText("Image")
        self.ui.input2.setText("Output")
        self.ui.operation.setText("RGB -> LUV")
        self.RGBtoLUV()

    def Segmentation_process(self):
        self.comboBox.clear()
        self.ui.input1.setText("Image")
        self.ui.input2.setText("Output")
        self.ui.operation.setText("Segmentation")
        self.comboBox.setDisabled(False)
        options=["select","K-means","Region Growing","Agglomerative","Mean Shift"]
        self.comboBox.addItems(options)
    
    def Thresholding_process(self):
        self.comboBox.clear()
        self.ui.input1.setText("Image")
        self.ui.input2.setText("Output")
        self.ui.operation.setText("Thresholding")
        self.comboBox.setDisabled(False)
        options=["select","Local Otsu","Global Otsu","Local Spectral","Global Spectral","Local Optimal","Global Optimal"]
        self.comboBox.addItems(options)

    def show_boxes(self):
        self.comboBox_W.setVisible(True)          
        self.comboBox_H.setVisible(True)
        self.comboBox_C.setVisible(False)
        self.ui.Select_W.setVisible(True)          
        self.ui.Select_H.setVisible(True)  
        self.ui.Select_C.setVisible(False)  
        self.comboBox_W.setDisabled(False)          
        self.comboBox_H.setDisabled(False)

    def hide_boxes(self):
        self.comboBox_W.setVisible(False)          
        self.comboBox_H.setVisible(False)
        self.comboBox_C.setVisible(False)
        self.ui.Select_W.setVisible(False)          
        self.ui.Select_H.setVisible(False)          
        self.ui.Select_C.setVisible(False)  
        self.comboBox_W.setDisabled(True)          
        self.comboBox_H.setDisabled(True)
        self.comboBox_W.setCurrentIndex(0)
        self.comboBox_H.setCurrentIndex(0)

    def show_boxes_clusters(self):
        self.comboBox_W.setVisible(False)          
        self.comboBox_H.setVisible(False)
        self.comboBox_C.setVisible(True)
        self.ui.Select_W.setVisible(False)          
        self.ui.Select_H.setVisible(False)  
        self.ui.Select_C.setVisible(True)  
        self.comboBox_C.setDisabled(False)

    def hide_boxes_clusters(self):
        self.comboBox_W.setVisible(False)          
        self.comboBox_H.setVisible(False)
        self.comboBox_C.setVisible(False)
        self.ui.Select_W.setVisible(False)          
        self.ui.Select_H.setVisible(False)          
        self.ui.Select_C.setVisible(False)  
        self.comboBox_C.setCurrentIndex(0)

    def show_Image(self, Image, time):
        plt.imsave("./output/Output.png",Image ,cmap = plt.cm.gray)
        qpixmap = QPixmap("./output/Output.png")
        self.ui.image2.setPixmap(qpixmap)
        t=float("{:.2f}".format((time)))
        self.ui.time.setText(str(t)+"s")
        
    def Thresholding(self):
        img = cv2.imread(filepath)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgGray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        npImg = np.array(imgGray)

        if self.comboBox.currentText()== "Local Otsu":
            self.show_boxes()
            if self.comboBox_H.currentText() != "None" and self.comboBox_W.currentText() != "None":
                x = int(self.comboBox_H.currentText())
                y = int(self.comboBox_W.currentText())
                t1 = time.time()
                Img = self.localThresholding(image, x, y, self.otsuThresholding)
                t2 = time.time()          
                self.show_Image(Img,t2-t1)

        elif self.comboBox.currentText()== "Global Otsu":
            self.hide_boxes()
            t1 = time.time()
            Img = self.otsuThresholding(image)
            t2 = time.time()          
            self.show_Image(Img,t2-t1)

        elif self.comboBox.currentText()== "Local Spectral":
            self.show_boxes()
            if self.comboBox_H.currentText() != "None" and self.comboBox_W.currentText() != "None":
                x = int(self.comboBox_H.currentText())
                y = int(self.comboBox_W.currentText())
                t1 = time.time()
                Img = self.localThresholding(image, x, y, self.spectralThresholding)
                t2 = time.time()          
                self.show_Image(Img,t2-t1)

        elif self.comboBox.currentText()== "Global Spectral":
            self.hide_boxes()
            t1 = time.time()
            Img = self.spectralThresholding(image)
            t2 = time.time()          
            self.show_Image(Img,t2-t1)

        elif self.comboBox.currentText()== "Local Optimal":
            self.show_boxes()
            if self.comboBox_H.currentText() != "None" and self.comboBox_W.currentText() != "None":
                x = int(self.comboBox_H.currentText())
                y = int(self.comboBox_W.currentText())
                t1 = time.time()
                localOptThreshold = self.localThresholding(npImg, x, y, self.getOptimmalThreshold)
                Img = npImg > localOptThreshold
                t2 = time.time()          
                self.show_Image(Img,t2-t1)
            
        elif self.comboBox.currentText()== "Global Optimal":
            self.hide_boxes()
            t1 = time.time()
            Img = self.globaloptimalThreshold(npImg)
            t2 = time.time()          
            self.show_Image(Img,t2-t1)

        else:
            return None
        
    def Segmentation(self):
        img = cv2.imread(filepath)
        
        if self.comboBox.currentText()== "K-means":
            self.show_boxes_clusters()
            if self.comboBox_C.currentText() != "None":
                C_num = int(self.comboBox_C.currentText())
                t1 = time.time()
                self.im= Image.open(filepath)
                self.img_width, self.img_height = self.im.size
                self.px = self.im.load()
                result = self.startKmeans(C_num)
                Img = self.drawWindow(result)
                t2 = time.time() 
                self.show_Image(Img,t2-t1)         

        elif self.comboBox.currentText()== "Region Growing":
            self.hide_boxes_clusters()   
            t1 = time.time()      
            seed_points = []
            for i in range(3):
                x = np.random.randint(0, img.shape[0])
                y = np.random.randint(0, img.shape[1])
                seed_points.append(Point(x, y))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            Img = self.Region_growing(img_gray, seed_points, 10)
            t2 = time.time()          
            self.show_Image(Img,t2-t1)         

        elif self.comboBox.currentText()== "Agglomerative":
            self.show_boxes_clusters()
            if self.comboBox_C.currentText() != "None":
                n_clusters = int(self.comboBox_C.currentText())
                t1 = time.time()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
                pixels = img.reshape(img.shape[0]*img.shape[1],3)
                agglo = AgglomerativeClustering(k=n_clusters, initial_k=25)
                agglo.fit(pixels)
                new_img = [[agglo.predict_center(list(pixel)) for pixel in row] for row in img]
                new_img = np.array(new_img, np.uint8)
                Img = cv2.cvtColor(new_img.astype('float32'), cv2.COLOR_Luv2RGB)
                t2 = time.time()
                self.show_Image(Img,t2-t1)         

        elif self.comboBox.currentText()== "Mean Shift":
            self.hide_boxes_clusters()          
            t1 = time.time()
            res = cv2.resize(img, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
            imgluv= cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
            clf = Mean_Shift()
            clf.create_feature_space(imgluv)
            clf.fit()
            output_image=clf.output
            Img = cv2.cvtColor(output_image.astype('float32'), cv2.COLOR_Luv2RGB)
            t2 = time.time()
            self.show_Image(Img,t2-t1)         

        else:
            return None
        
    def open_Image(self):
        global filepath
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "",
                        "*", options=options)
        qpixmap = QPixmap(filepath)
        self.ui.image1.setPixmap(qpixmap)

    def RGBtoLUV(self):
        img = cv2.imread(filepath)
        Image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t1=time.time()
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
                if x==0 and y==0 and z==0:
                    u_val = 0
                    v_val = 0
                else:
                    u = 4 * x / (x + 15 * y + 3 * z)
                    v = 9 * y / (x + 15 * y + 3 * z)
                    u_val = 255 / 354 * (13 * l_val * (u - 0.19793943) + 134)
                    v_val = 255 / 262 * (13 * l_val*(v - 0.46831096)+140)
                Image_LUV[i,j][0] = l_val
                Image_LUV[i,j][1] = u_val
                Image_LUV[i,j][2] = v_val
        Image_LUV = np.array(Image_LUV,np.uint8)
        t2=time.time()
        
        plt.imsave("./output/RGB_LUV.png",Image_LUV)
        qpixmap = QPixmap("./output/RGB_LUV.png")
        self.ui.image2.setPixmap(qpixmap)

        t=float("{:.2f}".format((t2-t1)))
        self.ui.time.setText(str(t)+"s")

    def histogram(self, image):
        h, w = image.shape
        grayscale_array = []
        for px in range(0, h):
            for py in range(0, w):
                intensity = image[px][py]
                grayscale_array.append(intensity)
        bins = range(0, 255)
        img_histogram = np.histogram(grayscale_array, bins)
        return img_histogram

    def Global_thresholding(self, image, threshold):
        h, w = np.shape(image)
        # pixel threshold
        for px in range(0, h):
            for py in range(0, w):
                intensity = image[px][py]
                if (intensity <= threshold):
                    intensity = 0
                else:
                    intensity = 255
                image[px][py] = intensity

        return image

    def localThresholding(self, img, nx, ny, thresholdingOption):

        grayImage = np.copy(img)

        if len(img.shape) > 2:
            grayImage = self.grayImg(grayImage)
        else:
            pass

        YMax, XMax = grayImage.shape
        subImage = np.zeros((YMax, XMax))
        yWindowSize = YMax // ny
        xWindowSize = XMax // nx
        xWindow = []
        yWindow = []
        for i in range(0, nx):
            xWindow.append(xWindowSize * i)

        for i in range(0, ny):
            yWindow.append(yWindowSize * i)

        xWindow.append(XMax)
        yWindow.append(YMax)
        for x in range(0, nx):
            for y in range(0, ny):
                if thresholdingOption == self.getOptimmalThreshold:

                    newimg = grayImage[yWindow[y]
                        :yWindow[y + 1], xWindow[x]:xWindow[x + 1]]
                    im, bins = np.histogram(newimg, range(257))
                    subImage[yWindow[y]:yWindow[y + 1], xWindow[x]:xWindow[x + 1]] = self.getOptimmalThreshold(im, 200
                                                                                                        )
                else:
                    subImage[yWindow[y]:yWindow[y + 1], xWindow[x]:xWindow[x + 1]] = thresholdingOption(
                        grayImage[yWindow[y]:yWindow[y + 1], xWindow[x]:xWindow[x + 1]])

        return subImage

    def otsuThresholding(self, img):
        grayImage = np.copy(img)
        if len(img.shape) > 2:
            grayImage = self.grayImg(img)
        else:
            pass
        yWindowSize, xWindowSize = grayImage.shape
        # get pixels values probabilities using histogram
        HistValues = plt.hist(grayImage.ravel(), 256)
        # get the total number of pixels
        total_pixels = xWindowSize*yWindowSize
        current_max, threshold = 0, 0
        sumT, sumF, sumB = 0, 0, 0
        for i in range(0, 256):
            # getting the sum of probabilities of all pixels values
            sumT += i * HistValues[0][i]
        weightB, weightF = 0, 0
        varBetween, meanB, meanF = 0, 0, 0

        # Iterating on all pixels' values to get the best threshold
        for i in range(0, 256):
            weightB += HistValues[0][i]
            weightF = total_pixels - weightB
            # Check if the pixels values represented in one value
            if weightF == 0:
                break
            sumB += i*HistValues[0][i]
            sumF = sumT - sumB
            meanB = sumB/weightB
            meanF = sumF/weightF
            varBetween = weightB * weightF
            varBetween *= (meanB-meanF)*(meanB-meanF)
            if varBetween > current_max:
                current_max = varBetween
                threshold = i

        # print("threshold is:", threshold)
        ostu_image = self.Global_thresholding(grayImage, threshold)
        return ostu_image

    def spectralThresholding(self, img):

        grayImage = np.copy(img)
        if len(img.shape) > 2:
            grayImage = self.grayImg(grayImage)
        else:
            pass
        # Get Image Dimensions
        yWindowSize, xWindowSize = grayImage.shape
        # Get The Values of The Histogram Bins
        HistValues = plt.hist(grayImage .ravel(), 256)[0]
        # Calculate The Probability Density Function
        PDF = HistValues / (yWindowSize * xWindowSize)
        # Calculate The Cumulative Density Function
        CDF = np.cumsum(PDF)
        OptimalLow = 1
        OptimalHigh = 1
        MaxVariance = 0
        # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
        Global = np.arange(0, 256)
        GMean = sum(Global * PDF) / CDF[-1]
        for LowT in range(1, 254):
            for HighT in range(LowT + 1, 255):
                # Background Intensities Array
                Back = np.arange(0, LowT)
                # Low Intensities Array
                Low = np.arange(LowT, HighT)
                # High Intensities Array
                High = np.arange(HighT, 256)
                # Get Low Intensities CDF
                CDFL = np.sum(PDF[LowT:HighT])
                # Get Low Intensities CDF
                CDFH = np.sum(PDF[HighT:256])
                # Calculation Mean of Background & The Object (Foreground), Based on CDF & PDF
                BackMean = sum(Back * PDF[0:LowT]) / CDF[LowT]
                LowMean = sum(Low * PDF[LowT:HighT]) / CDFL
                HighMean = sum(High * PDF[HighT:256]) / CDFH
                # Calculate Cross-Class Variance
                Variance = (CDF[LowT] * (BackMean - GMean) ** 2 + (CDFL * (LowMean - GMean) ** 2) + (
                    CDFH * (HighMean - GMean) ** 2))
                # Filter Out Max Variance & It's Threshold
                if Variance > MaxVariance:
                    MaxVariance = Variance
                    OptimalLow = LowT
                    OptimalHigh = HighT
            """
        Apply Double Thresholding To Image to get the Lowest Allowed Value using Low Threshold Ratio/Intensity and the Minimum Value To Be Boosted using High Threshold Ratio/Intensity
        """
        # Create Empty Array
        ThresholdedImage = np.zeros(grayImage.shape)

        HighPixel = 255
        LowPixel = 128

        # Find Position of Strong & Weak Pixels
        HighRow,  HighCol = np.where(grayImage >= OptimalHigh)
        LowRow, LowCol = np.where(
            (grayImage <= OptimalHigh) & (grayImage >= OptimalLow))

        # Apply Thresholding
        ThresholdedImage[HighRow, HighCol] = HighPixel
        ThresholdedImage[LowRow, LowCol] = LowPixel

        return ThresholdedImage

    def globaloptimalThreshold(self, grayimg):
        im, bins = np.histogram(grayimg, range(257))
        optimalthreshold = self.getOptimmalThreshold(im, 200)
        # print(optimalthreshold)
        # Map threshold with the original image
        segImg = grayimg > optimalthreshold
        return segImg

    def getOptimmalThreshold(self, im, threshold):
        # devide image into two sections "Background & foreGround"
        # im, bins = np.histogram(im, range(257))
        back = im[:threshold]
        fore = im[threshold:]
        # Compute the centroids or Mean
        mBack = (back*np.arange(0, threshold)).sum()/back.sum()
        mFore = (fore*np.arange(threshold, len(im))).sum()/fore.sum()
        # New threshold
        NewThreshold = int(np.round((mBack+mFore)/2))
        # print(mBack, mFore, NewThreshold)
        # Recursion with the newthreshold till the threshold is the same "const"
        if(NewThreshold != threshold):
            return self.getOptimmalThreshold(im, NewThreshold)
        return NewThreshold

    def getGrayDiff(self, img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))

    def selectConnects(self, p):
        if p != 0:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1),
                        Point(1, 0), Point(1, 1), Point(0, 1),
                        Point(-1, 1), Point(-1, 0)]
        else:
            connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]

        return connects

    def Region_growing(self, img, seeds, thresh, p = 1):

        height, weight = img.shape
        seedMark = np.zeros(img.shape)
        seedList = []

        for seed in seeds:
            seedList.append(seed)
        label = 1
        connects = self.selectConnects(p)

        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)

            seedMark[currentPoint.x, currentPoint.y] = label

            for i in range(8):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y

                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue

                grayDiff = self.getGrayDiff(img, currentPoint, Point(tmpX, tmpY))

                if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                    seedMark[tmpX, tmpY] = label
                    seedList.append(Point(tmpX, tmpY))

        return seedMark

    def converged(self, centroids, old_centroids):

        if len(old_centroids) == 0:
            return False


        if len(centroids) <= 5:
            a = 1
        elif len(centroids) <= 10:
            a = 2
        else:
            a = 4

        for i in range(0, len(centroids)):
            cent = centroids[i]
            old_cent = old_centroids[i]

            if ((int(old_cent[0]) - a) <= cent[0] <= (int(old_cent[0]) + a)) and ((int(old_cent[1]) - a) <= cent[1] <= (int(old_cent[1]) + a)) and ((int(old_cent[2]) - a) <= cent[2] <= (int(old_cent[2]) + a)):
                continue
            else:
                return False

        return True

    def getMin(self, pixel, centroids):
        minDist = 9999
        minIndex = 0

        for i in range(0, len(centroids)):
            d = np.sqrt(int((centroids[i][0] - pixel[0]))**2 + int((centroids[i][1] - pixel[1]))**2 + int((centroids[i][2] - pixel[2]))**2)
            if d < minDist:
                minDist = d
                minIndex = i

        return minIndex

    def assignPixels(self, centroids):
        clusters = {}

        for x in range(0, self.img_width):
            for y in range(0, self.img_height):
                p = self.px[x, y]
                minIndex = self.getMin(self.px[x, y], centroids)

                try:
                    clusters[minIndex].append(p)
                except KeyError:
                    clusters[minIndex] = [p]

        return clusters

    def adjustCentroids(self, centroids, clusters):
        new_centroids = []
        keys = sorted(clusters.keys())

        for k in keys:
            n = np.mean(clusters[k], axis=0)
            new = (int(n[0]), int(n[1]), int(n[2]))
            # print(str(k) + ": " + str(new))
            new_centroids.append(new)

        return new_centroids

    def startKmeans(self, someK):

        centroids = []
        old_centroids = []
        rgb_range = ImageStat.Stat(self.im).extrema
        i = 1

        for k in range(0, someK):

            cent = self.px[np.random.randint(0, self.img_width), np.random.randint(0, self.img_height)]
            centroids.append(cent)
        

        while not self.converged(centroids, old_centroids) and i <= 20:
            # print("Iteration #" + str(i))
            i += 1

            old_centroids = centroids 								
            clusters = self.assignPixels(centroids) 						
            centroids = self.adjustCentroids(old_centroids, clusters) 	

        # print(centroids)
        return centroids

    def drawWindow(self, result):

        img = Image.new('RGB', (self.img_width, self.img_height), "white")
        p = img.load()

        for x in range(img.size[0]):
            for y in range(img.size[1]):
                RGB_value = result[self.getMin(self.px[x, y], result)]
                p[x, y] = RGB_value
        # Convert the pixels into an array using numpy
        array = np.array(img, dtype=np.uint8)
        return  array

    def grayImg(self, img):
        # Apply gray scale
        gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8) 

        return gray_img         
                

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())

