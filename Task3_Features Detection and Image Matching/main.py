import numpy as np
import cv2
from libs import sift ,Harris ,Image_matching
import time
from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog , QLabel , QMessageBox ,QComboBox
from PyQt5.QtGui import QIcon, QPixmap
from mainGui import Ui_MainWindow
import sys 
import qdarkstyle

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.ui.actionHarris.triggered.connect(lambda: self.harris())
        self.ui.actionSIFT.triggered.connect(lambda: self.SIFT())
        self.ui.actionImage_Matching.triggered.connect(lambda: self.Matching())
        self.ui.actionOpen_Image1.triggered.connect(lambda: self.open_image1())
        self.ui.actionOpen_Image2.triggered.connect(lambda: self.open_image2())
        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(345, 260, 111, 71)
        options=["select","NCC","SSD"]
        self.comboBox.addItems(options)
        self.comboBox.setDisabled(True)
        self.comboBox.currentTextChanged.connect(lambda: self.process_matching())

    def harris(self):
        self.comboBox.setDisabled(True)
        self.ui.input1.setText("Input")
        self.ui.input2.setText("Output")
        self.ui.output.setText("")
        self.ui.operation.setText("Harris")
        output,t=self.process_harris()
        t=float("{:.2f}".format(t))
        self.ui.time.setText(str(t)+"s")
        qpixmap = QPixmap(output)
        self.ui.image2.setPixmap(qpixmap)

    def SIFT(self):
       self.comboBox.setDisabled(True)
       self.ui.input1.setText("Input")
       self.ui.input2.setText("Output")
       self.ui.output.setText("")
       self.ui.operation.setText("SIFT")
       output,t=self.process_SIFT()
       t=float("{:.2f}".format(t))
       self.ui.time.setText(str(t)+"s")
       qpixmap = QPixmap(output)
       self.ui.image2.setPixmap(qpixmap)

    def open_image1(self):
        global filepath1
        options = QFileDialog.Options()
        filepath1, _ = QFileDialog.getOpenFileName(self, "",
                        "*", options=options)
        qpixmap = QPixmap(filepath1)
        self.ui.image1.setPixmap(qpixmap)

    def open_image2(self):
        global filepath2
        options = QFileDialog.Options()
        filepath2, _ = QFileDialog.getOpenFileName(self, "",
                        "*", options=options)
        qpixmap = QPixmap(filepath2)
        self.ui.image2.setPixmap(qpixmap)
       
    def Matching(self):
        self.ui.input1.setText("Image1")
        self.ui.input2.setText("Image2")
        self.ui.output.setText("Output")
        self.ui.operation.setText("Matching")
        self.comboBox.setDisabled(False)
      
    def process_SIFT(self):
        img1 = cv2.imread(filepath1, 0) 
        t1 = time.time()          
        kp1, des1 = sift.computeKeypointsAndDescriptors(img1)
        t2 = time.time()
        fig, ax = plt.subplots(figsize=(80, 80))
        ax.imshow(img1, 'gray')
        plt.axis('off')
        for pnt in kp1:
            ax.scatter(pnt.pt[0], pnt.pt[1], s=pnt.size*50, c="red")

        plt.savefig("./output/SIFTDescriptor.png",bbox_inches = 'tight',pad_inches=0)

        return "./output/SIFTDescriptor.png", t2-t1

    def process_harris(self):

        img = cv2.imread(filepath1)
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t1 = time.time()
        r = Harris.getHarrisRespone(imggray)
        corners = Harris.getHarrisIndices(r)
        cornerImg = np.copy(img)
        cornerImg[corners == 1] = [255, 0, 0]
        t2 = time.time()
        plt.imsave('./output/harrisRes.jpg', cornerImg)
        return "./output/harrisRes.jpg" , t2-t1

    def process_matching(self):
        img1 = cv2.imread(filepath1)
        img2 = cv2.imread(filepath2)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # sift
        
        if self.comboBox.currentText()== "SSD":
            t1=time.time()
            keypoints_1, descriptors_1 = sift.computeKeypointsAndDescriptors(img1)
            keypoints_2, descriptors_2 = sift.computeKeypointsAndDescriptors(img2)
            matches =Image_matching.apply_feature_matching(descriptors_1, descriptors_2, Image_matching.calculate_ssd)
            matches = sorted(matches, key=lambda x: x.distance, reverse=True)

            matched_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                            matches[:30], img2, flags=2)
            t2=time.time()
            plt.imsave("./output/matching_ssd.png",matched_image)
            qpixmap = QPixmap("./output/matching_ssd.png")

            self.ui.image3.setPixmap(qpixmap)
        elif self.comboBox.currentText()== "NCC":
            t1=time.time()
            keypoints_1, descriptors_1 = sift.computeKeypointsAndDescriptors(img1)
            keypoints_2, descriptors_2 = sift.computeKeypointsAndDescriptors(img2)

            matches = Image_matching.apply_feature_matching(descriptors_1, descriptors_2, Image_matching.calculate_ncc)
            matches = sorted(matches, key=lambda x: x.distance, reverse=True)

            matched_image_2 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                            matches[:30], img2, flags=2)
            t2=time.time()
            plt.imsave("./output/matching_ncc.png",matched_image_2)
            qpixmap = QPixmap("./output/matching_ncc.png")
            self.ui.image3.setPixmap(qpixmap)

        t=float("{:.2f}".format((t2-t1)/60))
        self.ui.time.setText(str(t)+"m")        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())

