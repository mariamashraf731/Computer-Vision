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
import os
import scipy.io 


np.random.seed(42)

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
        options = ["Select","Face Detection","Face Recogniton","ROC Curve"]
        self.comboBox.addItems(options)
        self.comboBox.setDisabled(True)
        self.comboBox.currentTextChanged.connect(lambda: self.choose_process())

    def choose_process(self):
        if self.comboBox.currentText() == "Face Detection":
            self.face_detection_process()
        if self.comboBox.currentText() == "Face Recogniton":
            self.face_recognition_process()
        if self.comboBox.currentText() == "ROC Curve":
            self.Thresholding()
        

    def face_detection_process(self):
        image = cv2.imread(filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t1 = time.time()
        detected_faces = self.detect_faces(source=image)
        faced_image = self.draw_faces(source=image_rgb, faces=detected_faces)
        print(f"Found {len(detected_faces)} Faces!")
        print(detected_faces)
        print(faced_image)
        t2 = time.time()
        self.show_Image(faced_image, t2-t1)

    def face_recognition_process(self):
        mat_contents= scipy.io.loadmat(os.path.join('allFaces.mat') )
        faces=mat_contents['faces']
        M=int(mat_contents['m'])
        N=int(mat_contents['n'])
        nfaces=np.ndarray.flatten(mat_contents['nfaces'])

        traindata=(faces[:,:np.sum(nfaces[:30])])
        meanfaces=np.mean(traindata,axis=1)
        normalizedtrain=traindata-np.tile(meanfaces,(traindata.shape[1],1)).T
        L = ((normalizedtrain.T).dot (normalizedtrain)) /normalizedtrain.shape[1]
        cov_matrix = np.cov(normalizedtrain.T)
        cov_matrix = np.divide(cov_matrix,normalizedtrain.shape[1])
        eigenvalues, eigenvectors = np.linalg.eig(L)  
        eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]
        eig_pairs.sort(reverse=True)
        eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
        eigvectors_sort = list([eig_pairs[index][1] for index in range(len(eigenvalues))])
        Cumulative_var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)
        scores_Idx = np.where(Cumulative_var_comp_sum>=0.9)[0]
        eigvectors_CVF= eigenvectors[:,scores_Idx]
        proj_data = (np.dot(normalizedtrain,eigvectors_CVF)).T
        w = np.array([np.dot(proj_data,i) for i in normalizedtrain.T])
        testface=faces[:,np.sum(nfaces[:20])]
        testmean = np.subtract(testface,meanfaces)
        w_unknown = np.dot(proj_data,testmean)
        diff  = w - w_unknown
        norms = np.linalg.norm(diff, axis=1)
        normssorted=np.sort(norms)  
        normssorted=normssorted[:6]
        related_images=[]
      



    def detect_faces(self, source: np.ndarray, scale_factor: float = 1.1, min_size: int = 50) -> list:

        src = np.copy(source)
        if len(src.shape) > 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        else:
            pass

        # Create the haar cascade
        repo_root = os.path.dirname(os.getcwd())
        sys.path.append(repo_root)

        cascade_path = r"src\haarcascade_frontalface_default.xml"

        face_cascade = cv2.CascadeClassifier(cascade_path)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            image=src,
            scaleFactor=scale_factor,
            minNeighbors=5,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return faces


    def draw_faces(self ,source: np.ndarray, faces: list, thickness: int = 10) -> np.ndarray:
        """
        Draw rectangle around each face in the given faces list
        :return:
        """

        src = np.copy(source)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img=src, pt1=(x, y), pt2=(x + w, y + h),
                        color=(0, 255, 0), thickness=thickness)

        return src


    def show_Image(self, Image, time):
        plt.imsave("./output/Output.png",Image ,cmap = plt.cm.gray)
        qpixmap = QPixmap("./output/Output.png")
        self.ui.image2.setPixmap(qpixmap)
        t=float("{:.2f}".format((time)))
        self.ui.time.setText(str(t)+"s")

        
    def open_Image(self):
        global filepath
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "",
                        "*", options=options)
        qpixmap = QPixmap(filepath)
        self.ui.image1.setPixmap(qpixmap)
        self.comboBox.setDisabled(False)
        self.ui.input1.setText("Image")
        self.ui.input2.setText("Output")

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

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())

