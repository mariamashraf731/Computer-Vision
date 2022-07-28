import matplotlib.pyplot as plt
import numpy as np
import cv2
import time


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


imgrgb = cv2.imread('images/dog.jpg')
res = cv2.resize(imgrgb, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
imgluv= cv2.cvtColor(res, cv2.COLOR_RGB2Luv)
t1=time.time()
clf = Mean_Shift()
clf.create_feature_space(imgluv)
clf.fit()
output_image=clf.output
output_imgrgb= cv2.cvtColor(output_image.astype('float32'), cv2.COLOR_Luv2RGB)
plt.imshow((output_imgrgb * 255).astype(np.uint8))
cv2.imwrite('./output_meanshift_resized.jpg',(output_imgrgb * 255).astype(np.uint8))
plt.show()
t2=time.time()
print(t2-t1)
