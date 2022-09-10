import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt

 
# Read face image from zip file on the fly
faces = {}
with zipfile.ZipFile("Dataset/archive.zip") as facezip:
    for filename in facezip.namelist():
        if not filename.endswith(".pgm"):
            continue # not a face picture
        with facezip.open(filename) as image:
            # If we extracted files from zip, we can use cv2.imread(filename) instead
            faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
 
# Show sample faces using matplotlib
fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
faceimages = list(faces.values())[-16:] # take last 16 images
for i in range(16):
    axes[i%4][i//4].imshow(faceimages[i], cmap="gray")
print("Showing sample faces")
#plt.show()
 
# Print some details
faceshape = list(faces.values())[0].shape
print("Face image shape:", faceshape)
 
classes = set(filename.split("/")[0] for filename in faces.keys())
print("Number of classes:", len(classes))
print("Number of images:", len(faces))
 
# Take classes 1-39 for eigenfaces, keep entire class 40 and
# image 10 of class 39 as out-of-sample test
facematrix = []
facelabel = []
for key,val in faces.items():
    if key.startswith("s40/"):
        continue # this is our test set
    if key == "s39/10.pgm":
        continue # this is our test set
    facematrix.append(val.flatten())
    facelabel.append(key.split("/")[0])
 
# Create a NxM matrix with N images and M pixels per image
facematrix = np.array(facematrix)

meanfaces=np.mean(facematrix,axis=0)
normalizedtrain=facematrix-np.tile(meanfaces,(facematrix.shape[0],1))
# since each row is an image vector
L = (normalizedtrain.dot(normalizedtrain.T)) /len(classes)
# find the eigenvalues and the eigenvectors of L
eigenvalues, eigenvectors = np.linalg.eig(L)



# since each column is an image vector
#U,S,VT=np.linalg.svd(normalizedtrain,full_matrices=0)
L = ((normalizedtrain.T).dot (normalizedtrain)) /normalizedtrain.shape[1]
cov_matrix = np.cov(normalizedtrain.T)

cov_matrix = np.divide(cov_matrix,normalizedtrain.shape[1])

# find the eigenvalues and the eigenvectors of L
eigenvalues, eigenvectors = np.linalg.eig(L)

eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

# Sort the eigen pairs in descending order:
eig_pairs.sort(reverse=True)
eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = list([eig_pairs[index][1] for index in range(len(eigenvalues))])


# Get the first 10 people as training data 
Cumulative_var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)
# get the cdf 
scores_Idx=np.where(Cumulative_var_comp_sum>=0.9)[0]
eigvectors_CDF= eigenvectors[:,scores_Idx]
reduced_data = np.array(eigvectors_CDF)
proj_data = (np.dot(normalizedtrain,reduced_data)).T
for i in range(proj_data.shape[0]):
    img = proj_data[i].reshape(faceshape[0],faceshape[1])
    plt.imshow(img, cmap='jet')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    plt.show()


