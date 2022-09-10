import os 
import scipy.io 
import numpy as np 
import matplotlib.pyplot as plt


mat_contents= scipy.io.loadmat (os.path.join('allFaces.mat') )
faces=mat_contents['faces']
M=int(mat_contents['m'])
N=int(mat_contents['n'])
Nfaces=(mat_contents['nfaces'])
nfaces=np.ndarray.flatten(mat_contents['nfaces'])
   
# Get the first 30 people as training data 

traindata=(faces[:,:np.sum(nfaces[:30])])

meanfaces=np.mean(traindata,axis=1)
normalizedtrain=traindata-np.tile(meanfaces,(traindata.shape[1],1)).T
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

Cumulative_var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)

#eigvectors_Length=len(var_comp_sum)
# get the cdf 
#cdf = np.arange(eigvectors_Length) / float(eigvectors_Length-1)
scores_Idx=np.where(Cumulative_var_comp_sum>=0.9)[0]
#eigvectors_sortcdf=var_comp_sum[scores_Idx]
#selected_Scores=np.nonzero(np.in1d(var_comp_sum ,eigvectors_sortcdf))[0]
eigvectors_CDF= eigenvectors[:,scores_Idx]



# Show cumulative proportion of varaince with respect to components
#print("Cumulative proportion of variance explained vector: \n%s" %var_comp_sum)

# x-axis for number of principal components kept
#num_comp = range(1,len(eigvalues_sort)+1)
#plt.title('Cum. Prop. Variance Explain and Components Kept')
#plt.xlabel('Principal Components')
#plt.ylabel('Cum. Prop. Variance Expalined')

#plt.scatter(num_comp, var_comp_sum)
#plt.show()


reduced_data = np.array(eigvectors_CDF)

proj_data = (np.dot(normalizedtrain,reduced_data)).T




for i in range(proj_data.shape[0]):
    img = proj_data[i].reshape(M,N).T
    plt.imshow(img, cmap='jet')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    plt.show()




