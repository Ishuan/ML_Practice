import numpy as np
from numpy import linalg as la

my_data = np.loadtxt('./dataset_1.csv',skiprows=1,delimiter=',')

print ("Data present in the .csv file: \n",my_data)

x,y,z = my_data.transpose()

mean_x = np.mean(x)
mean_y = np.mean(y)
mean_z = np.mean(z)

scaled_data = np.column_stack([my_data[:,0]-mean_x,my_data[:,1]-mean_y,my_data[:,2]-mean_z])
scaled_data_trans = np.transpose(scaled_data)
covar_matrix = np.dot(scaled_data_trans,scaled_data)/1000
print ("----------------------")
print("Covariance Matrix: \n",covar_matrix)
print ("----------------------")

#Finding Eigen values

eigen_val, eigen_vector = la.eig(covar_matrix)
print("Eigen Values: \n",eigen_val)
print("Eigen Vector: \n",eigen_vector)

feature_vector = np.column_stack([eigen_vector[:,0],eigen_vector[:,2]])
print ("----------------------")
print("Feature Vector: \n",feature_vector)
print ("----------------------")

newdata = np.dot(feature_vector.T,scaled_data_trans)
print("PCA DATA: \n",newdata)
print ("----------------------")


