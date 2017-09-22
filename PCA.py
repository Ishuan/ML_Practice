import numpy as np
from numpy import linalg as la

my_data = np.loadtxt('./dataset_1.csv',skiprows=1,delimiter=',')
print ("Data present in the .csv file: \n",my_data)

def PCA(input):
    x,y,z = my_data.transpose()

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_z = np.mean(z)

    scaled_data = np.column_stack([my_data[:,0]-mean_x,my_data[:,1]-mean_y,my_data[:,2]-mean_z])
    scaled_data_trans = np.transpose(scaled_data)
    covar_matrix = np.dot(scaled_data_trans,scaled_data)/len(x-1)
    print ("----------------------")
    print("Covariance Matrix: \n",covar_matrix)
    print ("----------------------")

    #Finding Eigen values

    eigen_val, eigen_vector = la.eig(covar_matrix)
    print("Eigen Values: \n",eigen_val)
    print("Eigen Vector: \n",eigen_vector)

    #Finding Max Eigen Values

    eigen_max_1 = np.argmax(eigen_val)
    eigen_max_1_vector = (eigen_vector[:,eigen_max_1])

    #deleting the max values
    eigen_new_val = np.delete(eigen_val,eigen_max_1)
    eigen_new_vector = np.delete(eigen_vector,eigen_max_1,1)

    #Finding the next max values
    eigen_max_2 = np.argmax(eigen_new_val)
    eigen_max_2_vector = (eigen_new_vector[:,eigen_max_2])

    print("PC1: \n",eigen_max_1_vector)
    print("PC2: \n",eigen_max_2_vector)

    feature_vector = np.column_stack([eigen_max_1_vector,eigen_max_2_vector])

    print ("----------------------")
    print("Feature Vector: \n",feature_vector)
    print ("----------------------")

    newdata = np.dot(feature_vector.T,scaled_data_trans)
    print("PCA DATA: \n",newdata)
    print ("----------------------")

PCA(my_data)