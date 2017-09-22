import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


my_data = np.loadtxt('G:/UNCC/Subjects/ML/example HW-1/dataset_1.csv',skiprows=1,delimiter=',')

print (my_data)

x,y,z = my_data.transpose()
#print (x)
#print (y)
#print (z)

mean_x = np.mean(x)
mean_y = np.mean(y)
mean_z = np.mean(z)

print(mean_x)
print(mean_y)
print (mean_z)

new_data_set = np.column_stack([my_data[:,0]-mean_x,my_data[:,1]-mean_y,my_data[:,2]-mean_z])
new_data_set_trans = np.transpose(new_data_set)
new_prod = np.dot(new_data_set_trans,new_data_set)/1000
print ("----------------------")
print("Covariance Matrix: \n",new_prod)
print ("----------------------")
#print(new_data_set)


#Finding Eigen values

eigen_val, eigen_vector = la.eig(new_prod)
print("Eigen Values: \n",eigen_val)
print("Eigen Vector: \n",eigen_vector)

feature_vector = np.column_stack([eigen_vector[:,0],eigen_vector[:,2]])
print ("----------------------")
print("Feature Vector: \n",feature_vector)
print ("----------------------")

newdata = np.dot(feature_vector.T,new_data_set.T)
print ("----------------------")
print(newdata)
print ("----------------------")



#Finding variance of x
var_x = np.var(x)
print ("Variance for x:",var_x)
#Finding variance of y
var_y = np.var(y)
print ("Variance for y:",var_y)
#Finding variance of z
var_z = np.var(z)
print ("Variance for z:",var_z)

cov_x_y = np.cov(x,y)
#print (cov_x_y)

cov_x_z = np.cov(x,z)
#print (cov_x_z)

cov_y_z = np.cov(y,z)
#print (cov_y_z)


#forming the covariance matrix for PCA:
cov_PCA = np.concatenate((cov_x_y,cov_y_z))
#print (cov_PCA)



new = cov_x_z[0][1]
#print (new)


#Covariance matrix

my_data_transpose = np.transpose(my_data)
print ('transpose:',my_data_transpose)

#taking the dot product of the matrix
my_data_prod = np.dot(my_data_transpose,my_data)
print (my_data_prod)

new = my_data_prod/1000
print (new)

#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.scatter(x,y, color='green', s=4)
#fig.show()

