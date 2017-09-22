#For calculating covariance and variance
import numpy as np

my_data = np.loadtxt("G:/UNCC/Subjects/ML/dataset_1.csv",skiprows=1,delimiter=",")
print("Data present in the .csv file: \n",my_data)

def cal_var(x):
    n = len(x)
    mean = sum(x)/n
    variance = sum((x-mean)**2)/(n-1)
    return variance

print ("Variance of X:",cal_var(my_data[:,0]))
print ("Variance of Y:",cal_var(my_data[:,1]))
print ("Variance of Z:",cal_var(my_data[:,2]))


print("Covariance between a x and y: \n",np.cov((my_data[:,0]),(my_data[:,1])))
print("Covariance between a y and z: \n",np.cov((my_data[:,1]),(my_data[:,2])))