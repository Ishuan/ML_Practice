import numpy as np
import matplotlib.pyplot as plt


my_data = np.loadtxt("G:/UNCC/Subjects/ML/example HW-1/dataset_1.csv",delimiter=',',skiprows=1)

print (my_data)

x,y,z = my_data.transpose()
print (x)
print (y)
print (z)

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
print (cov_x_y)

cov_x_z = np.cov(x,z)
print (cov_x_z)

cov_y_z = np.cov(y,z)
print (cov_y_z)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x,y, color='green', s=4)
fig.show()

