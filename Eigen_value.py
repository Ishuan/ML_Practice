#for Question-3 calculating the eigen values
import numpy as np
from numpy import linalg as la
mat = np.array([[0,-1],[2,3]])
#Finding eigen values
eigen_val, eigen_vector = la.eig(mat)
print(eigen_val)
print (eigen_vector)