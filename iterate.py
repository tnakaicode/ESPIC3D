import numpy as np

# Solves A x = B iteratively
# This function is equivalent to the Jacobi relaxation method.
def iterative(A,B,tol):
  maxIterations = 10000
  x = np.copy(B)
  for j in range(maxIterations):
    oldX = np.copy(x)
    for i in np.ndindex(x.shape):
      if A[i][i] != 0.0:  
        x[i] = oldX[i] + (1.0/A[i][i])*(B[i] - np.dot(A,oldX)[i])
    if np.allclose(oldX,x,tol/10.0,0.0) == True:
      break
  return x
