import numpy as np

# Solves A x = B iteratively
# This function is equivalent to the Jacobi relaxation method.
def iterate(A,B,relTol,absTol):
  maxIterations = 10000
  x = np.copy(B)
  for j in range(maxIterations):
    oldX = np.copy(x)
    for i in np.ndindex(x.shape):
      if A[i][i] != 0.0:  
        x[i] = oldX[i] + (1.0/A[i][i])*(B[i] - np.dot(A,oldX)[i])
    if np.allclose(oldX,x,relTol/10.0,absTol/10.0) == True:
      break
  return x

# Solves A x = B directly
def direct(A,B):
  return np.linalg.solve(A,B)
