import numpy as np
import math

maxIterations = 10000

# Solves A x = B iteratively
# This function is equivalent to the Jacobi relaxation method.
def iterative(A,B,tol):
  x = np.zeros(((B.shape)[0]))
  oldX = np.ones(((B.shape)[0]))
  for j in xrange(maxIterations):
    oldX = np.copy(x)
    for i in np.ndindex(x.shape):
      if A[i][i] != 0.0:
        x[i] = oldX[i] + (1.0/A[i][i])*(B[i] - np.dot(A,oldX)[i])
    if np.allclose(oldX,x,tol/10.0,0.0) == True:
      break
  return x
