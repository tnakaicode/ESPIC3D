import numpy as np
import math

# Solves A x = B iteratively
# This function is equivalent to the Jacobi relaxation method.
def iterative(A,B,tol):
  x = np.zeros(((B.shape)[0]))
  oldX = np.copy(x)
  count = 0
  while np.allclose(oldX,x,tol/10.0,0.0) == False or count == 0:
    count = count + 1
    oldX = np.copy(x)
    for i in np.ndindex(x.shape):
      x[i] = oldX[i] + (1.0/A[i][i])*(B[i] - np.dot(A,oldX)[i])
  return x
