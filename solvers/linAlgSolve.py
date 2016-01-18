import numpy as np

# Solves A x = B iteratively
# This function is equivalent to the Gauss Seidel  method.
def iterate(A,B,relTol,absTol):
  maxIterations = 10000
  x = np.copy(B)
  alphaRel = 100.0
  alphaAbs = 100.0
  relTol = relTol/alphaRel
  absTol = absTol/alphaAbs
  for j in range(maxIterations):
    oldX = np.copy(x)
    # JACOBI
    # AdotOldX = np.dot(A,oldX)
    for i in range(x.shape[0]):
      # should really first try to rearrange A,B to make A[i,i] != 0 if possible
      if A[i,i] != 0.0:
        # JACOBI
        # x[i] = oldX[i] + (1.0/A[i][i])*(B[i] - AdotOldX[i])

        # GAUSS - SEIDEL
        x[i] = (1.0/A[i][i])*(B[i] - np.dot(A[i,:i],x[:i]) - np.dot(A[i,i+1:], oldX[i+1:]))
    if np.allclose(oldX,x,relTol,absTol) == True:
      break
  return x

# Solves A x = B directly
def direct(A,B):
  return np.linalg.solve(A,B)
