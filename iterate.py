from numpy import *
import math

def keepGoing(old,new,tol):
  tol = tol/100.0
  stillGTtol = 0
  for i in ndindex(old.shape):
    if abs(old[i] - new[i]) > tol:
      stillGTtol = stillGTtol + 1
  if stillGTtol == 0:
    return 0
  else:
    return 1

# Solves A x = B iteratively
# This function is equivalent to the Jacobi relaxation method.
def iterative(A,B,tol):
  x = zeros(((B.shape)[0]))
  oldX = copy(x)
  count = 0
  while keepGoing(oldX,x,tol) == 1 or count == 0:
    count = count + 1
    oldX = copy(x)
    for i in ndindex(x.shape):
      x[i] = oldX[i] + (1.0/A[i][i])*(B[i] - dot(A,oldX)[i])
  return x
