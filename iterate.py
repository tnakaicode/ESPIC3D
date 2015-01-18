from numpy import *
import math

# TO DO;
# iterative won't work if D[i][i] = 0
# decide how to handle that

def keepGoing(A,x,B,tol):
  # TEMPORARY
  tol = 1.0e-10
  stillGTtol = 0
  for i in ndindex(x.shape):
    if dot(A,x)[i] - B[i] > tol:
      stillGTtol = stillGTtol + 1
  if stillGTtol == 0:
    return 0
  else:
    return 1

def iterative(D,potBC,tol):
  x = copy(potBC)
  while keepGoing(D,x,potBC,tol):
    tempX = copy(x)
    for i in ndindex(x.shape):
      x[i] = tempX[i] + (1/D[i][i])*(potBC[i] - dot(D,tempX)[i])
  return x
