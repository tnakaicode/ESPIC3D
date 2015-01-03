from numpy import *
import math

def keepGoing(A,x,B,tol):
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
