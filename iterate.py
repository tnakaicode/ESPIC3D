from numpy import *
import math

# TO DO;
# iterative won't work if D[i][i] = 0
# decide how to handle that

def keepGoing(old,new,tol):
  # TEMPORARY
  tol = tol/100.0
  stillGTtol = 0
  for i in ndindex(old.shape):
    if abs(old[i] - new[i]) > tol:
      stillGTtol = stillGTtol + 1
  if stillGTtol == 0:
    return 0
  else:
    return 1

def iterative(x,D,potBC,tol):
  oldX = copy(x)
  count = 0
  while keepGoing(oldX,x,tol) == 1 or count == 0:
    count = count + 1
    oldX = copy(x)
    for i in ndindex(x.shape):
      x[i] = oldX[i] + (1.0/D[i][i])*(potBC[i] - dot(D,oldX)[i])
  return x
