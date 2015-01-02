from numpy import *
import math

# TO DO
# 1. will need to allow neumann BCs
# 2. Poisson
# 3. Look at http://wiki.scipy.org/PerformancePython

# D * potential = - DX^2 * charge / eps_0
#Needed later
#L = 1.0
#X0 = 0.0
#DX = L/N
#DX2 = pow(DX,2.0)
#EPS0 = 8.854187817e-12

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

def laplace1D(N,V0,VN,type,tol):
  pts = N + 1
  D = zeros(shape=(pts,pts))
  potBC = zeros(shape=(pts))

  for row in xrange(N+1):
    if row == 0 or row == N:
      D[row][row] = 1.0
      if row == 0:
        potBC[row] = V0
      if row == N:
        potBC[row] = VN
    else:
      for col in xrange(N+1):
        if col == row-1 or col == row+1:
          D[row][col] = 1.0
        elif col == row:
          D[row][col] = -2.0  

  if type == "direct":
    return linalg.solve(D,potBC)
  elif type == "iterative":
    return iterative(D,potBC,tol)
  else:
    return "invalid type"
