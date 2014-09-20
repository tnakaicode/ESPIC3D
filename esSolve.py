from numpy import *
import math

# D * potential = - DX^2 * charge / eps_0
#Needed later
#L = 1.0
#X0 = 0.0
#DX = L/N
#DX2 = pow(DX,2.0)
#EPS0 = 8.854187817e-12

N = 10
V0 = 1.0
VN = 2.0

def laplace1D(N,V0,VN):
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

  return linalg.solve(D,potBC)
