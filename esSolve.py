from numpy import *
import math
import iterate,direct

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

##################################

######
# 1D #
######

def laplace1D(N,V0,VN,type,tol):
  pts = N + 1
  D = zeros(shape=(pts,pts))
  potBC = zeros(shape=(pts))

  for row in xrange(pts):
    if row == 0 or row == N:
      D[row][row] = 1.0
      if row == 0:
        potBC[row] = V0
      if row == N:
        potBC[row] = VN
    else:
      for col in xrange(pts):
        if col == row-1 or col == row+1:
          D[row][col] = 1.0
        elif col == row:
          D[row][col] = -2.0  

  if type == "direct":
    return direct.directly(D,potBC)
  elif type == "iterative":
    return iterate.iterative(D,potBC,tol)
  else:
    return "invalid type"

######
# 2D #
######

# PHI[(Ny+1)*i + j] = phi[i][j]
# V0x, VNx, V0y, VNy are each arrays
# phi[0][j] = PHI[j] = V0x
# phi[i][0] = PHI[(Ny+1)*i] = V0y
# phi[Nx][j] = PHI[(Ny+1)*Nx + j] = VNx
# phi[i][Ny] = PHI[(Ny+1)*i + Ny] = VNy
# Should there be a check that V0x[0] = V0y[0] ?
def laplace2D(Nx,V0x,VNx,Ny,V0y,VNy,type,tol):
  pts = (Nx + 1)*(Ny + 1)
  D = zeros(shape=(pts,pts))
  potBC = zeros(shape=(pts))

  for row in xrange(pts):
    if row == 0 or row == N:
      D[row][row] = 1.0
      if row == 0:
        potBC[row] = V0
      if row == N:
        potBC[row] = VN
    else:
      for col in xrange(pts):
        if col == row-1 or col == row+1:
          D[row][col] = 1.0
        elif col == row:
          D[row][col] = -2.0  

  if type == "direct":
    return direct.directly(D,potBC)
  elif type == "iterative":
    return iterate.iterative(D,potBC,tol)
  else:
    return "invalid type"

  # 2D should return a 2d array!

######
# 3D #
######
# PHI[(Nz+1)*(Ny+1)*i+(Nz+1)*j+k] = phi[i][j][k]

  # 3D should return a 3d array!
