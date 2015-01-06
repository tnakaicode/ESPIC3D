from numpy import *
import math
import iterate,direct

# TO DO
# 1. will need to allow neumann BCs
# 2. Poisson
# 3. Look at http://wiki.scipy.org/PerformancePython

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
      D[row][row-1] = 1.0
      D[row][row+1] = 1.0
      D[row][row] = -2.0  

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
# Should there be a check that V0x[0] = V0y[0] ?

def laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,type,tol):
  pts = (NX + 1)*(NY + 1)
  D = zeros(shape=(pts,pts))
  potBC = zeros(shape=(pts))
  PHI = zeros(shape=(pts))

  rowsNotBC = list()
  for i in xrange(pts):
    rowsNotBC.append(i)

# phi[i][0] = PHI[(Ny+1)*i] = V0y[i]
# phi[i][Ny] = PHI[(Ny+1)*i + Ny] = VNy[i]
  for i in xrange(NX+1):
    V0yIndex = (NY+1)*i
    VNyIndex = (NY+1)*i + NY
    potBC[V0yIndex] = V0y[i]
    potBC[VNyIndex] = VNy[i]
    D[V0yIndex][V0yIndex] = 1.0
    D[VNyIndex][VNyIndex] = 1.0
    rowsNotBC.remove(V0yIndex)
    rowsNotBC.remove(VNyIndex)

# phi[0][j] = PHI[j] = V0x[j]
# phi[Nx][j] = PHI[(Ny+1)*Nx + j] = VNx[j]
  for j in xrange(NY+1):
    V0xIndex = j
    VNxIndex = (NY+1)*NX + j
    potBC[V0xIndex] = V0x[j]
    potBC[VNxIndex] = VNx[j] 
    D[V0xIndex][V0xIndex] = 1.0
    D[VNxIndex][VNxIndex] = 1.0
    if V0xIndex in rowsNotBC:
      rowsNotBC.remove(V0xIndex)
    if VNxIndex in rowsNotBC:
      rowsNotBC.remove(VNxIndex)

  for row in rowsNotBC:
# For now just multiply by DX^2
    coeff1 = pow(DX/DY,2.0)
    coeff2 = -2.0*(1.0 + coeff1)

    D[row][row - (NY+1)] = 1.0
    D[row][row + (NY+1)] = 1.0
    D[row][row - 1] = coeff1
    D[row][row + 1] = coeff1
    D[row][row] = coeff2  

  if type == "direct":
    PHI = direct.directly(D,potBC)
  elif type == "iterative":
    PHI = iterate.iterative(D,potBC,tol)
  else:
    print "invalid type"

  phi = empty((NX+1,NY+1))
  for i,j in ndindex(phi.shape):
    phi[i][j] = PHI[(NY+1)*i + j]

  return phi

######
# 3D #
######

# PHI[(Nz+1)*(Ny+1)*i+(Nz+1)*j+k] = phi[i][j][k]
