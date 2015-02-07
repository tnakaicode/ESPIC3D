from numpy import *
import math
import iterate,direct

# TO DO
# 1. Neumann BCs
# 2. Poisson
# 3. Parallelize
# 4. Look at http://wiki.scipy.org/PerformancePython

def solver(solType,phi,D,potBC,tol):
  if solType == "direct":
    return direct.directly(D,potBC)
  elif solType == "iterative":
    return iterate.iterative(phi,D,potBC,tol)
  else:
    print "invalid type"
    return 0

######
# 1D #
######

def laplace1D(N,V0,VN,solType,tol):
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

  phi = zeros((pts))

  return solver(solType,phi,D,potBC,tol)

######
# 2D #
######

# PHI[(Ny+1)*i + j] = phi[i][j]
# V0x, VNx, V0y, VNy are each arrays
# Should there be a check that V0x[0] = V0y[0] ?

def laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,solType,tol):
  phi = zeros((NX+1,NY+1))

  pts = (NX + 1)*(NY + 1)
  D = zeros(shape=(pts,pts))
  potBC = zeros(shape=(pts))
  PHI = zeros(shape=(pts))

  rowsNotBC = list()
  for i in xrange(pts):
    rowsNotBC.append(i)

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

# phi[i][0] = PHI[(Ny+1)*i] = V0y[i]
# phi[i][Ny] = PHI[(Ny+1)*i + Ny] = VNy[i]
  for i in xrange(NX+1):
    V0yIndex = (NY+1)*i
    VNyIndex = (NY+1)*i + NY
    potBC[V0yIndex] = V0y[i]
    potBC[VNyIndex] = VNy[i]
    D[V0yIndex][V0yIndex] = 1.0
    D[VNyIndex][VNyIndex] = 1.0
    if V0yIndex in rowsNotBC:
      rowsNotBC.remove(V0yIndex)
    if VNyIndex in rowsNotBC:
      rowsNotBC.remove(VNyIndex)

  for row in rowsNotBC:
    coeff1 = pow(DX,2.0)
    coeff3 = pow(DY,2.0)
    coeff2 = -2.0*(coeff1+coeff3)
 
    D[row][row - (NY+1)] = coeff3
    D[row][row + (NY+1)] = coeff3
    D[row][row - 1] = coeff1
    D[row][row + 1] = coeff1
    D[row][row] = coeff2  

  PHI = zeros((pts))

  PHI = solver(solType,PHI,D,potBC,tol)

  for i,j in ndindex(phi.shape):
    phi[i][j] = PHI[(NY+1)*i + j]

  return phi

######
# 3D #
######

# PHI[(Nz+1)*(Ny+1)*i+(Nz+1)*j+k] = phi[i][j][k]
# V0x, VNx, V0y, VNy, V0z, VNz are each arrays
# Should there be a check that V0x[0] = V0y[0] ?

def laplace3D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,NZ,DZ,V0z,VNz,solType,tol):
  phi = zeros((NX+1,NY+1,NZ+1))

  pts = (NX + 1)*(NY + 1)*(NZ + 1)
  D = zeros(shape=(pts,pts))
  potBC = zeros(shape=(pts))
  PHI = zeros(shape=(pts))

  rowsNotBC = list()
  for i in xrange(pts):
    rowsNotBC.append(i)

# phi[0][j][k] = PHI[(Nz+1)*j+k] = V0x[j][k]
# phi[Nx][j][k] = PHI[(Nz+1)*(Ny+1)*NX+(Nz+1)*j+k] = VNx[j][k]
  for j in xrange(NY+1):
    for k in xrange(NZ+1):
      V0xIndex = (NZ+1)*j+k
      VNxIndex = (NZ+1)*(NY+1)*NX+(NZ+1)*j+k
      potBC[V0xIndex] = V0x[j][k]
      potBC[VNxIndex] = VNx[j][k] 
      D[V0xIndex][V0xIndex] = 1.0
      D[VNxIndex][VNxIndex] = 1.0
      if V0xIndex in rowsNotBC:
        rowsNotBC.remove(V0xIndex)
      if VNxIndex in rowsNotBC:
        rowsNotBC.remove(VNxIndex)

# phi[i][0][k] = PHI[(Nz+1)*(Ny+1)*i+k] = V0y[i]
# phi[i][Ny][k] = PHI[(Nz+1)*(Ny+1)*i+(Nz+1)*NY+k] = VNy[i]
  for i in xrange(NX+1):
    for k in xrange(NZ+1):
      V0yIndex = (NZ+1)*(NY+1)*i+k
      VNyIndex = (NZ+1)*(NY+1)*i+(NZ+1)*NY+k
      potBC[V0yIndex] = V0y[i][k]
      potBC[VNyIndex] = VNy[i][k]
      D[V0yIndex][V0yIndex] = 1.0
      D[VNyIndex][VNyIndex] = 1.0
      if V0yIndex in rowsNotBC:
        rowsNotBC.remove(V0yIndex)
      if VNyIndex in rowsNotBC:
        rowsNotBC.remove(VNyIndex)

# phi[i][j][0] = PHI[(Nz+1)*(Ny+1)*i+(Nz+1)*j] = V0z[i][j]
# phi[i][j][NZ] = PHI[(Nz+1)*(Ny+1)*i+(Nz+1)*j+NZ] = VNz[i][j]
  for i in xrange(NX+1):
    for j in xrange(NY+1):
      V0zIndex = (NZ+1)*(NY+1)*i+(NZ+1)*j
      VNzIndex = (NZ+1)*(NY+1)*i+(NZ+1)*j+NZ
      potBC[V0zIndex] = V0z[i][j]
      potBC[VNzIndex] = VNz[i][j] 
      D[V0zIndex][V0zIndex] = 1.0
      D[VNzIndex][VNzIndex] = 1.0
      if V0zIndex in rowsNotBC:
        rowsNotBC.remove(V0zIndex)
      if VNzIndex in rowsNotBC:
        rowsNotBC.remove(VNzIndex)

  for row in rowsNotBC:
    coeffX = pow(DY*DZ,2.0)
    coeffY = pow(DX*DZ,2.0)
    coeffZ = pow(DX*DY,2.0)
    coeffXYZ = -2.0*(coeffX+coeffY+coeffZ)
 
    D[row][row - (NZ+1)*(NY+1)] = coeffX
    D[row][row + (NZ+1)*(NY+1)] = coeffX
    D[row][row - (NZ+1)] = coeffY
    D[row][row + (NZ+1)] = coeffY
    D[row][row - 1] = coeffZ
    D[row][row + 1] = coeffZ
    D[row][row] = coeffXYZ

  PHI = zeros((pts))

  PHI = solver(solType,PHI,D,potBC,tol)

  for i,j,k in ndindex(phi.shape):
    phi[i][j][k] = PHI[(NZ+1)*(NY+1)*i+(NZ+1)*j+k]

  return phi
