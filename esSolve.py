from numpy import *
import iterate,direct

######################################################
# TO DO
# 1. Neumann BCs
# 2. Make a single laplace solver that 1d,2d,3d call
# 3. Poisson
# 4. Parallelize
# 5. Look at http://wiki.scipy.org/PerformancePython
# 6. Should there be a check that V0x[0] = V0y[0] ?
######################################################

# PHI[i] = phi[i]
# PHI[(Ny+1)*i + j] = phi[i][j]
# PHI[(Nz+1)*(Ny+1)*i+(Nz+1)*j+k] = phi[i][j][k]
def oneDindex(N,i,j,k):
  return (N[2]+1)*(N[1]+1)*i+(N[2]+1)*j+k

def useBCs(index1,index2,V1,V2,potBC,D,rowsNotBC):
  potBC[index1] = V1
  potBC[index2] = V2
  D[index1][index1] = 1.0
  D[index2][index2] = 1.0
  if index1 in rowsNotBC:
    rowsNotBC.remove(index1)
  if index2 in rowsNotBC:
    rowsNotBC.remove(index2)

def laplace1D(NX,DX,V0x,VNx,solType,tol):
  return laplace([NX,0,0],[DX,1.0,1.0],[V0x],[VNx],solType,tol)

def laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,solType,tol):
  return laplace([NX,NY,0],[DX,DY,1.0],[V0x,V0y],[VNx,VNy],solType,tol)

def laplace3D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,NZ,DZ,V0z,VNz,solType,tol):
  return laplace([NX,NY,NZ],[DX,DY,DZ],[V0x,V0y,V0z],[VNx,VNy,VNz],solType,tol)

### General Laplace Solver ###
def laplace(N,D,V0,VN,solType,tol):
  NX = N[0]
  NY = N[1]
  NZ = N[2]
  DX = D[0]
  DY = D[1]
  DZ = D[2]
  pts = (NX+1)*(NY+1)*(NZ+1)
  D = zeros(shape=(pts,pts))
  potBC = zeros(shape=(pts))
  PHI = zeros(shape=(pts))
  rowsNotBC = [i for i in xrange(pts)]

  for j in xrange(NY+1):
    for k in xrange(NZ+1):
      if NY == 0 and NZ == 0:
        useBCs(oneDindex(N,0,j,k),oneDindex(N,NX,j,k),V0[0],VN[0],potBC,D,rowsNotBC)
      if NY != 0 and NZ == 0:
        useBCs(oneDindex(N,0,j,k),oneDindex(N,NX,j,k),V0[0][j],VN[0][j],potBC,D,rowsNotBC)
      if NY != 0 and NZ != 0:
        useBCs(oneDindex(N,0,j,k),oneDindex(N,NX,j,k),V0[0][j][k],VN[0][j][k],potBC,D,rowsNotBC)

  if NY > 0:
    for i in xrange(NX+1):
      for k in xrange(NZ+1):
        if NZ == 0:
          useBCs(oneDindex(N,i,0,k),oneDindex(N,i,NY,k),V0[1][i],VN[1][i],potBC,D,rowsNotBC)
        else:
          useBCs(oneDindex(N,i,0,k),oneDindex(N,i,NY,k),V0[1][i][k],VN[1][i][k],potBC,D,rowsNotBC)

    if NZ > 0:
      for i in xrange(NX+1):
        for j in xrange(NY+1):
          useBCs(oneDindex(N,i,j,0),oneDindex(N,i,j,NZ),V0[2][i][j],VN[2][i][j],potBC,D,rowsNotBC)

  for row in rowsNotBC:
    coeffX = pow(DY*DZ,2.0)
    if NY == 0.0:
      coeffY = 0.0
    else:
      coeffY = pow(DX*DZ,2.0)
    if NZ == 0.0:
      coeffZ = 0.0
    else:
      coeffZ = pow(DX*DY,2.0)
    coeffXYZ = -2.0*(coeffX+coeffY+coeffZ)

    D[row][row - (NZ+1)*(NY+1)] = coeffX
    D[row][row + (NZ+1)*(NY+1)] = coeffX
    if NY != 0.0:
      D[row][row - (NZ+1)] = coeffY
      D[row][row + (NZ+1)] = coeffY
    if NZ != 0.0:
      D[row][row - 1] = coeffZ
      D[row][row + 1] = coeffZ
    D[row][row] = coeffXYZ

  if solType == "direct":
    PHI = direct.directly(D,potBC)
  elif solType == "iterative":
    PHI = iterate.iterative(D,potBC,tol)
  else:
    print "invalid type"

  if NY == 0 and NZ == 0:
    phi = zeros(NX+1)
    for i in xrange(len(phi)):
      phi[i] = PHI[oneDindex(N,i,0,0)]

  elif NY != 0 and NZ == 0:
    phi = zeros((NX+1,NY+1))
    for i,j in ndindex(phi.shape):
      phi[i][j] = PHI[oneDindex(N,i,j,0)]

  else:
    phi = zeros((NX+1,NY+1,NZ+1))
    for i,j,k in ndindex(phi.shape):
      phi[i][j][k] = PHI[oneDindex(N,i,j,k)]

  return phi
