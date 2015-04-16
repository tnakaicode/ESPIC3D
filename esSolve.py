from numpy import *
import iterate,direct

# This function puts a 3d array on a grid with
# indicies (i,j,k) into a 1d array. 
#   PHI[i] = phi[i]
#   PHI[(Ny+1)*i + j] = phi[i][j]
#   PHI[(Nz+1)*(Ny+1)*i+(Nz+1)*j+k] = phi[i][j][k]
def indexTo1D(N,i,j,k):
  return (N[2]+1)*(N[1]+1)*i+(N[2]+1)*j+k

def useBCs(index1,index2,V1,V2,potBC,D,rowsNotBC):
  indexes = [index1,index2]
  V = [V1,V2]
  for i in xrange(2):
    index = indexes[i]
    myV = V[i]
    if index not in rowsNotBC:
      if potBC[index] != myV:
        print "inconsistent BCs"
    else:  
      potBC[index] = myV
      D[index][index] = 1.0
      rowsNotBC.remove(index)

def laplace1D(NX,DX,V0x,VNx,solType,tol):
  return laplace([NX,0,0],[DX,1.0,1.0],[V0x],[VNx],solType,tol)

def laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,solType,tol):
  return laplace([NX,NY,0],[DX,DY,1.0],[V0x,V0y],[VNx,VNy],solType,tol)

def laplace3D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,NZ,DZ,V0z,VNz,solType,tol):
  return laplace([NX,NY,NZ],[DX,DY,DZ],[V0x,V0y,V0z],[VNx,VNy,VNz],solType,tol)

### General Laplace Solver ###
def laplace(N,D,V0,VN,solType,tol):
  (NX,NY,NZ) = (N[0],N[1],N[2])
  (DX,DY,DZ) = (D[0],D[1],D[2])
  pts = (NX+1)*(NY+1)*(NZ+1)
  D = zeros(shape=(pts,pts))
  potBC = zeros(shape=(pts))
  PHI = zeros(shape=(pts))
  rowsNotBC = [i for i in xrange(pts)]

  # Set up rows corresponding to a boundary condition
  for j in xrange(NY+1):
    for k in xrange(NZ+1):
      if NY == 0 and NZ == 0:
        useBCs(indexTo1D(N,0,j,k),indexTo1D(N,NX,j,k),V0[0],VN[0],potBC,D,rowsNotBC)
      if NY != 0 and NZ == 0:
        useBCs(indexTo1D(N,0,j,k),indexTo1D(N,NX,j,k),V0[0][j],VN[0][j],potBC,D,rowsNotBC)
      if NY != 0 and NZ != 0:
        useBCs(indexTo1D(N,0,j,k),indexTo1D(N,NX,j,k),V0[0][j][k],VN[0][j][k],potBC,D,rowsNotBC)

  if NY > 0:
    for i in xrange(NX+1):
      for k in xrange(NZ+1):
        if NZ == 0:
          useBCs(indexTo1D(N,i,0,k),indexTo1D(N,i,NY,k),V0[1][i],VN[1][i],potBC,D,rowsNotBC)
        else:
          useBCs(indexTo1D(N,i,0,k),indexTo1D(N,i,NY,k),V0[1][i][k],VN[1][i][k],potBC,D,rowsNotBC)

    if NZ > 0:
      for i in xrange(NX+1):
        for j in xrange(NY+1):
          useBCs(indexTo1D(N,i,j,0),indexTo1D(N,i,j,NZ),V0[2][i][j],VN[2][i][j],potBC,D,rowsNotBC)

  # Set up rows not corresponding to a boundary condition
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

  # Solve linear system D * PHI = potBC
  if solType == "direct":
    PHI = direct.directly(D,potBC)
  elif solType == "iterative":
    PHI = iterate.iterative(D,potBC,tol)
  else:
    print "invalid type"

  # Convert 1-d PHI to dim-D phi
  if NY == 0 and NZ == 0:
    phi = zeros(NX+1)
    for i in xrange(len(phi)):
      phi[i] = PHI[indexTo1D(N,i,0,0)]

  elif NY != 0 and NZ == 0:
    phi = zeros((NX+1,NY+1))
    for i,j in ndindex(phi.shape):
      phi[i][j] = PHI[indexTo1D(N,i,j,0)]

  else:
    phi = zeros((NX+1,NY+1,NZ+1))
    for i,j,k in ndindex(phi.shape):
      phi[i][j][k] = PHI[indexTo1D(N,i,j,k)]

  return phi
