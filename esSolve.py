import numpy as np
import linAlgSolve

# Converts grid indices (i,j,k) to 1d array indices
#   1-D: fieldAs1DArray[i] = fieldOn1DGrid[i]
#   2-D: fieldAs1DArray[(Ny+1)*i + j] = fieldOn2DGrid[i][j]
#   3-D: fieldAs1DArray[(Nz+1)*(Ny+1)*i+(Nz+1)*j+k] = fieldOn3DGrid[i][j][k]
def indexTo1D(N,i,j,k):
  return (N[2]+1)*(N[1]+1)*i+(N[2]+1)*j+k

# Updates D and potBC for rows that contain a boundary conditon
def applyBCs(index1,index2,V1,V2,potBC,D,rowsNotBC):
  indexes = [index1,index2]
  V = [V1,V2]
  for i in range(2):
    index = indexes[i]
    myV = V[i]
    # This will need to be updated when we add Neumann BCs
    if index not in rowsNotBC:
      if potBC[index] != myV:
        print("inconsistent BCs")
    else: 
      potBC[index] = myV
      D[index][index] = 1.0
      rowsNotBC.remove(index)

# Return dimension given number of grid indices in each dimension
def dimension(N):
  if N[1] == 0 and N[2] == 0:
    return 1
  elif N[2] == 0:
    return 2
  else:
    return 3

# Takes 1-D array and puts it on the computational grid
# returning a d-D array
def put1DArrayOnGrid(N,array):
  (NX,NY,NZ) = (N[0],N[1],N[2])
  dim = dimension(N)

  if dim == 1:
    arrayOnGrid = np.zeros(NX+1)
    for i in range(len(arrayOnGrid)):
      arrayOnGrid[i] = array[indexTo1D(N,i,0,0)]
  
  elif dim == 2:
    arrayOnGrid = np.zeros((NX+1,NY+1))
    for i,j in np.ndindex(arrayOnGrid.shape):
      arrayOnGrid[i][j] = array[indexTo1D(N,i,j,0)]

  elif dim == 3:
    arrayOnGrid = np.zeros((NX+1,NY+1,NZ+1))
    for i,j,k in np.ndindex(arrayOnGrid.shape):
      arrayOnGrid[i][j][k] = array[indexTo1D(N,i,j,k)]

  return arrayOnGrid

# Solve linear system A x = B
def solveLinearSystem(A,B,solType,relTol,absTol):
  if solType == "direct":
    x = linAlgSolve.direct(A,B)
  elif solType == "iterative":
    x = linAlgSolve.iterate(A,B,relTol,absTol)
  else:
    sys.exit("esSolve::solveForPotential() -- invalid solution type")

  return x

# Solve linear system A x = B but return x
# on the computational grid
def solveForPotential(N,D,potBC,solType,relTol,absTol):
  return put1DArrayOnGrid(N,solveLinearSystem(D,potBC,solType,relTol,absTol))

# can i assign a variable to these default params?

def laplace1D(NX,DX,V0x,VNx,solType,relTol=0.1,absTol=0.1):
  return laplace([NX,0,0],[DX,1.0,1.0],[V0x],[VNx],solType,relTol,absTol)

def laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,solType,relTol=0.1,absTol=0.1):
  return laplace([NX,NY,0],[DX,DY,1.0],[V0x,V0y],[VNx,VNy],solType,relTol,absTol)

def laplace3D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,NZ,DZ,V0z,VNz,solType,relTol=0.1,absTol=0.1):
  return laplace([NX,NY,NZ],[DX,DY,DZ],[V0x,V0y,V0z],[VNx,VNy,VNz],solType,relTol,absTol)

### General Laplace Solver ###
def laplace(N,D,V0,VN,solType,relTol=0.1,absTol=0.1):
  (NX,NY,NZ) = (N[0],N[1],N[2])
  (DX,DY,DZ) = (D[0],D[1],D[2])
  pts = (NX+1)*(NY+1)*(NZ+1)
  D = np.zeros(shape=(pts,pts))
  potBC = np.zeros(shape=(pts))
  rowsNotBC = [i for i in range(pts)]
  dim = dimension(N)

  # Set up rows corresponding to a boundary condition
  for j in range(NY+1):
    for k in range(NZ+1):
      if dim == 1:
        applyBCs(indexTo1D(N,0,j,k),indexTo1D(N,NX,j,k),V0[0],VN[0],potBC,D,rowsNotBC)
      elif dim == 2:
        applyBCs(indexTo1D(N,0,j,k),indexTo1D(N,NX,j,k),V0[0][j],VN[0][j],potBC,D,rowsNotBC)
      elif dim == 3:
        applyBCs(indexTo1D(N,0,j,k),indexTo1D(N,NX,j,k),V0[0][j][k],VN[0][j][k],potBC,D,rowsNotBC)

  if dim == 2 or dim == 3:
    for i in range(NX+1):
      for k in range(NZ+1):
        if dim == 2:
          applyBCs(indexTo1D(N,i,0,k),indexTo1D(N,i,NY,k),V0[1][i],VN[1][i],potBC,D,rowsNotBC)
        elif dim == 3:
          applyBCs(indexTo1D(N,i,0,k),indexTo1D(N,i,NY,k),V0[1][i][k],VN[1][i][k],potBC,D,rowsNotBC)

  if dim == 3:
    for i in range(NX+1):
      for j in range(NY+1):
        applyBCs(indexTo1D(N,i,j,0),indexTo1D(N,i,j,NZ),V0[2][i][j],VN[2][i][j],potBC,D,rowsNotBC)

  # Set up rows not corresponding to a boundary condition
  for row in rowsNotBC:
    coeffX = pow(DY*DZ,2.0)

    if dim == 2 or dim == 3:
      coeffY = pow(DX*DZ,2.0)
    else:
      coeffY = 0.0

    if dim == 3:
      coeffZ = pow(DX*DY,2.0)
    else:
      coeffZ = 0.0

    coeffXYZ = -2.0*(coeffX+coeffY+coeffZ)

    D[row][row - (NZ+1)*(NY+1)] = coeffX
    D[row][row + (NZ+1)*(NY+1)] = coeffX

    if dim == 2 or dim == 3:
      D[row][row - (NZ+1)] = coeffY
      D[row][row + (NZ+1)] = coeffY

    if dim == 3:
      D[row][row - 1] = coeffZ
      D[row][row + 1] = coeffZ

    D[row][row] = coeffXYZ

  return solveForPotential(N,D,potBC,solType,relTol,absTol)
