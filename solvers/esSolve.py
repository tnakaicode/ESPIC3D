import sys
import numpy as np
import linAlgSolveCy
import linAlgSolve

# Converts grid indices (i,j,k) to 1d array indices
#   1-D: fieldAs1DArray[i]                          = fieldOn1DGrid[i]
#   2-D: fieldAs1DArray[(Ny+1)*i + j]               = fieldOn2DGrid[i][j]
#   3-D: fieldAs1DArray[(Nz+1)*(Ny+1)*i+(Nz+1)*j+k] = fieldOn3DGrid[i][j][k]
def gridIndexTo1DIndex(N,i,j,k):
  return (N[2]+1)*(N[1]+1)*i+(N[2]+1)*j+k

# Updates D and potBC for rows that contain a boundary conditon
def applyBC(index,indexNeighbor,BC,D,potBC,M,rowsNotBC):
  if index not in rowsNotBC:
    if potBC[index] != BC[1]:
      # should probably just tell user that BCs overlap and will use first one
      print("inconsistent BCs")

  else:
    M[index][index] = 1.0

    if BC[0] == "dirichlet":
      potBC[index] = BC[1]

    elif BC[0] == "neumann":
      M[index][indexNeighbor] = -1.0
  
      # corresponds to 0
      if index < indexNeighbor:
        potBC[index] = BC[1]*D

      # corresponds to N
      else:
        potBC[index] = -BC[1]*D

    else:
      print("invalid bc type")

    rowsNotBC.remove(index)

# Return dimension given number of grid indices in each dimension
def dimension(N):
  if N[1] == 0 and N[2] == 0:
    return 1
  elif N[2] == 0:
    return 2
  else:
    return 3

# Takes 1-D array and puts it on the computational grid returning a d-D array
def put1DArrayOnGrid(N,array):
  (NX,NY,NZ) = (N[0],N[1],N[2])
  dim = dimension(N)

  if dim == 1:
    arrayOnGrid = np.zeros(NX+1)
    for i in range(len(arrayOnGrid)):
      arrayOnGrid[i] = array[gridIndexTo1DIndex(N,i,0,0)]
  
  elif dim == 2:
    arrayOnGrid = np.zeros((NX+1,NY+1))
    for i,j in np.ndindex(arrayOnGrid.shape):
      arrayOnGrid[i][j] = array[gridIndexTo1DIndex(N,i,j,0)]

  elif dim == 3:
    arrayOnGrid = np.zeros((NX+1,NY+1,NZ+1))
    for i,j,k in np.ndindex(arrayOnGrid.shape):
      arrayOnGrid[i][j][k] = array[gridIndexTo1DIndex(N,i,j,k)]

  return arrayOnGrid

# Solve linear system M x = B for x
def solveLinearSystem(M,B,solType,relTol,absTol,useCython=True):
  if solType == "direct":
    if useCython:
      x = linAlgSolveCy.direct(M,B)
    else:
      x = linAlgSolve.direct(M,B)
  elif solType == "jacobi":
    if useCython:
      x = linAlgSolveCy.jacobi(M,B,relTol,absTol)
    else:
      x = linAlgSolve.jacobi(M,B,relTol,absTol)
  elif solType == "gaussSeidel":
    if useCython:
      x = linAlgSolveCy.gaussSeidel(M,B,relTol,absTol)
    else:
      x = linAlgSolve.gaussSeidel(M,B,relTol,absTol)
  else:
    sys.exit("esSolve::solveLinearSystem() -- invalid solution type")

  return x

# Solve linear system M x = potBC but return x on the computational grid
def solveForPotential(N,M,potBC,solType,relTol,absTol,useCython=True):
  return put1DArrayOnGrid(N,solveLinearSystem(M,potBC,solType,relTol,absTol,useCython))

# call this potentialToNegGradPot?
def potentialToElectricField(potential,D):
  dim = len(potential.shape)

  DX = D[0]
  if dim == 1:
    electricField = np.zeros((potential.shape[0],1))
  else:
    DY = D[1]
    if dim == 2:
      electricField = np.zeros((potential.shape[0],potential.shape[1],2))
    else:
      DZ = D[2]
      electricField = np.zeros((potential.shape[0],potential.shape[1],potential.shape[2],3))

  # refactor ?
  if dim == 1:
    for i_x in range(potential.shape[0]):
      if i_x == 0:
        electricField[i_x] = -(potential[i_x+1] - potential[i_x])/DX
      elif i_x == potential.shape[0] - 1:
        electricField[i_x] = -(potential[i_x]   - potential[i_x-1])/DX
      else:
        electricField[i_x] = -(potential[i_x+1] - potential[i_x-1])/(2.0*DX)

  if dim == 2:
    for i_x in range(potential.shape[0]):
      for i_y in range(potential.shape[1]):
        if i_x == 0:
          electricField[i_x][i_y][0] = -(potential[i_x+1][i_y] - potential[i_x][i_y])/DX
        elif i_x == potential.shape[0] - 1:
          electricField[i_x][i_y][0] = -(potential[i_x][i_y]   - potential[i_x-1][i_y])/DX
        else:
          electricField[i_x][i_y][0] = -(potential[i_x+1][i_y] - potential[i_x-1][i_y])/(2.0*DX)

        if i_y == 0:
          electricField[i_x][i_y][1] = -(potential[i_x][i_y+1] - potential[i_x][i_y])/DY
        elif i_y == potential.shape[1] - 1:
          electricField[i_x][i_y][1] = -(potential[i_x][i_y]   - potential[i_x][i_y-1])/DY
        else:
          electricField[i_x][i_y][1] = -(potential[i_x][i_y+1] - potential[i_x][i_y-1])/(2.0*DY)
  
  if dim == 3:
    for i_x in range(potential.shape[0]):
      for i_y in range(potential.shape[1]):
        for i_z in range(potential.shape[2]):
          if i_x == 0:
            electricField[i_x][i_y][i_z][0] = -(potential[i_x+1][i_y][i_z] - potential[i_x][i_y][i_z])/DX
          elif i_x == potential.shape[0] - 1:
            electricField[i_x][i_y][i_z][0] = -(potential[i_x][i_y][i_z]   - potential[i_x-1][i_y][i_z])/DX
          else:
            electricField[i_x][i_y][i_z][0] = -(potential[i_x+1][i_y][i_z] - potential[i_x-1][i_y][i_z])/(2.0*DX)

          if i_y == 0:
            electricField[i_x][i_y][i_z][1] = -(potential[i_x][i_y+1][i_z] - potential[i_x][i_y][i_z])/DY
          elif i_y == potential.shape[1] - 1:
            electricField[i_x][i_y][i_z][1] = -(potential[i_x][i_y][i_z]   - potential[i_x][i_y-1][i_z])/DY
          else:
            electricField[i_x][i_y][i_z][1] = -(potential[i_x][i_y+1][i_z] - potential[i_x][i_y-1][i_z])/(2.0*DY)

          if i_z == 0:
            electricField[i_x][i_y][i_z][2] = -(potential[i_x][i_y][i_z+1] - potential[i_x][i_y][i_z])/DZ
          elif i_z == potential.shape[2] - 1:
            electricField[i_x][i_y][i_z][2] = -(potential[i_x][i_y][i_z]   - potential[i_x][i_y][i_z-1])/DZ
          else:
            electricField[i_x][i_y][i_z][2] = -(potential[i_x][i_y][i_z+1] - potential[i_x][i_y][i_z-1])/(2.0*DZ)

  return electricField

# assigns values to M and potBC in M x = potBC, using boundary conditions BC0 and BCN
def setupBCRows(N,D,BC0,BCN,M,potBC,rowsNotBC):
  (NX,NY,NZ) = (N[0],N[1],N[2])
  (DX,DY,DZ) = (D[0],D[1],D[2])

  dim = dimension(N)

  (BCX_0, BCX_NX) = (BC0[0],BCN[0])

  if dim > 1:
    (BCY_0,BCY_NY) = (BC0[1],BCN[1])

  if dim > 2:
    (BCZ_0,BCZ_NZ) = (BC0[2],BCN[2])

  for j in range(NY+1):
    for k in range(NZ+1):
      if dim == 1:
        BCX_0_values  = BCX_0.getValues()
        BCX_NX_values = BCX_NX.getValues()
      elif dim == 2:
        BCX_0_values  = BCX_0.getValues()[j]
        BCX_NX_values = BCX_NX.getValues()[j]
      elif dim == 3:
        BCX_0_values  = BCX_0.getValues()[j][k]
        BCX_NX_values = BCX_NX.getValues()[j][k]

      applyBC(gridIndexTo1DIndex(N,0,j,k),gridIndexTo1DIndex(N,1,j,k),
              [BCX_0.getType(),BCX_0_values],DX,potBC,M,rowsNotBC)
      applyBC(gridIndexTo1DIndex(N,NX,j,k),gridIndexTo1DIndex(N,NX-1,j,k),
              [BCX_NX.getType(),BCX_NX_values],DX,potBC,M,rowsNotBC)

  if dim == 2 or dim == 3:
    for i in range(NX+1):
      for k in range(NZ+1):
        if dim == 2:
          BCY_0_values  = BCY_0.getValues()[i]
          BCY_NY_values = BCY_NY.getValues()[i]
        elif dim == 3:
          BCY_0_values  = BCY_0.getValues()[i][k]
          BCY_NY_values = BCY_NY.getValues()[i][k]

        applyBC(gridIndexTo1DIndex(N,i,0,k),gridIndexTo1DIndex(N,i,1,k),
                [BCY_0.getType(),BCY_0_values],DY,potBC,M,rowsNotBC)
        applyBC(gridIndexTo1DIndex(N,i,NY,k),gridIndexTo1DIndex(N,i,NY-1,k),
                [BCY_NY.getType(),BCY_NY_values],DY,potBC,M,rowsNotBC)

  if dim == 3:
    for i in range(NX+1):
      for j in range(NY+1):
        applyBC(gridIndexTo1DIndex(N,i,j,0),gridIndexTo1DIndex(N,i,j,1),
                [BCZ_0.getType(),BCZ_0.getValues()[i][j]],DZ,potBC,M,rowsNotBC)
        applyBC(gridIndexTo1DIndex(N,i,j,NZ),gridIndexTo1DIndex(N,i,j,NZ-1),
                [BCZ_NZ.getType(),BCZ_NZ.getValues()[i][j]],DZ,potBC,M,rowsNotBC)

# assigns values to M in M x = potBC, for rows that do not correspond to BCs
def setupNonBCRows(N,D,M,rowsNotBC):
  (NX,NY,NZ) = (N[0],N[1],N[2])
  (DX,DY,DZ) = (D[0],D[1],D[2])
  dim = dimension(N)

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

    M[row][row - (NZ+1)*(NY+1)] = coeffX
    M[row][row + (NZ+1)*(NY+1)] = coeffX

    if dim == 2 or dim == 3:
      M[row][row - (NZ+1)] = coeffY
      M[row][row + (NZ+1)] = coeffY

    if dim == 3:
      M[row][row - 1] = coeffZ
      M[row][row + 1] = coeffZ

    M[row][row] = coeffXYZ

# can i assign a variable to these default params?
def laplace1D(NX,DX,BCX_0,BCX_NX, \
              solType,relTol=0.1,absTol=0.1,useCython=True):
  return laplace([NX,0,0],[DX,1.0,1.0],[BCX_0],[BCX_NX], \
                 solType,relTol,absTol,useCython)

def laplace2D(NX,DX,BCX_0,BCX_NX,NY,DY,BCY_0,BCY_NY, \
              solType,relTol=0.1,absTol=0.1,useCython=True):
  return laplace([NX,NY,0],[DX,DY,1.0],[BCX_0,BCY_0],[BCX_NX,BCY_NY], \
                 solType,relTol,absTol,useCython)

def laplace3D(NX,DX,BCX_0,BCX_NX,NY,DY,BCY_0,BCY_NY,NZ,DZ,BCZ_0,BCZ_NZ, \
              solType,relTol=0.1,absTol=0.1,useCython=True):
  return laplace([NX,NY,NZ],[DX,DY,DZ],[BCX_0,BCY_0,BCZ_0],[BCX_NX,BCY_NY,BCZ_NZ], \
                 solType,relTol,absTol,useCython)

# General Laplace Solver
def laplace(N,D,BC0,BCN,solType,relTol=0.1,absTol=0.1,useCython=True):
  (NX,NY,NZ) = (N[0],N[1],N[2])
  (DX,DY,DZ) = (D[0],D[1],D[2])
  pts        = (NX+1)*(NY+1)*(NZ+1)
  M          = np.zeros(shape=(pts,pts))
  potBC      = np.zeros(shape=(pts))
  rowsNotBC  = range(pts)
  dim        = dimension(N)

  setupBCRows(N,D,BC0,BCN,M,potBC,rowsNotBC)
  setupNonBCRows(N,D,M,rowsNotBC)

  return solveForPotential(N,M,potBC,solType,relTol,absTol,useCython)
