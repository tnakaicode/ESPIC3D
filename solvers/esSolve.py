import sys
import numpy as np
import linAlgSolveCy
import linAlgSolve

# Converts grid indices (i,j,k) to 1d array indices
#   1-D: fieldAs1DArray[i] = fieldOn1DGrid[i]
#   2-D: fieldAs1DArray[(Ny+1)*i + j] = fieldOn2DGrid[i][j]
#   3-D: fieldAs1DArray[(Nz+1)*(Ny+1)*i+(Nz+1)*j+k] = fieldOn3DGrid[i][j][k]
def gridIndexTo1DIndex(N,i,j,k):
  return (N[2]+1)*(N[1]+1)*i+(N[2]+1)*j+k

# Updates D and potBC for rows that contain a boundary conditon

#look everywhere where V is used for a BC and change to something like "BC"

def applyBCs(index1,index1right,index2,index2left,V1,V2,D,potBC,M,rowsNotBC):
  indexes = [[index1,index1right],[index2,index2left]]
  V = [V1,V2]
  for i in range(2):
    index = indexes[i][0]
    indexNearby = indexes[i][1]
    bcType = V[i][0]
    myV = V[i][1]
    # This will need to be updated when we add Neumann BCs
    if index not in rowsNotBC:
      if potBC[index] != myV:
        print("inconsistent BCs")
    else:
      M[index][index] = 1.0

      if bcType == "d":
        potBC[index] = myV

      elif bcType == "n":
        M[index][indexNearby] = -1.0
  
        # corresponds to 0
        if index == index1:
          potBC[index] = myV*D

        # corresponds to N
        elif index == index2:
          potBC[index] = -myV*D

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
  (N0,N1,N2) = (N[0],N[1],N[2])
  dim = dimension(N)

  if dim == 1:
    arrayOnGrid = np.zeros(N0+1)
    for i in range(len(arrayOnGrid)):
      arrayOnGrid[i] = array[gridIndexTo1DIndex(N,i,0,0)]
  
  elif dim == 2:
    arrayOnGrid = np.zeros((N0+1,N1+1))
    for i,j in np.ndindex(arrayOnGrid.shape):
      arrayOnGrid[i][j] = array[gridIndexTo1DIndex(N,i,j,0)]

  elif dim == 3:
    arrayOnGrid = np.zeros((N0+1,N1+1,N2+1))
    for i,j,k in np.ndindex(arrayOnGrid.shape):
      arrayOnGrid[i][j][k] = array[gridIndexTo1DIndex(N,i,j,k)]

  return arrayOnGrid

# Solve linear system M x = B
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

# assigns values to M and potBC in M x = potBC, using boundary conditions
def setupBCRows(N,D,V0,VN,M,potBC,rowsNotBC):
  (N0,N1,N2) = (N[0],N[1],N[2])
  (D0,D1,D2) = (D[0],D[1],D[2])
  dim = dimension(N)
  for j in range(N1+1):
    for k in range(N2+1):
      if dim == 1:
        applyBCs(gridIndexTo1DIndex(N,0,j,k),gridIndexTo1DIndex(N,1,j,k), \
                 gridIndexTo1DIndex(N,N0,j,k),gridIndexTo1DIndex(N,N0-1,j,k), \
                 [V0[0][0],V0[0][1]],[VN[0][0],VN[0][1]], \
                 D0,potBC,M,rowsNotBC)

      elif dim == 2:
        applyBCs(gridIndexTo1DIndex(N,0,j,k),gridIndexTo1DIndex(N,1,j,k), \
                 gridIndexTo1DIndex(N,N0,j,k),gridIndexTo1DIndex(N,N0-1,j,k), \
                 [V0[0][0],V0[0][1][j]],[VN[0][0],VN[0][1][j]], \
                 D0,potBC,M,rowsNotBC)

      elif dim == 3:
        applyBCs(gridIndexTo1DIndex(N,0,j,k),gridIndexTo1DIndex(N,1,j,k), \
                 gridIndexTo1DIndex(N,N0,j,k),gridIndexTo1DIndex(N,N0-1,j,k), \
                 [V0[0][0],V0[0][1][j][k]],[VN[0][0],VN[0][1][j][k]], \
                 D0,potBC,M,rowsNotBC)

  if dim == 2 or dim == 3:
    for i in range(N0+1):
      for k in range(N2+1):
        if dim == 2:
          applyBCs(gridIndexTo1DIndex(N,i,0,k),gridIndexTo1DIndex(N,i,1,k), \
                   gridIndexTo1DIndex(N,i,N1,k),gridIndexTo1DIndex(N,i,N1-1,k), \
                   [V0[0][0],V0[1][1][i]],[VN[0][0],VN[1][1][i]], \
                   D1,potBC,M,rowsNotBC)

        elif dim == 3:
          applyBCs(gridIndexTo1DIndex(N,i,0,k),gridIndexTo1DIndex(N,i,1,k), \
                   gridIndexTo1DIndex(N,i,N1,k),gridIndexTo1DIndex(N,i,N1-1,k), \
                   [V0[0][0],V0[1][1][i][k]],[VN[0][0],VN[1][1][i][k]], \
                   D1,potBC,M,rowsNotBC)

  if dim == 3:
    for i in range(N0+1):
      for j in range(N1+1):
        applyBCs(gridIndexTo1DIndex(N,i,j,0), gridIndexTo1DIndex(N,i,j,1), \
                 gridIndexTo1DIndex(N,i,j,N2), gridIndexTo1DIndex(N,i,j,N2-1), \
                 [V0[0][0],V0[2][1][i][j]],[VN[0][0],VN[2][1][i][j]], \
                 D2,potBC,M,rowsNotBC)

# assigns values to M in M x = potBC, for rows that do not correspond to BCs
def setupNonBCRows(N,D,M,rowsNotBC):
  (N0,N1,N2) = (N[0],N[1],N[2])
  (D0,D1,D2) = (D[0],D[1],D[2])
  dim = dimension(N)

  for row in rowsNotBC:
    coeffX = pow(D1*D2,2.0)

    if dim == 2 or dim == 3:
      coeffY = pow(D0*D2,2.0)
    else:
      coeffY = 0.0

    if dim == 3:
      coeffZ = pow(D0*D1,2.0)
    else:
      coeffZ = 0.0

    coeffXYZ = -2.0*(coeffX+coeffY+coeffZ)

    M[row][row - (N2+1)*(N1+1)] = coeffX
    M[row][row + (N2+1)*(N1+1)] = coeffX

    if dim == 2 or dim == 3:
      M[row][row - (N2+1)] = coeffY
      M[row][row + (N2+1)] = coeffY

    if dim == 3:
      M[row][row - 1] = coeffZ
      M[row][row + 1] = coeffZ

    M[row][row] = coeffXYZ

# can i assign a variable to these default params?
def laplace1D(N0,D0,V0_0,V0_N0,solType,relTol=0.1,absTol=0.1,useCython=True):
  return laplace([N0,0,0],[D0,1.0,1.0],[V0_0],[V0_N0],solType,relTol,absTol,useCython)

def laplace2D(N0,D0,V0_0,V0_N0,N1,D1,V1_0,V1_N1,solType,relTol=0.1,absTol=0.1,useCython=True):
  return laplace([N0,N1,0],[D0,D1,1.0],[V0_0,V1_0],[V0_N0,V1_N1],solType,relTol,absTol,useCython)

def laplace3D(N0,D0,V0_0,V0_N0,N1,D1,V1_0,V1_N1,N2,D2,V2_0,V2_N2,solType,relTol=0.1,absTol=0.1,useCython=True):
  return laplace([N0,N1,N2],[D0,D1,D2],[V0_0,V1_0,V2_0],[V0_N0,V1_N1,V2_N2],solType,relTol,absTol,useCython)

# General Laplace Solver
def laplace(N,D,V0,VN,solType,relTol=0.1,absTol=0.1,useCython=True):
  (N0,N1,N2) = (N[0],N[1],N[2])
  (D0,D1,D2) = (D[0],D[1],D[2])
  pts        = (N0+1)*(N1+1)*(N2+1)
  M          = np.zeros(shape=(pts,pts))
  potBC      = np.zeros(shape=(pts))
  rowsNotBC  = range(pts)
  dim        = dimension(N)

  setupBCRows(N,D,V0,VN,M,potBC,rowsNotBC)
  setupNonBCRows(N,D,M,rowsNotBC)

  return solveForPotential(N,M,potBC,solType,relTol,absTol,useCython)
