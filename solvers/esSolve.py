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
def applyBCs(index1,index1right,index2,index2left,BC1,BC2,D,potBC,M,rowsNotBC):
  indexes = [[index1,index1right],[index2,index2left]]
  BC = [BC1,BC2]
  for i in range(2):
    index = indexes[i][0]
    indexNearby = indexes[i][1]
    bcType = BC[i][0]
    myBC = BC[i][1]
    # This will need to be updated when we add Neumann BCs
    if index not in rowsNotBC:
      if potBC[index] != myBC:
        print("inconsistent BCs")
    else:
      M[index][index] = 1.0

      if bcType == "d":
        potBC[index] = myBC

      elif bcType == "n":
        M[index][indexNearby] = -1.0
  
        # corresponds to 0
        if index == index1:
          potBC[index] = myBC*D

        # corresponds to N
        elif index == index2:
          potBC[index] = -myBC*D

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

# assigns values to M and potBC in M x = potBC, using boundary conditions BC0 and BCN
def setupBCRows(N,D,BC0,BCN,M,potBC,rowsNotBC):
  (NX,NY,NZ) = (N[0],N[1],N[2])
  (DX,DY,DZ) = (D[0],D[1],D[2])
  dim = dimension(N)
  for j in range(NY+1):
    for k in range(NZ+1):
      if dim == 1:
        applyBCs(gridIndexTo1DIndex(N,0,j,k),gridIndexTo1DIndex(N,1,j,k), \
                 gridIndexTo1DIndex(N,NX,j,k),gridIndexTo1DIndex(N,NX-1,j,k), \
                 [BC0[0][0],BC0[0][1]],[BCN[0][0],BCN[0][1]], \
                 DX,potBC,M,rowsNotBC)

      elif dim == 2:
        applyBCs(gridIndexTo1DIndex(N,0,j,k),gridIndexTo1DIndex(N,1,j,k), \
                 gridIndexTo1DIndex(N,NX,j,k),gridIndexTo1DIndex(N,NX-1,j,k), \
                 [BC0[0][0],BC0[0][1][j]],[BCN[0][0],BCN[0][1][j]], \
                 DX,potBC,M,rowsNotBC)

      elif dim == 3:
        applyBCs(gridIndexTo1DIndex(N,0,j,k),gridIndexTo1DIndex(N,1,j,k), \
                 gridIndexTo1DIndex(N,NX,j,k),gridIndexTo1DIndex(N,NX-1,j,k), \
                 [BC0[0][0],BC0[0][1][j][k]],[BCN[0][0],BCN[0][1][j][k]], \
                 DX,potBC,M,rowsNotBC)

  if dim == 2 or dim == 3:
    for i in range(NX+1):
      for k in range(NZ+1):
        if dim == 2:
          applyBCs(gridIndexTo1DIndex(N,i,0,k),gridIndexTo1DIndex(N,i,1,k), \
                   gridIndexTo1DIndex(N,i,NY,k),gridIndexTo1DIndex(N,i,NY-1,k), \
                   [BC0[0][0],BC0[1][1][i]],[BCN[0][0],BCN[1][1][i]], \
                   DY,potBC,M,rowsNotBC)

        elif dim == 3:
          applyBCs(gridIndexTo1DIndex(N,i,0,k),gridIndexTo1DIndex(N,i,1,k), \
                   gridIndexTo1DIndex(N,i,NY,k),gridIndexTo1DIndex(N,i,NY-1,k), \
                   [BC0[0][0],BC0[1][1][i][k]],[BCN[0][0],BCN[1][1][i][k]], \
                   DY,potBC,M,rowsNotBC)

  if dim == 3:
    for i in range(NX+1):
      for j in range(NY+1):
        applyBCs(gridIndexTo1DIndex(N,i,j,0), gridIndexTo1DIndex(N,i,j,1), \
                 gridIndexTo1DIndex(N,i,j,NZ), gridIndexTo1DIndex(N,i,j,NZ-1), \
                 [BC0[0][0],BC0[2][1][i][j]],[BCN[0][0],BCN[2][1][i][j]], \
                 DZ,potBC,M,rowsNotBC)

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
def laplace1D(NX,DX,BCX_0,BCX_NX,solType,relTol=0.1,absTol=0.1,useCython=True):
  return laplace([NX,0,0],[DX,1.0,1.0],[BCX_0],[BCX_NX],solType,relTol,absTol,useCython)

def laplace2D(NX,DX,BCX_0,BCX_NX,NY,DY,BCY_0,BCY_NY,solType,relTol=0.1,absTol=0.1,useCython=True):
  return laplace([NX,NY,0],[DX,DY,1.0],[BCX_0,BCY_0],[BCX_NX,BCY_NY],solType,relTol,absTol,useCython)

def laplace3D(NX,DX,BCX_0,BCX_NX,NY,DY,BCY_0,BCY_NY,NZ,DZ,BCZ_0,BCZ_NZ,solType,relTol=0.1,absTol=0.1,useCython=True):
  return laplace([NX,NY,NZ],[DX,DY,DZ],[BCX_0,BCY_0,BCZ_0],[BCX_NX,BCY_NY,BCZ_NZ],solType,relTol,absTol,useCython)

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
