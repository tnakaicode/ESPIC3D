import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../solvers')
import esSolve
import numpy as np
import math

absTol = 0.0
relTol = 1.0e-3

NX_1D = 10
LX_1D = 1.2
DX_1D = LX_1D / NX_1D
X0_1D = 1.0

NX_2D = 6
NY_2D = 7
LX_2D = 1.25
LY_2D = 2.3
DX_2D = LX_2D / NX_2D
DY_2D = LY_2D / NY_2D
X0_2D = 1.0
Y0_2D = 2.0

NX_3D = 5
NY_3D = 6
NZ_3D = 7
LX_3D = 1.25
LY_3D = 2.3
LZ_3D = 1.87
DX_3D = LX_3D / NX_3D
DY_3D = LY_3D / NY_3D
DZ_3D = LZ_3D / NZ_3D
X0_3D = 1.0
Y0_3D = 2.0
Z0_3D = 3.0

def V1_1D(x):
  return 1.0*x + 2.0

def V1_2D(x,y):
  return 0.5*(pow(x,2.0) - pow(y,2.0))

def V2_2D(x,y):
  return 2.0*x + 0.5*y

def V3_2D(x,y):
  return 2.0

def V4_2D(x,y):
  a = 0.1*2.0*math.pi/LX_2D
  return (math.cos(a*x)+math.sin(a*x))*(math.cosh(a*y)+math.sinh(a*y))
 
def V5_2D(x,y):
  return V1_2D(x,y) + V2_2D(x,y) + V3_2D(x,y)

def V1_3D(x,y,z):
  return 0.5*V5_2D(x,y) + 2.0*z

def V2_3D(x,y,z):
  return 0.5*V5_2D(x,z) + 2.0*y

def V3_3D(x,y,z):
  return 0.5*V5_2D(y,z) + 2.0*x

def V4_3D(x,y,z):
  a = 0.1*2.0*math.pi/max(LX_3D,LY_3D)
# should these be using np versions?
  return math.exp(a*x)*math.exp(a*y)*math.sin(math.sqrt(2.0)*a*z)


def test_laplace():
  def test(array1,array2):
    assert np.allclose(array1,array2,relTol,absTol)

  def test1D(func):
    potAccept1D = np.zeros(NX_1D+1)
    for i in range(NX_1D+1):
      potAccept1D[i] = func(X0_1D+DX_1D*i)

    # Boundary conditions
    V0_1D = potAccept1D[0]
    VN_1D = potAccept1D[NX_1D]

    potDirect1D      = esSolve.laplace1D(NX_1D,DX_1D,V0_1D,VN_1D,"direct")
    potJacobi1D      = esSolve.laplace1D(NX_1D,DX_1D,V0_1D,VN_1D,"jacobi",relTol,absTol)
    potGaussSeidel1D = esSolve.laplace1D(NX_1D,DX_1D,V0_1D,VN_1D,"gaussSeidel",relTol,absTol)

    test(potDirect1D,potAccept1D)
    test(potJacobi1D,potAccept1D)
    test(potGaussSeidel1D,potAccept1D)

  def test2D(func):
    potAccept2D = np.zeros((NX_2D+1,NY_2D+1))
    for i,j in np.ndindex(potAccept2D.shape):
      potAccept2D[i,j] = func(X0_2D+DX_2D*i,Y0_2D+DY_2D*j)

    # Boundary conditions
    V0x_2D = potAccept2D[0,:]
    VNx_2D = potAccept2D[NX_2D,:]
    V0y_2D = potAccept2D[:,0]
    VNy_2D = potAccept2D[:,NY_2D]

    potDirect2D      = esSolve.laplace2D(NX_2D,DX_2D,V0x_2D,VNx_2D,NY_2D,DY_2D,V0y_2D,VNy_2D,"direct")
    potJacobi2D      = esSolve.laplace2D(NX_2D,DX_2D,V0x_2D,VNx_2D,NY_2D,DY_2D,V0y_2D,VNy_2D,"jacobi",relTol,absTol)
    potGaussSeidel2D = esSolve.laplace2D(NX_2D,DX_2D,V0x_2D,VNx_2D,NY_2D,DY_2D,V0y_2D,VNy_2D,"gaussSeidel",relTol,absTol)

    test(potDirect2D,potAccept2D)
    test(potJacobi2D,potAccept2D)
    test(potGaussSeidel2D,potAccept2D)

  def test3D(func):
    potAccept3D = np.zeros((NX_3D+1,NY_3D+1,NZ_3D+1))
    for i,j,k in np.ndindex(potAccept3D.shape):
      # consider using np.fromfunction here
      potAccept3D[i,j,k] = func(X0_3D+DX_3D*i,Y0_3D+DY_3D*j,Z0_3D+DZ_3D*k)

    # Boundary conditions
    V0x_3D = potAccept3D[0,:,:]
    VNx_3D = potAccept3D[NX_3D,:,:]
    V0y_3D = potAccept3D[:,0,:]
    VNy_3D = potAccept3D[:,NY_3D,:]
    V0z_3D = potAccept3D[:,:,0]
    VNz_3D = potAccept3D[:,:,NZ_3D]

    potDirect3D      = esSolve.laplace3D(NX_3D,DX_3D,V0x_3D,VNx_3D,NY_3D,DY_3D,V0y_3D,VNy_3D,NZ_3D,DZ_3D,V0z_3D,VNz_3D,"direct")
    potJacobi3D      = esSolve.laplace3D(NX_3D,DX_3D,V0x_3D,VNx_3D,NY_3D,DY_3D,V0y_3D,VNy_3D,NZ_3D,DZ_3D,V0z_3D,VNz_3D,"jacobi",relTol,absTol)
    potGaussSeidel3D = esSolve.laplace3D(NX_3D,DX_3D,V0x_3D,VNx_3D,NY_3D,DY_3D,V0y_3D,VNy_3D,NZ_3D,DZ_3D,V0z_3D,VNz_3D,"gaussSeidel",relTol,absTol)

    test(potDirect3D,potAccept3D)
    test(potJacobi3D,potAccept3D)
    test(potGaussSeidel3D,potAccept3D)

  # Run the tests
  test1D(V1_1D)

  for testFunc in [V1_2D, V2_2D, V3_2D, V4_2D, V5_2D]:
    test2D(testFunc)

  for testFunc in [V1_3D, V2_3D, V3_3D, V4_3D]:
    test3D(testFunc)
