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

def E1_1D(x):
  return np.array([-1.0])

def V1_2D(x,y):
  return 0.5*(pow(x,2.0) - pow(y,2.0))

def E1_2D(x,y):
  return np.array([-x,y])

def V2_2D(x,y):
  return 2.0*x + 0.5*y

def E2_2D(x,y):
  return np.array([-2.0,-0.5])

def V3_2D(x,y):
  return 2.0

def E3_2D(x,y):
  return np.array([0.0,0.0])

def V4_2D(x,y):
  a = 0.1*2.0*math.pi/LX_2D
  return (math.cos(a*x) + math.sin(a*x))*(math.cosh(a*y) + math.sinh(a*y))

def E4_2D(x,y):
  a = 0.1*2.0*math.pi/LX_2D
  return np.array([-a*(math.cos(a*x) - math.sin(a*x))*(math.cosh(a*y) + math.sinh(a*y)), \
                   -a*(math.cos(a*x) + math.sin(a*x))*(math.cosh(a*y) + math.sinh(a*y))])

def V5_2D(x,y):
  return V1_2D(x,y) + V2_2D(x,y) + V3_2D(x,y)

def E5_2D(x,y):
  return np.array([E1_2D(x,y)[0] + E2_2D(x,y)[0] + E3_2D(x,y)[0], \
                   E1_2D(x,y)[1] + E2_2D(x,y)[1] + E3_2D(x,y)[1]])

def V1_3D(x,y,z):
  return 0.5*V5_2D(x,y) + 2.0*z

def E1_3D(x,y,z):
  return np.array([0.5*E5_2D(x,y)[0],0.5*E5_2D(x,y)[1],-2.0])

def V2_3D(x,y,z):
  return 0.5*V5_2D(x,z) + 2.0*y

def E2_3D(x,y,z):
  return np.array([0.5*E5_2D(x,z)[0],-2.0,0.5*E5_2D(x,z)[1]])

def V3_3D(x,y,z):
  return 0.5*V5_2D(y,z) + 2.0*x

def E3_3D(x,y,z):
  return np.array([-2.0,0.5*E5_2D(y,z)[0],0.5*E5_2D(y,z)[1]])

def V4_3D(x,y,z):
  a = 0.1*2.0*math.pi/max(LX_3D,LY_3D)
# should these be using np versions?
  return math.exp(a*x)*math.exp(a*y)*math.sin(math.sqrt(2.0)*a*z)

def E4_3D(x,y,z):
  a = 0.1*2.0*math.pi/max(LX_3D,LY_3D)
# should these be using np versions?
  return np.array([-a*math.exp(a*x)*math.exp(a*y)*math.sin(math.sqrt(2.0)*a*z), \
                   -a*math.exp(a*x)*math.exp(a*y)*math.sin(math.sqrt(2.0)*a*z), \
                   math.sqrt(2.0)*a*math.exp(a*x)*math.exp(a*y)*math.cos(math.sqrt(2.0)*a*z)])

def test_laplace():
  def test(array1,array2):
    assert np.allclose(array1,array2,relTol,absTol)

  def test1D(potential,field):
    potAccept1D = np.zeros(NX_1D+1)
    fieldAccept1D = np.zeros((NX_1D+1,1))
    for i in range(NX_1D+1):
      potAccept1D[i]   = potential(X0_1D+DX_1D*i)
      fieldAccept1D[i] = field(X0_1D+DX_1D*i)

    # Boundary conditions
    V0_1D = ["d",potAccept1D[0]]
    E0_1D = ["n",fieldAccept1D[0]]
    VN_1D = ["d",potAccept1D[NX_1D]]
    EN_1D = ["n",fieldAccept1D[NX_1D]]

    potDirect1D             = esSolve.laplace1D(NX_1D,DX_1D,V0_1D,VN_1D, \
                                                "direct",useCython=False)
    potDirect1D_Cython      = esSolve.laplace1D(NX_1D,DX_1D,V0_1D,VN_1D, \
                                                "direct",useCython=True)
    potJacobi1D             = esSolve.laplace1D(NX_1D,DX_1D,V0_1D,VN_1D, \
                                                "jacobi",relTol,absTol,useCython=False)
    potJacobi1D_Cython      = esSolve.laplace1D(NX_1D,DX_1D,V0_1D,VN_1D, \
                                                "jacobi",relTol,absTol,useCython=True)
    potGaussSeidel1D        = esSolve.laplace1D(NX_1D,DX_1D,V0_1D,VN_1D, \
                                                "gaussSeidel",relTol,absTol,useCython=False)
    potGaussSeidel1D_Cython = esSolve.laplace1D(NX_1D,DX_1D,V0_1D,VN_1D, \
                                                "gaussSeidel",relTol,absTol,useCython=True)

    test(potDirect1D,potAccept1D)
    test(potDirect1D_Cython,potAccept1D)
    test(potJacobi1D,potAccept1D)
    test(potJacobi1D_Cython,potAccept1D)
    test(potGaussSeidel1D,potAccept1D)
    test(potGaussSeidel1D_Cython,potAccept1D)

  def test2D(potential,field):
    potAccept2D = np.zeros((NX_2D+1,NY_2D+1))
    fieldAccept2D = np.zeros((NX_2D+1,NY_2D+1,2))
    for i,j in np.ndindex(potAccept2D.shape):
      potAccept2D[i,j] = potential(X0_2D+DX_2D*i,Y0_2D+DY_2D*j)
      fieldAccept2D[i,j] = field(X0_2D+DX_2D*i,Y0_2D+DY_2D*j)

    # Boundary conditions
    V0x_2D = ["d",potAccept2D[0,:]]
    E0x_2D = ["n",fieldAccept2D[0,:,0]]
    VNx_2D = ["d",potAccept2D[NX_2D,:]]
    ENx_2D = ["n",fieldAccept2D[NX_2D,:,0]]
    V0y_2D = ["d",potAccept2D[:,0]]
    E0y_2D = ["n",fieldAccept2D[:,0,1]]
    VNy_2D = ["d",potAccept2D[:,NY_2D]]
    ENy_2D = ["n",fieldAccept2D[:,NY_2D,1]]

    potDirect2D             = esSolve.laplace2D(NX_2D,DX_2D,V0x_2D,VNx_2D, \
                                                NY_2D,DY_2D,V0y_2D,VNy_2D, \
                                                #NY_2D,DY_2D,E0y_2D,VNy_2D, \
                                                "direct",useCython=False)
    potDirect2D_Cython      = esSolve.laplace2D(NX_2D,DX_2D,V0x_2D,VNx_2D, \
                                                NY_2D,DY_2D,V0y_2D,VNy_2D, \
                                                "direct",useCython=True)
    potJacobi2D             = esSolve.laplace2D(NX_2D,DX_2D,V0x_2D,VNx_2D, \
                                                NY_2D,DY_2D,V0y_2D,VNy_2D, \
                                                "jacobi",relTol,absTol,useCython=False)
    potJacobi2D_Cython      = esSolve.laplace2D(NX_2D,DX_2D,V0x_2D,VNx_2D, \
                                                NY_2D,DY_2D,V0y_2D,VNy_2D, \
                                                "jacobi",relTol,absTol,useCython=True)
    potGaussSeidel2D        = esSolve.laplace2D(NX_2D,DX_2D,V0x_2D,VNx_2D, \
                                                NY_2D,DY_2D,V0y_2D,VNy_2D, \
                                                "gaussSeidel",relTol,absTol,useCython=False)
    potGaussSeidel2D_Cython = esSolve.laplace2D(NX_2D,DX_2D,V0x_2D,VNx_2D, \
                                                NY_2D,DY_2D,V0y_2D,VNy_2D, \
                                                "gaussSeidel",relTol,absTol,useCython=True)

    test(potDirect2D,potAccept2D)
    test(potDirect2D_Cython,potAccept2D)
    test(potJacobi2D,potAccept2D)
    test(potJacobi2D_Cython,potAccept2D)
    test(potGaussSeidel2D,potAccept2D)
    test(potGaussSeidel2D_Cython,potAccept2D)

  def test3D(potential,field):
    potAccept3D   = np.zeros((NX_3D+1,NY_3D+1,NZ_3D+1))
    fieldAccept3D = np.zeros((NX_3D+1,NY_3D+1,NZ_3D+1,3))
    for i,j,k in np.ndindex(potAccept3D.shape):
      # consider using np.fromfunction here
      potAccept3D[i,j,k] = potential(X0_3D+DX_3D*i,Y0_3D+DY_3D*j,Z0_3D+DZ_3D*k)
      fieldAccept3D[i,j,k] = field(X0_3D+DX_3D*i,Y0_3D+DY_3D*j,Z0_3D+DZ_3D*k)

    # Boundary conditions
    V0x_3D = ["d",potAccept3D[0,:,:]]
    E0x_3D = ["d",fieldAccept3D[0,:,:,0]]
    VNx_3D = ["d",potAccept3D[NX_3D,:,:]]
    ENx_3D = ["d",fieldAccept3D[NX_3D,:,:,0]]
    V0y_3D = ["d",potAccept3D[:,0,:]]
    E0y_3D = ["d",fieldAccept3D[:,0,:,1]]
    VNy_3D = ["d",potAccept3D[:,NY_3D,:]]
    ENy_3D = ["d",fieldAccept3D[:,NY_3D,:,1]]
    V0z_3D = ["d",potAccept3D[:,:,0]]
    E0z_3D = ["d",fieldAccept3D[:,:,0,2]]
    VNz_3D = ["d",potAccept3D[:,:,NZ_3D]]
    ENz_3D = ["d",fieldAccept3D[:,:,NZ_3D,2]]

    potDirect3D             = esSolve.laplace3D(NX_3D,DX_3D,V0x_3D,VNx_3D, \
                                                NY_3D,DY_3D,V0y_3D,VNy_3D, \
                                                NZ_3D,DZ_3D,V0z_3D,VNz_3D, \
                                                "direct",useCython=False)
    potDirect3D_Cython      = esSolve.laplace3D(NX_3D,DX_3D,V0x_3D,VNx_3D, \
                                                NY_3D,DY_3D,V0y_3D,VNy_3D, \
                                                NZ_3D,DZ_3D,V0z_3D,VNz_3D, \
                                                "direct",useCython=True)
    potJacobi3D             = esSolve.laplace3D(NX_3D,DX_3D,V0x_3D,VNx_3D, \
                                                NY_3D,DY_3D,V0y_3D,VNy_3D, \
                                                NZ_3D,DZ_3D,V0z_3D,VNz_3D, \
                                                "jacobi",relTol,absTol,useCython=False)
    potJacobi3D_Cython      = esSolve.laplace3D(NX_3D,DX_3D,V0x_3D,VNx_3D, \
                                                NY_3D,DY_3D,V0y_3D,VNy_3D, \
                                                NZ_3D,DZ_3D,V0z_3D,VNz_3D, \
                                                "jacobi",relTol,absTol,useCython=True)
    potGaussSeidel3D        = esSolve.laplace3D(NX_3D,DX_3D,V0x_3D,VNx_3D, \
                                                NY_3D,DY_3D,V0y_3D,VNy_3D, \
                                                NZ_3D,DZ_3D,V0z_3D,VNz_3D, \
                                                "gaussSeidel",relTol,absTol,useCython=False)
    potGaussSeidel3D_Cython = esSolve.laplace3D(NX_3D,DX_3D,V0x_3D,VNx_3D, \
                                                NY_3D,DY_3D,V0y_3D,VNy_3D, \
                                                NZ_3D,DZ_3D,V0z_3D,VNz_3D, \
                                                "gaussSeidel",relTol,absTol,useCython=True)

    test(potDirect3D,potAccept3D)
    test(potDirect3D_Cython,potAccept3D)
    test(potJacobi3D,potAccept3D)
    test(potJacobi3D_Cython,potAccept3D)
    test(potGaussSeidel3D,potAccept3D)
    test(potGaussSeidel3D_Cython,potAccept3D)

  # Run the tests
  test1D(V1_1D,E1_1D)

  for testFuncs in [[V1_2D,E1_2D], [V2_2D,E2_2D], [V3_2D,E3_2D], [V4_2D,E4_2D], [V5_2D,E5_2D]]:
    test2D(testFuncs[0],testFuncs[1])

  for testFuncs in [[V1_3D,E1_3D], [V2_3D,E2_3D], [V3_3D,E3_3D], [V4_3D,E4_3D]]:
    test3D(testFuncs[0],testFuncs[1])
