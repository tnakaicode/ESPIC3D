from numpy import *
import esSolve, sys, test, time
import pytest

######################

#tol = float((sys.argv)[1])
absoluteTolerance = 1.0e-3
tol = 1.0e-3

def timeTook(start):
  print "That took " + str(time.time() - start) + " seconds."

######
# 1D #
######

print "*** 1D Tests ***"

start = time.time()

NX = 10
LX = 1.2
DX = LX / NX

# USE THIS
X0 = 0.0

def test1D(func):
  potAccept1D = zeros(NX+1)
  for i in xrange(NX+1):
    potAccept1D[i] = func(DX*i)

  V0 = potAccept1D[0]
  VN = potAccept1D[NX]

  potDirect1D = esSolve.laplace1D(NX,DX,V0,VN,"direct",tol)
  potIterative1D = esSolve.laplace1D(NX,DX,V0,VN,"iterative",tol)

  test.test(potDirect1D,potAccept1D,tol,"direct")
  test.test(potIterative1D,potAccept1D,tol,"iterative")

def V1_1D(x):
  return 1.0*x + 2.0

test1D(V1_1D)

timeTook(start)

#@pytest.fixture(params = [()])
#def data(request):
#  return request.param

#def test_1D(data):
#  calculated,accepted = data
#  assert np.allclose(calculated, accepted, 0.0, absoluteTolerance)

######
# 2D #
######

print " "
print "*** 2D Tests ***"

start = time.time()

NX = 6
NY = 7

LX = 1.25
LY = 2.3

DX = LX / NX
DY = LY / NY

# USE THIS
X0 = 0.0
Y0 = 0.0

def test2D(func):
  potAccept2D = zeros((NX+1,NY+1))
  for i,j in ndindex(potAccept2D.shape):
    potAccept2D[i][j] = func(DX*i,DY*j)

  # Boundary conditions
  V0x = [potAccept2D[0][j] for j in xrange(NY+1)]
  VNx = [potAccept2D[NX][j] for j in xrange(NY+1)]
  V0y = [potAccept2D[i][0] for i in xrange(NX+1)]
  VNy = [potAccept2D[i][NY] for i in xrange(NX+1)]

  potDirect2D = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"direct",tol)
  potIterative2D = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"iterative",tol)

  test.test(potDirect2D,potAccept2D,tol,"direct")
  test.test(potIterative2D,potAccept2D,tol,"iterative")

def V1_2D(x,y):
  return 0.5*(pow(x,2.0) - pow(y,2.0))

def V2_2D(x,y):
  return 2.0*x + 0.5*y

def V3_2D(x,y):
  return 2.0

def V4_2D(x,y):
  a = 0.1*2.0*pi/LX
  return (cos(a*x)+sin(a*x))*(cosh(a*y)+sinh(a*y))
 
def V5_2D(x,y):
  return V1_2D(x,y) + V2_2D(x,y) + V3_2D(x,y)

test2D(V1_2D)
test2D(V2_2D)
test2D(V3_2D)
test2D(V4_2D)
test2D(V5_2D)

timeTook(start)

######
# 3D #
######

print " "
print "*** 3D Tests ***"

start = time.time()

NX = 5
NY = 6
NZ = 7

LX = 1.25
LY = 2.3
LZ = 1.87

DX = LX / NX
DY = LY / NY
DZ = LZ / NZ

# USE THIS
X0 = 0.0
Y0 = 0.0
Z0 = 0.0

def test3D(func):
  potAccept3D = zeros((NX+1,NY+1,NZ+1))
  for i,j,k in ndindex(potAccept3D.shape):
    potAccept3D[i][j][k] = func(DX*i,DY*j,DZ*k)

  # Boundary conditions
  V0x = [[potAccept3D[0][j][k] for k in xrange(NZ+1)] for j in xrange(NY+1)]
  VNx = [[potAccept3D[NX][j][k] for k in xrange(NZ+1)] for j in xrange(NY+1)]
  V0y = [[potAccept3D[i][0][k] for k in xrange(NZ+1)] for i in xrange(NX+1)]
  VNy = [[potAccept3D[i][NY][k] for k in xrange(NZ+1)] for i in xrange(NX+1)]
  V0z = [[potAccept3D[i][j][0] for j in xrange(NY+1)] for i in xrange(NX+1)]
  VNz = [[potAccept3D[i][j][NZ] for j in xrange(NY+1)] for i in xrange(NX+1)]

  potDirect3D = esSolve.laplace3D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,NZ,DZ,V0z,VNz,"direct",tol)
  potIterative3D = esSolve.laplace3D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,NZ,DZ,V0z,VNz,"iterative",tol)

  test.test(potDirect3D,potAccept3D,tol,"direct")
  test.test(potIterative3D,potAccept3D,tol,"iterative")

def V1_3D(x,y,z):
  return 0.5*V5_2D(x,y) + 2.0*z

def V2_3D(x,y,z):
  return 0.5*V5_2D(x,z) + 2.0*y

def V3_3D(x,y,z):
  return 0.5*V5_2D(y,z) + 2.0*x

def V4_3D(x,y,z):
  a = 0.1*2.0*pi/max(LX,LY)
  return exp(a*x)*exp(a*y)*sin(sqrt(2.0)*a*z)

test3D(V1_3D)
test3D(V2_3D)
test3D(V3_3D)
test3D(V4_3D)

timeTook(start)
