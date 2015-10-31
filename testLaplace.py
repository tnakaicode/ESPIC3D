#!/usr/bin/python

import numpy as np
import esSolve, sys, time
import math
#import pytest

######################

absoluteTolerance = 0.0
relativeTolerance = 1.0e-3

def timeTook(start):
  print "That took " + str(time.time() - start) + " seconds."

def test(array1,array2,testName):
  if (np.allclose(array1,array2,relativeTolerance,absoluteTolerance)):
    result = "PASS"
  else:
    result = "FAIL"
  print testName + ": " + result

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
  potAccept1D = np.zeros(NX+1)
  for i in xrange(NX+1):
    potAccept1D[i] = func(DX*i)

  V0 = potAccept1D[0]
  VN = potAccept1D[NX]

  potDirect1D    = esSolve.laplace1D(NX,DX,V0,VN,"direct",relativeTolerance)
  potIterative1D = esSolve.laplace1D(NX,DX,V0,VN,"iterative",relativeTolerance)

  test(potDirect1D,potAccept1D,"direct")
  test(potIterative1D,potAccept1D,"iterative")

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
  potAccept2D = np.zeros((NX+1,NY+1))
  for i,j in np.ndindex(potAccept2D.shape):
    potAccept2D[i][j] = func(DX*i,DY*j)

  # Boundary conditions
  V0x = [potAccept2D[0][j]  for j in xrange(NY+1)]
  VNx = [potAccept2D[NX][j] for j in xrange(NY+1)]
  V0y = [potAccept2D[i][0]  for i in xrange(NX+1)]
  VNy = [potAccept2D[i][NY] for i in xrange(NX+1)]

  potDirect2D    = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"direct",relativeTolerance)
  potIterative2D = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"iterative",relativeTolerance)

  test(potDirect2D,potAccept2D,"direct")
  test(potIterative2D,potAccept2D,"iterative")

def V1_2D(x,y):
  return 0.5*(pow(x,2.0) - pow(y,2.0))

def V2_2D(x,y):
  return 2.0*x + 0.5*y

def V3_2D(x,y):
  return 2.0

def V4_2D(x,y):
  a = 0.1*2.0*math.pi/LX
  return (math.cos(a*x)+math.sin(a*x))*(math.cosh(a*y)+math.sinh(a*y))
 
def V5_2D(x,y):
  return V1_2D(x,y) + V2_2D(x,y) + V3_2D(x,y)

for testFunc in [V1_2D, V2_2D, V3_2D, V4_2D, V5_2D]:
  test2D(testFunc)

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
  potAccept3D = np.zeros((NX+1,NY+1,NZ+1))
  for i,j,k in np.ndindex(potAccept3D.shape):
    potAccept3D[i][j][k] = func(DX*i,DY*j,DZ*k)

  # Boundary conditions
  V0x = [[potAccept3D[0][j][k]  for k in xrange(NZ+1)] for j in xrange(NY+1)]
  VNx = [[potAccept3D[NX][j][k] for k in xrange(NZ+1)] for j in xrange(NY+1)]
  V0y = [[potAccept3D[i][0][k]  for k in xrange(NZ+1)] for i in xrange(NX+1)]
  VNy = [[potAccept3D[i][NY][k] for k in xrange(NZ+1)] for i in xrange(NX+1)]
  V0z = [[potAccept3D[i][j][0]  for j in xrange(NY+1)] for i in xrange(NX+1)]
  VNz = [[potAccept3D[i][j][NZ] for j in xrange(NY+1)] for i in xrange(NX+1)]

  potDirect3D    = esSolve.laplace3D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,NZ,DZ,V0z,VNz,"direct",relativeTolerance)
  potIterative3D = esSolve.laplace3D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,NZ,DZ,V0z,VNz,"iterative",relativeTolerance)

  test(potDirect3D,potAccept3D,"direct")
  test(potIterative3D,potAccept3D,"iterative")

def V1_3D(x,y,z):
  return 0.5*V5_2D(x,y) + 2.0*z

def V2_3D(x,y,z):
  return 0.5*V5_2D(x,z) + 2.0*y

def V3_3D(x,y,z):
  return 0.5*V5_2D(y,z) + 2.0*x

def V4_3D(x,y,z):
  a = 0.1*2.0*math.pi/max(LX,LY)
  return math.exp(a*x)*math.exp(a*y)*math.sin(math.sqrt(2.0)*a*z)

for testFunc in [V1_3D, V2_3D, V3_3D, V4_3D]:
  test3D(testFunc)

timeTook(start)
