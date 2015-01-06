from numpy import *
import esSolve
import sys
import test

# Relative error tolerance
tol = float((sys.argv)[1])

######
# 1D #
######

N = 10
V0 = 1.0
VN = 2.0
inc = (VN - V0)/N

potAccept1D = arange(V0, VN + inc, inc)
potDirect1D = esSolve.laplace1D(N,V0,VN,"direct",tol)
potIterative1D = esSolve.laplace1D(N,V0,VN,"iterative",tol)

test.test(potDirect1D,potAccept1D,tol,"direct")
test.test(potIterative1D,potAccept1D,tol,"iterative")

######
# 2D #
######

NX = 10
NY = 16

LX = 1.0
LY = 2.0

DX = LX / NX
DY = LY / NY

# V(x,y) = x^2 - y^2
potAccept2D = empty((NX+1,NY+1))

for i,j in ndindex(potAccept2D.shape):
  potAccept2D[i][j] = pow(DX*i,2.0) - pow(DY*j,2.0)   

V0x = [potAccept2D[0][j]  for j in xrange(NY+1)]
VNx = [potAccept2D[NX][j] for j in xrange(NY+1)]
V0y = [potAccept2D[i][0]  for i in xrange(NX+1)]
VNy = [potAccept2D[i][NY] for i in xrange(NX+1)]

potDirect2D = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"direct",tol)
potIterative2D = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"iterative",tol)

test.test(potDirect2D,potAccept2D,tol,"direct")
test.test(potIterative2D,potAccept2D,tol,"iterative")

######
# 3D #
######
