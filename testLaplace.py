from numpy import *
import esSolve
import sys
import test

# Relative error tolerance
tol = (sys.argv)[1]

######
# 1D #
######

Nx = 10
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
NY = 15

LX = 1.0
LY = 2.0

DX = LX / NX
DY = LY / NY

# V(x,y) = x^2 - y^2
potAccept2D =  

potDirect2D = esSolve.laplace2D(Nx,V0x,VNx,Ny,V0y,VNy,"direct",tol)
potIterative2D = esSolve.laplace2D(Nx,V0x,VNx,Ny,V0y,VNy,"iterative",tol)

test.test(potDirect2D,potAccept2D,tol,"direct")
test.test(potIterative2D,potAccept2D,tol,"iterative")

######
# 3D #
######
