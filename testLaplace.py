from numpy import *
import esSolve, sys, test, time

#tol = float((sys.argv)[1])
tol = 1.0e-3

def timeTook(start):
  print "That took " + str(time.time() - start) + " seconds."

######
# 1D #
######

print "*** 1D Tests ***"

start = time.time()

N = 10
V0 = 1.0
VN = 2.0
inc = (VN - V0)/N

potAccept1D = arange(V0, VN + inc, inc)
potDirect1D = esSolve.laplace1D(N,V0,VN,"direct",tol)
potIterative1D = esSolve.laplace1D(N,V0,VN,"iterative",tol)

test.test(potDirect1D,potAccept1D,tol,"direct")
test.test(potIterative1D,potAccept1D,tol,"iterative")

timeTook(start)

######
# 2D #
######

print " "
print "*** 2D Tests ***"

start = time.time()

NX = 16
NY = 18

LX = 1.25
LY = 2.3

DX = LX / NX
DY = LY / NY

def V1(x,y):
  return pow(x,2.0) - pow(y,2.0)

#def V2(x,y):
#  return 2.0*x + 3.0*y + 1.0

# V(x,y) = x^2 - y^2
potAccept2D = zeros((NX+1,NY+1))
for i,j in ndindex(potAccept2D.shape):
  potAccept2D[i][j] = V1(DX*i,DY*j)

# Boundary conditions
V0x = [potAccept2D[0][j]  for j in xrange(NY+1)]
VNx = [potAccept2D[NX][j] for j in xrange(NY+1)]
V0y = [potAccept2D[i][0]  for i in xrange(NX+1)]
VNy = [potAccept2D[i][NY] for i in xrange(NX+1)]

potDirect2D = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"direct",tol)
potIterative2D = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"iterative",tol)

test.test(potDirect2D,potAccept2D,tol,"direct")
test.test(potIterative2D,potAccept2D,tol,"iterative")

timeTook(start)

######
# 3D #
######

# V(x,y,z) = x^2 - y^2 + C*z
