from numpy import *
import esSolve

tol = 1.0e-5
N = 10
V0 = 1.0
VN = 2.0
inc = (VN-V0)/N

potAccept = arange(V0, VN + inc, inc)
pot = esSolve.laplace1D(N,V0,VN)

numPass = 0
numTest = 0

for i in ndindex(pot.shape):
  numTest = numTest + 1
  if 100.0*abs(pot[i]/potAccept[i]-1.0) <= tol:
    numPass = numPass + 1

if numPass == numTest:
  print "pass"
else:
  print "fail"
