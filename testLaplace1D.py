from numpy import *
import esSolve
import sys

tol = (sys.argv)[1]

def test(calc,accept,tol,name):
  numPass = 0
  numTest = 0

  for i in ndindex(calc.shape):
    numTest = numTest + 1
    if 100.0*abs(calc[i]/accept[i]-1.0) <= tol:
      numPass = numPass + 1

  if numPass == numTest:
    print name + ": pass"
  else:
    print name + ": fail"

N = 10
V0 = 1.0
VN = 2.0
inc = (VN-V0)/N

potAccept = arange(V0, VN + inc, inc)
potDirect = esSolve.laplace1D(N,V0,VN,"direct",tol)
potIterative = esSolve.laplace1D(N,V0,VN,"iterative",tol)

test(potDirect,potAccept,tol,"direct")
test(potIterative,potAccept,tol,"iterative")
