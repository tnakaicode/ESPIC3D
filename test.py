from numpy import *

def test(calc,accept,tol,name):
  numPass = 0
  numTest = 0

  for i in ndindex(calc.shape):
    numTest = numTest + 1
    if abs(calc[i]-accept[i]) <= tol:
      numPass = numPass + 1

  if numPass == numTest:
    print name + ": pass"
  else:
    print name + ": fail (" + str(100.0*numPass/numTest) + " percent pass)"
