from numpy import *

def test(calc,accept,tol,name):
  numPass = 0
  numTest = 0

  for i in ndindex(calc.shape):
    numTest = numTest + 1
    if accept[i] != 0.0:
      if 100.0*abs(calc[i]/accept[i]-1.0) <= tol:
        numPass = numPass + 1
    else:
      # rethink what to do here
      if abs(calc[i] - accept[i]) <= tol:
        numPass = numPass + 1

  if numPass == numTest:
    print name + ": pass (" + str(numPass) + " of " + str(numTest) + " pass)"
  else:
    print name + ": fail (" + str(numPass) + " of " + str(numTest) + " pass)"
