from numpy import *
import math

# Solves A x = B directly
def directly(A,B):
  # for now, just use linalg to solve directly
  return linalg.solve(A,B)
