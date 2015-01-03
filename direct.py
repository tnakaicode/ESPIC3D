from numpy import *
import math

# for now, just use linalg
# to solve directly
def directly(D,potBC):
  return linalg.solve(D,potBC)
