from numpy import *
import math

# for now, just use linalg to solve directly
# might implement own solver lateri

def directly(D,potBC):
  return linalg.solve(D,potBC)
