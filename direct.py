from numpy import *
import math

# TO DO
# 1. Implement own direct solver instead of linalg.solve?
##########################################################

# Solves A x = B directly
def directly(A,B):
  return linalg.solve(A,B)
