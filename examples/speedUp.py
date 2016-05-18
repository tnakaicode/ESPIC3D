import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../solvers')
import esSolve
from mpl_toolkits.mplot3d import axes3d

NX = 35
NY = 40
LX = 1.25
LY = 2.3
DX = LX / NX
DY = LY / NY
X0 = 1.0
Y0 = 2.0

amplitude = 2.0

def nonGroundedWall(Yindex):
  return amplitude * np.sin(np.pi * Yindex / NY)

# Boundary conditions
V0x = ["d",np.zeros((NY+1))]
VNx = ["d",np.fromfunction(nonGroundedWall, (NY+1,))]

V0y = ["d",np.zeros((NX+1))]
VNy = ["d",np.zeros((NX+1))]

start = time.clock()
potential_NoCython = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"gaussSeidel",relTol=0.0,absTol=1.0e-3,useCython=False)
end = time.clock()
print("That took",round(end-start,1),"seconds.")

start = time.clock()
potential_Cython = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"gaussSeidel",relTol=0.0,absTol=1.0e-3,useCython=True)
end = time.clock()
print("That took",round(end-start,1),"seconds.")
