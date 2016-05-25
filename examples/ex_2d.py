import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../solvers')
import esSolve
from mpl_toolkits.mplot3d import axes3d
from dirichlet import dirichlet as dirBC

NX = 25
NY = 30
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
V0x = dirBC(np.zeros((NY+1)))
VNx = dirBC(np.fromfunction(nonGroundedWall, (NY+1,)))

V0y = dirBC(np.zeros((NX+1)))
VNy = dirBC(np.zeros((NX+1)))

start = time.clock()

potential_0 = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"gaussSeidel",relTol=0.0,absTol=1.0)
potential_3 = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"gaussSeidel",relTol=0.0,absTol=1.0e-3)

end = time.clock()

print("That took",round(end-start,1),"seconds.")

def plot2Darrays(arrays):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  X = np.linspace(X0, X0 + LX, num=arrays[0][0].shape[0])
  Y = np.linspace(Y0, Y0 + LY, num=arrays[0][0].shape[1])
  X, Y = np.meshgrid(X, Y, indexing='ij')

  for array in arrays:
    ax.plot_wireframe(X, Y, array[0][:],label='absolute tolerance = ' + array[1],color=array[2])

  plt.legend(loc='best')
  plt.show()

plot2Darrays([[potential_0,'1.0e-0','green'],[potential_3,'1.0e-3','blue']])
