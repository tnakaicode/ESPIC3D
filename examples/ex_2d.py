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

potential = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"gaussSeidel",relTol=0.0,absTol=1.0e-3)

end = time.clock()

print("That took",round(end-start,1),"seconds.")

def plot2Darray(array2D):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  X = np.linspace(X0, X0 + LX, num=array2D.shape[0])
  Y = np.linspace(Y0, Y0 + LY, num=array2D.shape[1])
  X, Y = np.meshgrid(X, Y, indexing='ij')

  ax.plot_wireframe(X, Y, array2D[:])
  plt.show()

plot2Darray(potential)
