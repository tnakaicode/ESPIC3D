import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../solvers')
import esSolve
from mpl_toolkits.mplot3d import axes3d

NX = 22
NY = 23
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
V0x = np.zeros((NY+1))
VNx = np.fromfunction(nonGroundedWall, (NY+1,))

V0y = np.zeros((NX+1))
VNy = np.zeros((NX+1))

potential = esSolve.laplace2D(NX,DX,V0x,VNx,NY,DY,V0y,VNy,"iterative",relTol=0.0,absTol=1.0e-3)

def plot2Darray(array2D):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  X = np.arange(array2D.shape[0])
  Y = np.arange(array2D.shape[1])
  X, Y = np.meshgrid(X, Y, indexing='ij')

  ax.plot_wireframe(X, Y, array2D[:])
  plt.show()

plot2Darray(potential)
