import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../solvers')
import matplotlib.pyplot as plt
import esSolve
from dirichlet import dirichlet as dirBC

NX   = 100
LX   = 1.2

DX   = LX / NX

V0   = dirBC(1.0)
VN   = dirBC(2.0)

X0 = 1.5

pot1D = esSolve.laplace1D(NX,DX,V0,VN,"gaussSeidel",relTol=0.0,absTol=1.0e-3,useCython=False)

# could use X0 here to plot versus position instead of index
plt.plot(pot1D)
plt.show()
