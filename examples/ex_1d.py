import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../solvers')
import matplotlib.pyplot as plt
import esSolve

NX   = 10
LX   = 1.2

DX   = LX / NX

V0   = 1.0
VN   = 2.0

pot1D = esSolve.laplace1D(NX,DX,V0,VN,"iterative",relTol=0.0,absTol=1.0e-3)

plt.plot(pot1D)
plt.show()