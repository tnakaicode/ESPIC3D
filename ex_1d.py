import matplotlib.pyplot as plt

import esSolve

NX_1D   = 10
LX_1D   = 1.2

DX_1D   = LX_1D / NX_1D

V0_1D   = 1.0
VN_1D   = 2.0

pot1D = esSolve.laplace1D(NX_1D,DX_1D,V0_1D,VN_1D,"iterative",relTol=0.0,absTol=1.0e-3)

plt.plot(pot1D)
plt.show()
