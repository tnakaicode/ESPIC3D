import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../solvers')
import matplotlib.pyplot as plt
import esSolve
from particle import particle
import scipy.constants
from dirichlet import dirichlet as dirBC

NX     = 100
LX     = 1.2
X0     = 1.5
DX     = LX / NX
DT     = 1.0e-6
steps  = 128
deltaV = 1.0e-3 # 1.0 mV
V_min  = 1.0
V0     = dirBC(V_min)
VN     = dirBC(V_min + deltaV)

# Particle
mass        = scipy.constants.electron_mass
charge      = -scipy.constants.elementary_charge
X0_particle = X0
V0_particle = 0.0

pot1D = esSolve.laplace1D(NX,DX,V0,VN,"gaussSeidel",relTol=0.0,absTol=1.0e-3*(deltaV),useCython=False)

electron = particle(mass,charge,[X0_particle],[V0_particle])

electricFieldOnGrid = esSolve.potentialToElectricField(pot1D,[DX])

positions = []

for step in xrange(steps):
  electricFieldAtPoint = esSolve.electricFieldAtPoint(electricFieldOnGrid,[DX],[X0],electron.position)

  if step == 0:
    # back velocity up 1/2 step
    electron.velocity = electron.velocity - 0.5*DT*(charge/mass)*electricFieldAtPoint
 
  electron.push(DT,electricFieldAtPoint)  
  
  positions.append(electron.position[0])

# advance velocity by one step
electron.velocity = electron.velocity + 0.5*DT*(charge/mass)*electricFieldAtPoint

plt.plot(positions)
plt.show()

print electron.velocity
