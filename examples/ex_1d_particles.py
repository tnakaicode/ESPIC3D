import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../solvers')
import matplotlib.pyplot as plt
import esSolve
from particle import particle
import scipy.constants
from dirichlet import dirichlet as dirBC

# Grid
NX     = 20
LX     = 1.2
X0     = 1.5
DX     = LX / NX

# Time steps
DT     = 1.0e-6
steps  = 128

# Boundary conditions
deltaV = 1.0e-3 # 1.0 mV
V_min  = 1.0
V0     = dirBC(V_min)
VN     = dirBC(V_min + deltaV)

# Particle
mass        = scipy.constants.electron_mass
charge      = -scipy.constants.elementary_charge
X0_particle = X0
V0_particle = 0.0

electron = particle(mass,charge,[X0_particle],[V0_particle])

# Solve for potential
pot1D = esSolve.laplace1D(NX,DX,V0,VN,"gaussSeidel",relTol=0.0,absTol=1.0e-3*(deltaV),useCython=False)

# Compute E = - grad V on grid
electricFieldOnGrid = esSolve.potentialToElectricField(pot1D,[DX])
 
positions  = [electron.position[0]]
velocities = [electron.velocity[0]]

for step in xrange(steps):
  # Compute E at particle position
  electricFieldAtPoint = esSolve.electricFieldAtPoint(electricFieldOnGrid,[DX],[X0],electron.position)

  if step == 0:
    # Back velocity up 1/2 step
    electron.velocity = electron.velocity - 0.5*DT*(charge/mass)*electricFieldAtPoint
 
  # Push particle
  electron.push(DT,electricFieldAtPoint)  
  
  positions.append(electron.position[0])
  velocities.append(electron.velocity[0])

# Advance velocity by one step
electron.velocity = electron.velocity + 0.5*DT*(charge/mass)*electricFieldAtPoint

seconds = [i*DT for i in xrange(steps + 1)]

plt.plot(seconds,positions)
plt.show()

plt.plot(seconds,velocities)
plt.show()

print electron.position[0]
print electron.velocity[0]
