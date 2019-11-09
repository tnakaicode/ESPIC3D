import scipy.constants
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join('./'))
#sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '../solvers/')

from solvers.dirichlet import dirichlet as dirBC
from solvers.particle import particle
from solvers.particleUtils import velocityToMomentum, momentumToVelocity
from solvers.esSolve import laplace1D, potentialToElectricField, electricFieldAtPoint

# Grid
NX = 20
LX = 1.2
X0 = 1.5
DX = LX / NX

# Boundary conditions
deltaV_NR = 1.0e-3  # 1.0 mV
deltaV_R = 1.0e6  # 1.0 MV
V_min = 1.0
V0 = dirBC(V_min)
VN_NR = dirBC(V_min + deltaV_NR)
VN_R = dirBC(V_min + deltaV_R)

# Particle
mass = scipy.constants.electron_mass
charge = -scipy.constants.elementary_charge
X0_particle = X0
V0_particle = 0.0

electron_NR = particle(mass, charge, [X0_particle], [V0_particle])
electron_R_Rpush = particle(mass, charge, [X0_particle], [V0_particle])
electron_R_NRpush = particle(mass, charge, [X0_particle], [V0_particle])

# Time steps
T_NR = 0.99*pow(-2.0*mass*pow(LX, 2.0)/(charge*deltaV_NR), 0.5)
T_R = 0.99*pow(-2.0*mass*pow(LX, 2.0)/(charge*deltaV_R), 0.5)
steps = 100
DT_NR = T_NR/steps
DT_R = T_R/steps

# Solve for potential
pot1D_NR = esSolve.laplace1D(NX, DX, V0, VN_NR, "gaussSeidel",
                             relTol=0.0, absTol=1.0e-3*(deltaV_NR), useCython=False)
pot1D_R = esSolve.laplace1D(NX, DX, V0, VN_R, "gaussSeidel",
                            relTol=0.0, absTol=1.0e-3*(deltaV_R), useCython=False)

# Compute E = - grad V on grid
electricFieldOnGrid_NR = esSolve.potentialToElectricField(pot1D_NR, [DX])
electricFieldOnGrid_R = esSolve.potentialToElectricField(pot1D_R, [DX])

positions_NR = [electron_NR.position[0]]
velocities_NR = [electron_NR.velocity[0]]
positions_R_Rpush = [electron_R_Rpush.position[0]]
velocities_R_Rpush = [electron_R_Rpush.velocity[0]]
positions_R_NRpush = [electron_R_NRpush.position[0]]
velocities_R_NRpush = [electron_R_NRpush.velocity[0]]

for step in xrange(steps+1):
    # Compute E at particle position
    electricFieldAtPoint_NR = esSolve.electricFieldAtPoint(
        electricFieldOnGrid_NR, [DX], [X0], electron_NR.position)
    electricFieldAtPoint_R_Rpush = esSolve.electricFieldAtPoint(
        electricFieldOnGrid_R, [DX], [X0], electron_R_Rpush.position)
    electricFieldAtPoint_R_NRpush = esSolve.electricFieldAtPoint(
        electricFieldOnGrid_R, [DX], [X0], electron_R_NRpush.position)

    if step == 0:
        # Back velocity up 1/2 step
        electron_NR.velocity = electron_NR.velocity - \
            0.5*DT_NR*(charge/mass)*electricFieldAtPoint_NR
        electron_R_NRpush.velocity = electron_R_NRpush.velocity - \
            0.5*DT_R*(charge/mass)*electricFieldAtPoint_R_NRpush

        momentum_0 = particleUtils.velocityToMomentum(
            electron_R_Rpush.mass, electron_R_Rpush.velocity)
        momentum_minusHalf = momentum_0 - 0.5*DT_R * \
            electron_R_Rpush.charge*electricFieldAtPoint_R_Rpush
        electron_R_Rpush.velocity = particleUtils.momentumToVelocity(
            electron_R_Rpush.mass, momentum_minusHalf)

    elif step == steps:
        # Advance velocity by 1/2 step
        electron_NR.velocity = electron_NR.velocity + \
            0.5*DT_NR*(charge/mass)*electricFieldAtPoint_NR
        electron_R_NRpush.velocity = electron_R_NRpush.velocity + \
            0.5*DT_R*(charge/mass)*electricFieldAtPoint_R_NRpush

        momentum_N = particleUtils.velocityToMomentum(
            electron_R_Rpush.mass, electron_R_Rpush.velocity)
        momentum_plusHalf = momentum_N + 0.5*DT_R * \
            electron_R_Rpush.charge*electricFieldAtPoint_R_Rpush
        electron_R_Rpush.velocity = particleUtils.momentumToVelocity(
            electron_R_Rpush.mass, momentum_plusHalf)

    else:
        # Push particle
        electron_NR.push(DT_NR, electricFieldAtPoint_NR)
        electron_R_Rpush.push_relativistic(DT_R, electricFieldAtPoint_R_Rpush)
        electron_R_NRpush.push(DT_R, electricFieldAtPoint_R_NRpush)

    positions_NR.append(electron_NR.position[0])
    velocities_NR.append(electron_NR.velocity[0])
    positions_R_Rpush.append(electron_R_Rpush.position[0])
    velocities_R_Rpush.append(electron_R_Rpush.velocity[0])
    positions_R_NRpush.append(electron_R_NRpush.position[0])
    velocities_R_NRpush.append(electron_R_NRpush.velocity[0])

seconds_NR = [i*DT_NR for i in xrange(steps + 2)]
seconds_R = [i*DT_R for i in xrange(steps + 2)]

# eric: subtlety here where ther eis a 0.5*DT for some

plt.plot(seconds_NR, velocities_NR)
plt.show()
plt.plot(seconds_R, velocities_R_NRpush)
plt.plot(seconds_R, velocities_R_Rpush)
plt.show()

plt.plot(seconds_NR, positions_NR)
plt.show()
plt.plot(seconds_R, positions_R_NRpush)
plt.plot(seconds_R, positions_R_Rpush)
plt.show()

# print electron.position[0]
# print electron.velocity[0]
