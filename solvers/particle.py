import numpy as np

class particle(object):
  def __init__(self,*args):
    mass           = args[0]
    charge         = args[1]
    position       = args[2]
    velocity       = args[3]
    chargeOverMass = charge/mass

  # Boris push
  # need to use momentum for relativistic
  # google relativistic boris push
  def push(Vi,E,B,dt):
    t = 0.5*dt*chargeOverMass*B
    s = 2.0*t/(1.0 + np.dot(t,t))

    Vminus = Vi     + 0.5*dt*chargeOverMass*E
    Vprime = Vminus + np.cross(Vminus,t)
    Vplus  = Vminus + np.cross(Vprime,s)
    Vf     = Vplus  + 0.5*dt*chargeOverMass*E
