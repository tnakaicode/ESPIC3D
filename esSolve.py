from numpy import *

X0 = 0.0
L = 1.0
N = 10
DX = L/N

pts = N+1

V0 = 1.0
V5 = 3.0
VN = 2.0

# D * potential = - DX^2 * charge / eps_0

D = zeros(shape=(pts,pts))
potBC = zeros(shape=(pts))

for row in xrange(N+1):
  if row == 0 or row == 5 or row == N:
    D[row][row] = 1.0
    if row == 0:
      potBC[row] = V0
    if row == 5:
      potBC[row] = V5
    if row == N:
      potBC[row] = VN
  else:
    for col in xrange(N+1):
      if col == row-1 or col == row+1:
        D[row][col] = 1.0
      elif col == row:
        D[row][col] = -2.0  

potential = linalg.solve(D,potBC)

print potential
