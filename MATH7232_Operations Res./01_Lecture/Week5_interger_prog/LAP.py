import math
import random
from gurobipy import *
import pylab

#p = (x,y)
#math.hypot: return square
def Distance(p1, p2):
  return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

Num = 50
N = range(Num)
Square = 1000
random.seed(Num)

#Set position
#random.randient(a,b) return random interger between a and b
Plant = [(random.randint(0,Square), random.randint(0,Square)) for i in N]
Job = [(random.randint(0,Square), random.randint(0,Square)) for i in N]

D = [[Distance(Plant[i], Job[j]) for j in N] for i in N]

m = Model('Assignment')

#X: job assined to plant (0 or 1)
X = [[m.addVar() for j in N] for i in N]

m.update()

m.setObjective(quicksum(D[i][j]*X[i][j] for i in N for j in N))

C1 = [m.addConstr(quicksum(X[i][j] for j in N) == 1) for i in N] # job assined to only plant
C2 = [m.addConstr(quicksum(X[i][j] for i in N) == 1) for j in N] # plant assined to only job

m.optimize()

[pylab.plot(Plant[i][0], Plant[i][1], color='black', marker='*') for i in N]
[pylab.plot([Plant[i][0], Job[j][0]], [Plant[i][1], Job[j][1]], 'black') 
  for i in N for j in N if X[i][j].x > 0.99]

pylab.show()
