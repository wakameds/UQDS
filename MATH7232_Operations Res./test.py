from gurobipy import *

m = Model()

X1 = m.addVar()
X2 = m.addVar()

m.setObjective(4*X1 + 2*X2, GRB.MAXIMIZE)

m.addConstr(20*X1 + 50*X2 <= 480)
m.addConstr(3*X1 + X2 <= 50)
m.addConstr(0.25*X1 + 0.3*X2 <= 6)

m.optimize()

print("X1 =",X1.x," X2 =",X2.x)