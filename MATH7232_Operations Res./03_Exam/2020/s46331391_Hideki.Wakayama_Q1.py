from gurobipy import *

Warehouses = ['A','B','C','D','E']

# Supply at each warehouse
Supply = [208, 209, 224, 259, 268]

# Demand for each customer
Demand = [79, 62, 104, 134, 117, 66, 53, 91, 82, 119]

# Transport cost from warehouse to customer
Costs = [
    [27, 23, 24, 33, 31, 26, 33, 28, 32, 32],
    [30, 24, 26, 34, 38, 33, 40, 37, 30, 23],
    [23, 23, 42, 23, 36, 34, 30, 24, 27, 38],
    [37, 41, 38, 25, 37, 34, 40, 40, 23, 35],
    [31, 36, 24, 38, 34, 31, 39, 24, 44, 38]
]

W = range(len(Supply))
C = range(len(Demand))


m = Model("Widget")

#Variables
X = {(w,c): m.addVar() for w in W for c in C}


#Objective
m.setObjective(quicksum(Costs[w][c]*X[w,c] for w in W for c in C),GRB.MINIMIZE)

#Constraints
for c in C:
    m.addConstr(Demand[c] <= quicksum(X[w,c] for w in W)) 


for w in W:
    m.addConstr(Supply[w] >= quicksum(X[w,c] for c in C)) 

m.optimize()
print("Min cost=$",round(m.objVal))

for w in W:
    print("Warehouse",w,sum(X[w,c].x for c in C), "tonne")

for w in W:
    print("Warehouse",w,"to customer", [X[w,c].x for c in C], "tonne")


print("Sensitivity Analysis - Variables")
for w in W:
    for c in C:
        print(w,round(X[w,c].obj,3),X[w,c].x,round(X[w,c].RC,3),round(X[w,c].SAObjLow,3),round(X[w,c].SAObjUp,3))
