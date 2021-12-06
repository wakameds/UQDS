from gurobipy import *

#Set
Options = ["TV","Social Media","Print","Radio","Cinema","In-store Marketing"]
O=range(len(Options))


#Data
Customers=[1000000,200000,300000,400000,450000,450000]
Cost =[500000,150000,300000,250000,250000,100000]
Designer=[700,250,200,200,300,400]
Salesman=[200,100,100,100,100,1000]

#Model
m = Model("AJPCo")


#Variables
X = {o: m.addVar(vtype=GRB.BINARY) for o in O}

m.setObjective(quicksum(Customers[o]*X[o] for o in O), GRB.MAXIMIZE)


#Constraints
m.addConstr(quicksum(Cost[o]*X[o] for o in O) <= 1400000)
m.addConstr(quicksum(Designer[o]*X[o] for o in O) <= 1500)
m.addConstr(quicksum(Salesman[o]*X[o] for o in O) <= 1200)
m.addConstr(X[1]+X[4] <= 1)

for o in O:
    if o == 5:
        m.addConstr(X[3]+X[4] >= 1)

m.optimize()
print("Customers",m.objVal)

for o in O:
    if X[o].x>.9:
        print(Options[o],X[o].x)
