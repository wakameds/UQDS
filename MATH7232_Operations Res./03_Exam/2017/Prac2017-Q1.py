from gurobipy import *
#Set
Packages = range(15)
Sections = ["A","B","C","D"]

P = range(15)
S = range(4)

#Data
Mass =[70,90,100,110,120,130,150,180,210,220,250,280,340,350,400]

Eachcontain = 3
Maxweight = 1000

#Variables
m = Model("Compartment")
X = {(s,p): m.addVar(vtype=GRB.BINARY) for s in S for p in P}

#Objective
m.setObjective(quicksum(X[s,p] for s in S for p in P), GRB.MAXIMIZE)

#Constraints
for s in S:
    m.addConstr(quicksum(X[s,p] for p in P) >= Eachcontain)

for s in S:
    m.addConstr(quicksum(X[s,p]*Mass[p] for p in P) <= Maxweight)

m.addConstr(quicksum(X[0,p]*Mass[p] for p in P)-quicksum(X[3,p]*Mass[p] for p in P) == 0)
m.addConstr(quicksum(X[1,p]*Mass[p] for p in P)-quicksum(X[2,p]*Mass[p] for p in P) == 0)

for p in P:
    m.addConstr(quicksum(X[s,p] for s in S)==1)

m.optimize()

print(m.ObjVal)

for s in S:
    for p in P:
        if X[s,p].x > .9:
            print(Sections[s],":Pack",Packages[p],Mass[p])
            
for s in S:
    print([X[s,p].x for p in P]) 