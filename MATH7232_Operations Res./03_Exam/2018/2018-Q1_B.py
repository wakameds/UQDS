from gurobipy import *

Ports = ['Manly','Cleveland','Dunwich']
P = range(len(Ports))
B = range(18)

Travel = [
	[29, 27, 21], [39, 18, 30], [40, 20, 31], [33, 19, 27], [35, 29, 36], [21, 23, 20],
	[30, 41, 32], [37, 27, 36], [20, 25, 34], [36, 28, 20], [24, 23, 25], [38, 22, 40], 
	[39, 19, 27], [30, 18, 28], [40, 20, 32], [21, 32, 40], [23, 18, 20], [31, 18, 20]
]

Maxcapa = [8,8,6]

#Model
m = Model("Boart")

#Variables
X = {(b,p): m.addVar(vtype=GRB.BINARY) for b in B for p in P}
Y = m.addVar(vtype=GRB.INTEGER)
#Objective
m.setObjective(Y, GRB.MINIMIZE)


#Constraints
for b in B:
    m.addConstr(quicksum(X[b,p] for p in P) == 1)
    for p in P:
        m.addConstr(X[b,p]*Travel[b][p] <= Y)
    
for p in P:
    m.addConstr(quicksum(X[b,p] for b in B) <= Maxcapa[p])

m.optimize()



print("Total travel time:", sum(X[b,p].x*Travel[b][p] for b in B for p in P))
print("Travel time for last boat:", m.objVal)

for p in P:
    print(Ports[p],":Boart", [b for b in B if X[b,p].x >.9])
    
    
   