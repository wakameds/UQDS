from gurobipy import *

Ports = ['Manly','Cleveland','Dunwich']
P = range(len(Ports))
B = range(18)

Travel = [
	[29, 27, 21], [39, 18, 30], [40, 20, 31], [33, 19, 27], [35, 29, 36], [21, 23, 20],
	[30, 41, 32], [37, 27, 36], [20, 25, 34], [36, 28, 20], [24, 23, 25], [38, 22, 40], 
	[39, 19, 27], [30, 18, 28], [40, 20, 32], [21, 32, 40], [23, 18, 20], [31, 18, 20]
]

Capacity = [8, 8, 6]

m = Model()

# Part A

# assign boat b to port p
X = { (b,p): m.addVar(vtype=GRB.BINARY) for p in P for b in B}

m.setObjective(quicksum(Travel[b][p]*X[b,p] for p in P for b in B))

for b in B:
	m.addConstr(quicksum(X[b,p] for p in P) == 1)

for p in P:
	m.addConstr(quicksum(X[b,p] for b in B) <= Capacity[p])

m.optimize()

for p in P:
	print(p,[b for b in B if X[b,p].x > .99])
print(m.objVal)

# Part B
Z = m.addVar()

for b in B:
	m.addConstr(Z >= quicksum(Travel[b][p]*X[b,p] for p in P))


m.setObjective(Z)

m.optimize()

for p in P:
	print(p,[b for b in B if X[b,p].x > .99])
print(m.objVal)

# Comparison
print('Total is now ',sum(Travel[b][p]*X[b,p].x for p in P for b in B))