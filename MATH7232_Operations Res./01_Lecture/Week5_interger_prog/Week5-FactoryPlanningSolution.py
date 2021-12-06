from gurobipy import *

# Set up your data
profit = [10, 6, 8, 4, 11, 9, 3]
P = range(len(profit))

n = [4, 2, 3, 1, 1]
M = range(len(n))

# usage[P][M]
usage = [
    [0.5, 0.1, 0.2, 0.05, 0.00],
    [0.7, 0.2, 0.0, 0.03, 0.00],
    [0.0, 0.0, 0.8, 0.00, 0.01],
    [0.0, 0.3, 0.0, 0.07, 0.00],
    [0.3, 0.0, 0.0, 0.10, 0.05],
    [0.2, 0.6, 0.0, 0.00, 0.00],
    [0.5, 0.0, 0.6, 0.08, 0.05]
    ]

# months
T = range(6)

# maintenance[T][M]
maint = [
    [1, 0, 0, 0, 0],
    [0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1]
    ]

# market[P][T]
market = [
    [ 500, 600, 300, 200,   0, 500],
    [1000, 500, 600, 300, 100, 500],
    [ 300, 200,   0, 400, 500, 100],
    [ 300,   0,   0, 500, 100, 300],
    [ 800, 400, 500, 200,1000,1100],
    [ 200, 300, 400,   0, 300, 500],
    [ 100, 150, 100, 100,   0,  60]
    ]

maxstore = 100
storecost = 0.5
endstore = 50
initialstore = 0
monthhours = 16*24

fp = Model("Factory Planning")

# Variables  (vtype is needed to set interger)
X = {(p,t): fp.addVar(vtype=GRB.INTEGER) for p in P for t in T} #Unit made product
S = {(p,t): fp.addVar(vtype=GRB.INTEGER) for p in P for t in T} #Store in the end on a manth
Y = {(p,t): fp.addVar(vtype=GRB.INTEGER) for p in P for t in T} #Unit sold
# Q2 -> Z[t,m] is number of machines m to maintain in month t
Z = {(t,m): fp.addVar(vtype=GRB.INTEGER) for t in T for m in M}

fp.setObjective(quicksum(profit[p]*Y[p,t] for p in P for t in T) -
                quicksum(storecost*S[p,t] for p in P for t in T), GRB.MAXIMIZE)

# Example of defining constraints with dictionary comprehension
# maxstorecons = {(p,t): fp.addConstr(S[p,t] <= maxstore) for p in P for t in T}

for p in P:
    fp.addConstr(S[p,0] == X[p,0] - Y[p,0] + initialstore)
    fp.addConstr(S[p,5] == endstore)
    
    for t in T:
        fp.addConstr(Y[p,t] <= market[p][t])
        fp.addConstr(S[p,t] <= maxstore)
        
        if t > 0:
            fp.addConstr(S[p,t] == X[p,t] - Y[p,t] + S[p,t-1])

# Q2 -> ensure we maintain the same number of each machine as in Q1 data
for m in M:
    fp.addConstr(quicksum(Z[t,m] for t in T) == sum(maint[t][m] for t in T))

# Q2 extension -> smooth maintainance so not so much happens in a single month
for t in T:
    fp.addConstr(quicksum(Z[t,m] for m in M) <= 2)
    
for m in M:
    for t in T:
        fp.addConstr(quicksum(usage[p][m]*X[p,t] for p in P) <= 
                     monthhours*(n[m]-Z[t,m]))

fp.optimize()

print("Profit = $",fp.objVal)
print("Sales")
for p in P:
    print(p, [int(Y[p,t].x) for t in T])
  
print("Production")
for p in P:
    print(p, [int(X[p,t].x) for t in T])

print("Storage")
for p in P:
    print(p, [int(S[p,t].x) for t in T])

print("Maintain")
for m in M:
    print(m, [int(Z[t,m].x) for t in T])

