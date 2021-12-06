from gurobipy import *

# Sets
Nodes = [0,3,4,5,6,7,8]
T = range(4) #weeks

# SSNodes = [0,1,2,8,9]  # sources and sinks

Arcs = {
        'Line1': (0,4),
        'Line2': (0,3),
        'Line3': (0,3),
        'Line4': (3,4),
        'Unload1': (4,5),
        'Unload2': (4,5),
        'Bypass': (5,7),
        'Stacker1': (5,6),
        'Stacker2': (5,6),
        'Stacker3': (5,6),
        'Stacker4': (5,6),
        'Reclaim1': (6,7),
        'Reclaim2': (6,7),
        'Reclaim3': (6,7),
        'Load1': (7,8),
        'Load2': (7,8),
        'Back': (8,0)
    }

# Data
throughput = {
        'Line1': 100,
        'Line2': 60,
        'Line3': 60,
        'Line4': 100,
        'Unload1': 80,
        'Unload2': 80,
        'Bypass': 20,
        'Stacker1': 40,
        'Stacker2': 40,
        'Stacker3': 40,
        'Stacker4': 40,
        'Reclaim1': 50,
        'Reclaim2': 50,
        'Reclaim3': 50,
        'Load1': 75,
        'Load2': 75
    }

maintain = {
    'Line3': 50,
    'Unload2': 15,
    'Bypass': 55,
    'Stacker1': 30,
    'Stacker2': 20,
    'Stacker3': 70,
    'Stacker4': 20,
    'Reclaim1': 35,
    'Reclaim2': 35,
    'Load1': 45
    }
days = [110 for t in T]

m = Model("Coal Line Maintenance")

# Variables
X = { (a,t): m.addVar() for a in Arcs for t in T}
Y = { (a,t): m.addVar(vtype=GRB.BINARY) for a in Arcs for t in T}

# Objective
m.setObjective(quicksum(X['Back',t] for t in T), GRB.MAXIMIZE)

# Constraints
for a in throughput:
    for t in T:
        m.addConstr(X[a,t] <= throughput[a]*(1 - Y[a,t]))

for n in Nodes:
    for t in T:
        #if n not in SSNodes:
            m.addConstr(quicksum(X[a,t] for a in Arcs if Arcs[a][1] == n) ==
                quicksum(X[a,t] for a in Arcs if Arcs[a][0] == n))

for t in T:
    m.addConstr(quicksum(Y[a,t]*maintain[a] for a in maintain) <= days[t])
    
for a in maintain:
    m.addConstr(quicksum(Y[a,t] for t in T) == 1)
    
# m.addConstr(X['Load1']+X['Load2'] == X['Line1']+X['Line2']+X['Line3'])

# Part C - 1
m.addConstr(Y['Bypass',0] == 1)

# Part C - 2
#Y['Stacker4',0] <= 0
#Y['Stacker4',1] <= Y['Stacker3',0]
#Y['Stacker4',2] <= Y['Stacker3',0] + Y['Stacker3',1]
for t in T:
    m.addConstr(Y['Stacker4',t] <= quicksum(Y['Stacker3',u] for u in T if u < t))

# Part C - 3
#Y['Stacker1',0] <= 0
#Y['Stacker1',1] <= 0
#Y['Stacker1',2] <= Y['Stacker2',0]
#Y['Stacker1',3] <= Y['Stacker2',0] + Y['Stacker2',1]
for t in T:
    m.addConstr(Y['Stacker1',t] <= quicksum(Y['Stacker2',u] for u in T if u+1 < t))

m.optimize()

print("Throughput")
for a in Arcs:
    print(a,[X[a,t].x for t in T])

print("\nMaintenance")    
for a in maintain:
    for t in T:
        if Y[a,t].x > .9:
            print("Maintain",a,"in Week",t+1)


