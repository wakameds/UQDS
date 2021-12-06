from gurobipy import *

# Sets
FSDs = ["FSD0","FSD1","FSD2","FSD3","FSD4"]
DDPs = ["DDP0","DDP1","DDP2","DDP3","DDP4","DDP5","DDP6","DDP7","DDP8","DDP9"]
Scenarios = ["0","1","2","3","4","5"]
F = range(len(FSDs))
D = range(len(DDPs))
S = range(len(Scenarios))

# Data
Costs1 = [68,81,57,39,106]
Costs2 = [[59,111,54,31,94,91,120,52,63,45],
          [77,133,76,48,116,113,142,72,60,54],
          [81,45,52,86,43,58,41,67,137,107],
          [76,61,45,73,54,65,61,60,125,96],
          [89,153,95,61,135,130,164,88,50,58]]
Demans = [268,283,101,153,271,262,118,174,253,156]
Cost = 156
Maxstockf = 1010
ScenarioProb = [0.03,0.09,0.12,0.21,0.27,0.28]

# Model
m = Model("Relief Supplies")

# Variables
X = {}
for f in F:
    X[f] = m.addVar()  #import variable X[f]

Y =  {(f,d,s): m.addVar() for f in F for d in D for s in S}

# Objective
m.setObjective(quicksum(Cost*X[f] for f in F) + 
               quicksum(Costs1[f]*X[f] for f in F) +
               quicksum(ScenarioProb[s]*Costs2[f][d]*Y[f,d,s] for s in S for f in F for d in D),
               GRB.MINIMIZE)

# Constraints
#1. Reach demand of all DDPs in all scenarios
for s in S:
    for d in D:
        m.addConstr(quicksum(Y[f,d,s] for f in F) >= Demans[d])

#2. Stock of each FSD is over total amount supplied to DDPs in all scenarios
for s in S:
    for f in F:
        m.addConstr(X[f]-quicksum(Y[f,d,s] for d in D) >= 0)

#3. Stosk of FSD is not over Maxstock of FSD
for f in F:
    m.addConstr(Maxstockf-X[f] >= 0)

#4. Amount supplied from each FSDs is not over half demand of each DDP in all scenarios
for s in S:
    for f in F:
        for d in D:
            m.addConstr(2*Y[f,d,s] - Demans[d] <= 0)

#5. Affect to transport from FSD to DDP
for d in D:
    m.addConstr(Y[1,d,0] == 0)

for d in D:
    m.addConstr(Y[3,d,0] == 0)

for d in D:
    m.addConstr(Y[1,d,1] == 0)

for d in D:
    m.addConstr(Y[0,d,2] == 0)

for d in D:
    m.addConstr(Y[4,d,2] == 0)

for d in D:
    m.addConstr(Y[2,d,3] == 0)

for d in D:
    m.addConstr(Y[0,d,4] == 0)

for d in D:
    m.addConstr(Y[2,d,4] == 0)

for d in D:
    m.addConstr(Y[0,d,5] == 0)


m.optimize()


print("**The Amount of CDRD to FSDs**")
for f in F:
    print("CDRD to", FSDs[f], X[f].x, "tonne")

print("**The total Amount of CDRD to FSDs**")
print(round(sum(X[f].x for f in F)), "tonne")


print("**The Cost of Purchase Products**")
print("$",round(sum(Cost * X[f].x for f in F)))


print("**The Cost of transport Products from CDRD to FDSs**")
print("$",round(sum(Costs1[f] * X[f].x for f in F)))



print("**The Amount of FSDs to DDPs**")
for s in S:
    print("***************************scenario",s,"***************************" )
    for f in F:
        for d in D:
            print(FSDs[f], "to", DDPs[d], Y[f,d,s].x, "tonne")

print("**The Cost of supply from FSDs to DDPs**")
for s in S:
    print("***************************scenario",s,"***************************")
    print("Cost with supply from FSDs to DDPs is $", round(sum(Costs2[f][d]*Y[f,d,s].x for f in F for d in D),))



print("**The total expected cost**")
print("Total cost = $", round(m.objVal))