from gurobipy import *
import random

# Data and ranges
nHospitalSites = 30
nSuburbs = 55
MaxSuburbsPerHospital = 7
MaxPopulation = 500000

H = range(nHospitalSites)
S = range(nSuburbs)

random.seed(3)

FixedCost = [random.randint(5000000,10000000) for h in H]
Population = [random.randint(60000,90000) for s in S]

# Travel distance - multiply by population moved to get travel cost
Dist = [[random.randint(0,50) for s in S] for h in H]

# Set up model and set the gap on the answer to 0
m = Model()
m.setParam('MIPGap', 0)

# End of stub

Y = {h: m.addVar(vtype=GRB.BINARY) for h in H}
X = {(s,h): m.addVar(vtype=GRB.BINARY) for s in S for h in H}

m.update()

m.setObjective(quicksum(FixedCost[h]*Y[h] for h in H)+\
               quicksum(Dist[h][s]*Population[s]*X[s,h] for h in H for s in S))

for h in H:
    m.addConstr(quicksum(X[s,h] for s in S)<=MaxSuburbsPerHospital*Y[h])
    m.addConstr(quicksum(Population[s]*X[s,h] for s in S)<=MaxPopulation*Y[h])

for s in S:
    m.addConstr(quicksum(X[s,h] for h in H)==1)
#    for h in H:
#        m.addConstr(X[h][s]<=Y[h])

m.optimize()

for h in H:
    if Y[h].x > 0.9:
    	print ("Hospital",h, "Suburbs",[s for s in S if X[s,h].x > .9])
        #print FixedCost[h], sum(X[h][s].x for s in S), \
        #sum(Population[s]*X[h][s].x for s in S), sum(Dist[h][s]*Population[s]*X[h][s].x for s in S)
print(m.objVal)