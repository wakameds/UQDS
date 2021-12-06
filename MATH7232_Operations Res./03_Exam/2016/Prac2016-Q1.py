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

#Variables
X = { (h): m.addVar(vtype=GRB.BINARY) for h in H}
Y = { (h,s): m.addVar(vtype=GRB.BINARY) for h in H for s in S}


#Objective
m.setObjective(quicksum(X[h]*FixedCost[h] for h in H)+
               quicksum(Y[h,s]*Population[s]*Dist[h][s] for h in H for s in S),GRB.MINIMIZE)

#Constraints
for h in H:
    m.addConstr(quicksum(Y[h,s] for s in S) <= MaxSuburbsPerHospital*X[h])
    
for h in H:
    m.addConstr(quicksum(Population[s]*Y[h,s] for s in S) <= MaxPopulation*X[h])
    
for s in S:
    m.addConstr(quicksum(Y[h,s] for h in H)==1)

m.optimize()
print("Min cost=$",round(m.objVal))
for h in H:
    if X[h].x >.9:
        print("Hos",h," Sub",[s for s in S if Y[h,s].x >.9])
    
    

    



