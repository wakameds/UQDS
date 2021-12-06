from gurobipy import *

# Sets
FSDs = ["FSD0","FSD1","FSD2","FSD3","FSD4","FSD5","FSD6","FSD7","FSD8"]
DDPs = ["DDP0","DDP1","DDP2","DDP3","DDP4","DDP5","DDP6","DDP7","DDP8","DDP9"]
F = range(len(FSDs))
D = range(len(DDPs))
N = range(5)

# Data
Costs1 = [68,81,57,39,106,39,41,104,78]
Costs2 = [[59,111,54,31,94,91,120,52,63,45],
          [77,133,76,48,116,113,142,72,60,54],
          [81,45,52,86,43,58,41,67,137,107],
          [76,61,45,73,54,65,61,60,125,96],
          [89,153,95,61,135,130,164,88,50,58],
          [67,96,47,47,81,84,101,54,91,68],
          [58,73,31,49,59,64,78,43,99,71],
          [121,50,97,133,66,82,33,110,184,153],
          [112,54,84,119,63,80,39,99,170,140]
          ]


Demans = [268,283,101,153,271,262,118,174,253,156]
Cost = 156
Maxstockf = 1010
trackcapa = 100
trackcost = 20000

# Model
m = Model("Relief Supplies")

# Variables
X = {}
for f in F:
    X[f] = m.addVar()  #the amount of transport goods to FSD

Y = {(f,d): m.addVar() for f in F for d in D}

Z = [m.addVar(vtype=GRB.BINARY) for f in F] #if goods are stocked at FSD Z=1, if not, Z=0

CF = {(f): m.addVar(vtype=GRB.INTEGER) for f in F} #the count with transport from CDRD to FSD

FD = {(f,d): m.addVar(vtype=GRB.INTEGER) for f in F for d in D} #the count with transport from FSD to DDP

T = {(f,n): m.addVar(vtype=GRB.BINARY) for f in F for n in N} #the number of allocated trucks at FSD

# Objective
m.setObjective(quicksum(Cost*X[f] for f in F) + 
               quicksum(trackcapa*Costs1[f]*CF[f] for f in F) +
               quicksum(trackcapa*Costs2[f][d]*FD[f,d] for f in F for d in D)+
               quicksum(trackcost*T[f,n] for f in F for n in N),
               GRB.MINIMIZE)

# Constraints
#1. Reach demand of all DDPs
for d in D:
    m.addConstr(quicksum(Y[f,d] for f in F) >= Demans[d])

#2. Stock of each FSD is over total amount supplied to DDPs in all scenarios
for f in F:
    m.addConstr(X[f]*Z[f]-quicksum(Y[f,d] for d in D) >= 0)

#3. Stosk of FSD is not over Maxstock of FSD
for f in F:
    m.addConstr(Maxstockf-X[f] >= 0)

#4.  select less than 5 FSDs
m.addConstr(quicksum(Z[f] for f in F) <= 5)

#5 select from 5FSD to 8FSDs at most one
m.addConstr(Z[5]+Z[6]+Z[7]+Z[8] <= 1)

#6 count of transport to FSD
for f in F:
    m.addConstr(X[f]/trackcapa <= CF[f])
    
#7 count of transport to DDP
for f in F:
    for d in D:
        m.addConstr(Y[f,d]/trackcapa <= FD[f,d])
 
#8
for f in F:
    m.addConstr(quicksum(3*Costs2[f][d]*FD[f,d] for d in D) <= 720*quicksum(T[f,n] for n in N))   

#9 The number of track
for f in F:
    m.addConstr(quicksum(T[f,n] for n in N)<=6)

#10 The number of track
for f in F:
    for n in N:
        m.addConstr(T[f,n] <= 720)


m.optimize()



print("**The total expected cost**")
print("Total cost = $", round(m.objVal))

print("/purchased cost")
print("$", round(sum(Cost*X[f].x for f in F)))

print("/distributing cost")
print("$",round(sum(trackcapa*Costs1[f]*CF[f].x for f in F) +
               sum(trackcapa*Costs2[f][d]*FD[f,d].x for f in F for d in D)))


print("/track cost")
print("$", round(sum(trackcost*T[f,n].x for f in F for n in N)))


print("***the amount of relief good****")
print([round(X[f].x,1)for f in F])


print("***the amount of relief good to DDP****")
for f in F:
    print(f, [round(Y[f,d].x,1)for d in D])


print("****allocated track at FSD****")
for f in F:
    print(f,[int(sum(T[f,n].x for n in N))])


print("****count with transport from CDRD to FSD******")
print([int(CF[f].x) for f in F])


print("****time for transport from FSD to DDP******")
for f in F:
    print(f,[Costs2[f][d]*3 for d in D])

print("****count with transport from FSD to DDP******")
for f in F:
    print(f,[int(FD[f,d].x)for d in D])

print("****sum of time with transport from FSD to DDP by DDP and FSD******")
for f in F:
    print(f,[int(FD[f,d].x)*Costs2[f][d]*3 for d in D])

print("****sum of time with transport from FSD to DDP******")
for f in F:
    print(f,[sum(int(FD[f,d].x)*Costs2[f][d]*3 for d in D)])

#SA
# RC is reduced cost   

