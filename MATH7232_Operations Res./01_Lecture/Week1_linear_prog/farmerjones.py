from gurobipy import *

#Set
Cakes =["Chocolate", "Plain"]
Ingredients = ["Time", "Eggs", "Milk"]

C = range(len(Cakes))  #generate a list of number of Cakes
I = range(len(Ingredients))

#Data
price =[4,2]
available =[8*60, 30, 5] #Data of ingredients
usage = [
        [20, 50],   #Time
        [4,1],      #Eggs
        [0.25,0.2]  #Milk
]
        
m = Model("Farmer Jones")

#Variables
X = {} #create dictionary
for c in C:
    X[c] = m.addVar(vtype=GRB.INTEGER) # create variable: X[0] = X0, X[1]=X1
    
#Objective
m.setObjective(quicksum(price[c]*X[c] for c in C),
               GRB.MAXIMIZE) #Max = C0*X0+C1*X1

#Constraints
for i in I:
    m.addConstr(quicksum(usage[i][c]*X[c] for c in C) <= available[i])
    #a00*X0 + a11*X1

m.optimize()

for c in C:
    print("Bake", X[c].x, Cakes[c])
print("Revanue is", m.objVal)

