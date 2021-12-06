from gurobi import *

#Set
Oils =["Veg1","Veg2", "Oil1","Oil2","Oil3"]
O = range(len(Oils))

#Data
Cost =[110, 120, 130, 110, 115]
Hard =[8.8, 6.1, 2.0, 4.2, 5.0]
IsVeg =[True,True,False,False,False]
MaxVeg = 200
MaxNonVeg = 250
Sell = 150
MinH = 3
MaxH = 6

# Model        
m = Model("Oil Blending")

#Variables
X = {}
for i in O:
    X[i] = m.addVar()

#Objective
m.setObjective(quicksum((Sell-Cost[i])*X[i] for i in O), GRB.MAXIMIZE)

m.addConstr(quicksum(X[i] for i in O if IsVeg[i]) <= MaxVeg)
m.addConstr(quicksum(X[i] for i in O if not IsVeg[i]) <= MaxNonVeg)
m.addConstr(quicksum((Hard[i]-MaxH)*X[i] for i in O) <= 0)
m.addConstr(quicksum((Hard[i]-MinH)*X[i] for i in O) >= 0)

m.optimize()

print("Profit is", m.objVal)
for i in O:
    print(Oils[i], "process", round(X[i].x,2), "tonnes")
    
print("Checking hardness=", sum([Hard[i]*X[i].x for i in O])/sum([X[i].x for i in O]))

