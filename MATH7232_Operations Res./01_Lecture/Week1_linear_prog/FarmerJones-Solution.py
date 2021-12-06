import matplotlib.pyplot as plt
import numpy as np

d = np.linspace(0,30,400)
x1, x2 = np.meshgrid(d,d)
plt.imshow(((20*x1+50*x2 <= 480) & (4*x1+x2<=30) & (0.25*x1+0.2*x2<=5) & (0<=x1) & (0<=x2)),
           extent=(x1.min(),x1.max(),x2.min(),x2.max()), origin="lower", cmap="Greys", alpha = 0.2)

#plot the lines defining the constraints
x1 = np.linspace(0, 30, 2000)
x3 = (480-20*x1)/50
x4 = (30-4*x1)
x5 = (5-0.25*x1)/0.2

z1 = 1000
z2 = 2500
x6 = (z1-400*x1)/200
x7 = (z2-400*x1)/200


#Make plot
plt.plot(x1, x3, label=r'Oven: $20x_1+50x_2 \leq480$')
plt.plot(x1, x4, label=r'Eggs: $4x_1+x_2 \leq 30$')
plt.plot(x1, x5, label=r'Milk: $0.25x_1+0.2x_2 \leq 5$')
#plt.plot(x1, x6, '--', color = 'k')


plt.xlim(0,35)
plt.ylim(0,35)
plt.legend(bbox_to_anchor=(0.95, 0.95), borderaxespad=0.1)
plt.xlabel(r'$x_1 (choc)$')
plt.ylabel(r'$x_2 (pound)$')
plt.annotate('Feasible region', xy=(5,5), xytext=(10,0.8), arrowprops=dict(arrowstyle='->'))
plt.show()


#%%
from gurobipy import *

#Sets
Cakes = ["Chocolate","Plain"]
Ingredients = ["Time","Eggs","Milk"]

C = range(len(Cakes))
I = range(len(Ingredients))

#Data
price = [400, 200]
available = [8*60, 30, 5]
usage = [
        [20,50],
        [4,1],
        [0.25,0.2]
        ]

m = Model("LP")

#decide variables
X = {}
for c in C:
    X[c] = m.addVar(vtype=GRB.INTEGER) #set integer

#objective function
m.setObjective(quicksum(price[c]*X[c] for c in C), GRB.MAXIMIZE)

#constraints
for i in I:
    m.addConstr(quicksum(usage[i][c]*X[c] for c in C) <= available[i])

#solver
m.optimize()

for c in C:
    print("Bake",X[c].x,Cakes[c])
print("Revenue is", m.objVal)


