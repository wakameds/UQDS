from gurobipy import *

Grid = [
[0,0,0, 1,0,0, 0,0,0],
[0,2,4, 0,5,0, 0,0,0],
[0,0,0, 0,8,0, 3,7,5],

[9,0,0, 0,0,0, 4,0,0],
[0,7,0, 0,0,0, 0,3,0],
[0,0,2, 0,0,0, 0,0,8],

[1,5,8, 0,9,0, 0,0,0],
[0,0,0, 0,6,0, 9,1,0],
[0,0,0, 0,0,3, 0,0,0]]

N = range(9)
K = range(1,10)
m = Model()

X = {(i,j,k): m.addVar(vtype=GRB.BINARY) for i in N for j in N for k in K}

PreAssign = {
    (i,j): m.addConstr(X[i,j,Grid[i][j]]==1)
    for i in N for j in N if Grid[i][j]>0}
OnePerSquare = {
    (i,j): m.addConstr(quicksum(X[i,j,k] for k in K)==1)
    for i in N for j in N}
EachValueInRow = {
    (i,k): m.addConstr(quicksum(X[i,j,k] for j in N)==1)
    for i in N for k in K}        
EachValueInCol = {
    (j,k): m.addConstr(quicksum(X[i,j,k] for i in N)==1)
    for j in N for k in K}        
EachValueInSubSquares = {
    (ii,jj,k): m.addConstr(quicksum(X[i,j,k] 
                        for i in range(3*ii,3*ii+3)
                        for j in range(3*jj,3*jj+3))==1)
    for ii in range(3) for jj in range(3) for k in K}

m.optimize()

print('---+---+---')
for i in N:
    if i==3 or i==6:
        print('---+---+---')
    for j in N:
        if j==3 or j==6:
            print('|', end='')
        for k in K:
            if X[i,j,k].x > 0.9:
                print(k,sep='',end='')
    print('')
print('---+---+---')
