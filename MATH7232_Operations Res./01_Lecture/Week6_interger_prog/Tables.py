from gurobipy import *

# The size of the area is Rows x Cols
Rows = range(7)
Cols = range(18)
# Rubbish contains any unusable squares
Rubbish = [(6,8),(6,9)]
# Are tables allowed to touch or not
TablesTouching = True

# Return a list of the neighbours of (i,j)
def Neighbours(i,j):
    nList = []
    if i > 0:
        nList.append((i-1,j))
    if i < Rows[-1]:
        nList.append((i+1,j))
    if j > 0:
        nList.append((i,j-1))
    if j < Cols[-1]:
        nList.append((i,j+1))
    return nList    

m = Model()

# x[i,j]=1 => table
x = {(i,j): 
    m.addVar(vtype=GRB.BINARY)
    for i in Rows for j in Cols}

# x[i,j]=1 => chair
y = {(i,j): 
    m.addVar(vtype=GRB.BINARY)
    for i in Rows for j in Cols}

m.setObjective(
    quicksum(y[i,j] for i in Rows for j in  Cols), 
    GRB.MAXIMIZE)

ChairMustBeBesideTable = {
    (i,j): m.addConstr(y[i,j]<=
        quicksum(x[i1,j1] for (i1,j1) in Neighbours(i,j)))
    for i in Rows for j in Cols}

ChairCantBeOnTable = {
    (i,j): m.addConstr(x[i,j]+y[i,j]<=1)
    for i in Rows for j in Cols}

RubbishSquaresEmpty = {
    (i,j): m.addConstr(x[i,j]+y[i,j]==0)
    for (i,j) in Rubbish}

if not TablesTouching:
    TablesCantTouch = {
    (i,j): m.addConstr(x[i,j]+x[i+1,j]+
                    x[i,j+1]+x[i+1,j+1]<=1)
    for i in Rows[:-1] for j in Cols[:-1]}

m.optimize()

for i in Rows:
    Line = ''
    for j in Cols:
        if x[i,j].x > 0.9:
            Line += '#'
        elif y[i,j].x > 0.9:
            Line += '.'
        else:
            Line += ' '
    print(Line)
