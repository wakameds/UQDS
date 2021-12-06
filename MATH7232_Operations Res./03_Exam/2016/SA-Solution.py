import math
import random
from gurobipy import *
import pylab

def Distance(p1, p2):
  return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

nLoc = 1500
N = range(nLoc)
Square = 1000
random.seed(nLoc)
Pos = [(random.randint(0,Square), random.randint(0,Square)) for i in N]

# D is the cost matrix
D = [[Distance(Pos[i],Pos[j]) for j in N] for i in N]

# Set up the random initial assignment
Assignment = list(N)
random.shuffle(Assignment)

# Generic SA function

def RunSA(Initial,Cost,ChooseNeigh,MoveToNeigh,T,M,alpha):
    E = Cost(Initial)
    Best = E
    CostArr = [E]
    BestArr = [Best]
    for i in range(M):
        delta,neighbour = ChooseNeigh(Initial)
        if delta < 0 or math.exp(-delta/T) > random.random():
            MoveToNeigh(Initial,neighbour)
            E += delta
            if E < Best:
                Best = E
        CostArr.append(E)
        BestArr.append(Best)
        T *= alpha
	# This code shows the progress
    print(E,T)
    pylab.plot(range(M+1),CostArr)
    pylab.plot(range(M+1),BestArr)
    pylab.show()

# Problem-specific functions

def Cost(Assignment):
    return sum(D[i][Assignment[i]] for i in N)

def ChooseNeigh(Assignment):
    i = random.choice(N)
    j = random.choice(N)
    CurrCost = D[i][Assignment[i]] + D[j][Assignment[j]]
    Assignment[i],Assignment[j] = Assignment[j],Assignment[i]
    NewCost = D[i][Assignment[i]] + D[j][Assignment[j]]
    Assignment[i],Assignment[j] = Assignment[j],Assignment[i]
    return NewCost-CurrCost,(i,j)

def MoveToNeigh(Assignment, neigh):
    i,j = neigh
    Assignment[i],Assignment[j] = Assignment[j],Assignment[i]

RunSA(Assignment,Cost,ChooseNeigh,MoveToNeigh,100000,1000000,.99995)