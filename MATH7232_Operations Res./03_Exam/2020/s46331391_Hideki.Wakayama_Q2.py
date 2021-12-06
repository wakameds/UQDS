#stage: rolling dice
#state: sum of point by the stage, color we roll
#action: roll the color dice
#value: maximise the prob for win

_V = {}
def V(t,s,i):
    if t==50:
        if s > 0:
            return (1,"Win")
        elif s<=0:
            return (0,"lose")
    elif (t,s,i) not in _V:
        if i == "red":
            nextred = 0.5*V(t+1,s+1,"red")[0]+0.5*V(t+1,s-1,"red")[0]
            nextblue = 0.5*V(t+1,s+1,"blue")[0]+0.5*V(t+1,s-1,"blue")[0]
        elif i == "blue":
            nextred = 1/3*V(t+1,s+4,"red")[0]+2/3*V(t+1,s-2,"red")[0]
            nextblue = 1/3*V(t+1,s+4,"blue")[0]+2/3*V(t+1,s-2,"blue")[0]
        _V[t,s,i]=max((nextred,'next red'),(nextblue,'next blue'))
    return _V[t,s,i]


print("choice red",V(0,0,"red"))
print("chice blue",V(0,0,"blue"))


