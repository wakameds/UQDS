#Data
Sale = [14,8,17,22,12,6]
A = range(9)

#Stage: week(t)
#State: the number of books(s) in the beginning of week(t)
#Action: the number of purchased box(a)
#Value: maxmised profit

# Sold number of book is Demand[t], if s+a is over Sale[t]
# Otherwise "s+10a" is sold number in the stage 


_V = {}
def V(t,s):
    if t == 6:
        return (1*s-0.5*s, 0)
    elif (t,s) not in _V:
        _V[t,s] = max((12*min(Sale[t],s+10*a)-(0.5*s+50*a)+V(t+1,s+10*a-min(Sale[t],s+10*a))[0], a,s+10*a-Sale[t]) for a in A)
    return _V[t,s]
    


#State: the number of books(s) in the beginning of week(t) and announced(m)
#m = 1 if announced, otherwise m =0
#if anounced, demand become double
#Action: the number of purchased box(a)
#Value: maxmised expected profit



A2 = range(17) 
_V2 = {}
def V2(t,s,m):
    if t == 6:
        return (1*s - 0.5*s,0)
    else:
        if (t,s,m) not in _V2:
            if m ==0:
                _V2[t,s,0]=max(((0.3*(12*min(2*Sale[t],s+10*a)-(0.5*s+50*a)+V2(t+1, s+10*a-min(2*Sale[t],s+10*a),1)[0]))
                +0.7*(12*min(Sale[t],s+10*a)-(0.5*s+50*a)+V2(t+1,s+10*a-min(Sale[t],s+10*a),0)[0]), a) for a in A2)
            elif m == 1:
                _V2[t,s,1]=max((12*min(2*Sale[t],s+10*a)-(0.5*s+50*a)+V2(t+1, s+10*a-min(2*Sale[t],s+10*a),1)[0],a)for a in A2)
    return _V2[t,s,m]

         
print(V(0,0))
print(V2(0,0,0))       