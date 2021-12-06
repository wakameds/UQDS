#Stage: Week
#State: Condition at the beginning of day
#Action: take maintenance or not
#Value: minimum expected cost

_V = {}
def V(t,s):
    if t == 3:
        return (0,'End')
    elif (t,s) not in _V:
        if s == 'G':
            take = 0.95*(400+V(t+1,'G')[0])+0.05*(400+300+V(t+1,'B')[0])
            no_take = 0.8*(0+V(t+1, 'G')[0])+0.2*(300+V(t+1,'B')[0])       
        elif s == 'B':
            take = 0.9*(400+V(t+1,'G')[0])+0.1*(400+300+V(t+1,'B')[0])
            no_take = 0.1*(0+V(t+1, 'G')[0])+0.9*(300+V(t+1,'B')[0])
            
        _V[t,s] = min((take,'take'),(no_take,'no_take'))
    return _V[t,s]

print(V(0,'G'))