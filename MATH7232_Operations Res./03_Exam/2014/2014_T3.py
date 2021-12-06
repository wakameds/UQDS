#Stage: stamps
#State: value we can purchase(s)
#action: buy stamp
#value minimise the number of stamps

price = [12,7,1]

_P = {}
def stamp(t,s):
    if t == 2:
        return (s/price[2],0)
    elif (t,s) not in _P:
        _P[t,s] = min((a + stamp(t+1, s-price[t]*a)[0],a) for a in range(int(s/price[t])+1))
    return _P[t,s]
    
print(stamp(0,14))


#stage: game round
#state: money(s)
#action: bet(a)
#Value: maximise probability at least $5

def bet(t,s):
    if t == 3:
        if s>=5:
            return (1,"Good")
        elif s<5:
            return (0,'fail')
    else:
        return max((0.4*bet(t+1,s+a)[0]+0.6*bet(t+1,s-a)[0],a) for a in range(s+1))

print(bet(0,2))
    
    
    