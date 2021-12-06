Sales = [14,8,17,22,12,6]

# Stages: weeks, t
# State: books on hand at the start of week, s
# Actions: boxes to purchase
# Value function: V(t,s) = maximum total profit of sales 
# given we are in week t with s books in storage at start of week.

def V(t,s):
    if t == 6:
        return (1*s - 0.5*s,0)
    else:
        best = (-100,0)
        for a in range(9):
            sold = min(Sales[t], s + 10*a)
            profit = 12*sold - 0.5*s - 50*a + V(t+1, s+10*a-sold)[0]
            if profit > best[0]:
                best = (profit,a,s+10*a-sold)
        return best

print(V(0,0))
    
# Stages: weeks, t
# State: books on hand at the start of week, s; has movie been announced?, m
# Actions: boxes to purchase
# Value function: V2(t,s,m) = maximum total profit of sales 
# given we are in week t with s books in storage at start of week and movie 
# has been announced (m=1) or not announced (m=0)

# memoisation
_V2 = {}
def V2(t,s,m):
    if t == 6:
        return (1*s - 0.5*s,0)
    else:
        if (t,s,m) not in _V2:
            best = (-100,0)
            for a in range(17):
                # if movie not announced
                sold0 = min(Sales[t], s + 10*a)
                profit0 = 12*sold0 - 0.5*s - 50*a + V2(t+1, s+10*a-sold0, 0)[0]
                
                # if movie has been announced
                sold1 = min(2*Sales[t], s + 10*a)
                profit1 = 12*sold1 - 0.5*s - 50*a + V2(t+1, s+10*a-sold1, 1)[0]
            
                if m == 1:
                    profit = profit1
                else:
                    profit = 0.3*profit1 + 0.7*profit0
                    
                if profit > best[0]:
                    best = (profit,a)
            _V2[t,s,m] = best
        return _V2[t,s,m]

print(V2(0,0,0))