#stage:month(t)
#state: enventry at the beginning of month(s)
#action: order book
#Value: minimise the cost
#want to know V(0,0)

demand=[1,1,2,2]
pcost=4
ucost=2


def V(t,s):
    if t == 3:
        if s >= demand[t]:
            return (2*s,'End')
        elif s < demand[t]:
            return (4+2*s,'End')
    else:
        if s >= demand[t]:
            order = min((4+2*s + V(t+1,s+a-demand[t])[0],a) for a in range(demand[t]-s,7))        
            non = (2*s + V(t+1,s-demand[t])[0],0)
            return min(order,non)

        elif s < demand[t]:
            return min((4+2*s + V(t+1,s+a-demand[t])[0],a) for a in range(demand[t]-s,7))                           




def V2(t,s,k):
    if t==3:
        if s>=k+demand[t]:
            return (k + 2*s,0,k+demand[t])
        else:
            return (k + 2*s + 4,k+demand[t]-s,k+demand[t])
    else:
        order = min(min((k + 2*s + 4 + V2(t+1,s+b-c,k+demand[t]-c)[0],b,c) for c in range(0,min(s+b+1,demand[t]+1)))for b in range(1,7))
        NoOrder = min((k + 2*s + V2(t+1,s-c,k+demand[t]-c)[0],0,c) for c in range(0,min(s+1,demand[t]+1)))
        return min(order,NoOrder)


print(V(0,0))
print(V2(0,0,0))