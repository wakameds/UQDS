#Stage: month
#Data: d
#Data
Demands = [1,2,3,3,2,2]
stock = 0.1 #transfer cost
order = 20 #cost per order
order2 = 2 #order cost per unit


# State: stock amount at t
# Action: order book (a)
# Value: minimaze cost


_V ={}
def V(t,s):
    if t == 5:
        if s >= 2:
            return ((s-Demands[t])*1,0)
        else:
            a = Demands[t]-s
            return ((s+a-Demands[t])*1 + 20 + a*20,a)
        
    elif (t,s) not in _V:
        best = [1000,0] 
        for a in range(max(0,Demands[t]-s),7):
            c = 1*s
            if a>0:
                c = c+ 20 + 20*a
            c = c + V(t+1, s+a-Demands[t])[0]

            if c < best[0]:
                best = c, a
        return best

