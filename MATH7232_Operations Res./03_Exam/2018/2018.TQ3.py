#Stage: listen roynd
#State: skip count(s) and song type(i)
#Actipn: skip or not 
#Value: maxmise expexted motivation value
#Want V(1,5,i)

p=[0.5,0.2,0.1,0.2]
m=[10,5,2,-2]

J = range(len(p))


_V = {}
def V(t,s,i):
    if t == 12:
        return (0,"End")
    else:
        if (t,s,i) not in _V:
            if s ==0:
                listen = m[i]+sum(p[j]*V(t+1,0,j)[0] for j in J)
                skip = 0
            elif s != 0:
                listen = m[i]+sum(p[j]*V(t+1,s,j)[0] for j in J)
                skip = sum(p[j]*V(t,s-1,j)[0] for j in J)
            _V[t,s,i] = max((listen,"listen"), (skip,"skip"))
    return _V[t,s,i]

print(V(0,5,0))